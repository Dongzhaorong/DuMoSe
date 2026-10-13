"""Five-fold CV with training-only text guidance and patient-disjoint splits."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from nodule.data import NoduleDataset, read_metadata, collate_nodules
from nodule.splits import (stratified_group_kfold_split,stratified_group_trainval_split,
                           assert_no_patient_overlap,stratified_kfold_split,stratified_trainval_split)
from nodule.metrics import compute_binary_metrics
from nodule.model import DualBranchNoduleNet
from nodule.objectives import alignment_loss, uncertainty_to_variance
from nodule.utils import seed_everything,save_config

MODEL_KEYS=('vol_size','morph_depth','morph_base_channels','sem_patch','sem_dim',
            'sem_depth','sem_heads','embed_dim','film_hidden','xattn_heads','adapter_tokens',
            'use_film','use_mha','use_gamma')


def get_args():
    p=argparse.ArgumentParser(description='Paper-oriented pulmonary nodule 5-fold training')
    p.add_argument('--config',default='configs/paper.json')
    pre,_=p.parse_known_args()
    config=json.loads(Path(pre.config).read_text(encoding='utf-8'))
    p.set_defaults(**config)
    p.add_argument('--root_dir',required=True,help='Pre-extracted nodule ROI directory')
    p.add_argument('--data_csv',required=True)
    p.add_argument('--out_dir',default='runs/paper')
    p.add_argument('--cxr_bert_path',default=None,help='Local CXR-BERT directory or HF model ID')
    p.add_argument('--text_max_len',type=int,default=512)
    p.add_argument('--local_files_only',action='store_true')
    p.add_argument('--trust_remote_code',action='store_true',help='Only enable after reviewing remote model code')
    p.add_argument('--amp',action='store_true')
    p.add_argument('--pos_weight_auto',action='store_true',help='Optional weighted BCE, not paper default')
    p.add_argument('--epochs',type=int)
    p.add_argument('--batch_size',type=int)
    p.add_argument('--folds',type=int)
    p.add_argument('--num_workers',type=int)
    p.add_argument('--lr',type=float)
    p.add_argument('--seed',type=int)
    p.add_argument('--align_weight',type=float)
    p.add_argument('--uncertainty_mode',choices=['heteroscedastic','none'])
    p.add_argument('--threshold',type=float)
    p.add_argument('--patient_level_split',action=argparse.BooleanOptionalAction)
    args=p.parse_args()
    if args.align_weight>0 and not args.cxr_bert_path:
        p.error('For image-text alignment supply --cxr_bert_path (no bundled weights).')
    if args.align_weight>0 and args.batch_size<2:
        p.error('InfoNCE requires at least two image-text pairs in a batch.')
    return args


def make_loader(root,records,args,train=False):
    dataset=NoduleDataset(root,records,train=train,vol_size=args.vol_size,iso_spacing=args.iso_spacing)
    return DataLoader(dataset,batch_size=args.batch_size,shuffle=train,
                      drop_last=train,collate_fn=collate_nodules,
                      num_workers=args.num_workers,pin_memory=torch.cuda.is_available())


def build_model(args,device):
    return DualBranchNoduleNet(**{key:getattr(args,key) for key in MODEL_KEYS}).to(device)


def collect_predictions(model,loader,device,criterion=None):
    model.eval()
    names,truth,probabilities=[],[],[]
    losses=[]
    with torch.no_grad():
        for x,y,_texts,batch_names,_u in loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            logits,_,_=model(x)
            if criterion is not None: losses.append((float(criterion(logits,y).item()),len(y)))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            truth.extend(y.int().cpu().tolist())
            names.extend(batch_names)
    metrics=compute_binary_metrics(truth,probabilities)
    loss=sum(v*n for v,n in losses)/max(1,sum(n for _,n in losses))
    return names,np.asarray(truth),np.asarray(probabilities),metrics,loss


def train_epoch(model,text_encoder,loader,optimizer,device,criterion,args,u95,scaler):
    model.train()
    if text_encoder is not None: text_encoder.train()
    total,bce_sum,align_sum=0,0.,0.
    amp_enabled=bool(args.amp and device.type=='cuda')
    for x,y,texts,_names,u in loader:
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        with torch.autocast('cuda',enabled=amp_enabled) if amp_enabled else torch.autocast('cpu',enabled=False):
            logits,_fused,visual=model(x)
            bce=criterion(logits.float(),y)
        align=bce.new_zeros(())
        if args.align_weight>0:
            valid=[i for i,txt in enumerate(texts) if txt.strip()]
            if len(valid)>=2:
                selected=torch.tensor(valid,dtype=torch.long,device=device)
                txt=text_encoder([texts[i] for i in valid],device)
                sigma2=None
                if args.uncertainty_mode=='heteroscedastic':
                    sigma2=uncertainty_to_variance(u[selected.cpu()].to(device),u95,
                                            args.sigma2_min,args.sigma2_max)
                align=alignment_loss(visual.index_select(0,selected),txt,model.logit_scale,sigma2)
        loss=bce+args.align_weight*align
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        else:
            loss.backward();optimizer.step()
        total+=len(y);bce_sum+=float(bce.detach())*len(y);align_sum+=float(align.detach())*len(y)
    return {'bce':bce_sum/max(total,1),'align':align_sum/max(total,1)}


def save_csv(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['VolumeName','Grow','Text','PatientID','uncertainty'])
        for rec in records: w.writerow([rec.filename,rec.label,rec.text,rec.patient_id,rec.uncertainty])


def save_predictions(path,names,truth,probs,threshold):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['VolumeName','y_true','y_prob','y_pred'])
        for name,y,p in zip(names,truth,probs):w.writerow([name,int(y),float(p),int(p>=threshold)])


def main():
    args=get_args()
    seed_everything(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    save_config(out/'run_config.json',vars(args))
    records=read_metadata(args.data_csv)
    if args.align_weight>0 and any(not rec.text for rec in records):
        raise ValueError('Paper-aligned text guidance requires non-empty Text for each sample.')
    labels=np.array([r.label for r in records],dtype=int)
    groups=np.array([r.patient_id for r in records])
    if args.patient_level_split:
        folds=stratified_group_kfold_split(labels,groups,args.folds,args.seed)
    else:
        folds=stratified_kfold_split(labels,args.folds,args.seed)
    results=[]
    for fold,(tv_idx,test_idx) in enumerate(folds,1):
        seed_everything(args.seed+fold*1000)
        if args.patient_level_split:
            tr_idx,val_idx=stratified_group_trainval_split(tv_idx,labels,groups,args.val_ratio,args.seed+fold*1000)
            assert_no_patient_overlap(groups,tr_idx,val_idx,test_idx)
        else:
            tr_idx,val_idx=stratified_trainval_split(tv_idx,labels,args.val_ratio,args.seed+fold*1000)
        train_records=[records[i] for i in tr_idx]
        val_records=[records[i] for i in val_idx]
        test_records=[records[i] for i in test_idx]
        if min(len(train_records),len(val_records),len(test_records))==0:
            raise ValueError(f'Fold {fold} has empty split; more patient groups are required.')
        if args.uncertainty_mode=='heteroscedastic' and args.align_weight>0:
            train_u=np.array([r.uncertainty for r in train_records],dtype=float)
            if not np.all(np.isfinite(train_u)):
                raise ValueError('Uncertainty-aware alignment needs var_* or missing_fields for every TRAIN row')
            u95=max(float(np.quantile(train_u,.95)),1e-6)  # TRAIN only
        else: u95=1.
        dir_fold=out/f'fold_{fold:02d}';dir_fold.mkdir(parents=True,exist_ok=True)
        for split,data in [('train',train_records),('val',val_records),('test',test_records)]:
            save_csv(dir_fold/f'{split}_split.csv',data)
        loaders={split:make_loader(args.root_dir,data,args,train=(split=='train'))
                 for split,data in [('train',train_records),('val',val_records),('test',test_records)]}
        if len(loaders['train'])==0:
            raise ValueError('Training split too small for batch size; reduce --batch_size.')
        model=build_model(args,device)
        text=None
        if args.align_weight>0:
            from nodule.text import CXRBertTextEncoder
            text=CXRBertTextEncoder(args.cxr_bert_path,embed_dim=args.embed_dim,
                 max_length=args.text_max_len,local_files_only=args.local_files_only,
                 trust_remote_code=args.trust_remote_code).to(device)
        params=list(model.parameters())+([] if text is None else [p for p in text.parameters() if p.requires_grad])
        optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay)
        if args.pos_weight_auto:
            labels_train=np.array([r.label for r in train_records]);pw=(labels_train==0).sum()/max((labels_train==1).sum(),1)
            criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pw),device=device))
        else: criterion=nn.BCEWithLogitsLoss()
        scaler=torch.amp.GradScaler('cuda',enabled=(args.amp and device.type=='cuda'))
        best_f1=-1.
        with (dir_fold/'history.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['epoch','train_bce','train_align','val_bce','val_f1','val_auc','val_acc']);writer.writeheader()
            for epoch in range(1,args.epochs+1):
                tr=train_epoch(model,text,loaders['train'],optimizer,device,criterion,args,u95,scaler)
                _n,_y,_p,vm,vl=collect_predictions(model,loaders['val'],device,criterion)
                writer.writerow(dict(epoch=epoch,train_bce=tr['bce'],train_align=tr['align'],
                        val_bce=vl,val_f1=vm['f1'],val_auc=vm['auc'],val_acc=vm['acc']))
                f.flush()
                if vm['f1']>best_f1:
                    best_f1=vm['f1']
                    torch.save({'model':model.state_dict(),
                                'text_adapter':None if text is None else text.adapter.state_dict(),
                                'model_config':{key:getattr(args,key) for key in MODEL_KEYS},
                                'threshold':args.threshold,'epoch':epoch,'u95':u95,
                                'val_f1':best_f1},dir_fold/'best.pt')
                print(f'fold {fold}/{args.folds} epoch {epoch}/{args.epochs} train_bce={tr["bce"]:.4f} val_f1={vm["f1"]:.4f} val_auc={vm["auc"]:.4f}',flush=True)
        # Load best validation model and evaluate test ONCE.
        ckpt=torch.load(dir_fold/'best.pt',map_location=device,weights_only=True)
        model.load_state_dict(ckpt['model'])
        names,truth,prob,metrics,loss=collect_predictions(model,loaders['test'],device,criterion)
        save_predictions(dir_fold/'test_predictions.csv',names,truth,prob,args.threshold)
        (dir_fold/'test_metrics.json').write_text(json.dumps(dict(metrics,test_bce=loss,best_epoch=ckpt['epoch']),indent=2),encoding='utf-8')
        results.append(dict(fold=fold,**metrics))
        print(f'fold {fold} TEST: {metrics}')
        del model,text,optimizer,loaders
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    with (out/'cv_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['fold','acc','prec','recall','spe','f1','auc','tp','tn','fp','fn'])
        w.writeheader();w.writerows(results)
    summary={key:{'mean':float(np.nanmean([r[key] for r in results])),
                  'std':float(np.nanstd([r[key] for r in results],ddof=1)) if len(results)>1 else 0.}
             for key in ('acc','prec','recall','spe','f1','auc')}
    (out/'cv_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Saved CV results:',out)

if __name__=='__main__':main()
