"""Predict one already-cropped 3D nodule from a saved fold checkpoint (CT only)."""
import argparse
from pathlib import Path
import numpy as np
import torch
from nodule.model import DualBranchNoduleNet
from nodule.preprocessing import preprocess_roi


def predict(volume_path,checkpoint,iso_spacing=1.0):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=DualBranchNoduleNet(**payload['model_config']).to(device)
    model.load_state_dict(payload['model'],strict=True)
    model.eval()
    image=preprocess_roi(volume_path,vol_size=payload['model_config']['vol_size'],iso_spacing=iso_spacing)
    x=torch.from_numpy(np.ascontiguousarray(image[None,None])).to(device)
    with torch.no_grad():
        logits,_,_=model(x)
        prob=float(torch.sigmoid(logits).item())
    threshold=float(payload.get('threshold',0.5))
    return prob,int(prob>=threshold),threshold

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--volume',required=True,help='A target-nodule-centred ROI (.nii.gz/.nii/.mha/.mhd/.npy)')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--iso_spacing',type=float,default=1.0)
    args=parser.parse_args()
    p,label,thr=predict(args.volume,args.checkpoint,args.iso_spacing)
    print(f'Malignancy probability: {p:.6f} | predicted label: {label} | threshold: {thr:.3f}')
