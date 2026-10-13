import numpy as np
import torch
from nodule.model import DualBranchNoduleNet
from nodule.objectives import alignment_loss,uncertainty_to_variance
from nodule.splits import stratified_group_kfold_split,assert_no_patient_overlap
from nodule.preprocessing import center_crop_or_pad,clip_and_normalize_hu


def tiny_model():
    return DualBranchNoduleNet(vol_size=16,morph_depth=18,morph_base_channels=8,
      sem_patch=8,sem_dim=32,sem_depth=1,sem_heads=4,embed_dim=64,
      film_hidden=32,xattn_heads=4,adapter_tokens=4)


def test_forward_and_gate_has_gradient():
    torch.manual_seed(9)
    model=tiny_model().train()
    x=torch.randn(2,1,16,16,16)
    logits,fused,cls,gate=model(x,return_gate=True)
    assert logits.shape==(2,) and fused.shape==(2,64) and cls.shape==(2,64)
    assert gate.shape==(2,1)
    logits.sum().backward()
    assert model.gamma_generator[0].weight.grad is not None
    assert model.gamma_generator[0].weight.grad.abs().sum()>0


def test_uncertainty_loss():
    v=torch.randn(4,32,requires_grad=True)
    t=torch.randn(4,32)
    u=torch.tensor([0.,.5,1.,2.])
    sigma=uncertainty_to_variance(u,2.,1.,10.)
    assert (sigma[1:]>=sigma[:-1]).all()
    loss=alignment_loss(v,t,torch.tensor(1.),sigma)
    assert torch.isfinite(loss)
    loss.backward();assert v.grad is not None


def test_splits_do_not_leak_patients():
    y=np.array([0,0,0,0,1,1,1,1,0,1,0,1])
    ids=np.array([f'P{i//2}' for i in range(12)])
    splits=stratified_group_kfold_split(y,ids,n_splits=2,seed=42)
    for tv,te in splits:
        assert not (set(ids[tv]) & set(ids[te]))


def test_image_normalization_and_shape():
    x=np.random.default_rng(0).uniform(-1500,700,(14,18,15)).astype('float32')
    result=center_crop_or_pad(clip_and_normalize_hu(x),(16,16,16))
    assert result.shape==(16,16,16) and result.min()>=0 and result.max()<=1
