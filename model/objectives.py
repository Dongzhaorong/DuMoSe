"""Symmetric InfoNCE with paper Eq. (6)-(11) heteroscedastic correction."""
import torch
from torch.nn import functional as F

def uncertainty_to_variance(u, u95, sigma2_min=1., sigma2_max=10.):
    if sigma2_min < 1 or sigma2_max < sigma2_min:
        raise ValueError('Require 1 <= sigma2_min <= sigma2_max')
    if not torch.isfinite(u).all():
        raise ValueError('Uncertainty contains missing/NaN values; provide var_* or missing_fields')
    normalized=torch.clamp(u.float()/max(float(u95),1e-6),min=0.,max=1.)
    return sigma2_min+(sigma2_max-sigma2_min)*normalized

def alignment_loss(image_features,text_features,logit_scale,sigma2=None):
    if image_features.size(0)<2: return image_features.sum()*0
    v=F.normalize(image_features.float(),dim=-1)
    t=F.normalize(text_features.float(),dim=-1)
    logits=logit_scale.exp().clamp(max=100)*(v@t.T)
    labels=torch.arange(len(v),device=v.device)
    loss=0.5*(F.cross_entropy(logits,labels,reduction='none')+F.cross_entropy(logits.T,labels,reduction='none'))
    if sigma2 is not None:
        sigma2=sigma2.to(loss.device).float().flatten()
        loss=loss/sigma2+torch.log(sigma2)
    return loss.mean()
