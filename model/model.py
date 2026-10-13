"""Dual-branch visual framework: ResNet / ViT -> FiLM -> Adapter/MHA -> gamma gate -> classifier."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .encoders import ResNet3DEncoder,ViT3DEncoder,ViT3DSemanticTower

class TokenAdapter(nn.Module):
    """Compress semantic patch tokens before cross-attention (implementation of paper's Adapter).

    Adaptive 1D pooling over raster-ordered patch tokens is an explicit implementation
    choice: the manuscript does not specify the Adapter's precise downsampling operator.
    """
    def __init__(self, dim=512, num_tokens=16):
        super().__init__()
        self.pool=nn.AdaptiveAvgPool1d(num_tokens)
        self.proj=nn.Sequential(nn.Linear(dim,dim),nn.LayerNorm(dim),nn.ReLU(inplace=True))

    def forward(self,tokens):
        return self.proj(self.pool(tokens.transpose(1,2)).transpose(1,2))

class DualBranchNoduleNet(nn.Module):
    """No text input is needed for `forward` or inference."""
    def __init__(self,vol_size=48,morph_depth=101,morph_base_channels=64,sem_patch=8,
                 sem_dim=256,sem_depth=12,sem_heads=8,embed_dim=512,film_hidden=256,
                 xattn_heads=8,adapter_tokens=16,cls_hidden=256,
                 use_film=True,use_mha=True,use_gamma=True):
        super().__init__()
        self.morphology=ResNet3DEncoder(depth=morph_depth,base_channels=morph_base_channels)
        vit=ViT3DEncoder(vol_size=vol_size,patch_size=sem_patch,dim=sem_dim,depth=sem_depth,num_heads=sem_heads)
        self.semantic=ViT3DSemanticTower(vit,embed_dim=embed_dim,use_mlp=True)
        self.use_film,self.use_mha,self.use_gamma=use_film,use_mha,use_gamma
        self.global_projector=nn.Sequential(nn.Linear(embed_dim,embed_dim),nn.LayerNorm(embed_dim),nn.ReLU(inplace=True))
        self.film=nn.Sequential(nn.Linear(embed_dim,film_hidden),nn.ReLU(inplace=True),nn.Linear(film_hidden,2*self.morphology.out_channels))
        self.morph_projector=nn.Linear(self.morphology.out_channels,embed_dim)
        self.adapter=TokenAdapter(embed_dim,adapter_tokens)
        self.attention=nn.MultiheadAttention(embed_dim,xattn_heads,batch_first=True)
        self.gamma_generator=nn.Sequential(nn.Linear(embed_dim,128),nn.ReLU(inplace=True),nn.Linear(128,1),nn.Sigmoid())
        self.refine=nn.Sequential(nn.Linear(embed_dim,embed_dim),nn.ReLU(inplace=True),nn.LayerNorm(embed_dim))
        self.classifier=nn.Sequential(nn.Linear(embed_dim,cls_hidden),nn.ReLU(inplace=True),nn.Dropout(.2),nn.Linear(cls_hidden,1))
        self.logit_scale=nn.Parameter(torch.tensor(math.log(1/.07)))

    def forward(self,volume,return_gate=False):
        morphology=self.morphology(volume)
        sem_cls,sem_tokens=self.semantic(volume)
        if self.use_film:
            alpha,beta=self.film(self.global_projector(sem_cls)).chunk(2,dim=-1)
            morphology=morphology*(1.+alpha)+beta
        query=self.morph_projector(morphology)
        gate=torch.ones(query.shape[0],1,device=query.device,dtype=query.dtype)
        if self.use_mha:
            patches=self.adapter(sem_tokens)
            evidence,_=self.attention(query.unsqueeze(1),patches,patches,need_weights=False)
            evidence=evidence.squeeze(1)
            if self.use_gamma: gate=self.gamma_generator(query)
            # DO NOT normalize evidence after scaling: this would nullify a scalar gate.
            query=query+gate*evidence
        fused=self.refine(query)
        logits=self.classifier(fused).squeeze(-1)
        if return_gate: return logits,fused,sem_cls,gate
        return logits,fused,sem_cls
