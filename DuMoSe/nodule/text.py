"""CXR-BERT for image-text alignment during training only."""
import torch
from torch import nn
from torch.nn import functional as F

class CXRBertTextEncoder(nn.Module):
    """Frozen pretrained language backbone and trainable dimension adapter.

    The adapter is required when BERT hidden size differs from the shared 512-D space.
    Training a random frozen projection would produce non-semantic embeddings.
    """
    def __init__(self, model_name_or_path, embed_dim=512, max_length=512, local_files_only=False, trust_remote_code=False):
        super().__init__()
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError('Install transformers to train with CXR-BERT') from exc
        self.tokenizer=AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only,trust_remote_code=trust_remote_code)
        self.backbone=AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only,trust_remote_code=trust_remote_code)
        for p in self.backbone.parameters(): p.requires_grad_(False)
        self.backbone.eval()
        self.max_length=int(max_length)
        self.adapter=nn.Linear(int(self.backbone.config.hidden_size),embed_dim)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, texts, device):
        batch=self.tokenizer(texts,return_tensors='pt',padding=True,truncation=True,max_length=self.max_length)
        batch={k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            hidden=self.backbone(**batch).last_hidden_state[:,0,:]
        return F.normalize(self.adapter(hidden.float()),dim=-1)
