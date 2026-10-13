"""Repeatable seeds and experiment metadata."""
import os,random,json,sys,platform
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def save_config(path,config):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    payload=dict(config=config,python=sys.version,platform=platform.platform(),
                  torch=torch.__version__,created_at=datetime.now().isoformat())
    Path(path).write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
