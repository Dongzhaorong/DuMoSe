"""CSV loading, uncertainty extraction and CT-text dataset."""
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import preprocess_roi
from .augment import augment_roi

@dataclass(frozen=True)
class NoduleSample:
    filename: str
    label: int
    text: str
    uncertainty: float | None
    patient_id: str


def read_metadata(path):
    """Columns: VolumeName, Grow, Text, PatientID, var_* OR missing_fields.

    var_* represents inter-reader variance, excluding malignancy.
    missing_fields is the number of 'unknown' descriptive fields in private reports.
    The two uncertainty sources should not be mixed within one dataset.
    """
    from .splits import infer_patient_id
    rows=[]
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or 'VolumeName' not in reader.fieldnames:
            raise ValueError('CSV must contain VolumeName')
        label_key='Grow' if 'Grow' in reader.fieldnames else 'label'
        if label_key not in reader.fieldnames:
            raise ValueError('CSV must contain Grow or label')
        var_fields=[key for key in reader.fieldnames if key.lower().startswith('var_') and 'malig' not in key.lower()]
        if var_fields and 'missing_fields' in reader.fieldnames:
            raise ValueError('Choose var_* or missing_fields for uncertainty, not both.')
        for row in reader:
            name=(row.get('VolumeName') or '').strip()
            if not name: continue
            raw_label=float(row[label_key])
            if raw_label not in (0.,1.): raise ValueError(f'Labels must be 0/1: {name}')
            label=int(raw_label)
            uncertainty=None
            if var_fields:
                vals=[float(row[k]) for k in var_fields if row.get(k) not in ('',None)]
                if vals:
                    if any(not math.isfinite(v) or v<0 for v in vals):
                        raise ValueError(f'Invalid variance: {name}')
                    uncertainty=float(np.sqrt(vals).mean())
            elif row.get('uncertainty') not in ('',None):
                uncertainty=float(row['uncertainty'])
                if not math.isfinite(uncertainty) or uncertainty<0:
                    raise ValueError(f'Invalid uncertainty: {name}')
            elif row.get('missing_fields') not in ('',None):
                uncertainty=float(row['missing_fields'])
                if uncertainty<0: raise ValueError(f'Negative missing field count: {name}')
            pid=(row.get('PatientID') or '').strip() or infer_patient_id(name)
            rows.append(NoduleSample(name,label,(row.get('Text') or '').strip(),uncertainty,pid))
    if not rows: raise ValueError('No cases in CSV')
    return rows


class NoduleDataset(Dataset):
    def __init__(self, root, records, train=False, vol_size=48, iso_spacing=1.0, use_spacing=True):
        self.root=Path(root)
        self.records=list(records)
        self.train=bool(train)
        self.vol_size=vol_size
        self.iso_spacing=iso_spacing
        self.use_spacing=use_spacing

    def __len__(self): return len(self.records)

    def _path(self, name):
        path=self.root/name
        if path.exists(): return path
        for ext in ('.nii.gz','.nii','.npy','.mha','.mhd'):
            path=self.root/(name+ext)
            if path.exists(): return path
        raise FileNotFoundError(f'Missing ROI for {name} under {self.root}')

    def __getitem__(self, index):
        rec=self.records[index]
        vol=preprocess_roi(self._path(rec.filename),vol_size=self.vol_size,
              iso_spacing=self.iso_spacing,use_spacing=self.use_spacing)
        if self.train: vol=augment_roi(vol)
        return (torch.from_numpy(np.ascontiguousarray(vol[None],dtype=np.float32)),
                torch.tensor(float(rec.label)), rec.text, rec.filename,
                torch.tensor(float('nan') if rec.uncertainty is None else rec.uncertainty))


def collate_nodules(batch):
    volumes,labels,texts,names,uncertainties=zip(*batch)
    return torch.stack(volumes),torch.stack(labels),list(texts),list(names),torch.stack(uncertainties)
