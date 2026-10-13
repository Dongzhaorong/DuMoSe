# Dual-Branch Complementary Visual Framework for Pulmonary Nodule Malignancy Prediction


## Files

| File                      | Responsibility |
|---------------------------| --- |
| `module/model.py`         | Dual-branch network, FiLM, Adapter, MHA and γ-Generator |
| `module/encoders.py`      | 3D ResNet and 3D ViT |
| `module/text.py`          | Frozen CXR-BERT, trainable dimensional adapter |
| `module/preprocessing.py` | CT ROI loading, HU windowing, spacing and centre crop/pad |
| `module/augment.py`       | Training-only flips, rotations and gamma jitter |
| `module/data.py`          | CSV metadata, uncertainties, datasets and batching |
| `module/splits.py`        | Stratified 5-fold and patient-disjoint train/val/test splits |
| `module/objectives.py`    | Symmetric InfoNCE and heteroscedastic loss |
| `module/metrics.py`       | ACC, precision, recall, specificity, F1, AUC |
| `train.py`                | Training, fold checkpoints, test predictions and summary |
| `predict.py`              | CT-only inference on a nodule-centred ROI |
| `configs/paper.json`      | Settings from the manuscript plus explicit implementation choices |
| `tests/`                  | Lightweight model/data/loss/split tests |

## Setup

Python ≥ 3.10; install PyTorch appropriate for your GPU first. Then:

```bash
pip install -r requirements.txt
pytest -q
```

`transformers` retrieves a pretrained model only when requested. If your local CXR-BERT requires custom remote code, review that code **before** enabling `--trust_remote_code`; otherwise leave it disabled. No pretrained weights, private data or internal paths are bundled.

## Expected metadata

Extract a **nodule-centred ROI** first. For example, in `data/rois/` each `VolumeName` points to `.nii.gz`, `.nii`, `.mha`, `.mhd`, or `.npy` in ZYX axis convention (for `.npy`); `.npy` intensities must be **raw HU**, not values previously normalized to [0,1]. The CSV should include:

```csv
VolumeName,Grow,Text,PatientID,var_subtlety,var_margin,var_spiculation
LIDC-IDRI-0001_n1.npy,0,"...description...",LIDC-IDRI-0001,0.25,0.50,0.00
```

- `Grow`: **0 benign, 1 malignant**; prepare these labels and ROI crops **externally**, consistent with the manuscript's filtering protocol.
- `Text`: standardized clinical description of **the target nodule only**; exclude diagnosis/pathology indicators to prevent label leakage.
- `PatientID`: strongly recommended. All nodules from one patient are restricted to the same partition. If missing, a name-based heuristic is used; check its output before relying on it.
- Public LIDC: `var_*` per-reader **attribute variance** (not standard deviation, and exclude `var_malignancy`). The mean square root across available attributes yields uncertainty.
- Private clinical reports: instead of `var_*`, use `missing_fields`, i.e. number of unknown descriptive fields in the standardized report (must be calculated beforehand); never blend the two uncertainty schemes in a dataset.
- Fold export files also contain an `uncertainty` column with the **already-computed raw uncertainty score**; this can be read back, although using the original attribute variances is preferred.
- `Text` and uncertainty are required for the paper-aligned alignment run, although a vision-only training mode is available.

## Train and test

```bash
python train.py --config configs/paper.json \
  --root_dir /path/to/roi_volumes --data_csv /path/to/metadata.csv \
  --cxr_bert_path /path/to/BiomedVLP-CXR-BERT-specialized \
  --local_files_only --out_dir runs/lidc
```

The paper-aligned configuration uses ResNet-101, 12-layer ViT, AdamW, `lr=1e-4`, batch size 32, 150 epochs and patient-grouped 5-fold CV. CXR-BERT weights must be provided by the user. Training from scratch on full 3D ROIs requires substantial GPU memory; decrease batch size if necessary and disclose the deviation.

For a vision-only ablation (no CXR-BERT dependency):

```bash
python train.py --config configs/paper.json --root_dir /path/to/roi_volumes \
  --data_csv /path/to/metadata.csv --align_weight 0 --uncertainty_mode none
```

For a prediction with a best fold's weights:

```bash
python predict.py --volume /path/to/nodule.nii.gz \
  --checkpoint runs/lidc/fold_01/best.pt
```

Each fold saves `best.pt`, `history.csv`, the three split CSVs, `test_predictions.csv` and `test_metrics.json`. Overall mean/standard deviation is in `cv_summary.json`. The test set is evaluated only after selecting a checkpoint using validation F1; **no test-set threshold selection** is performed.



## License

No license is assigned automatically. Choose and add a license only after confirming co-author, institutional and third-party code requirements.
