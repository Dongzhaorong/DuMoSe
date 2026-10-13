"""Patient-grouped stratified CV; no subject appears in different subsets."""
import re
from pathlib import Path
import numpy as np

def infer_patient_id(filename):
    """Only safe by default for recognizable LIDC naming. Otherwise supply PatientID."""
    base=Path(filename).name
    found=re.search(r'LIDC-IDRI-\d+',base,re.IGNORECASE)
    if found: return found.group(0).upper()
    return base.split('_')[0]

def _unique_in_order(arr):
    seen = set()
    out = []
    for x in arr:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return np.asarray(out, dtype=object)

def _patient_group_labels(y: np.ndarray, patient_ids: np.ndarray, unique_pids: np.ndarray):
    """Patient-level label for stratification only.

    If a patient has any positive nodule, this patient is treated as positive
    for stratified splitting. This does NOT change nodule-level labels.
    """
    y = np.asarray(y).astype(int)
    patient_ids = np.asarray(patient_ids).astype(str)

    group_y = []
    mixed_pids = []

    for pid in unique_pids:
        idx = np.where(patient_ids == str(pid))[0]
        vals = y[idx]

        if vals.min() != vals.max():
            mixed_pids.append(str(pid))

        group_y.append(int(vals.max()))

    if len(mixed_pids) > 0:
        print(
            f"[Warning] {len(mixed_pids)} patients have mixed nodule labels. "
            f"Patient-level stratification uses max label per patient. "
            f"Examples: {mixed_pids[:10]}"
        )

    return np.asarray(group_y, dtype=int)

def stratified_group_kfold_split(y: np.ndarray,
                                 patient_ids: np.ndarray,
                                 n_splits: int,
                                 seed: int):
    """Patient-level stratified K-fold.

    All nodules from the same patient are assigned to the same fold.
    """
    y = np.asarray(y).astype(int)
    patient_ids = np.asarray(patient_ids).astype(str)
    rng = np.random.RandomState(seed)

    unique_pids = _unique_in_order(patient_ids)
    group_y = _patient_group_labels(y, patient_ids, unique_pids)

    pos_pids = unique_pids[group_y == 1].copy()
    neg_pids = unique_pids[group_y == 0].copy()

    if len(pos_pids) < n_splits or len(neg_pids) < n_splits:
        print(
            f"[Warning] positive patients={len(pos_pids)}, negative patients={len(neg_pids)}, "
            f"folds={n_splits}. Some folds may contain only one class."
        )

    rng.shuffle(pos_pids)
    rng.shuffle(neg_pids)

    pos_folds = np.array_split(pos_pids, n_splits)
    neg_folds = np.array_split(neg_pids, n_splits)

    folds = []
    all_idx = np.arange(len(y), dtype=int)

    for k in range(n_splits):
        test_pids = set(
            np.concatenate([pos_folds[k], neg_folds[k]])
            .astype(str)
            .tolist()
        )

        test_mask = np.asarray([pid in test_pids for pid in patient_ids], dtype=bool)
        test_idx = all_idx[test_mask].astype(int)
        trainval_idx = all_idx[~test_mask].astype(int)

        rng.shuffle(test_idx)
        rng.shuffle(trainval_idx)

        folds.append((trainval_idx, test_idx))

    return folds

def stratified_group_trainval_split(trainval_idx: np.ndarray,
                                    y: np.ndarray,
                                    patient_ids: np.ndarray,
                                    val_ratio: float,
                                    seed: int):
    """Patient-level stratified train/val split inside each trainval fold."""
    y = np.asarray(y).astype(int)
    patient_ids = np.asarray(patient_ids).astype(str)
    trainval_idx = np.asarray(trainval_idx).astype(int)

    rng = np.random.RandomState(seed)

    tv_pids = _unique_in_order(patient_ids[trainval_idx])
    tv_group_y = _patient_group_labels(y, patient_ids, tv_pids)

    pos_pids = tv_pids[tv_group_y == 1].copy()
    neg_pids = tv_pids[tv_group_y == 0].copy()

    rng.shuffle(pos_pids)
    rng.shuffle(neg_pids)

    n_val_groups = int(round(len(tv_pids) * float(val_ratio)))
    n_val_groups = max(1, min(len(tv_pids) - 1, n_val_groups))

    n_pos = len(pos_pids)
    n_neg = len(neg_pids)
    p_pos = n_pos / max(n_pos + n_neg, 1)

    n_val_pos = int(round(n_val_groups * p_pos))
    n_val_pos = max(0, min(n_pos, n_val_pos))
    n_val_neg = n_val_groups - n_val_pos

    if n_val_neg > n_neg:
        n_val_neg = n_neg
        n_val_pos = n_val_groups - n_val_neg

    val_pids = set(
        np.concatenate([pos_pids[:n_val_pos], neg_pids[:n_val_neg]])
        .astype(str)
        .tolist()
    )

    val_mask = np.asarray([patient_ids[i] in val_pids for i in trainval_idx], dtype=bool)

    val_idx = trainval_idx[val_mask].astype(int)
    train_idx = trainval_idx[~val_mask].astype(int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return train_idx, val_idx

def assert_no_patient_overlap(patient_ids: np.ndarray,
                              train_idx: np.ndarray,
                              val_idx: np.ndarray,
                              test_idx: np.ndarray):
    patient_ids = np.asarray(patient_ids).astype(str)

    tr_p = set(patient_ids[np.asarray(train_idx, dtype=int)].tolist())
    va_p = set(patient_ids[np.asarray(val_idx, dtype=int)].tolist())
    te_p = set(patient_ids[np.asarray(test_idx, dtype=int)].tolist())

    if len(tr_p & va_p) > 0:
        raise RuntimeError(f"Patient leakage between train and val: {list(tr_p & va_p)[:10]}")
    if len(tr_p & te_p) > 0:
        raise RuntimeError(f"Patient leakage between train and test: {list(tr_p & te_p)[:10]}")
    if len(va_p & te_p) > 0:
        raise RuntimeError(f"Patient leakage between val and test: {list(va_p & te_p)[:10]}")

def _take_stratified(idx_pos: np.ndarray, idx_neg: np.ndarray, n_take: int, rng: np.random.RandomState):
    n_pos, n_neg = len(idx_pos), len(idx_neg)
    n_all = n_pos + n_neg
    p = n_pos / max(n_all, 1)
    take_pos = int(round(n_take * p))
    take_pos = max(0, min(n_pos, take_pos))
    take_neg = n_take - take_pos
    if take_neg > n_neg:
        take_neg = n_neg
        take_pos = n_take - take_neg
    taken = np.concatenate([idx_pos[:take_pos], idx_neg[:take_neg]], axis=0)
    rng.shuffle(taken)
    rem_pos = idx_pos[take_pos:]
    rem_neg = idx_neg[take_neg:]
    return taken.astype(int), rem_pos.astype(int), rem_neg.astype(int)

def stratified_kfold_split(y: np.ndarray, n_splits: int, seed: int):
    y = np.asarray(y).astype(int)
    rng = np.random.RandomState(seed)

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    pos_folds = np.array_split(idx_pos, n_splits)
    neg_folds = np.array_split(idx_neg, n_splits)

    folds = []
    all_idx = np.arange(len(y), dtype=int)
    for k in range(n_splits):
        test_idx = np.concatenate([pos_folds[k], neg_folds[k]], axis=0).astype(int)
        rng.shuffle(test_idx)

        mask = np.zeros(len(y), dtype=bool)
        mask[test_idx] = True
        trainval_idx = all_idx[~mask].astype(int)
        rng.shuffle(trainval_idx)
        folds.append((trainval_idx, test_idx))
    return folds

def stratified_trainval_split(trainval_idx: np.ndarray, y: np.ndarray, val_ratio: float, seed: int):
    y = np.asarray(y).astype(int)
    trainval_idx = np.asarray(trainval_idx).astype(int)
    rng = np.random.RandomState(seed)

    y_tv = y[trainval_idx]
    idx_pos = trainval_idx[np.where(y_tv == 1)[0]]
    idx_neg = trainval_idx[np.where(y_tv == 0)[0]]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_val = int(round(len(trainval_idx) * float(val_ratio)))
    n_val = max(1, min(len(trainval_idx) - 1, n_val))

    val_idx, pos_rem, neg_rem = _take_stratified(idx_pos, idx_neg, n_val, rng)
    train_idx = np.concatenate([pos_rem, neg_rem], axis=0).astype(int)
    rng.shuffle(train_idx)
    return train_idx, val_idx
