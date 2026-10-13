"""Binary classification metrics on held-out predictions."""
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_prob = np.asarray(y_prob).astype(np.float32).reshape(-1)
    y_pred = (y_prob >= float(threshold)).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(len(y_true), 1)
    recall = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 0.0 if (prec + recall) == 0 else (2.0 * prec * recall / (prec + recall))

    auc = float("nan")
    if roc_auc_score is not None and (y_true.min() != y_true.max()):
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = float("nan")

    return dict(acc=float(acc), prec=float(prec), recall=float(recall), spe=float(spe), f1=float(f1), auc=float(auc),
                tp=tp, tn=tn, fp=fp, fn=fn)

def find_best_threshold(y_true, y_prob,
                        mode: str = "f1",
                        num_steps: int = 401,
                        beta: float = 2.0,
                        target_spe: float = 0.85):
    """Grid search threshold in [0,1].

    mode:
      - "f1": maximize F1
      - "f2": maximize F_beta (beta default 2.0; emphasize recall)
      - "recall_at_spe": maximize recall subject to spe >= target_spe

    Returns: (best_thr, best_metrics_dict, best_score, score_name)
    """
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_prob = np.asarray(y_prob).astype(np.float32).reshape(-1)

    best_thr = 0.5
    best_score = -1.0
    best_mets = None

    def score_from_mets(mets):
        if mode == "f1":
            return float(mets["f1"]), "f1"
        if mode == "f2":
            s = _fbeta(float(mets["prec"]), float(mets["recall"]), float(beta))
            return float(s), f"f{beta:g}"
        if mode == "recall_at_spe":
            if float(mets["spe"]) >= float(target_spe):
                return float(mets["recall"]), f"recall@spe>={target_spe:g}"
            return -1.0, f"recall@spe>={target_spe:g}"
        raise ValueError(f"Unknown thr mode: {mode}")

    for thr in np.linspace(0.0, 1.0, int(num_steps)):
        mets = compute_binary_metrics(y_true, y_prob, threshold=float(thr))
        s, _name = score_from_mets(mets)
        if s > best_score:
            best_score = s
            best_thr = float(thr)
            best_mets = mets

    # Fallback for recall_at_spe if no threshold satisfies constraint
    if mode == "recall_at_spe" and (best_mets is None or best_score < 0):
        best_thr = 0.5
        best_mets = compute_binary_metrics(y_true, y_prob, threshold=best_thr)
        best_score, score_name = float(best_mets["f1"]), "f1"
        return best_thr, best_mets, best_score, score_name

    score_name = score_from_mets(best_mets)[1] if best_mets is not None else mode
    return best_thr, best_mets, float(best_score), score_name

def _fbeta(prec: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = (b2 * prec + recall)
    return 0.0 if denom <= 0 else (1.0 + b2) * prec * recall / denom
