"""The official metric: macro-averaged F0.5 over Source 1 entities, singletons included.

Per entity with predicted set S and true set T:
    TP = |S ∩ T|
    F0.5 = 1.25·TP / (|S| + 0.25·|T|)          (algebraically == 1.25·P·R / (0.25·P + R))
    T = ∅, S = ∅  -> 1.0   (correct singleton)
    T = ∅, S ≠ ∅  -> 0.0   (false merge on a singleton)
    T ≠ ∅, S = ∅  -> 0.0
"""
import numpy as np
import pandas as pd


def f05_single(pred: set, truth: set) -> float:
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    tp = len(pred & truth)
    return 1.25 * tp / (len(pred) + 0.25 * len(truth))


def macro_f05(pred: dict, truth: dict, ids) -> float:
    ids = list(ids)
    return float(np.mean([f05_single(set(pred.get(i, ())), set(truth.get(i, ()))) for i in ids])) if ids else 0.0


def report(pred: dict, truth: dict, ids, country_of: dict = None) -> dict:
    """Macro F0.5 plus diagnostics that explain *where* the score is lost."""
    ids = list(ids)
    scores, tp, n_pred, n_true = [], 0, 0, 0
    sing_ok = sing_n = 0
    for i in ids:
        p, t = set(pred.get(i, ())), set(truth.get(i, ()))
        scores.append(f05_single(p, t))
        tp += len(p & t); n_pred += len(p); n_true += len(t)
        if not t:
            sing_n += 1; sing_ok += int(not p)
    out = {
        "macro_f05": float(np.mean(scores)) if scores else 0.0,
        "micro_precision": tp / n_pred if n_pred else 0.0,
        "micro_recall": tp / n_true if n_true else 0.0,
        "singleton_share": sing_n / max(1, len(ids)),
        "singleton_accuracy": sing_ok / sing_n if sing_n else float("nan"),
        "n_entities": len(ids),
    }
    if country_of:
        by = {}
        for i, s in zip(ids, scores):
            by.setdefault(country_of.get(i, "?"), []).append(s)
        out["macro_f05_by_country"] = {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())}
    return out


def f05_per_entity(ent: np.ndarray, pred: np.ndarray, label: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """F0.5 of every entity from pair arrays (ent: entity position per pair; n_true: |T| per entity,
    including true matches that are not among the pairs; entities without pairs predict nothing)."""
    E = len(n_true)
    pred, label = np.asarray(pred, bool), np.asarray(label, bool)
    tp = np.bincount(ent, weights=(pred & label).astype(np.float64), minlength=E)
    npred = np.bincount(ent, weights=pred.astype(np.float64), minlength=E)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_true == 0, (npred == 0).astype(np.float64),
                        np.where(npred == 0, 0.0, 1.25 * tp / (npred + 0.25 * n_true)))


def report_pairs(ent, pred, label, n_true, country=None) -> dict:
    """report() for pair arrays; country: label per entity position."""
    f = f05_per_entity(ent, pred, label, n_true)
    pred, label = np.asarray(pred, bool), np.asarray(label, bool)
    tp, npred, ntrue = float((pred & label).sum()), float(pred.sum()), float(n_true.sum())
    npred_e = np.bincount(ent, weights=pred.astype(np.float64), minlength=len(n_true))
    sing = n_true == 0
    out = {"macro_f05": float(f.mean()) if len(f) else 0.0,
           "micro_precision": tp / npred if npred else 0.0,
           "micro_recall": tp / ntrue if ntrue else 0.0,
           "singleton_share": float(sing.mean()) if len(f) else 0.0,
           "singleton_accuracy": float((npred_e[sing] == 0).mean()) if sing.any() else float("nan"),
           "n_entities": int(len(f))}
    if country is not None:
        out["macro_f05_by_country"] = {str(k): round(float(v), 4) for k, v in pd.Series(f).groupby(np.asarray(country)).mean().items()}
    return out


def _self_test():
    # the worked example from the problem statement
    p = {"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]}
    t = {"S1-00001": ["S2-00047", "S3-00812"]}
    assert abs(macro_f05(p, t, ["S1-00001"]) - 0.714) < 1e-3
    assert f05_single(set(), set()) == 1.0
    assert f05_single({"a"}, set()) == 0.0
    assert f05_single(set(), {"a"}) == 0.0
    P, R = 2 / 3, 1.0
    assert abs(f05_single({"a", "b", "c"}, {"a", "c"}) - 1.25 * P * R / (0.25 * P + R)) < 1e-12


_self_test()
