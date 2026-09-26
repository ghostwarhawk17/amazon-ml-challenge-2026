"""LightGBM pair classifiers trained with grouped K-fold (MIT licence; far below the 8B-parameter limit).

Folds are over Source 1 entities, so all candidate pairs of one entity land in the same fold.
Every training pair gets an out-of-fold prediction (stage 2 stacks on it, and the decision rule is
tuned on it). Test pairs get the fold models' average, taken in log-odds space so it stays as sharp
as a single model's prediction.
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


def make_folds(entities: np.ndarray, strata: np.ndarray, n_folds: int, seed: int) -> pd.Series:
    """fold per entity row (Series indexed by row), stratified so every fold looks like the whole set."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(entities), np.int8)
    offset = 0
    for key in np.unique(strata):
        idx = np.flatnonzero(strata == key)
        rng.shuffle(idx)
        folds[idx] = (offset + np.arange(len(idx))) % n_folds
        offset += len(idx)  # continue the round-robin so small strata don't all start at fold 0
    return pd.Series(folds, index=entities)


def train_cv(df: pd.DataFrame, cols, mcfg, label: str = "label", fold: str = "fold"):
    """One booster per fold, early-stopped on its held-out fold. Returns (oof probabilities, boosters)."""
    params = dict(mcfg.lgb_params, seed=mcfg.seed)
    folds = df[fold].to_numpy()
    X = df[cols].to_numpy(np.float32)
    y = df[label].to_numpy()
    t0 = time.time()
    # bin the features once; every fold trains on a subset of the same Dataset
    full = lgb.Dataset(X, y, feature_name=list(cols), free_raw_data=False, params={"max_bin": params.get("max_bin", 255)})
    full.construct()
    print(f"[cv] {len(df):,} pairs x {len(cols)} features binned in {time.time() - t0:.0f}s")
    oof = np.zeros(len(df), dtype=np.float64)
    boosters = []
    for k in range(int(folds.max()) + 1):
        t1 = time.time()
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        dtr, dva = full.subset(tr), full.subset(va)
        b = lgb.train(params, dtr, num_boost_round=mcfg.num_boost_round, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(mcfg.early_stopping_rounds, verbose=False)])
        oof[va] = b.predict(X[va], num_iteration=b.best_iteration)
        print(f"[cv] fold {k}: {len(tr):,} train / {len(va):,} held-out pairs | best iteration {b.best_iteration} "
              f"| held-out logloss {b.best_score['val']['binary_logloss']:.5f} | {time.time() - t1:.0f}s", flush=True)
        boosters.append(b)
    return oof, boosters


def predict_cv(boosters, X) -> np.ndarray:
    """Average of the fold models in log-odds space, returned as probabilities."""
    X = X.to_numpy(np.float32) if isinstance(X, pd.DataFrame) else X
    z = np.mean([b.predict(X, num_iteration=b.best_iteration if b.best_iteration > 0 else None, raw_score=True)
                 for b in boosters], axis=0)
    return 1.0 / (1.0 + np.exp(-z))


def save_boosters(boosters, model_dir: str, prefix: str):
    for k, b in enumerate(boosters):
        b.save_model(os.path.join(model_dir, f"{prefix}_fold{k}.txt"),
                     num_iteration=b.best_iteration if b.best_iteration > 0 else None)


def load_boosters(model_dir: str, prefix: str):
    out, k = [], 0
    while os.path.exists(os.path.join(model_dir, f"{prefix}_fold{k}.txt")):
        out.append(lgb.Booster(model_file=os.path.join(model_dir, f"{prefix}_fold{k}.txt")))
        k += 1
    return out


def importance(boosters, cols) -> pd.DataFrame:
    gain = np.mean([b.feature_importance("gain") for b in boosters], axis=0)
    return (pd.DataFrame({"feature": cols, "gain": gain})
              .sort_values("gain", ascending=False).reset_index(drop=True))
