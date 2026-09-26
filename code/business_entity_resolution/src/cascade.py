"""Two-stage cascade: stage-1 scores prune the candidates, stage 2 re-scores what survives.

Stage 1 is LightGBM on the hand-crafted + retrieval features over every retrieved pair. Its
out-of-fold probability `p1` decides which pairs go on to stage 2 (and to the GPU cross-encoders).
Pruned pairs are never matched, so the surviving set is exactly what candidate_pairs.tsv contains.

The organisers rank a smaller candidate set per Source 1 entity higher (on top of the F0.5 score), so
pruning is a trade-off between pairs per entity and the F0.5 that stays reachable. `choose_pruning`
scans a grid of p1 cutoffs x per-entity caps and reports, for each point, pairs per entity, pair recall
and the oracle macro F0.5 (the score of a perfect matcher on the surviving pairs). It keeps the
smallest candidate set whose oracle score drops by at most `max_oracle_drop` -- or, when
`target_pairs_per_entity` is set, the best oracle score within that budget.

Stage 2 is LightGBM on the same features plus p1, any extra model scores (cross-encoders) and where
each score sits among the competing pairs of the same S1 entity and of the same candidate record.
"""
import os

import numpy as np
import pandas as pd


def rank_within(s1: np.ndarray, p: np.ndarray) -> np.ndarray:
    """1-based rank of each pair's p among the pairs of its S1 entity (1 = highest)."""
    order = np.lexsort((-p, s1))
    s1o = s1[order]
    starts = np.flatnonzero(np.r_[True, s1o[1:] != s1o[:-1]])
    ends = np.r_[starts[1:], len(s1o)]
    r = np.empty(len(p), np.int32)
    r[order] = np.arange(len(p)) - np.repeat(starts, ends - starts) + 1
    return r


def oracle_f05(ent: np.ndarray, label: np.ndarray, n_true: np.ndarray, keep: np.ndarray = None) -> float:
    """Macro F0.5 of a perfect matcher that may only pick kept pairs. ent: entity position per pair."""
    lab = label.astype(bool) if keep is None else (label.astype(bool) & keep)
    tp = np.bincount(ent, weights=lab.astype(np.float64), minlength=len(n_true))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(n_true == 0, 1.0, np.where(tp == 0, 0.0, 1.25 * tp / (tp + 0.25 * n_true)))
    return float(f.mean()) if len(n_true) else 0.0


def prune_mask(s1: np.ndarray, p1: np.ndarray, cutoff: float, cap: int) -> np.ndarray:
    keep = p1 >= cutoff
    if cap:
        keep &= rank_within(s1, p1) <= cap
    return keep


def choose_pruning(s1, p1, label, ent, n_true, ccfg):
    """-> (cutoff, cap, grid). Every grid point lists pairs per entity, pair recall and the oracle F0.5."""
    n_ent, n_pos = len(n_true), max(1, int(n_true.sum()))
    ranks = rank_within(s1, p1)
    base = oracle_f05(ent, label, n_true)
    grid = []
    for cap in ccfg.caps:
        for c in sorted({0.0, *ccfg.cutoffs}):
            keep = (p1 >= c) & ((ranks <= cap) if cap else True)
            orc = oracle_f05(ent, label, n_true, keep)
            grid.append({"cutoff": c, "cap": cap, "pairs": int(keep.sum()), "pairs_per_entity": float(keep.sum() / max(1, n_ent)),
                         "pair_recall": float((label.astype(bool) & keep).sum() / n_pos),
                         "oracle_f05": orc, "oracle_drop": base - orc})
    if not ccfg.prune:
        return 0.0, 0, grid
    if ccfg.target_pairs_per_entity:
        ok = [g for g in grid if g["pairs_per_entity"] <= ccfg.target_pairs_per_entity] or [min(grid, key=lambda g: g["pairs"])]
        best = max(ok, key=lambda g: (g["oracle_f05"], -g["pairs"]))
    else:
        ok = [g for g in grid if g["oracle_drop"] <= ccfg.max_oracle_drop + 1e-12]
        best = min(ok, key=lambda g: (g["pairs"], -g["oracle_f05"]))
    return best["cutoff"], best["cap"], grid


def add_score_context(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rank / gap-to-best of a score among the pairs of the same S1 entity and of the same candidate.

    Pairs without the score (e.g. a model that only scored part of the pairs) get NaN context.
    """
    v = df[col].astype(np.float64)
    missing = v.isna().to_numpy()
    vf = v.fillna(-1.0)
    g1, g2 = vf.groupby(df["s1"]), vf.groupby(df["cand"])
    ctx = {
        f"{col}_rank_s1": g1.rank(ascending=False, method="min"),
        f"{col}_gap_s1": g1.transform("max") - vf,
        f"{col}_rank_cand": g2.rank(ascending=False, method="min"),
        f"{col}_gap_cand": g2.transform("max") - vf,
        f"{col}_sum_s1": g1.transform("sum"),  # expected number of matches of the entity
    }
    # runner-up score of this entity: a strong second candidate means ambiguity
    srt = pd.DataFrame({"s1": df["s1"], "v": v}).dropna().sort_values(["s1", "v"], ascending=[True, False])
    second = srt[srt.groupby("s1").cumcount() == 1].set_index("s1")["v"]
    ctx[f"{col}_s1_second"] = df["s1"].map(second)
    for name, x in ctx.items():
        x = np.asarray(x, dtype=np.float32).copy()
        if name != f"{col}_s1_second":
            x[missing] = np.nan
        df[name] = x
    return df


def stage2_features(df: pd.DataFrame, score_cols) -> pd.DataFrame:
    """Stage-1 features + p1 + extra scores, plus competition context for every score."""
    df = df.copy()
    df["n_kept_s1"] = df.groupby("s1")["cand"].transform("size").astype(np.float32)
    df["n_kept_cand"] = df.groupby("cand")["s1"].transform("size").astype(np.float32)
    for c in ("p1", *score_cols):
        df = add_score_context(df, c)
    return df


def merge_extra(df: pd.DataFrame, extra_dir: str, name: str, which: str) -> pd.DataFrame:
    """Left-join one extra model's scores (extra/<name>_<which>.parquet: s1, cand, p) as column x_<name>."""
    path = os.path.join(extra_dir, f"{name}_{which}.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"[extra] missing {path}: run the model that writes it, or drop '{name}' from --extra")
    ex = pd.read_parquet(path)
    col = f"x_{name}"
    ex = ex.rename(columns={"p": col})[["s1", "cand", col]]
    ex[col] = ex[col].astype(np.float32)
    out = df.merge(ex, on=["s1", "cand"], how="left")
    cov = float(out[col].notna().mean()) if len(out) else 1.0
    stray = len(ex) - int(out[col].notna().sum())
    print(f"[extra] {name} ({which}): scores for {cov:.1%} of pairs"
          + (f" | {stray:,} scored pairs are not in the current candidate set" if stray else ""))
    if stray:
        print(f"[extra] WARNING: {name} was probably made from an older `prepare` run; re-run it on the current pairs")
    return out
