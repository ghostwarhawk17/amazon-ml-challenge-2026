"""Turning pair probabilities into per-entity match lists that maximise macro F0.5.

Two strategies are compared on out-of-fold predictions and the winner is saved:

1. threshold     : keep candidates with p >= t (t tuned for macro F0.5, not for accuracy/AUC)
2. expected_f    : per entity, pick the set S that maximises E[F0.5(S, T)] under the model's
                   probabilities (exact, via Poisson-binomial dynamic programming). Under
                   independence the optimum is always a top-k prefix of the sorted probabilities,
                   and k = 0 (predict "no match") is chosen whenever P(entity is a singleton) is
                   higher than the expected score of any non-empty set. This is exactly the
                   structure of the metric: singletons are worth a full 1.0, false merges on them 0.

Optional "one_to_one": in the training ground truth no Source 2/3 record belongs to more than one
Source 1 entity, so each candidate can be kept only for its highest-probability S1 entity. Whether this
helps is still decided on validation data.

Everything works on aligned pair arrays (s1, cand, p[, label]) of int row numbers; macro F0.5 of a
rule is computed with bincounts, so a grid search over hundreds of thousands of entities is fast.
"""
import numpy as np
import pandas as pd

from .metrics import f05_per_entity


def _pb_forward(p):
    pmfs = [np.array([1.0])]
    for q in p:
        prev = pmfs[-1]
        new = np.zeros(len(prev) + 1)
        new[:-1] += prev * (1 - q)
        new[1:] += prev * q
        pmfs.append(new)
    return pmfs


def expected_f05_topk(p: np.ndarray, max_items: int = 25):
    """p sorted descending. Returns (best_k, best_expected_f05)."""
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    n_all = len(p)
    if n_all == 0:
        return 0, 1.0
    n = min(n_all, max_items)
    fw = _pb_forward(p[:n])                 # pmf of #true among the first k
    bw = _pb_forward(p[::-1])               # pmf of #true among the last m (reversed order)
    best_k, best_v = 0, float(bw[n_all][0])  # k = 0: prob. that nothing is a match
    for k in range(1, n + 1):
        pa, pb = fw[k], bw[n_all - k]
        a = np.arange(k + 1)[:, None]
        b = np.arange(n_all - k + 1)[None, :]
        F = 1.25 * a / (k + 0.25 * (a + b))
        v = float(pa @ F @ pb)
        if v > best_v:
            best_k, best_v = k, v
    return best_k, best_v


def one_to_one(cand: np.ndarray, p: np.ndarray) -> np.ndarray:
    """p with every pair that is not its candidate's best S1 claimant set to 0 (ties: first pair wins)."""
    best = pd.Series(p).groupby(cand).transform("max").to_numpy()
    out = np.where(p >= best, p, 0.0)
    keep = out > 0
    dup = pd.DataFrame({"c": cand[keep], "p": out[keep]}).duplicated(keep="first").to_numpy()
    idx = np.flatnonzero(keep)[dup]
    out[idx] = 0.0
    return out


def select(s1: np.ndarray, cand: np.ndarray, p: np.ndarray, params: dict, max_items: int = 25) -> np.ndarray:
    """Boolean mask over the pairs: which pairs are predicted matches under the decision rule."""
    p = np.asarray(p, np.float64)
    if params.get("one_to_one"):
        p = one_to_one(cand, p)
    if params["strategy"] == "threshold":
        return (p >= params["t"]) & (p > 0)
    q = np.clip(p, 0, 1) ** params.get("gamma", 1.0)
    order = np.lexsort((-q, s1))
    s1o, qo = s1[order], q[order]
    starts = np.flatnonzero(np.r_[True, s1o[1:] != s1o[:-1]])
    ends = np.r_[starts[1:], len(s1o)]
    rank = np.arange(len(s1o)) - np.repeat(starts, ends - starts)
    k = np.zeros(len(starts), np.int64)
    for g, (a, b) in enumerate(zip(starts, ends)):
        qq = qo[a:b]
        qq = qq[qq > 1e-4]  # near-zero pairs cannot change the best set; skipping them keeps the DP small
        if len(qq):
            k[g] = expected_f05_topk(qq, max_items)[0]
    mask = np.zeros(len(p), bool)
    mask[order] = rank < np.repeat(k, ends - starts)
    return mask & (p > 0)


def macro_f05_pairs(ent: np.ndarray, pred: np.ndarray, label: np.ndarray, n_true: np.ndarray) -> float:
    """Macro F0.5 from pair arrays (see metrics.f05_per_entity)."""
    f = f05_per_entity(ent, pred, label, n_true)
    return float(f.mean()) if len(f) else 0.0


def tune(s1, cand, p, label, entities, n_true, dcfg, max_entities: int = 200_000, seed: int = 42):
    """Grid-search decision rules for macro F0.5 on out-of-fold predictions.

    entities / n_true: every evaluated S1 row and its number of true matches (so entities whose
    candidates were all pruned still count). Expected-F rules are searched on at most max_entities
    entities (the rule has 1-2 parameters); the winner is then re-scored on all of them.
    """
    ent_pos = pd.Series(np.arange(len(entities)), index=entities)
    ent = ent_pos.reindex(s1).to_numpy()
    label = np.asarray(label, bool)
    sub = np.ones(len(entities), bool)
    if len(entities) > max_entities:
        sub[:] = False
        sub[np.random.default_rng(seed).choice(len(entities), max_entities, replace=False)] = True
    in_sub = sub[ent]
    sub_pos = np.cumsum(sub) - 1

    grid = []
    for o2o in dcfg.try_one_to_one:
        for t in dcfg.thresholds:
            grid.append({"strategy": "threshold", "t": float(t), "one_to_one": o2o})
        if dcfg.try_expected_f:
            grid.extend({"strategy": "expected_f", "gamma": float(g), "one_to_one": o2o} for g in dcfg.calib_gammas)
    results = []
    for prm in grid:
        if prm["strategy"] == "threshold":
            m = select(s1, cand, p, prm)
            sc = macro_f05_pairs(ent, m, label, n_true)
        else:
            # one-to-one needs every claimant of a candidate, so it runs on all pairs before sub-sampling
            m = select(s1, cand, p, prm, dcfg.max_items_expected_f) if not prm["one_to_one"] else None
            if m is None:
                q = one_to_one(cand, np.asarray(p, np.float64))
                m = np.zeros(len(p), bool)
                m[in_sub] = select(s1[in_sub], cand[in_sub], q[in_sub], {**prm, "one_to_one": False},
                                   dcfg.max_items_expected_f)
            sc = macro_f05_pairs(sub_pos[ent[in_sub]], m[in_sub], label[in_sub], n_true[sub])
        results.append((sc, prm))
    results.sort(key=lambda r: -r[0])
    best = results[0][1]
    best_all = macro_f05_pairs(ent, select(s1, cand, p, best, dcfg.max_items_expected_f), label, n_true)
    return best, best_all, results


def decide(s1, cand, p, params: dict, all_s1, max_items: int = 25) -> dict:
    """{s1 row: [cand rows]} for every row in all_s1 (empty list = no match)."""
    m = select(np.asarray(s1), np.asarray(cand), np.asarray(p), params, max_items)
    out = {int(i): [] for i in all_s1}
    for a, b in zip(np.asarray(s1)[m].tolist(), np.asarray(cand)[m].tolist()):
        out[a].append(b)
    return out


def _self_test():
    # one very likely match -> take it
    assert expected_f05_topk(np.array([0.95]))[0] == 1
    # one unlikely match -> predicting empty is better (singleton credit)
    assert expected_f05_topk(np.array([0.3]))[0] == 0
    # brute-force check of the DP against enumeration for a small case
    import itertools
    p = np.array([0.9, 0.6, 0.35, 0.1])
    def brute(k):
        v = 0.0
        for bits in itertools.product([0, 1], repeat=len(p)):
            pr = np.prod([q if b else 1 - q for q, b in zip(p, bits)])
            T = {i for i, b in enumerate(bits) if b}
            S = set(range(k))
            if not T and not S: f = 1.0
            elif not T or not S: f = 0.0
            else: f = 1.25 * len(S & T) / (len(S) + 0.25 * len(T))
            v += pr * f
        return v
    k_best, v_best = expected_f05_topk(p)
    bk = max(range(len(p) + 1), key=brute)
    assert k_best == bk and abs(v_best - brute(bk)) < 1e-9, (k_best, bk, v_best, brute(bk))
    # pair-array macro F0.5 == the worked example of the problem statement (0.714), plus a correct singleton
    ent, pred, lab = np.array([0, 0, 0]), np.array([True, True, True]), np.array([True, False, True])
    assert abs(macro_f05_pairs(ent, pred, lab, np.array([2, 0])) - (0.7142857 + 1.0) / 2) < 1e-6
    # one-to-one keeps only the best claimant of a candidate
    assert list(one_to_one(np.array([7, 7, 8]), np.array([0.9, 0.8, 0.5]))) == [0.9, 0.0, 0.5]


_self_test()
