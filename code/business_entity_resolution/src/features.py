"""Pairwise features for (Source 1 record, candidate record) pairs.

Pairs are int32 row numbers into the records table (src/data.py), so every record attribute is a
numpy take, not a string merge over millions of rows. Features are built in chunks of S1 entities
(`build_features`), which keeps the test set (~1.7M entities x tens of candidates) within memory.

All features are similarity / agreement measures, never raw identities: there is no country
one-hot, so the model transfers to countries it has not seen (France). Missing information is NaN
(LightGBM routes it natively), so "both addresses empty" never looks like "addresses identical".

Context features only look at the S1 entity's own candidates (all in the same chunk) and at the
retrieval neighbourhood of both records (computed over the whole country by src/retrieval.py). Context
across the S1 entities that compete for one candidate is added after pruning, in stage 2
(src/cascade.py), when every surviving pair is in memory.
"""
import math
import time
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

REC_COLS = ["name_norm", "name_core", "name_alt", "addr_norm", "addr_core", "addr_nums", "has_landmark",
            "name_acr", "source", "name_indic", "addr_indic"]
NOT_FEATURES = {"s1", "cand", "label", "fold"}


def idf_table(recs: pd.DataFrame) -> dict:
    """Token -> inverse document frequency over one split's records (names and addresses separately)."""
    out = {}
    for col in ("name_core", "addr_core"):
        df = Counter()
        for s in recs[col].tolist():
            df.update(set(s.split()))
        n = max(1, len(recs))
        out[col] = {t: math.log(n / c) for t, c in df.items()}
        out[col + "_default"] = math.log(n)  # unseen token: as rare as it gets
    return out


def _sim(a, b, scorer, scale=100.0):
    return (cpdist(list(a), list(b), scorer=scorer, workers=-1, dtype=np.float32) / scale).astype(np.float32)


def _lens(a) -> np.ndarray:
    return np.fromiter(map(len, a), np.float32, len(a))


def _mask_empty(x, empty):
    x = x.astype(np.float32, copy=True)
    x[empty] = np.nan
    return x


def _set_feats(A, B, idf: dict, default: float):
    """Jaccard, overlap-with-smaller, and IDF-weighted Jaccard of token sets (rare shared tokens count more)."""
    n = len(A)
    jac = np.full(n, np.nan, np.float32)
    ovl = np.full(n, np.nan, np.float32)
    wj = np.full(n, np.nan, np.float32)
    get = idf.get
    for i in range(n):
        a, b = A[i], B[i]
        if not a or not b:
            continue
        inter = a & b
        ni = len(inter)
        jac[i] = ni / (len(a) + len(b) - ni)
        ovl[i] = ni / min(len(a), len(b))
        wu = sum(get(t, default) for t in a | b)
        wj[i] = sum(get(t, default) for t in inter) / wu if wu > 0 else np.nan
    return jac, ovl, wj


def _num_feats(A, B):
    """House / unit numbers: Jaccard, first number equal, no number in common (NaN when a side has none)."""
    n = len(A)
    jac = np.full(n, np.nan, np.float32)
    first_eq = np.full(n, np.nan, np.float32)
    conflict = np.full(n, np.nan, np.float32)
    for i in range(n):
        la, lb = A[i], B[i]
        if la and lb:
            sa, sb = set(la), set(lb)
            ni = len(sa & sb)
            jac[i] = ni / len(sa | sb)
            first_eq[i] = float(la[0] == lb[0])
            conflict[i] = float(ni == 0)
    return jac, first_eq, conflict


def _group_rank(key: np.ndarray, v: np.ndarray):
    """Rank (1 = best) and gap to the best value of v within groups of key."""
    s = pd.Series(np.nan_to_num(v, nan=-1.0))
    g = s.groupby(key)
    return (g.rank(ascending=False, method="min").to_numpy(np.float32),
            (g.transform("max") - s).to_numpy(np.float32))


def _chunk_features(p: pd.DataFrame, recs_cols: dict, idf: dict) -> pd.DataFrame:
    s1, cand = p["s1"].to_numpy(), p["cand"].to_numpy()
    L = {c: v[s1] for c, v in recs_cols.items()}
    R = {c: v[cand] for c, v in recs_cols.items()}
    f = {"s1": s1, "cand": cand}

    # ---- retrieval (bi-encoder) ---------------------------------------------------------------
    cos = p["cos"].to_numpy(np.float32)
    f["cos"] = cos
    f["fwd_rank"] = p["fwd_rank"].to_numpy(np.float32)
    f["rev_rank"] = p["rev_rank"].to_numpy(np.float32)
    f["cos_gap_s1"] = p["s1_top1"].to_numpy(np.float32) - cos        # 0 = the S1 record's nearest record
    f["cos_gap_cand"] = p["cand_top1"].to_numpy(np.float32) - cos    # 0 = this S1 is the record's nearest S1
    f["cand_margin"] = (p["cand_top1"] - p["cand_top2"]).to_numpy(np.float32)  # how contested the record is
    f["s1_density"] = (p["s1_top1"] - p["s1_top10"]).to_numpy(np.float32)      # crowded neighbourhood

    # ---- name ------------------------------------------------------------------------------------
    ln, rn = L["name_norm"], R["name_norm"]
    lc, rc = L["name_core"], R["name_core"]
    la, ra = L["name_alt"], R["name_alt"]
    llen, rlen = _lens(lc), _lens(rc)
    empty_n = (_lens(ln) == 0) | (_lens(rn) == 0)
    empty_c = (llen == 0) | (rlen == 0)
    f["name_ratio"] = _mask_empty(_sim(ln, rn, fuzz.ratio), empty_n)
    f["name_tset"] = _mask_empty(_sim(ln, rn, fuzz.token_set_ratio), empty_n)
    for key, scorer in (("ratio", fuzz.ratio), ("tset", fuzz.token_set_ratio), ("tsort", fuzz.token_sort_ratio),
                        ("partial", fuzz.partial_ratio), ("wratio", fuzz.WRatio)):
        f[f"core_{key}"] = _mask_empty(_sim(lc, rc, scorer), empty_c)
    f["core_jw"] = _mask_empty(_sim(lc, rc, JaroWinkler.normalized_similarity, 1.0), empty_c)
    f["core_lev"] = _mask_empty(_sim(lc, rc, Levenshtein.normalized_similarity, 1.0), empty_c)
    # "X formerly/aka/DBA Y": best over (core|alt) x (core|alt)
    best_tset, best_ratio = f["core_tset"].copy(), f["core_ratio"].copy()
    for a, b in ((lc, ra), (la, rc), (la, ra)):
        best_tset = np.fmax(best_tset, _sim(a, b, fuzz.token_set_ratio))
        best_ratio = np.fmax(best_ratio, _sim(a, b, fuzz.ratio))
    f["core_best_tset"], f["core_best_ratio"] = best_tset, best_ratio
    lcomp = np.array([s.replace(" ", "") for s in lc], dtype=object)
    rcomp = np.array([s.replace(" ", "") for s in rc], dtype=object)
    f["compact_ratio"] = _mask_empty(_sim(lcomp, rcomp, fuzz.ratio), empty_c)  # "globalcardiology" vs "global cardiology"
    f["compact_partial"] = _mask_empty(_sim(lcomp, rcomp, fuzz.partial_ratio), empty_c)
    f["compact_eq"] = ((lcomp == rcomp) & ~empty_c).astype(np.float32)
    lacr, racr = L["name_acr"], R["name_acr"]
    f["acronym_match"] = (((lacr != "") & (lacr == rcomp)) | ((racr != "") & (racr == lcomp))
                          | ((lacr != "") & (lacr == racr))).astype(np.float32)
    lt = [s.split() for s in lc.tolist()]
    rt = [s.split() for s in rc.tolist()]
    f["first_tok_eq"] = np.array([bool(a) and bool(b) and a[0] == b[0] for a, b in zip(lt, rt)], np.float32)
    f["last_tok_eq"] = np.array([bool(a) and bool(b) and a[-1] == b[-1] for a, b in zip(lt, rt)], np.float32)
    f["core_jacc"], f["core_overlap_min"], f["core_idf_jacc"] = _set_feats(
        [frozenset(x) for x in lt], [frozenset(x) for x in rt], idf["name_core"], idf["name_core_default"])
    f["core_len_min"], f["core_len_max"] = np.minimum(llen, rlen), np.maximum(llen, rlen)
    f["core_len_ratio"] = np.where(np.maximum(llen, rlen) > 0, np.minimum(llen, rlen) / np.maximum(np.maximum(llen, rlen), 1),
                                   np.nan).astype(np.float32)
    f["core_ntok_diff"] = np.abs(np.array([len(a) - len(b) for a, b in zip(lt, rt)], np.float32))
    ld = [{t for t in a if any(ch.isdigit() for ch in t)} for a in lt]
    rd = [{t for t in b if any(ch.isdigit() for ch in t)} for b in rt]
    f["name_digit_conflict"] = np.array([float(bool(a) and bool(b) and not (a & b)) for a, b in zip(ld, rd)], np.float32)
    f["name_digit_match"] = np.array([float(bool(a & b)) for a, b in zip(ld, rd)], np.float32)
    f["name_indic"] = R["name_indic"].astype(np.float32)
    f["name_has_alt"] = ((la != lc) | (ra != rc)).astype(np.float32)

    # ---- address ---------------------------------------------------------------------------------
    lad, rad = L["addr_core"], R["addr_core"]
    la_len, ra_len = _lens(lad), _lens(rad)
    empty_a = (la_len == 0) | (ra_len == 0)
    for key, scorer in (("ratio", fuzz.ratio), ("tset", fuzz.token_set_ratio), ("tsort", fuzz.token_sort_ratio),
                        ("partial", fuzz.partial_ratio)):
        f[f"addr_{key}"] = _mask_empty(_sim(lad, rad, scorer), empty_a)
    f["addr_jw"] = _mask_empty(_sim(lad, rad, JaroWinkler.normalized_similarity, 1.0), empty_a)
    f["addr_full_tset"] = _mask_empty(_sim(L["addr_norm"], R["addr_norm"], fuzz.token_set_ratio), empty_a)
    f["addr_jacc"], f["addr_overlap_min"], f["addr_idf_jacc"] = _set_feats(
        [frozenset(s.split()) for s in lad.tolist()], [frozenset(s.split()) for s in rad.tolist()],
        idf["addr_core"], idf["addr_core_default"])
    f["num_jacc"], f["num_first_eq"], f["num_conflict"] = _num_feats(
        [s.split() for s in L["addr_nums"].tolist()], [s.split() for s in R["addr_nums"].tolist()])
    f["landmark_n"] = (L["has_landmark"].astype(np.float32) + R["has_landmark"].astype(np.float32))
    f["addr_missing_n"] = (la_len == 0).astype(np.float32) + (ra_len == 0).astype(np.float32)
    f["addr_len_ratio"] = np.where(np.maximum(la_len, ra_len) > 0, np.minimum(la_len, ra_len) / np.maximum(np.maximum(la_len, ra_len), 1),
                                   np.nan).astype(np.float32)
    f["addr_indic"] = R["addr_indic"].astype(np.float32)

    # ---- cross / context within the S1 entity --------------------------------------------------
    f["cand_is_s3"] = (R["source"] == "S3").astype(np.float32)
    f["name_x_addr"] = (np.nan_to_num(f["core_best_tset"]) * np.nan_to_num(f["addr_tset"], nan=0.5)).astype(np.float32)
    feats = pd.DataFrame(f)
    for c in ("core_best_tset", "addr_tset", "name_x_addr", "core_idf_jacc", "addr_idf_jacc"):
        feats[f"{c}_rank_s1"], feats[f"{c}_gap_s1"] = _group_rank(s1, feats[c].to_numpy())
    src_key = s1.astype(np.int64) * 2 + (R["source"] == "S3")
    feats["cos_rank_s1_src"], _ = _group_rank(src_key, cos)
    feats["n_cand_s1"] = feats.groupby("s1")["cand"].transform("size").astype(np.float32)
    feats["n_cand_s1_src"] = pd.Series(src_key).groupby(src_key).transform("size").to_numpy(np.float32)
    return feats


def build_features(pairs: pd.DataFrame, recs: pd.DataFrame, idf: dict, chunk_entities: int = 100_000,
                   keep=None, log: bool = True) -> pd.DataFrame:
    """Features for all pairs, S1 entity by S1 entity in chunks.

    keep: optional function(feats_chunk) -> boolean mask, applied to each chunk before it is kept
    (the test set is scored and pruned chunk by chunk, so the full feature matrix never exists).
    """
    t0 = time.time()
    recs_cols = {c: recs[c].to_numpy() for c in REC_COLS}
    for c in ("name_norm", "name_core", "name_alt", "addr_norm", "addr_core", "addr_nums", "name_acr", "source"):
        recs_cols[c] = recs_cols[c].astype(object)
    pairs = pairs.sort_values(["s1", "cand"], kind="stable").reset_index(drop=True)
    s1 = pairs["s1"].to_numpy()
    ents = np.unique(s1)
    bounds = np.searchsorted(s1, ents[::chunk_entities].tolist() + [s1.max() + 1 if len(s1) else 0])
    out, n_in = [], 0
    for j in range(len(bounds) - 1):
        chunk = pairs.iloc[bounds[j]:bounds[j + 1]]
        n_in += len(chunk)
        feats = _chunk_features(chunk, recs_cols, idf)
        if keep is not None:
            feats = feats[keep(feats)]
        out.append(feats)
        if log and len(bounds) > 2:
            el = time.time() - t0
            print(f"[features] chunk {j + 1}/{len(bounds) - 1}: {n_in:,}/{len(pairs):,} pairs, "
                  f"{el / 60:.1f} min, ~{el / n_in * (len(pairs) - n_in) / 60:.1f} min left", flush=True)
    feats = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if log:
        print(f"[features] {len(feature_columns(feats))} features for {n_in:,} pairs"
              + (f" ({len(feats):,} kept)" if keep is not None else "") + f" in {time.time() - t0:.0f}s")
    return feats


def peer_features(kept: pd.DataFrame, recs: pd.DataFrame, confident: float = 0.5) -> pd.DataFrame:
    """Does a candidate look like the entity's other likely matches? (stage 2 only; needs p1)

    An entity has ~3.7 true matches and they describe the same business, so they resemble each other. A
    weak candidate ("BP" for "Bowling Pharmacie SAS") that shares its address with the entity's confident
    candidates is probably a match too, and one that resembles none of them probably is not. Computed on
    the pruned pairs (a few per entity), so comparing every candidate with its peers is cheap.
    """
    k = pd.DataFrame({"i": np.arange(len(kept)), "s1": kept["s1"].to_numpy(), "cand": kept["cand"].to_numpy(),
                      "p1": kept["p1"].to_numpy(np.float32)})
    m = k.merge(k[["s1", "cand", "p1"]], on="s1", suffixes=("", "_o"))
    m = m[m["cand"].to_numpy() != m["cand_o"].to_numpy()]
    out = pd.DataFrame(index=np.arange(len(kept)))
    if len(m):
        names, addrs = recs["name_core"].to_numpy(object), recs["addr_core"].to_numpy(object)
        c, o = m["cand"].to_numpy(), m["cand_o"].to_numpy()
        sn = _sim(names[c], names[o], fuzz.token_set_ratio)
        sa = _sim(addrs[c], addrs[o], fuzz.token_set_ratio)
        w = m["p1_o"].to_numpy(np.float64)
        conf = w >= confident
        g = pd.DataFrame({"i": m["i"].to_numpy(), "sn": sn, "sa": sa, "w": w, "wsn": w * sn, "wsa": w * sa,
                          "sn_c": np.where(conf, sn, np.nan), "sa_c": np.where(conf, sa, np.nan), "conf": conf})
        agg = g.groupby("i").agg(peer_name_max=("sn", "max"), peer_addr_max=("sa", "max"),
                                 peer_name_conf_max=("sn_c", "max"), peer_addr_conf_max=("sa_c", "max"),
                                 n_conf_peers=("conf", "sum"), w=("w", "sum"), wsn=("wsn", "sum"), wsa=("wsa", "sum"))
        agg["peer_name_wmean"] = agg["wsn"] / agg["w"].where(agg["w"] > 0)
        agg["peer_addr_wmean"] = agg["wsa"] / agg["w"].where(agg["w"] > 0)
        out = out.join(agg.drop(columns=["w", "wsn", "wsa"]))
    for col in ("peer_name_max", "peer_addr_max", "peer_name_conf_max", "peer_addr_conf_max", "n_conf_peers",
                "peer_name_wmean", "peer_addr_wmean"):
        kept[col] = (out[col].to_numpy(np.float32) if col in out else np.full(len(kept), np.nan, np.float32))
    kept["n_conf_peers"] = kept["n_conf_peers"].fillna(0)
    return kept


def feature_columns(feats: pd.DataFrame):
    return [c for c in feats.columns if c not in NOT_FEATURES]
