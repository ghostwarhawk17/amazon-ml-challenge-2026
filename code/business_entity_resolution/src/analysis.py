"""Where is macro F0.5 lost? Printed after stage 2 (out-of-fold, training 'rank' entities) and saved to
<model-dir>/error_report.json, with example errors in <model-dir>/error_examples.tsv.

  - true pairs: lost by retrieval, lost by pruning, missed by the matcher, found
  - macro F0.5 lost per kind of entity: singletons given a false merge, entities with a false merge,
    entities with only misses -- overall and per country
  - miss and false-merge rates by pair type (transliterated text, "formerly" names, Source 3, no address)
"""
import json
import os

import numpy as np
import pandas as pd

from . import data
from .metrics import f05_per_entity

FLAGS = {"transliterated_name": "name_indic", "transliterated_address": "addr_indic", "formerly_or_aka_name": "name_has_alt",
         "candidate_from_S3": "cand_is_s3"}


def error_report(pairs: pd.DataFrame, pred: np.ndarray, E: pd.DataFrame, model_dir: str, n_examples: int = 40) -> dict:
    """pairs: pruned training pairs (s1, cand, label, flag features); pred: predicted-match mask over them;
    E: entities parquet (s1, n_true, n_retrieved_true, n_kept_true, country)."""
    ents = E["s1"].to_numpy()
    ent = pd.Series(np.arange(len(ents)), index=ents).reindex(pairs["s1"].to_numpy()).to_numpy()
    lab = pairs["label"].to_numpy().astype(bool)
    pred = np.asarray(pred, bool)
    n_true = E["n_true"].to_numpy(np.float64)
    tp = np.bincount(ent, weights=(pred & lab).astype(np.float64), minlength=len(ents))
    fp = np.bincount(ent, weights=(pred & ~lab).astype(np.float64), minlength=len(ents))
    f = f05_per_entity(ent, pred, lab, n_true)
    T = max(1.0, n_true.sum())
    rep = {"true_pairs": {
        "lost_by_retrieval": float((n_true - E["n_retrieved_true"]).sum() / T),
        "lost_by_pruning": float((E["n_retrieved_true"] - E["n_kept_true"]).sum() / T),
        "missed_by_matcher": float((E["n_kept_true"].to_numpy() - tp).sum() / T),
        "found": float(tp.sum() / T)},
        "false_merges_per_1000_entities": float(1000 * fp.sum() / max(1, len(ents)))}

    def loss(mask):
        return {"singleton_with_false_merge": float(((1 - f) * (mask & (n_true == 0) & (fp > 0))).sum() / max(1, mask.sum())),
                "false_merge_on_matched_entity": float(((1 - f) * (mask & (n_true > 0) & (fp > 0))).sum() / max(1, mask.sum())),
                "misses_only": float(((1 - f) * (mask & (fp == 0) & (n_true > 0))).sum() / max(1, mask.sum())),
                "macro_f05": float(f[mask].mean()) if mask.any() else float("nan")}

    country = E["country"].to_numpy()
    rep["macro_f05_lost_by"] = {"all": loss(np.ones(len(ents), bool)),
                                **{str(c): loss(country == c) for c in np.unique(country)}}
    by = {}
    for name, col in FLAGS.items():
        if col not in pairs:
            continue
        on = pairs[col].to_numpy() > 0
        for v, m in (("yes", on), ("no", ~on)):
            pos, neg = m & lab, m & ~lab
            by[f"{name}={v}"] = {"pairs": int(m.sum()), "true_pairs": int(pos.sum()),
                                 "miss_rate": float((pos & ~pred).sum() / max(1, pos.sum())),
                                 "false_merge_rate": float((neg & pred).sum() / max(1, neg.sum()))}
    rep["by_pair_type"] = by

    # examples: the most confident mistakes of each kind, with the records as written
    score = pairs["p_final"].to_numpy() if "p_final" in pairs else np.zeros(len(pairs))
    fn = np.flatnonzero(lab & ~pred)
    fp_i = np.flatnonzero(~lab & pred)
    pick = np.r_[fn[np.argsort(score[fn])][:n_examples], fp_i[np.argsort(-score[fp_i])][:n_examples]]
    if len(pick):
        recs = data.load(model_dir, "train", ["entity_id", "business_name", "business_address"])
        rows = pairs.iloc[pick]
        a, b = rows["s1"].to_numpy(), rows["cand"].to_numpy()
        pd.DataFrame({"error": np.where(lab[pick], "missed", "false_merge"), "p": score[pick],
                      "s1_id": recs["entity_id"].to_numpy()[a], "s1_name": recs["business_name"].to_numpy()[a],
                      "s1_address": recs["business_address"].to_numpy()[a],
                      "cand_id": recs["entity_id"].to_numpy()[b], "cand_name": recs["business_name"].to_numpy()[b],
                      "cand_address": recs["business_address"].to_numpy()[b]}
                     ).to_csv(os.path.join(model_dir, "error_examples.tsv"), sep="\t", index=False)
    with open(os.path.join(model_dir, "error_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)

    t = rep["true_pairs"]
    print(f"[errors] true pairs: {t['found']:.2%} found | missed by retrieval {t['lost_by_retrieval']:.2%}, by pruning "
          f"{t['lost_by_pruning']:.2%}, by the matcher {t['missed_by_matcher']:.2%} | "
          f"{rep['false_merges_per_1000_entities']:.1f} false merges per 1,000 entities")
    for k, v in rep["macro_f05_lost_by"].items():
        print(f"[errors] {k:>8}: macro F0.5 {v['macro_f05']:.4f} | lost to false merges on singletons "
              f"{v['singleton_with_false_merge']:.4f}, false merges on matched entities {v['false_merge_on_matched_entity']:.4f}, "
              f"misses only {v['misses_only']:.4f}")
    for k, v in by.items():
        print(f"[errors] {k:<30} {v['true_pairs']:>9,} true pairs | miss rate {v['miss_rate']:.3%} | "
              f"false-merge rate {v['false_merge_rate']:.4%}")
    print(f"[errors] details: {os.path.join(model_dir, 'error_report.json')}, examples: error_examples.tsv")
    return rep
