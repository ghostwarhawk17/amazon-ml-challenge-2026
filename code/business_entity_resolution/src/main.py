"""End-to-end CLI for the ranking half of the pipeline (after src.data and src.retrieval).

    python -m src.data build        --data-dir ../../dataset --model-dir models    # normalise + cache (CPU)
    python -m src.retrieval train   --data-dir ../../dataset --model-dir models    # fine-tune bi-encoder (GPU)
    python -m src.retrieval index   --data-dir ../../dataset --model-dir models    # candidate pairs (GPU)
    python -m src.main prepare      --data-dir ../../dataset --model-dir models    # stage 1 + pruning (CPU)
    [python -m src.cross_encoder --name mdeberta ...]                              # optional (GPU)
    python -m src.main stage2       --data-dir ../../dataset --model-dir models --out-dir ../../output [--extra mdeberta]

    python -m src.main score --pred X.tsv --truth Y.tsv    # official macro F0.5 on any predictions
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import data
from .analysis import FLAGS, error_report
from .cascade import choose_pruning, merge_extra, oracle_f05, prune_mask, stage2_features
from .config import PipelineConfig
from .decision import decide, select, tune
from .features import REC_COLS, build_features, feature_columns, idf_table, peer_features
from .io_utils import read_tsv, write_submission
from .metrics import report, report_pairs
from .model import importance, make_folds, predict_cv, save_boosters, train_cv
from .normalize import normalize_country

LOAD_COLS = sorted(set(REC_COLS) | {"entity_id", "country_norm", "split"})


def _paths(model_dir: str) -> dict:
    for d in ("cache", "pairs", "extra"):
        os.makedirs(os.path.join(model_dir, d), exist_ok=True)
    j = lambda *p: os.path.join(model_dir, *p)
    return {"meta_stage1": j("meta_stage1.json"), "meta": j("meta.json"),
            "cands_train": j("cache", "cands_train.parquet"), "cands_test": j("cache", "cands_test.parquet"),
            "train_cache": j("cache", "stage2_train.parquet"), "test_cache": j("cache", "stage2_test.parquet"),
            "entities": j("cache", "train_entities.parquet"),
            "train_pairs": j("pairs", "train_pruned.parquet"), "test_pairs": j("pairs", "test_pruned.parquet"),
            "extra": j("extra")}


def _retrieved(path: str, cfg) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"[prepare] {path} not found: run `python -m src.retrieval index` first")
    c = pd.read_parquet(path)
    keep = (c["fwd_rank"] <= cfg.blocking.k_fwd) | (c["rev_rank"] <= cfg.blocking.k_rev)
    return c[keep.to_numpy()].reset_index(drop=True)


def _sample_entities(recs: pd.DataFrame, truth: dict, max_n: int, seed: int):
    """'rank' S1 rows (never used to train the bi-encoder), sampled per country x has-match stratum."""
    rank = np.flatnonzero(recs["split"].to_numpy() == "rank")
    strata = recs["country_norm"].to_numpy()[rank] + "|" + np.where([bool(truth[i]) for i in rank], "m", "s")
    if max_n and len(rank) > max_n:
        rng = np.random.default_rng(seed)
        pick = []
        for key in np.unique(strata):
            idx = np.flatnonzero(strata == key)
            pick.append(rng.choice(idx, size=int(round(max_n * len(idx) / len(rank))), replace=False))
        sel = np.sort(np.concatenate(pick))
        rank, strata = rank[sel], strata[sel]
    return rank.astype(np.int32), strata


def _label(pairs: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    key = lambda a, b: a.astype(np.int64) * (1 << 32) + b.astype(np.int64)
    return np.isin(key(pairs["s1"].to_numpy(), pairs["cand"].to_numpy()),
                   key(gt["s1"].to_numpy(), gt["cand"].to_numpy())).astype(np.int8)


def _tune_and_report(s1, cand, p, label, ents, n_true, country, dcfg, tag):
    best, best_all, results = tune(s1, cand, p, label, ents, n_true, dcfg)
    ent = pd.Series(np.arange(len(ents)), index=ents).reindex(s1).to_numpy()
    rep = report_pairs(ent, select(s1, cand, p, best, dcfg.max_items_expected_f), label, n_true, country)
    rep["naive_threshold_0.5"] = report_pairs(ent, p >= 0.5, label, n_true)["macro_f05"]
    print(f"[{tag}] top decision settings (out-of-fold macro F0.5; expected-F rules on up to "
          f"{dcfg.max_tune_entities:,} entities):")
    for sc, prm in results[:5]:
        print(f"         {sc:.4f}  {prm}")
    print(f"[{tag}] naive p>=0.5 : {rep['naive_threshold_0.5']:.4f}   ->   chosen {best}: {rep['macro_f05']:.4f}")
    print(f"[{tag}] {json.dumps(rep)}")
    return best, rep, results


def _print_grid(grid, chosen):
    caps = sorted({g["cap"] for g in grid})
    print("[cascade] pruning grid: pairs per entity / oracle macro F0.5 (rows: p1 cutoff, columns: cap per entity, 0 = none)")
    print("          cutoff " + "".join(f"{('cap ' + str(c)) if c else 'no cap':>17}" for c in caps))
    for c in sorted({g["cutoff"] for g in grid}):
        cells = {g["cap"]: g for g in grid if g["cutoff"] == c}
        row = "".join(f"{cells[k]['pairs_per_entity']:>8.2f} / {cells[k]['oracle_f05']:.4f}" + ("*" if (c, k) == chosen else " ")
                      for k in caps)
        print(f"        {c:>8g} {row}")


def cmd_prepare(args, cfg: PipelineConfig):
    """Features + stage 1 (K-fold) on sampled 'rank' entities, pruning, then the same for the test set."""
    t0 = time.time()
    P = _paths(args.model_dir)
    recs = data.load(args.model_dir, "train", LOAD_COLS)
    truth, gt = data.load_truth(args.data_dir, recs)
    multi = int(gt["cand"].duplicated(keep=False).sum())
    print(f"[gt] {len(gt):,} true pairs | singleton share {np.mean([not v for v in truth.values()]):.3f} | "
          f"S2/S3 records linked to >1 S1 entity: {multi}")
    ents, strata = _sample_entities(recs, truth, cfg.model.max_train_entities, cfg.model.seed)
    n_true = np.array([len(truth[i]) for i in ents], np.float64)
    country = recs["country_norm"].to_numpy()[ents]

    cands = _retrieved(P["cands_train"], cfg)
    cands = cands[np.isin(cands["s1"].to_numpy(), ents)].reset_index(drop=True)
    idf = idf_table(recs)
    print(f"[prepare] {len(ents):,} 'rank' entities, {len(cands):,} retrieved pairs ({len(cands) / len(ents):.1f}/entity) "
          f"| {time.time() - t0:.0f}s")
    feats = build_features(cands, recs, idf, cfg.blocking.chunk_entities)
    del cands
    feats["label"] = _label(feats, gt)
    ent_pos = pd.Series(np.arange(len(ents)), index=ents)
    ent = ent_pos.reindex(feats["s1"].to_numpy()).to_numpy()
    s1, cand, lab = feats["s1"].to_numpy(), feats["cand"].to_numpy(), feats["label"].to_numpy()
    br = {"pair_recall": float(lab.sum() / max(1, n_true.sum())), "pairs_per_entity": len(feats) / len(ents),
          "oracle_macro_f05": oracle_f05(ent, lab, n_true)}
    print(f"[blocking] retrieval on 'rank' entities: pair recall {br['pair_recall']:.4f} | "
          f"{br['pairs_per_entity']:.1f} pairs/entity | oracle macro F0.5 ceiling {br['oracle_macro_f05']:.4f}")

    # ---- stage 1: K-fold LightGBM on every retrieved pair
    cols1 = feature_columns(feats)
    folds = make_folds(ents, strata, cfg.model.n_folds, cfg.model.seed)
    feats["fold"] = folds.reindex(s1).to_numpy()
    print(f"[stage1] {cfg.model.n_folds} folds | {len(feats):,} pairs | {len(cols1)} features | positive rate {lab.mean():.4f}")
    oof1, b1 = train_cv(feats, cols1, cfg.model)
    feats["p1"] = oof1.astype(np.float32)
    dec1, rep1, _ = _tune_and_report(s1, cand, oof1, lab, ents, n_true, country, cfg.decision, "stage1")

    # ---- prune: the survivors are stage 2's input and the candidate_pairs.tsv content
    cutoff, cap, grid = choose_pruning(s1, oof1, lab, ent, n_true, cfg.cascade)
    _print_grid(grid, (cutoff, cap))
    g = next(x for x in grid if x["cutoff"] == cutoff and x["cap"] == cap)
    print(f"[cascade] keep p1 >= {cutoff:g}, at most {cap or 'all'} per entity: {g['pairs']:,} of {len(feats):,} pairs "
          f"({g['pairs_per_entity']:.2f}/entity) | pair recall {g['pair_recall']:.4f} | oracle {g['oracle_f05']:.4f} "
          f"(-{g['oracle_drop']:.4f})")
    keep = prune_mask(s1, oof1, cutoff, cap)
    # where the entities' true pairs survive, for the error report after stage 2
    pd.DataFrame({"s1": ents, "n_true": n_true, "country": country,
                  "n_retrieved_true": np.bincount(ent, weights=lab, minlength=len(ents)),
                  "n_kept_true": np.bincount(ent[keep], weights=lab[keep], minlength=len(ents))}
                 ).to_parquet(P["entities"], index=False)
    kept = peer_features(feats[keep].reset_index(drop=True), recs)
    kept.to_parquet(P["train_cache"], index=False)
    kept[["s1", "cand", "label", "fold", "p1"]].to_parquet(P["train_pairs"], index=False)
    save_boosters(b1, args.model_dir, "stage1")
    importance(b1, cols1).to_csv(os.path.join(args.model_dir, "feature_importance_stage1.csv"), index=False)
    del feats, kept, recs

    # ---- test: same features, stage-1 fold models and pruning, chunk by chunk
    trecs = data.load(args.model_dir, "test", LOAD_COLS)
    tcands = _retrieved(P["cands_test"], cfg)

    def score_and_prune(f):
        f["p1"] = predict_cv(b1, f[cols1]).astype(np.float32)
        return prune_mask(f["s1"].to_numpy(), f["p1"].to_numpy(), cutoff, cap)

    tkept = build_features(tcands, trecs, idf_table(trecs), cfg.blocking.chunk_entities, keep=score_and_prune)
    tkept = peer_features(tkept, trecs)
    tkept.to_parquet(P["test_cache"], index=False)
    tkept[["s1", "cand", "p1"]].to_parquet(P["test_pairs"], index=False)
    ts1 = np.flatnonzero(trecs["source"].to_numpy() == "S1")
    per = pd.Series(ts1).map(tkept.groupby("s1").size()).fillna(0).to_numpy()
    summ = pd.DataFrame({"country": trecs["country_norm"].to_numpy()[ts1], "n": per}).groupby("country")["n"].agg(
        entities="size", no_candidates=lambda x: float((x == 0).mean()), pairs_per_entity="mean")
    print(f"[cascade] test: {len(tkept):,} of {len(tcands):,} retrieved pairs kept\n{summ.round(3).to_string()}")

    meta = {"stage1_features": cols1, "cutoff": cutoff, "cap": cap, "n_folds": cfg.model.n_folds, "config": cfg.to_dict(),
            "blocking_train": br, "cascade_grid": grid, "stage1_oof_report": rep1, "stage1_decision": dec1,
            "test_pairs_per_entity": float(per.mean())}
    with open(P["meta_stage1"], "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[done] prepare finished in {(time.time() - t0) / 60:.1f} min -> {args.model_dir}")


def cmd_stage2(args, cfg: PipelineConfig):
    """Stage 2 (K-fold) on the pruned pairs + extra model scores, decision tuning, test prediction."""
    t0 = time.time()
    P = _paths(args.model_dir)
    with open(P["meta_stage1"]) as f:
        meta1 = json.load(f)
    tr = pd.read_parquet(P["train_cache"])
    te = pd.read_parquet(P["test_cache"])
    extras = [e.strip() for e in (args.extra or "").split(",") if e.strip()]
    for name in extras:
        tr = merge_extra(tr, P["extra"], name, "oof")
        te = merge_extra(te, P["extra"], name, "test")
    tr = stage2_features(tr, [f"x_{n}" for n in extras])
    te = stage2_features(te, [f"x_{n}" for n in extras])
    cols2 = feature_columns(tr)
    E = pd.read_parquet(P["entities"])
    ents, n_true, country = E["s1"].to_numpy(), E["n_true"].to_numpy(), E["country"].to_numpy()

    print(f"[stage2] {len(cols2)} features ({len(extras)} extra model scores) | {len(tr):,} training pairs")
    oof2, b2 = train_cv(tr, cols2, cfg.model)
    s1, cand, lab = tr["s1"].to_numpy(), tr["cand"].to_numpy(), tr["label"].to_numpy()
    dec2, rep2, results = _tune_and_report(s1, cand, oof2, lab, ents, n_true, country, cfg.decision, "stage2")
    f1 = meta1["stage1_oof_report"]["macro_f05"]
    use_stage2 = rep2["macro_f05"] > f1
    print(f"[stage2] out-of-fold macro F0.5: stage 1 alone {f1:.4f} -> stage 2 {rep2['macro_f05']:.4f} | "
          f"the submission uses {'stage 2' if use_stage2 else 'stage 1 (stage 2 did not beat it)'}")
    p_oof = oof2 if use_stage2 else tr["p1"].to_numpy(np.float64)
    dec_oof = dec2 if use_stage2 else meta1["stage1_decision"]
    flags = [c for c in FLAGS.values() if c in tr]
    error_report(tr[["s1", "cand", "label", *flags]].assign(p_final=p_oof),
                 select(s1, cand, p_oof, dec_oof, cfg.decision.max_items_expected_f), E, args.model_dir)

    # ---- test: fold-averaged stage 2 (or stage 1), tuned decision rule, candidate file = pruned pairs
    trecs = data.load(args.model_dir, "test", ["entity_id", "source", "country_norm"])
    ts1 = np.flatnonzero(trecs["source"].to_numpy() == "S1")
    p2 = predict_cv(b2, te[cols2])
    p_final, dec_final = (p2, dec2) if use_stage2 else (te["p1"].to_numpy(np.float64), meta1["stage1_decision"])
    pred = decide(te["s1"].to_numpy(), te["cand"].to_numpy(), p_final, dec_final, ts1, cfg.decision.max_items_expected_f)
    write_submission(pred, te["s1"].to_numpy(), te["cand"].to_numpy(), trecs, args.out_dir)
    n_match = np.array([len(pred[i]) for i in ts1])
    summ = pd.DataFrame({"country": trecs["country_norm"].to_numpy()[ts1], "n": n_match}).groupby("country")["n"].agg(
        entities="size", predicted_singletons=lambda x: float((x == 0).mean()), mean_matches="mean")
    print(f"[predict] {'stage 2' if use_stage2 else 'stage 1'} with decision {dec_final}\n{summ.round(3).to_string()}")

    save_boosters(b2, args.model_dir, "stage2")
    importance(b2, cols2).to_csv(os.path.join(args.model_dir, "feature_importance_stage2.csv"), index=False)
    tr[["s1", "cand", "label", "fold", "p1"]].assign(p2=oof2.astype(np.float32)).to_parquet(
        os.path.join(args.model_dir, "oof_scored_pairs.parquet"), index=False)
    meta = {**meta1, "stage2_features": cols2, "extras": extras, "decision": dec_final,
            "final_stage": 2 if use_stage2 else 1, "stage2_decision": dec2, "stage2_oof_report": rep2,
            "decision_grid": [{"f05": sc, **prm} for sc, prm in results]}
    with open(P["meta"], "w") as f:
        json.dump(meta, f, indent=2, default=str)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        te[["s1", "cand"]].assign(p=p2).to_parquet(os.path.join(args.debug_dir, "test_scored_pairs.parquet"), index=False)
    print(f"[done] stage 2 finished in {(time.time() - t0) / 60:.1f} min")


def cmd_score(args):
    """Score any matching_results.tsv against a ground-truth file with the official macro F0.5."""
    pdf, tdf = read_tsv(args.pred), read_tsv(args.truth)
    split = lambda x: [i for i in str(x).split(",") if i]
    pred = dict(zip(pdf["source1_entity_id"], pdf["matched_entity_ids"].map(split)))
    truth = dict(zip(tdf["source1_entity_id"], tdf["matched_entity_ids"].map(split)))
    country_of = None  # per-country breakdown when the matching test_source1.tsv can be found
    for d in (os.path.dirname(os.path.abspath(args.truth)), args.data_dir):
        p = os.path.join(d, "test", "test_source1.tsv")
        if os.path.exists(p):
            s1 = read_tsv(p)
            country_of = dict(zip(s1["entity_id"].str.strip(), [normalize_country(x) for x in s1["country"]]))
            break
    out = report(pred, truth, list(truth), country_of)
    cand_path = os.path.join(os.path.dirname(os.path.abspath(args.pred)), "candidate_pairs.tsv")
    if os.path.exists(cand_path):  # the candidate set's size counts in the final ranking
        c = read_tsv(cand_path)
        cl = c["candidate_entity_ids"].map(split)
        ct = dict(zip(c["source1_entity_id"], cl))
        n_true = sum(len(v) for v in truth.values())
        out["candidates_per_entity"] = float(cl.map(len).mean())
        out["candidate_pair_recall"] = sum(len(set(v) & set(ct.get(k, []))) for k, v in truth.items()) / max(1, n_true)
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Business entity resolution (Amazon ML Challenge): ranking stages")
    ap.add_argument("command", choices=["prepare", "stage2", "all", "score"])
    ap.add_argument("--data-dir", default=os.path.join("..", "..", "dataset"))
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default=os.path.join("..", "..", "output"))
    ap.add_argument("--debug-dir", default=None, help="optional: dump scored test pairs here")
    ap.add_argument("--extra", default="", help="stage2/all: comma-separated extra model scores from <model-dir>/extra "
                                                "(e.g. mdeberta,xlmr_large)")
    ap.add_argument("--folds", type=int, default=None, help="number of CV folds (default 5)")
    ap.add_argument("--train-entities", type=int, default=None, help="'rank' entities to train on (0 = all)")
    ap.add_argument("--k-fwd", type=int, default=None, help="use retrieved pairs up to this forward rank")
    ap.add_argument("--k-rev", type=int, default=None, help="... or up to this reverse rank")
    ap.add_argument("--max-oracle-drop", type=float, default=None,
                    help="cascade: oracle F0.5 the pruning may cost (default 0.002)")
    ap.add_argument("--target-pairs", type=float, default=None,
                    help="cascade: instead, the best oracle F0.5 with at most this many pairs per entity")
    ap.add_argument("--no-prune", action="store_true", help="cascade: keep every retrieved pair for stage 2")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pred", help="score: a matching_results.tsv")
    ap.add_argument("--truth", help="score: a ground-truth tsv (source1_entity_id, matched_entity_ids)")
    args = ap.parse_args()

    cfg = PipelineConfig()
    if args.folds: cfg.model.n_folds = args.folds
    if args.train_entities is not None: cfg.model.max_train_entities = args.train_entities
    if args.seed is not None: cfg.model.seed = args.seed
    if args.k_fwd: cfg.blocking.k_fwd = args.k_fwd
    if args.k_rev is not None: cfg.blocking.k_rev = args.k_rev
    if args.max_oracle_drop is not None: cfg.cascade.max_oracle_drop = args.max_oracle_drop
    if args.target_pairs: cfg.cascade.target_pairs_per_entity = args.target_pairs
    if args.no_prune: cfg.cascade.prune = False

    if args.command == "score":
        cmd_score(args)
        return
    if args.command in ("prepare", "all"):
        cmd_prepare(args, cfg)
    if args.command in ("stage2", "all"):
        cmd_stage2(args, cfg)


if __name__ == "__main__":
    main()
