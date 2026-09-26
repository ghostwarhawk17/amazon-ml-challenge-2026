"""Load and normalise the official data once, cache it as parquet, and split the training entities.

    python -m src.data build --data-dir ../../dataset --model-dir models
    python -m src.data mini  --data-dir ../../dataset --out ../../dataset_mini --entities 20000

`build` writes, in <model-dir>:
  translit_dict.json            Indic word -> Latin word, learned from the training pairs (src/translit.py)
  cache/records_train.parquet   every train record with its normalised columns and, for S1 rows, a `split`
  cache/records_test.parquet    every test record with its normalised columns
Rows keep a fixed order (S1, S2, S3 files in turn), so a record is addressed by its row number everywhere
downstream: candidate pairs are pairs of int32 row numbers, not strings.

Training S1 entities are split once (stratified by country and has-match):
  "enc"  : fine-tunes the retrieval bi-encoder (src/retrieval.py)
  "rank" : everything after retrieval -- LightGBM, cross-encoders, the decision rule
The ranker never sees entities the bi-encoder was trained on, so its training candidates look like the
test candidates (a retriever scores its own training pairs too well).

`mini` cuts a small copy of the official *training* data with the same layout, for smoke tests on a
laptop. Its "test" part is held-out training entities, so the pipeline can be scored end to end with
`main score --truth <mini>/test_ground_truth.tsv`.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .io_utils import read_tsv
from .normalize import prepare
from .translit import has_indic, learn_dictionary

SPLITS = ("train", "test")


def paths(model_dir: str) -> dict:
    os.makedirs(os.path.join(model_dir, "cache"), exist_ok=True)
    return {"dict": os.path.join(model_dir, "translit_dict.json"),
            "train": os.path.join(model_dir, "cache", "records_train.parquet"),
            "test": os.path.join(model_dir, "cache", "records_test.parquet")}


def read_raw(data_dir: str, split: str) -> pd.DataFrame:
    frames = []
    for k in (1, 2, 3):
        df = read_tsv(os.path.join(data_dir, split, f"{split}_source{k}.tsv"))
        missing = [c for c in ("entity_id", "business_name", "business_address", "country") if c not in df.columns]
        if missing:
            raise ValueError(f"{split}_source{k}.tsv is missing columns {missing}")
        frames.append(df[["entity_id", "business_name", "business_address", "country"]])
    recs = pd.concat(frames, ignore_index=True)
    recs["entity_id"] = recs["entity_id"].str.strip()
    dup = recs["entity_id"].duplicated()
    if dup.any():
        print(f"[warn] {int(dup.sum())} duplicate entity_ids in {split}; keeping the first")
        recs = recs[~dup].reset_index(drop=True)
    return recs


def read_truth(data_dir: str) -> pd.DataFrame:
    """Ground-truth pairs [s1_id, cand_id] (entities without matches have no rows)."""
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    s1 = gt["source1_entity_id"].str.strip().to_numpy()
    lists = gt["matched_entity_ids"].str.split(",")
    n = lists.str.len().fillna(0).astype(int).to_numpy()
    pairs = pd.DataFrame({"s1_id": np.repeat(s1, n), "cand_id": [x.strip() for l in lists.dropna() for x in l]})
    return pairs[pairs["cand_id"] != ""].drop_duplicates().reset_index(drop=True)


def entity_split(s1: pd.DataFrame, matched: set, enc_frac: float, seed: int) -> np.ndarray:
    """'enc' / 'rank' per S1 row, stratified by country and has-match."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(s1), dtype=object)
    strata = s1["country_norm"].to_numpy() + "|" + np.where(s1["entity_id"].isin(matched), "m", "s")
    for key in np.unique(strata):
        idx = np.flatnonzero(strata == key)
        rng.shuffle(idx)
        n_enc = int(round(enc_frac * len(idx)))
        out[idx[:n_enc]], out[idx[n_enc:]] = "enc", "rank"
    return out


def build(data_dir: str, model_dir: str, enc_frac: float = 0.2, seed: int = 42, workers: int = None):
    t0 = time.time()
    p = paths(model_dir)
    raw = read_raw(data_dir, "train")
    gt = read_truth(data_dir)
    print(f"[data] train: {len(raw):,} records, {len(gt):,} true pairs ({time.time() - t0:.0f}s)")

    # Indic -> Latin dictionary from the true pairs whose S2/S3 side has Indic text (names and addresses)
    by_id = raw.set_index("entity_id")
    left, right = by_id.loc[gt["s1_id"]], by_id.loc[gt["cand_id"]]
    has = np.array([has_indic(n) or has_indic(a) for n, a in zip(right["business_name"], right["business_address"])])
    ex = [(a, b) for a, b in zip(left["business_name"].to_numpy()[has], right["business_name"].to_numpy()[has])]
    ex += [(a, b) for a, b in zip(left["business_address"].to_numpy()[has], right["business_address"].to_numpy()[has])]
    dictionary = learn_dictionary(ex)
    with open(p["dict"], "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=0, sort_keys=True)
    print(f"[data] transliteration dictionary: {len(dictionary):,} words from {int(has.sum()):,} pairs "
          f"({time.time() - t0:.0f}s)")
    del by_id, left, right, ex

    recs = prepare(raw, dictionary, workers)
    s1 = recs["source"] == "S1"
    recs["split"] = ""
    recs.loc[s1, "split"] = entity_split(recs[s1], set(gt["s1_id"]), enc_frac, seed)
    recs.to_parquet(p["train"], index=False)
    print(f"[data] train normalised: {recs['source'].value_counts().to_dict()} | "
          f"{recs.loc[s1, 'split'].value_counts().to_dict()} ({time.time() - t0:.0f}s)")
    _country_check(recs, "train")
    del recs, raw

    recs = prepare(read_raw(data_dir, "test"), dictionary, workers)
    recs["split"] = ""
    recs.to_parquet(p["test"], index=False)
    print(f"[data] test normalised: {recs['source'].value_counts().to_dict()} | "
          f"countries {recs['country_norm'].value_counts().to_dict()} ({time.time() - t0:.0f}s)")
    _country_check(recs, "test")


def _country_check(recs, split):
    """Retrieval runs within a country label, so a label spelled differently across sources never meets."""
    c1 = set(recs.loc[recs["source"] == "S1", "country_norm"])
    c23 = set(recs.loc[recs["source"] != "S1", "country_norm"])
    if c1 != c23:
        print(f"[warn] {split}: country labels differ between Source 1 and Sources 2/3 -- only in S1: "
              f"{sorted(c1 - c23)}, only in S2/S3: {sorted(c23 - c1)}. These records are never compared.")


def load(model_dir: str, split: str, columns=None) -> pd.DataFrame:
    path = paths(model_dir)[split]
    if not os.path.exists(path):
        raise SystemExit(f"[data] {path} not found: run `python -m src.data build` first")
    return pd.read_parquet(path, columns=columns)


def load_truth(data_dir: str, recs: pd.DataFrame):
    """Ground truth as row numbers: (truth {s1_row: set(cand_rows)}, pairs DataFrame [s1, cand])."""
    gt = read_truth(data_dir)
    row = pd.Series(np.arange(len(recs), dtype=np.int32), index=recs["entity_id"].to_numpy())
    pairs = pd.DataFrame({"s1": row.reindex(gt["s1_id"]).to_numpy(), "cand": row.reindex(gt["cand_id"]).to_numpy()})
    if pairs.isna().any().any():
        raise SystemExit("[data] the ground truth references ids that are not in the training records")
    pairs = pairs.astype(np.int32)
    truth = {int(i): set() for i in np.flatnonzero(recs["source"].to_numpy() == "S1")}
    for a, b in zip(pairs["s1"].to_numpy(), pairs["cand"].to_numpy()):
        truth[int(a)].add(int(b))
    return truth, pairs


# ----------------------------------------------------------------------------------------- mini dataset
def make_mini(data_dir: str, out_dir: str, n_entities: int = 20000, test_frac: float = 0.3,
              distractors: float = 1.0, seed: int = 42):
    """Small copy of the training data (same layout); test = held-out training entities + their truth.

    Keeps every match of the sampled S1 entities, plus `distractors` x as many other S2/S3 records of the
    same countries (unmatched records or matches of entities that were not sampled). Reads with pyarrow,
    so it runs in a few GB of RAM.
    """
    import pyarrow.csv as pcsv
    import pyarrow.compute as pc

    opts = dict(parse_options=pcsv.ParseOptions(delimiter="\t"),
                convert_options=pcsv.ConvertOptions(strings_can_be_null=False))
    rd = lambda f: pcsv.read_csv(os.path.join(data_dir, "train", f), **opts).to_pandas()
    rng = np.random.default_rng(seed)
    s1 = rd("train_source1.tsv")
    pick = pd.concat([g.sample(n=min(len(g), int(round(n_entities * len(g) / len(s1)))), random_state=seed)
                      for _, g in s1.groupby("country")])
    pick_ids = pick["entity_id"].to_numpy()
    part = dict(zip(pick_ids, np.where(rng.random(len(pick_ids)) < test_frac, "test", "train")))
    gt = read_truth(data_dir)
    g = gt[gt["s1_id"].isin(part)]
    cand_part = dict(zip(g["cand_id"], g["s1_id"].map(part)))
    countries = set(pick["country"])
    per_file = int(distractors * len(cand_part) / 2)
    frames = {1: pick}
    for k in (2, 3):
        t = pcsv.read_csv(os.path.join(data_dir, "train", f"train_source{k}.tsv"), **opts)
        mine = t.filter(pc.is_in(t["entity_id"], value_set=pa_array(list(cand_part))))
        rest = t.filter(pc.and_(pc.invert(pc.is_in(t["entity_id"], value_set=pa_array(list(cand_part)))),
                                pc.is_in(t["country"], value_set=pa_array(sorted(countries)))))
        take = rng.choice(rest.num_rows, size=min(rest.num_rows, per_file), replace=False)
        dis = rest.take(np.sort(take)).to_pandas()
        cand_part.update(zip(dis["entity_id"], np.where(rng.random(len(dis)) < test_frac, "test", "train")))
        frames[k] = pd.concat([mine.to_pandas(), dis], ignore_index=True)
        del t, mine, rest
    part.update(cand_part)
    for split in ("train", "test"):
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
        n23 = 0
        for k, df in frames.items():
            df = df[df["entity_id"].map(part) == split]
            n23 += len(df) if k > 1 else 0
            df.to_csv(os.path.join(out_dir, split, f"{split}_source{k}.tsv"), sep="\t", index=False)
        ids = pick_ids[np.array([part[i] == split for i in pick_ids])]
        lists = g[g["s1_id"].isin(ids)].groupby("s1_id")["cand_id"].apply(",".join)
        truth = pd.DataFrame({"source1_entity_id": ids,
                              "matched_entity_ids": pd.Series(ids).map(lists).fillna("").to_numpy()})
        name = os.path.join("train", "train_ground_truth.tsv") if split == "train" else "test_ground_truth.tsv"
        truth.to_csv(os.path.join(out_dir, name), sep="\t", index=False)
        print(f"[mini] {split}: {len(ids):,} S1 entities, {n23:,} S2/S3 records")
    print(f"[mini] written to {out_dir} (score test predictions against {os.path.join(out_dir, 'test_ground_truth.tsv')})")


def pa_array(values):
    import pyarrow as pa
    return pa.array(values, type=pa.string())


def main():
    ap = argparse.ArgumentParser(description="Normalise + cache the data, or cut a mini copy for smoke tests")
    ap.add_argument("command", choices=["build", "mini"])
    ap.add_argument("--data-dir", default=os.path.join("..", "..", "dataset"))
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--enc-frac", type=float, default=0.2, help="build: share of train S1 entities for the bi-encoder")
    ap.add_argument("--workers", type=int, default=None, help="build: CPU processes (default: all cores)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="mini: output folder")
    ap.add_argument("--entities", type=int, default=20000, help="mini: number of S1 entities")
    ap.add_argument("--distractors", type=float, default=1.0, help="mini: other S2/S3 records per matched record")
    args = ap.parse_args()
    if args.command == "build":
        build(args.data_dir, args.model_dir, args.enc_frac, args.seed, args.workers)
    else:
        make_mini(args.data_dir, args.out, args.entities, distractors=args.distractors, seed=args.seed)


if __name__ == "__main__":
    main()
