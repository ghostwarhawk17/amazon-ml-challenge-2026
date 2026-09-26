"""Reading the official TSVs and writing / self-validating the two submission files."""
import os

import numpy as np
import pandas as pd


def read_tsv(path: str) -> pd.DataFrame:
    # tab separator is mandatory: addresses and ID lists contain commas. Fields that contain a double quote
    # are CSV-escaped in the official files ("""ehpad Club SAS"), which pandas' default quoting decodes.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)


def write_submission(pred: dict, cand_s1: np.ndarray, cand_rows: np.ndarray, test_recs: pd.DataFrame, out_dir: str):
    """pred: {S1 row: [candidate rows]}; (cand_s1, cand_rows): the candidate pairs (candidate_pairs.tsv).

    Rows are positions in the test records table; ids are written in the order of test_source1.tsv.
    """
    os.makedirs(out_dir, exist_ok=True)
    ids = test_recs["entity_id"].to_numpy()
    src = test_recs["source"].to_numpy()
    s1_rows = np.flatnonzero(src == "S1")
    ok = np.isin(src[cand_rows], ["S2", "S3"])
    cands = pd.Series(cand_rows[ok]).groupby(cand_s1[ok]).apply(lambda x: list(dict.fromkeys(x))).to_dict()
    m_path = os.path.join(out_dir, "matching_results.tsv")
    c_path = os.path.join(out_dir, "candidate_pairs.tsv")
    n_pairs = n_match = 0
    with open(m_path, "w", encoding="utf-8", newline="\n") as fm, open(c_path, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for r in s1_rows.tolist():
            c = cands.get(r, [])
            cset = set(c)
            m = [x for x in dict.fromkeys(pred.get(r, [])) if x in cset]  # matches ⊆ candidates
            n_pairs += len(c)
            n_match += len(m)
            fm.write(f"{ids[r]}\t{','.join(sorted(ids[m]))}\n")
            fc.write(f"{ids[r]}\t{','.join(sorted(ids[c]))}\n")
    print(f"[write] {len(s1_rows):,} S1 entities | {n_pairs / max(1, len(s1_rows)):.2f} candidates and "
          f"{n_match / max(1, len(s1_rows)):.2f} matches per entity")
    problems = validate_files(m_path, c_path, test_recs)
    if problems:
        print("[validate] PROBLEMS:\n  - " + "\n  - ".join(problems))
    else:
        print(f"[validate] PASS  ->  {m_path}\n                     {c_path}")
    return m_path, c_path, problems


def validate_files(m_path: str, c_path: str, test_recs: pd.DataFrame):
    """Mirror of the rules in the problem statement (the official utils/validate_submission.py is authoritative)."""
    src = test_recs["source"].to_numpy()
    ids = test_recs["entity_id"].to_numpy()
    s1 = set(ids[src == "S1"])
    valid = set(ids[src != "S1"])
    problems, lists = [], {}
    for path, col in ((m_path, "matched_entity_ids"), (c_path, "candidate_entity_ids")):
        df = read_tsv(path)
        name = os.path.basename(path)
        if list(df.columns) != ["source1_entity_id", col]:
            problems.append(f"{name}: header must be source1_entity_id<TAB>{col}, got {list(df.columns)}")
            continue
        rows = df["source1_entity_id"].tolist()
        if len(rows) != len(set(rows)):
            problems.append(f"{name}: duplicate source1_entity_id rows")
        if set(rows) != s1:
            problems.append(f"{name}: {len(s1 - set(rows))} S1 ids missing, {len(set(rows) - s1)} unknown")
        d = {}
        for sid, lst in zip(rows, df[col]):
            items = [x for x in lst.split(",") if x] if lst else []
            if len(items) != len(set(items)):
                problems.append(f"{name}: duplicate ids in list for {sid}")
            bad = [x for x in items if x not in valid]
            if bad:
                problems.append(f"{name}: {sid} lists non-S2/S3 or unknown ids {bad[:3]}")
            d[sid] = set(items)
        lists[col] = d
    if len(lists) == 2:
        leak = sum(1 for sid, m in lists["matched_entity_ids"].items() if not m <= lists["candidate_entity_ids"].get(sid, set()))
        if leak:
            problems.append(f"{leak} entities have matches that are not in candidate_pairs.tsv")
    return problems[:20]
