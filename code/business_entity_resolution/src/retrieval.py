"""Candidate generation (blocking) with a fine-tuned multilingual bi-encoder and exact GPU top-k search.

    python -m src.retrieval train --model-dir models      # fine-tune on the "enc" entities (GPU, ~30 min)
    python -m src.retrieval index --model-dir models      # embed every record, top-k per country (GPU)

Why a bi-encoder: each split has ~10M Source 2/3 records, so comparing every S1 record with every
record of its country needs an index, not pairwise string scores. One matrix product on the GPU scores a
chunk of S1 records against millions of records at once. The encoder (multilingual-e5-small, MIT,
118M parameters) is fine-tuned with a contrastive loss on true pairs, so it learns this data's noise
(reordered addresses, legal-suffix games, typos, transliterated names, "formerly" names).

Retrieval runs in both directions within the same country label (an open set: France works as is):
  forward : each S1 record -> its k_fwd nearest S2/S3 records
  reverse : each S2/S3 record -> its k_rev nearest S1 records
A record belongs to at most one S1 entity, so "this S1 is the record's nearest S1" (reverse rank 1) is
strong evidence, and the reverse direction adds true pairs that a crowded S1 neighbourhood pushed out of
the forward top-k. The union is written to <model-dir>/cache/cands_{train,test}.parquet as int32 row
numbers into records_{split}.parquet, with the similarity and rank features the ranker uses.

The bi-encoder is trained only on "enc" entities (src/data.py); the ranker trains on "rank" entities,
whose candidates therefore look like test candidates.
"""
import argparse
import json
import os
import time

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from . import data
from .metrics import macro_f05

DEFAULT_BACKBONE = "intfloat/multilingual-e5-small"


def record_texts(recs: pd.DataFrame) -> list:
    # e5 models expect a "query: " prefix for symmetric tasks
    return ("query: " + recs["name_clean"] + " | " + recs["addr_clean"]).tolist()


class Packed:
    """Token ids of many texts in one flat int32 array (a Python list per text would need ~20 GB)."""

    def __init__(self, tok, texts, max_len: int, chunk: int = 200_000):
        flat, lens = [], []
        for s in range(0, len(texts), chunk):
            ids = tok(texts[s:s + chunk], truncation=True, max_length=max_len)["input_ids"]
            lens.append(np.fromiter((len(x) for x in ids), np.int32, len(ids)))
            flat.append(np.fromiter((t for x in ids for t in x), np.int32, int(lens[-1].sum())))
        self.lengths = np.concatenate(lens) if lens else np.zeros(0, np.int32)
        self.flat = np.concatenate(flat) if flat else np.zeros(0, np.int32)
        self.offsets = np.zeros(len(self.lengths) + 1, np.int64)
        np.cumsum(self.lengths, out=self.offsets[1:])
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.max_len = max_len

    def batch(self, idx):
        idx = np.asarray(idx)
        L = self.lengths[idx]
        cols = np.arange(int(L.max()))
        valid = cols[None, :] < L[:, None]
        ids = np.full(valid.shape, self.pad, np.int64)
        ids[valid] = self.flat[(self.offsets[idx][:, None] + cols[None, :])[valid]]
        return torch.from_numpy(ids), torch.from_numpy(valid.astype(np.int64))


class TextEncoder(torch.nn.Module):
    """Transformer + mean pooling + L2 normalisation (the e5 recipe)."""

    def __init__(self, backbone: str, revision: str = None, freeze_embeddings: bool = False):
        super().__init__()
        self.model = AutoModel.from_pretrained(backbone, revision=revision)
        if freeze_embeddings:
            self.model.get_input_embeddings().weight.requires_grad_(False)

    def forward(self, ids, mask):
        h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return torch.nn.functional.normalize(e.float(), dim=-1)


def _device(args):
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else None
    print(f"[retrieval] device {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'} | "
          f"{'bf16 autocast' if amp else 'fp32'}")
    if dev.type == "cpu":
        print("[retrieval] WARNING: no CUDA GPU -- fine for the mini dataset, far too slow for the real one")
    return dev, amp


@torch.inference_mode()
def encode(model, packed: Packed, device, amp, batch_size: int = 1024, log_every: int = 2000) -> np.ndarray:
    """Embeddings of every packed text (float16, L2-normalised), in input order."""
    model.eval()
    n = len(packed.lengths)
    out = None
    order = np.argsort(packed.lengths, kind="stable")
    t0 = time.time()
    for bi, s in enumerate(range(0, n, batch_size)):
        idx = order[s:s + batch_size]
        ids, mask = packed.batch(idx)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            e = model(ids.to(device), mask.to(device))
        if out is None:
            out = np.zeros((n, e.shape[1]), np.float16)
        out[idx] = e.cpu().numpy().astype(np.float16)
        if (bi + 1) % log_every == 0:
            done = s + len(idx)
            el = time.time() - t0
            print(f"[encode] {done:,}/{n:,} | {done / el:,.0f} texts/s | ~{(n - done) / (done / el) / 60:.1f} min left",
                  flush=True)
    return out if out is not None else np.zeros((0, 384), np.float16)


@torch.inference_mode()
def search(Q: np.ndarray, D: np.ndarray, k: int, device, q_chunk: int = 1024, d_shard: int = 1_000_000):
    """Exact inner-product top-k of every row of Q among the rows of D -> (idx int32, sim float32).

    D is streamed to the GPU one shard at a time and every query keeps a running top-k, so GPU memory is
    one shard + one (q_chunk x shard) score block + the running top-k, whatever the size of D or Q.
    """
    k = min(k, len(D))
    if len(Q) == 0 or k == 0:
        return np.zeros((len(Q), k), np.int32), np.zeros((len(Q), k), np.float32)
    dt = torch.float16 if device.type == "cuda" else torch.float32
    best_v = torch.full((len(Q), k), float("-inf"), dtype=dt, device=device)
    best_i = torch.zeros((len(Q), k), dtype=torch.int32, device=device)
    for off in range(0, len(D), d_shard):
        d = torch.from_numpy(D[off:off + d_shard]).to(device=device, dtype=dt)
        kk = min(k, d.shape[0])
        for s in range(0, len(Q), q_chunk):
            q = torch.from_numpy(Q[s:s + q_chunk]).to(device=device, dtype=dt)
            v, i = torch.topk(q @ d.T, kk, dim=1)
            v, j = torch.topk(torch.cat([best_v[s:s + len(q)], v], 1), k, dim=1)
            cand_i = torch.cat([best_i[s:s + len(q)], (i + off).to(torch.int32)], 1)
            best_v[s:s + len(q)], best_i[s:s + len(q)] = v, torch.gather(cand_i, 1, j)
        del d
    return best_i.cpu().numpy(), best_v.float().cpu().numpy()


# ------------------------------------------------------------------------------------------ training
def _batches(pairs: pd.DataFrame, batch_size: int, rng) -> list:
    """Batches of (s1, cand) row pairs from one country, one pair per S1 entity where possible.

    An entity's positives are spread over "levels" (its 1st pair, its 2nd pair, ...), and a batch is cut
    from one level, so the in-batch negatives are other entities' records. Within a level the pairs are
    sorted by the S1 record's city (second-to-last address part), so many negatives share a city.
    """
    out = []
    for _, g in pairs.groupby("country", sort=False):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        g = g.assign(level=g.groupby("s1").cumcount())
        g = g.sort_values(["level", "city", "tie"], kind="stable")
        idx = g.index.to_numpy()
        lev = g["level"].to_numpy()
        for L in np.unique(lev):
            sel = idx[lev == L]
            out.extend(sel[j:j + batch_size] for j in range(0, len(sel), batch_size) if len(sel[j:j + batch_size]) > 1)
    return [out[i] for i in rng.permutation(len(out))]


def cmd_train(args):
    t_start = time.time()
    device, amp = _device(args)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    recs = data.load(args.model_dir, "train", ["entity_id", "source", "country_norm", "split", "name_clean", "addr_clean"])
    _, gt = data.load_truth(args.data_dir, recs)
    split = recs["split"].to_numpy()
    pairs = gt[split[gt["s1"].to_numpy()] == "enc"].reset_index(drop=True)
    if args.limit_pairs and len(pairs) > args.limit_pairs:
        pairs = pairs.sample(n=args.limit_pairs, random_state=args.seed).reset_index(drop=True)
    s1_rows = pairs["s1"].to_numpy()
    pairs["country"] = recs["country_norm"].to_numpy()[s1_rows]
    pairs["city"] = pd.Series(recs["addr_clean"].to_numpy()[s1_rows]).str.split(", ").str[-2].fillna("").to_numpy()
    pairs["tie"] = rng.random(len(pairs))
    print(f"[train] {len(pairs):,} positive pairs from {pairs['s1'].nunique():,} 'enc' entities")

    tok = AutoTokenizer.from_pretrained(args.backbone, revision=args.revision)
    rows = np.unique(np.concatenate([pairs["s1"].to_numpy(), pairs["cand"].to_numpy()]))
    pos = np.full(len(recs), -1, np.int64)
    pos[rows] = np.arange(len(rows))
    packed = Packed(tok, record_texts(recs.iloc[rows]), args.max_len)
    print(f"[train] tokenised {len(rows):,} records | mean {packed.lengths.mean():.1f} tokens | "
          f"{np.mean(packed.lengths >= args.max_len):.2%} truncated at {args.max_len}")

    model = TextEncoder(args.backbone, args.revision, args.freeze_embeddings).to(device)
    if args.grad_checkpointing:
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                            **({"fused": True} if device.type == "cuda" else {}))
    epochs = [_batches(pairs, args.batch_size, rng) for _ in range(args.epochs)]
    n_steps = sum(len(e) for e in epochs)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * n_steps), n_steps)
    s1_all, c_all = pairs["s1"].to_numpy(), pairs["cand"].to_numpy()
    step, t0 = 0, time.time()
    model.train()
    for ep, batches in enumerate(epochs):
        run = n = 0
        for b in batches:
            s1r, cr = s1_all[b], c_all[b]
            ids_q, m_q = packed.batch(pos[s1r])
            ids_d, m_d = packed.batch(pos[cr])
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                q = model(ids_q.to(device), m_q.to(device))
                d = model(ids_d.to(device), m_d.to(device))
            logits = (q @ d.T) / args.temperature
            # two pairs of the same S1 entity in one batch are not negatives of each other
            same = torch.from_numpy(s1r[:, None] == s1r[None, :]).to(device)
            same.fill_diagonal_(False)
            logits = logits.masked_fill(same, float("-inf"))
            target = torch.arange(len(b), device=device)
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, target)
                          + torch.nn.functional.cross_entropy(logits.T, target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run, n = run + float(loss.detach()), n + 1
            if step % args.log_every == 0 or step == n_steps:
                el = time.time() - t0
                print(f"[train] epoch {ep + 1}/{args.epochs} step {step:,}/{n_steps:,} | loss {run / n:.4f} | "
                      f"{el / 60:.1f} min, ~{el / step * (n_steps - step) / 60:.1f} min left", flush=True)
                run = n = 0
    out = os.path.join(args.model_dir, "biencoder")
    model.model.save_pretrained(out)
    tok.save_pretrained(out)
    meta = {k: v for k, v in vars(args).items() if k != "func"}
    meta.update(n_pairs=len(pairs), steps=n_steps, minutes=(time.time() - t_start) / 60)
    with open(os.path.join(out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] saved the fine-tuned encoder to {out} ({(time.time() - t_start) / 60:.1f} min)")


# ------------------------------------------------------------------------------------------ indexing
def _encoder_path(args):
    if args.encoder:
        return args.encoder
    p = os.path.join(args.model_dir, "biencoder")
    if os.path.isdir(p):
        return p
    raise SystemExit(f"[index] no fine-tuned encoder in {p}: run `python -m src.retrieval train` first, "
                     f"or pass --encoder {DEFAULT_BACKBONE} to index with the pretrained model")


def candidates(recs: pd.DataFrame, emb: np.ndarray, k_fwd: int, k_rev: int, device, q_chunk: int) -> pd.DataFrame:
    """Forward + reverse top-k within each country -> one row per (S1 row, S2/S3 row) pair."""
    src = recs["source"].to_numpy()
    ctry = recs["country_norm"].to_numpy()
    parts = []
    for c in pd.unique(ctry):
        q_rows = np.flatnonzero((src == "S1") & (ctry == c)).astype(np.int32)
        d_rows = np.flatnonzero((src != "S1") & (ctry == c)).astype(np.int32)
        if len(q_rows) == 0 or len(d_rows) == 0:
            continue
        t0 = time.time()
        fi, fs = search(emb[q_rows], emb[d_rows], k_fwd, device, q_chunk)
        ri, rs = search(emb[d_rows], emb[q_rows], k_rev, device, q_chunk)
        kf, kr = fi.shape[1], ri.shape[1]
        fwd = pd.DataFrame({"s1": np.repeat(q_rows, kf), "cand": d_rows[fi.ravel()], "cos": fs.ravel(),
                            "fwd_rank": np.tile(np.arange(1, kf + 1, dtype=np.float32), len(q_rows))})
        rev = pd.DataFrame({"s1": q_rows[ri.ravel()], "cand": np.repeat(d_rows, kr), "cos_rev": rs.ravel(),
                            "rev_rank": np.tile(np.arange(1, kr + 1, dtype=np.float32), len(d_rows))})
        m = fwd.merge(rev, on=["s1", "cand"], how="outer")
        m["cos"] = m["cos"].fillna(m["cos_rev"])
        # neighbourhood of both sides: how close is the best (and 10th) match of the S1 record, and the best
        # (and 2nd) S1 claimant of the candidate -- also for pairs found by only one direction
        s1ctx = pd.DataFrame({"s1": q_rows, "s1_top1": fs[:, 0], "s1_top10": fs[:, min(9, kf - 1)]})
        cctx = pd.DataFrame({"cand": d_rows, "cand_top1": rs[:, 0], "cand_top2": rs[:, min(1, kr - 1)]})
        m = m.drop(columns=["cos_rev"]).merge(s1ctx, on="s1", how="left").merge(cctx, on="cand", how="left")
        parts.append(m)
        print(f"[index] {c}: {len(q_rows):,} S1 x {len(d_rows):,} S2/S3 -> {len(m):,} pairs "
              f"({len(m) / len(q_rows):.1f}/entity, {len(m) - len(fwd):,} only from the reverse search) "
              f"in {time.time() - t0:.0f}s", flush=True)
    cols = ["s1", "cand", "cos", "fwd_rank", "rev_rank", "s1_top1", "s1_top10", "cand_top1", "cand_top2"]
    out = pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)
    out[["s1", "cand"]] = out[["s1", "cand"]].astype(np.int32)
    for c in cols[2:]:
        out[c] = out[c].astype(np.float32)
    return out


def recall_report(cands: pd.DataFrame, recs: pd.DataFrame, gt: pd.DataFrame, truth: dict, ks=(1, 2, 3, 5, 10, 20, 30, 50)) -> dict:
    """Pair recall at forward rank <= k and for the whole union, on the 'rank' entities; oracle macro F0.5."""
    split = recs["split"].to_numpy()
    rank_s1 = np.flatnonzero(split == "rank")
    g = gt[split[gt["s1"].to_numpy()] == "rank"]
    c = cands[split[cands["s1"].to_numpy()] == "rank"]
    key = lambda d: d["s1"].to_numpy(np.int64) * (1 << 32) + d["cand"].to_numpy(np.int64)
    lab = np.isin(key(c), key(g)).astype(np.float64)
    n_true = max(1, len(g))
    rep = {f"recall@{k}": float(lab[c["fwd_rank"].to_numpy() <= k].sum() / n_true) for k in ks}
    rep["recall_reverse_only_added"] = float(lab[c["fwd_rank"].isna().to_numpy()].sum() / n_true)
    rep["recall_union"] = float(lab.sum() / n_true)
    rep["pairs_per_entity"] = float(len(c) / max(1, len(rank_s1)))
    pos = c[lab == 1]
    oracle = {int(i): [] for i in rank_s1}
    for a, b in zip(pos["s1"].to_numpy(), pos["cand"].to_numpy()):
        oracle[int(a)].append(int(b))
    rep["oracle_macro_f05"] = macro_f05(oracle, truth, rank_s1.tolist())
    return rep


def cmd_index(args):
    t_start = time.time()
    device, amp = _device(args)
    enc_path = _encoder_path(args)
    tok = AutoTokenizer.from_pretrained(enc_path, revision=args.revision if args.encoder else None)
    model = TextEncoder(enc_path, args.revision if args.encoder else None).to(device)
    print(f"[index] encoder {enc_path}")
    report = {"encoder": enc_path, "k_fwd": args.k_fwd, "k_rev": args.k_rev}
    for split in args.splits.split(","):
        t0 = time.time()
        recs = data.load(args.model_dir, split, ["entity_id", "source", "country_norm", "split", "name_clean", "addr_clean"])
        packed = Packed(tok, record_texts(recs), args.max_len)
        print(f"[index] {split}: tokenised {len(recs):,} records in {time.time() - t0:.0f}s | "
              f"{np.mean(packed.lengths >= args.max_len):.2%} truncated at {args.max_len}", flush=True)
        emb = encode(model, packed, device, amp, args.batch_size)
        del packed
        print(f"[index] {split}: embedded in {time.time() - t0:.0f}s", flush=True)
        cands = candidates(recs, emb, args.k_fwd, args.k_rev, device, args.q_chunk)
        del emb
        path = os.path.join(args.model_dir, "cache", f"cands_{split}.parquet")
        cands.to_parquet(path, index=False)
        print(f"[index] {split}: {len(cands):,} candidate pairs -> {path} ({time.time() - t0:.0f}s)")
        if split == "train":
            truth, gt = data.load_truth(args.data_dir, recs)
            rep = recall_report(cands, recs, gt, truth)
            report["train_rank_entities"] = rep
            print("[index] train ('rank' entities): " + " | ".join(f"{k} {v:.4f}" for k, v in rep.items()))
        del cands, recs
    with open(os.path.join(args.model_dir, "retrieval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[index] done in {(time.time() - t_start) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description="Bi-encoder retrieval: fine-tune, then build candidate pairs")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", "index"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", default=os.path.join("..", "..", "dataset"))
        p.add_argument("--model-dir", default="models")
        p.add_argument("--revision", default=None, help="pin the backbone's Hub commit (reproducibility)")
        p.add_argument("--max-len", type=int, default=64)
        p.add_argument("--cpu", action="store_true", help="force CPU (mini dataset only)")
        p.add_argument("--seed", type=int, default=42)
    t, i = sub.choices["train"], sub.choices["index"]
    t.add_argument("--backbone", default=DEFAULT_BACKBONE, help="Hugging Face id or local folder")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch-size", type=int, default=512, help="pairs per step (in-batch negatives)")
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--warmup", type=float, default=0.05)
    t.add_argument("--temperature", type=float, default=0.05)
    t.add_argument("--freeze-embeddings", action="store_true", help="freeze the 96M-parameter vocabulary matrix")
    t.add_argument("--grad-checkpointing", action="store_true")
    t.add_argument("--limit-pairs", type=int, default=0, help="smoke tests: cap the training pairs")
    t.add_argument("--log-every", type=int, default=100)
    t.set_defaults(func=cmd_train)
    i.add_argument("--encoder", default=None, help=f"default <model-dir>/biencoder; or e.g. {DEFAULT_BACKBONE}")
    i.add_argument("--splits", default="train,test")
    i.add_argument("--k-fwd", type=int, default=50, help="nearest S2/S3 records kept per S1 record")
    i.add_argument("--k-rev", type=int, default=5, help="nearest S1 records kept per S2/S3 record")
    i.add_argument("--batch-size", type=int, default=1024, help="texts per encoding batch")
    i.add_argument("--q-chunk", type=int, default=1024, help="queries per GPU matrix product")
    i.set_defaults(func=cmd_index)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
