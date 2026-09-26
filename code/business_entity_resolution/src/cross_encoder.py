"""Transformer cross-encoder, stacked into stage 2 as one extra feature. Runs on the GPU box.

    python -m src.main prepare --data-dir ../../dataset --model-dir models
    python -m src.cross_encoder --name mdeberta --backbone microsoft/mdeberta-v3-base --data-dir ../../dataset --model-dir models
    python -m src.main stage2  --data-dir ../../dataset --model-dir models --out-dir ../../output --extra mdeberta

For each fold group of the pruned training pairs (LightGBM's folds, or `--group-folds G` groups of
them: fold % G) it fine-tunes one model on the other groups, then scores the held-out group (the
out-of-fold feature stage 2 trains on) and all pruned test pairs. G = 2 halves the GPU time of the
default 5 on a set of millions of pairs. Each group's scores are written as soon as it finishes, so an
interrupted run resumes at the next group. Test scores are averaged over the models in log-odds space.

The model reads "name | address" as written but transliterated and lower-cased (the `name_clean` /
`addr_clean` columns, `--text clean`, default), not our canonical tokens: pretrained models know
"Pvt. Ltd." and "Boulevard" but not "hldg" or "trdg", and cannot compare "श्री" with "Sree".
`--text raw` feeds the untouched source fields instead.
Only the provided training pairs are used for fine-tuning (challenge rule: no external data).
"""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import time

os.environ.setdefault("USE_TF", "0")  # keep transformers from importing TensorFlow (slow, ABI clashes in conda envs)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from . import data


def record_texts(model_dir: str, split: str, mode: str) -> np.ndarray:
    """Text per record row: 'name | address' (clean: transliterated + lower-case; raw: as in the source files)."""
    cols = ["name_clean", "addr_clean"] if mode == "clean" else ["business_name", "business_address"]
    recs = data.load(model_dir, split, cols)
    a, b = (recs[c].map(lambda x: " ".join(str(x).split())) for c in cols)
    return (a + " | " + b).to_numpy(dtype=object)


class PairEncodings:
    """Token ids of every (S1 record, candidate record) pair, tokenised once and padded per batch."""

    def __init__(self, tok, left, right, max_len: int):
        enc = tok(left, right, truncation=True, max_length=max_len)
        self.keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
        self.data = {k: enc[k] for k in self.keys}
        self.lengths = np.fromiter((len(x) for x in enc["input_ids"]), dtype=np.int64, count=len(left))
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def batch(self, idx) -> dict:
        width = int(self.lengths[idx].max())
        out = {}
        for k in self.keys:
            arr = np.full((len(idx), width), self.pad_id if k == "input_ids" else 0, dtype=np.int64)
            for r, i in enumerate(idx):
                seq = self.data[k][i]
                arr[r, :len(seq)] = seq
            out[k] = torch.from_numpy(arr)
        return out


def length_grouped_batches(idx, lengths, batch_size: int, rng):
    """Batches of similar-length pairs (little padding) in random order."""
    idx = rng.permutation(idx)
    mega = batch_size * 50
    batches = []
    for s in range(0, len(idx), mega):
        chunk = idx[s:s + mega]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches.extend(chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size))
    return [batches[o] for o in rng.permutation(len(batches))]


def sorted_batches(idx, lengths, batch_size: int):
    idx = np.asarray(idx)[np.argsort(lengths[idx], kind="stable")]
    return [idx[j:j + batch_size] for j in range(0, len(idx), batch_size)]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_pairs(tr: pd.DataFrame, k: int, neg_per_entity: int, limit: int, seed: int) -> np.ndarray:
    """Pairs of the other fold groups: every positive + the hardest negatives (highest p1) of each entity."""
    d = tr[tr["group"] != k]
    if neg_per_entity > 0:
        neg = (d[d["label"] == 0].sort_values(["s1", "p1"], ascending=[True, False])
                                 .groupby("s1").head(neg_per_entity))
        d = pd.concat([d[d["label"] == 1], neg])
    idx = d.index.to_numpy()
    if limit and len(idx) > limit:
        idx = np.random.default_rng(seed).choice(idx, size=limit, replace=False)
    return np.sort(idx)


def load_model(args, device):
    model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=1, revision=args.revision,
                                                               ignore_mismatched_sizes=True)
    if args.freeze_embeddings:
        # the vocabulary matrix is 45-70% of these models' weights; freezing it saves memory and keeps the
        # embeddings of tokens never seen in training (French!) aligned with the rest
        model.get_input_embeddings().weight.requires_grad_(False)
    if args.grad_checkpointing:
        # non-reentrant: gradients still reach the layers when the (frozen) embeddings need none
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ce] {args.backbone}: {n_all / 1e6:.0f}M parameters, {n_train / 1e6:.0f}M trainable")
    return model.to(device)


def fit(model, enc: PairEncodings, idx, y, args, device, amp, seed: int):
    rng = np.random.default_rng(seed)
    epochs = [length_grouped_batches(idx, enc.lengths, args.batch_size, rng) for _ in range(args.epochs)]
    n_steps = sum(math.ceil(len(e) / args.grad_accum) for e in epochs)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr,
                            **({"fused": True} if device.type == "cuda" else {}))
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * n_steps), n_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    target = torch.as_tensor(y, dtype=torch.float32)
    model.train()
    step, t0 = 0, time.time()
    for ep, batches in enumerate(epochs):
        run, n = 0.0, 0
        for bi, b in enumerate(batches):
            inputs = {k: v.to(device) for k, v in enc.batch(b).items()}
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                logits = model(**inputs).logits.squeeze(-1)
            loss = loss_fn(logits.float(), target[torch.from_numpy(b)].to(device))
            (loss / args.grad_accum).backward()
            run, n = run + float(loss.detach()), n + 1
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(decay + no_decay, args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == n_steps:
                    el = time.time() - t0
                    print(f"[ce]   epoch {ep + 1}/{args.epochs} step {step:,}/{n_steps:,} | loss {run / n:.4f} | "
                          f"{el / 60:.1f} min elapsed, ~{el / step * (n_steps - step) / 60:.1f} min left", flush=True)
                    run, n = 0.0, 0


@torch.no_grad()
def predict_logits(model, enc: PairEncodings, idx, batch_size: int, device, amp) -> np.ndarray:
    model.eval()
    out = np.zeros(len(enc.lengths), dtype=np.float32)
    for b in sorted_batches(idx, enc.lengths, batch_size):
        inputs = {k: v.to(device) for k, v in enc.batch(b).items()}
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            out[b] = model(**inputs).logits.float().squeeze(-1).cpu().numpy()
    return out[np.asarray(idx)]


def _metrics(y, logit) -> dict:
    p = 1.0 / (1.0 + np.exp(-np.asarray(logit, dtype=np.float64)))
    out = {"n": int(len(y)), "positives": int(np.sum(y))}
    if 0 < out["positives"] < len(y):
        out.update(logloss=float(log_loss(y, p, labels=[0, 1])), auc=float(roc_auc_score(y, logit)))
    return out


def _fingerprint(args, tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    """What the saved fold scores depend on; a mismatch means they are stale. Hashes the pair ids, folds
    and labels (not the p1 floats), so re-running an identical `prepare` doesn't invalidate finished folds."""
    h = hashlib.sha1()
    for df, cols in ((tr, ["s1", "cand", "fold", "label"]), (te, ["s1", "cand"])):
        h.update(pd.util.hash_pandas_object(df[cols], index=False).to_numpy().tobytes())
    keys = ("backbone", "revision", "max_len", "epochs", "lr", "batch_size", "grad_accum", "weight_decay", "warmup",
            "neg_per_entity", "freeze_embeddings", "limit_train", "seed", "text", "group_folds")
    return {"pairs_sha1": h.hexdigest(), **{k: getattr(args, k) for k in keys}}


def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune a cross-encoder per fold; writes out-of-fold + test scores "
                                             "to <model-dir>/extra/<name>_{oof,test}.tsv for `main stage2 --extra <name>`")
    ap.add_argument("--name", required=True, help="short name, used by stage2 --extra (e.g. mdeberta)")
    ap.add_argument("--backbone", required=True, help="Hugging Face model id, e.g. microsoft/mdeberta-v3-base")
    ap.add_argument("--revision", default=None, help="pin the model's exact Hub commit (reproducibility)")
    ap.add_argument("--data-dir", default=os.path.join("..", "..", "dataset"))
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06, help="fraction of optimiser steps")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--neg-per-entity", type=int, default=20,
                    help="train on every positive + this many hardest negatives per entity (0 = all pairs)")
    ap.add_argument("--no-freeze-embeddings", dest="freeze_embeddings", action="store_false")
    ap.add_argument("--grad-checkpointing", action="store_true", help="less memory, ~30%% slower (large models)")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--text", choices=["clean", "raw"], default="clean",
                    help="clean: transliterated, lower-case name | address (default); raw: the source fields")
    ap.add_argument("--group-folds", type=int, default=0,
                    help="train G models on fold groups (fold %% G) instead of one per LightGBM fold (0 = one per fold)")
    ap.add_argument("--folds", default="", help="comma-separated subset of fold groups to run now (default: all)")
    ap.add_argument("--limit-train", type=int, default=0, help="smoke tests: cap the training pairs per fold")
    ap.add_argument("--save-models", action="store_true", help="also keep each fold's fine-tuned weights")
    ap.add_argument("--log-every", type=int, default=200, help="optimiser steps between progress lines")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="force CPU (smoke tests only)")
    return ap.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp = torch.bfloat16 if (device.type == "cuda" and args.precision == "bf16" and torch.cuda.is_bf16_supported()) else None
    print(f"[ce] device {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} | "
          f"precision {'bf16 autocast' if amp else 'fp32'}")
    if device.type == "cpu":
        print("[ce] WARNING: no CUDA GPU in use -- fine for a smoke test, far too slow for real training")

    pairs = os.path.join(args.model_dir, "pairs")
    train_file, test_file = os.path.join(pairs, "train_pruned.parquet"), os.path.join(pairs, "test_pruned.parquet")
    tr, te = pd.read_parquet(train_file), pd.read_parquet(test_file)
    tr["group"] = tr["fold"] % args.group_folds if args.group_folds else tr["fold"]
    folds = sorted(tr["group"].unique().tolist())
    run = [int(f) for f in args.folds.split(",") if f.strip()] or folds

    extra = os.path.join(args.model_dir, "extra")
    parts = os.path.join(extra, f"{args.name}_parts")
    os.makedirs(parts, exist_ok=True)
    fp, fp_path = _fingerprint(args, tr, te), os.path.join(parts, "fingerprint.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            old = json.load(f)
        if old != fp:
            diff = sorted(k for k in fp if old.get(k) != fp[k])
            raise SystemExit(f"[ce] {parts} holds folds made with different {diff}. Delete that folder or use a new --name.")
    with open(fp_path, "w") as f:
        json.dump(fp, f, indent=2)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.backbone, revision=args.revision)
    texts_tr, texts_te = record_texts(args.model_dir, "train", args.text), record_texts(args.model_dir, "test", args.text)
    enc_tr = PairEncodings(tok, texts_tr[tr["s1"].to_numpy()].tolist(), texts_tr[tr["cand"].to_numpy()].tolist(), args.max_len)
    enc_te = PairEncodings(tok, texts_te[te["s1"].to_numpy()].tolist(), texts_te[te["cand"].to_numpy()].tolist(), args.max_len)
    del texts_tr, texts_te
    lens = np.concatenate([enc_tr.lengths, enc_te.lengths])
    print(f"[ce] tokenised {len(tr):,} train + {len(te):,} test pairs in {time.time() - t0:.0f}s | "
          f"mean {lens.mean():.0f} tokens | {np.mean(lens >= args.max_len):.1%} truncated at {args.max_len}")

    y = tr["label"].to_numpy()
    for k in run:
        oof_path, test_path = os.path.join(parts, f"oof_fold{k}.parquet"), os.path.join(parts, f"test_fold{k}.parquet")
        if os.path.exists(oof_path) and os.path.exists(test_path):
            print(f"[ce] fold group {k}: already done, skipping")
            continue
        t0 = time.time()
        set_seed(args.seed + k)
        idx = training_pairs(tr, k, args.neg_per_entity, args.limit_train, args.seed + k)
        va = np.flatnonzero(tr["group"].to_numpy() == k)
        print(f"[ce] fold group {k}: training on {len(idx):,} pairs ({y[idx].mean():.1%} positive), scoring "
              f"{len(va):,} held-out + {len(te):,} test pairs")
        model = load_model(args, device)
        fit(model, enc_tr, idx, y, args, device, amp, args.seed + k)
        oof = predict_logits(model, enc_tr, va, args.eval_batch_size, device, amp)
        tst = predict_logits(model, enc_te, np.arange(len(te)), args.eval_batch_size, device, amp)
        tr.iloc[va][["s1", "cand"]].assign(logit=oof).to_parquet(oof_path, index=False)
        te[["s1", "cand"]].assign(logit=tst).to_parquet(test_path, index=False)
        print(f"[ce] fold group {k} done in {(time.time() - t0) / 60:.1f} min | held-out {json.dumps(_metrics(y[va], oof))}")
        if args.save_models:
            out = os.path.join(extra, f"{args.name}_fold{k}_model")
            model.save_pretrained(out)
            tok.save_pretrained(out)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    done = [k for k in folds if os.path.exists(os.path.join(parts, f"oof_fold{k}.parquet"))
            and os.path.exists(os.path.join(parts, f"test_fold{k}.parquet"))]
    if done != folds:
        print(f"[ce] fold groups finished: {done}; still missing {sorted(set(folds) - set(done))} -- run again to finish")
        return
    oof = pd.concat([pd.read_parquet(os.path.join(parts, f"oof_fold{k}.parquet")) for k in folds], ignore_index=True)
    tests = [pd.read_parquet(os.path.join(parts, f"test_fold{k}.parquet")) for k in folds]
    if any(len(t) != len(te) or not (t["cand"].to_numpy() == te["cand"].to_numpy()).all() for t in tests):
        raise SystemExit(f"[ce] test scores in {parts} don't line up with {test_file}; delete the folder and re-run")
    z = np.mean([t["logit"].to_numpy(np.float64) for t in tests], axis=0)

    sig = lambda v: 1.0 / (1.0 + np.exp(-np.asarray(v, dtype=np.float64)))
    oof.assign(p=sig(oof["logit"]).astype(np.float32))[["s1", "cand", "p"]].to_parquet(
        os.path.join(extra, f"{args.name}_oof.parquet"), index=False)
    te[["s1", "cand"]].assign(p=sig(z).astype(np.float32)).to_parquet(os.path.join(extra, f"{args.name}_test.parquet"), index=False)
    lab = oof.merge(tr[["s1", "cand", "label"]], on=["s1", "cand"], how="left")["label"].to_numpy()
    rep = {"oof": _metrics(lab, oof["logit"].to_numpy()), "args": vars(args), "fingerprint": fp,
           "minutes_this_run": (time.time() - t_start) / 60}
    with open(os.path.join(extra, f"{args.name}_meta.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[ce] out-of-fold {json.dumps(rep['oof'])}")
    print(f"[ce] wrote {extra}/{args.name}_oof.parquet and {args.name}_test.parquet -> use "
          f"`python -m src.main stage2 --extra {args.name}`")


if __name__ == "__main__":
    main()
