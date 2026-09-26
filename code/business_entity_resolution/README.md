# Business Entity Resolution: Amazon ML Challenge

For each Source 1 business, this pipeline finds the matching Source 2 and Source 3 records. Every stage is tuned for the official metric, **macro F0.5 per Source 1 entity, with singletons included**. It also aims for a **small candidate set per entity**, because the organisers rank a smaller `candidate_pairs.tsv` higher.

```
dataset/{train,test}
  -> normalise (transliterate Indic scripts, clean noise)                         src/data.py, normalize.py, translit.py
  -> retrieval: fine-tuned multilingual bi-encoder, exact top-k per country (GPU)  src/retrieval.py
  -> ~60 retrieved pairs/entity -> 80+ pair features -> stage 1: LightGBM, 5-fold src/features.py, model.py
  -> prune to a few pairs/entity (= candidate_pairs.tsv)                          src/cascade.py
  -> [optional, GPU: fine-tuned cross-encoder on the pruned pairs]                src/cross_encoder.py
  -> stage 2: LightGBM on features + stage-1 score (+ cross-encoder), 5-fold
  -> F0.5-optimal decision layer -> matching_results.tsv                          src/decision.py
```

## Folder layout (challenge root)

```
Amazon ML Challenge/
├── dataset/                 <- the official train/ and test/ folders
├── output/                  <- matching_results.tsv + candidate_pairs.tsv are written here
├── utils/                   <- the official validate_submission.py
├── Documentation_template.md  <- the methodology write-up for the final zip
└── code/business_entity_resolution/
    ├── src/                 all source code
    ├── README.md, requirements.txt, run_pipeline.bat, run_pipeline_gpu.bat
    └── models/              created by the pipeline (caches, encoder, LightGBM models, reports)
```

## Setup and run

Everything runs on one machine with a CUDA GPU (developed for a 12 GB RTX 4070 with 128 GB RAM; the real data has ~12M records per split).

```bash
cd "C:\Amazon ML Challenge\code\business_entity_resolution"
python -m venv .venv && .venv\Scripts\activate        # optional
pip install -r requirements.txt

run_pipeline.bat          # everything except the cross-encoder
run_pipeline_gpu.bat      # the same plus a fine-tuned mdeberta cross-encoder stacked into stage 2
```

Both scripts activate `.venv` if it exists, print one line per step, and log everything to `run_log.txt` / `run_log_gpu.txt` next to them (watch it live with `powershell Get-Content run_log.txt -Wait -Tail 30`). On an error they stop and show the end of the log.

Step by step (what the .bat files run):

| Command | Where | What it does |
|---|---|---|
| `python -m src.data build` | CPU, all cores | Reads the TSVs, learns the Indic-to-Latin word dictionary from the training pairs, normalises every record, splits the training S1 entities into `enc` (20%, bi-encoder) and `rank` (80%, everything else), caches `models/cache/records_{train,test}.parquet`. |
| `python -m src.retrieval train` | GPU | Fine-tunes `intfloat/multilingual-e5-small` on the `enc` entities' true pairs (contrastive, in-batch negatives). |
| `python -m src.retrieval index` | GPU | Embeds every record, runs exact top-k search within each country in both directions, writes `models/cache/cands_{train,test}.parquet` and prints recall@k on the `rank` entities. |
| `python -m src.main prepare` | CPU | Features for the sampled `rank` entities, stage-1 LightGBM (5-fold), the pruning grid, and the same features + stage-1 scores + pruning for the test set, chunk by chunk. |
| `python -m src.cross_encoder --name mdeberta ...` | GPU, optional | Fine-tunes a cross-encoder on the pruned pairs, per fold group, and scores the held-out and test pairs. |
| `python -m src.main stage2 [--extra mdeberta]` | CPU | Stage-2 LightGBM (5-fold), decision tuning, the error report, then writes and self-validates both output TSVs. If stage 2 does not beat stage 1 out-of-fold, the submission uses stage 1. |
| `python -m src.main score --pred X.tsv --truth Y.tsv` | CPU | Official macro F0.5 of any predictions, plus candidates per entity and candidate recall. |
| `python -m src.package --team <name>` | - | Builds `<name>_submission.zip` in the challenge root: `output/`, this code folder (without caches or models) and the filled-in `Documentation_template.md`. |

Useful flags for `main`: `--train-entities 600000` (0 = all `rank` entities), `--max-oracle-drop 0.002`, `--target-pairs 5`, `--no-prune`, `--k-fwd 50 --k-rev 5`, `--folds 5`, `--seed 42`. Other settings are in `src/config.py`.

### Smoke test on a laptop (no GPU)

`python -m src.data mini --out <folder>\dataset_mini --entities 20000` cuts a small copy of the training data (keep it outside the submission folder). Its `test/` part is held-out training entities, and `dataset_mini/test_ground_truth.tsv` lets you score the output end to end. Run the same commands with `--data-dir ..\..\dataset_mini --model-dir models_mini`, add `--cpu` to the retrieval steps and, for the retrieval training, `--limit-pairs 3000 --batch-size 64`. The mini pool has ~100x fewer distractors than the real data, so its scores are optimistic.

## Validation

- **Entity split.** The bi-encoder is trained only on `enc` entities. Every later model trains on `rank` entities, whose candidates were retrieved by an encoder that never saw them, just like the test entities.
- **5-fold, out-of-fold.** Folds are over `rank` S1 entities (stratified by country and has-match), so all candidates of an entity share a fold. Every training pair gets an out-of-fold prediction from a model that never saw it. The decision rule is tuned on these predictions, stage 2 and the cross-encoder stack on them without leakage, and test pairs get the fold models' average (in log-odds space).
- The reported macro F0.5 counts every sampled `rank` entity, including true matches that retrieval or pruning lost.

## Normalisation (`src/normalize.py`, `src/translit.py`)

The real data has noise that plain string similarity cannot handle:

- **Indic scripts.** About 15% of Source 2/3 records have names or addresses in Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada or Malayalam (`श्री सॉफ्टवेयर लिमिटेड` = "Sree Software Limited"). These nine scripts share one Unicode layout, so one rule-based transliterator covers all of them (consonants with the inherent vowel, vowel signs, virama, nasal signs, Hindi schwa deletion). On top of the rules, a word dictionary is learned from the training pairs: each Indic word is aligned with the most similar word of its matched Source 1 text, and words seen with at least 20 different entities are kept (learned on half of the training entities: 1,248 words such as `लिमिटेड` → limited, `प्राइवेट` → private, state names). On the other half of the entities this raises the share of Indic names with token-set similarity ≥ 80 to their true Source 1 name from 43% to 93%.
- **Names.** Digits used as letters (`C0mpanies`, `5ervices`), tags (`(ID: 81759)`, `#92781`, phone numbers), `@handle` / `#hashtag` / `domain.com` names, `<made-up name> formerly|aka|DBA <real name>` (the part after the marker is the name the other sources use), bracketed and repeated legal words, and canonical legal forms (Pvt/Private, Ltd/Limited, Corp/Corporation, SARL/SAS...).
- **Addresses.** Literal `null` / `<NULL>`, zero-padded house numbers, "City of ...", state names vs. codes per country (`CT` / `Connecticut`, `UP` / `Uttar Pradesh`; a lone component only, so `Fawn Ct` stays a court), street-type abbreviations in English and French (`R.` = rue, `ALL.` = allée, `AV` = avenue, `BD` = boulevard, `91 bis`, `ST.-NAZAIRE` = Saint-Nazaire), and landmark phrases. The data has no postal codes: 5-6 digit numbers are house numbers, and they are kept.
- Country is an open set. Every country-specific table falls back to "no change" for an unknown country (France).

## Blocking: dense retrieval (`src/retrieval.py`)

Each split has ~10M Source 2/3 records, so blocking has to be an index, not pairwise string scores.

- **Encoder.** `intfloat/multilingual-e5-small` (MIT, 118M parameters) reads `name | address` (transliterated, lower-case). It is fine-tuned with a symmetric contrastive loss on the `enc` entities' true pairs. Each batch comes from one country and holds one pair per entity; within a batch the pairs are sorted by city, so the in-batch negatives are hard.
- **Search.** Exact inner-product top-k on the GPU (one matrix product scores a chunk of S1 records against millions of records), within each country label:
  - forward: every S1 record → its 50 nearest S2/S3 records;
  - reverse: every S2/S3 record → its 5 nearest S1 records. No record belongs to more than one S1 entity, so "this S1 is the record's nearest S1" is strong evidence, and the reverse direction recovers true pairs that a crowded neighbourhood pushed out of the forward top-k.
- `index` prints recall@k, the union's recall and the oracle F0.5 ceiling on the `rank` entities.
- **Scaling further.** Exact search costs one matrix product per country. For billions of records the same embeddings go into an approximate nearest-neighbour index (e.g. IVF or HNSW), and the rest of the pipeline does not change.

## Candidate set: cascade and pruning (`src/cascade.py`)

1. **Stage 1** scores every retrieved pair with LightGBM on the retrieval features (cosine, forward/reverse rank, gap to each side's best match, neighbourhood density) and the string features.
2. **Pruning** keeps pairs with stage-1 probability `p1 >= cutoff` and at most `cap` pairs per entity. `prepare` prints a grid of cutoffs x caps with pairs per entity and the oracle macro F0.5. It picks the smallest candidate set whose oracle score drops by at most 0.002 (`--max-oracle-drop`), or the best oracle score within `--target-pairs N` pairs per entity.
3. **Stage 2** re-scores the surviving pairs with the same features, `p1`, any cross-encoder score, and each score's rank, gap and sum among the pairs of the same entity and of the same candidate record.

Per the rules, `candidate_pairs.tsv` is "the last filtering stage: whatever your model actually runs inference over", so it holds the **pruned** pairs that stage 2 scores.

## Features (`src/features.py`)

All features are similarities or agreements, never identities, so they transfer to France. There is no country one-hot. Missing values are NaN.

- Retrieval: cosine, forward and reverse rank, gap to the S1 record's best match and to the candidate's best S1 claimant, the candidate's top-1 vs. top-2 margin, neighbourhood density.
- Name: rapidfuzz ratio / token-set / token-sort / partial / WRatio / Jaro-Winkler / Levenshtein on the full and core names, best over "formerly/aka" alternatives, space-free comparison (`globalcardiology` vs. `global cardiology`), acronyms, Jaccard, overlap, **IDF-weighted Jaccard** (rare shared words count more), first/last token, digits in names, transliteration flags.
- Address: the same string similarities, Jaccard / overlap / IDF-weighted Jaccard, house-number agreement and conflict, landmarks, missing parts.
- Context within the entity: rank and gap of the main scores among the entity's candidates, per source; number of candidates.
- Stage 2 only, **peer agreement**: how similar the candidate's name and address are to the entity's other candidates, weighted by their stage-1 probability (and to the confident ones only). An entity has ~3.7 true matches that describe the same business, so a weak candidate (`BP` for "Bowling Pharmacie SAS") that shares its address with the confident ones is probably a match, and one that resembles none of them probably is not.

## Cross-encoder (`src/cross_encoder.py`, optional)

A cross-encoder reads both records together and outputs a match probability. For each fold group (`--group-folds 2`: folds 0/2/4 vs. 1/3), it fine-tunes on the other group's pruned pairs (all positives plus the 20 hardest negatives per entity), then scores the held-out group and every pruned test pair. `stage2 --extra <name>` stacks the result as a feature. It is resumable per fold group. Add a model to `--extra` only if it improves the stage-2 out-of-fold macro F0.5.

| `--name` | `--backbone` | Licence | Parameters |
|---|---|---|---|
| `mdeberta` | `microsoft/mdeberta-v3-base` | MIT | 0.28B |
| `xlmr_large` | `FacebookAI/xlm-roberta-large` (`--lr 1e-5`) | MIT | 0.56B |
| `bge_m3` | `BAAI/bge-reranker-v2-m3` (`--lr 1e-5`) | Apache-2.0 | 0.57B |

## How F0.5 is prioritised (`src/decision.py`)

Per entity, `F0.5 = 1.25·TP / (|pred| + 0.25·|true|)`. A singleton scores 1.0 only when you predict **nothing** for it.

1. The metric is implemented exactly and self-tests against the problem statement's worked example (0.714). Everything is selected on it.
2. The decision rule is tuned on out-of-fold predictions: a global threshold, or **expected-F0.5 maximisation** (per entity, the top-k set with the highest expected F0.5 under the model's probabilities, computed exactly with Poisson-binomial DP), each with or without a **one-to-one** constraint (each record kept only for its best S1 entity; the training ground truth never links a record to two entities).
3. Both LightGBM stages use plain log-loss with no re-weighting, so the probabilities stay calibrated for the expected-F0.5 rule.

## Rules compliance

- Models: multilingual-e5-small (MIT, 0.12B), LightGBM (MIT), optional cross-encoder mdeberta-v3-base (MIT, 0.28B). The total stays far below 8B parameters. Model revisions are pinned in the .bat files.
- Data: pretrained weights come from Hugging Face. Everything is fitted only on the provided training data (bi-encoder, transliteration dictionary, LightGBM, cross-encoder); IDF tables are computed on each split's own records. The state-name and abbreviation tables are hand-written normalisation rules, not a lookup. No external data, APIs, geocoding or address databases.
- Output: one row per test S1 entity; S2/S3 ids only; no duplicates; matches are a subset of candidates, and the candidates are exactly the pairs stage 2 scores. Checked automatically after writing (`[validate] PASS`); the .bat files also run the official `utils/validate_submission.py`.

## Troubleshooting

- `The tokenizer you are loading from 'models\biencoder' with an incorrect regex pattern ... fix_mistral_regex`: a false positive in transformers 4.57 when re-loading a saved XLM-R tokenizer; its token ids are identical to the original's.
- Windows without symlink rights can fail to populate the Hugging Face cache. Download the model to a folder (`huggingface_hub.snapshot_download(..., local_dir=...)`) and pass that folder as `--backbone`.

## Files written (in `models/`)

- `translit_dict.json`: the learned Indic-to-Latin word dictionary.
- `cache/records_{train,test}.parquet`, `cache/cands_{train,test}.parquet`: normalised records and retrieved pairs (int32 row numbers into the records).
- `biencoder/`: the fine-tuned retrieval encoder; `retrieval_report.json`: recall@k.
- `meta_stage1.json`, `meta.json`: features, pruning grid and choice, out-of-fold reports (overall, by country, singleton accuracy), decision rule and grid.
- `stage1_fold*.txt`, `stage2_fold*.txt`, `feature_importance_stage{1,2}.csv`: the LightGBM models.
- `pairs/{train,test}_pruned.parquet`: the pruned pairs (input to the cross-encoder); `extra/`: cross-encoder scores.
- `oof_scored_pairs.parquet`: out-of-fold `p1` and `p2` for every pruned training pair, for error analysis.
- `error_report.json`, `error_examples.tsv` (`src/analysis.py`): where macro F0.5 is lost on the out-of-fold predictions -- true pairs lost by retrieval / pruning / the matcher, score lost to false merges on singletons, to false merges on matched entities and to misses (overall and per country), miss and false-merge rates by pair type, and the most confident mistakes with the records as written.
