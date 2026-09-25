# CLAUDE.md — Business Entity Resolution (Amazon ML Challenge 2026)

This file is the single source of truth for what we are building, how, and why.
Every teammate and every coding agent (Claude Code etc.) reads this before touching the code.
If reality disagrees with this file, fix the file in the same PR.

Deadline: **27 Sep 2026, 23:59 IST**. Leaderboard: **5 submissions/day**. Final rank = **private LB** (evaluation across public + private).

---

## 1. The task in one paragraph

Three sources of business records (`entity_id`, `business_name`, `business_address`, `country`).
Source 1 (S1) is deduplicated. For every S1 entity, output the list of S2/S3 records that are the same
real-world business (0, 1 or many). Metric: **F0.5 computed per S1 entity, then macro-averaged** over
all S1 entities, singletons included (empty GT + empty prediction = 1.0; empty GT + any prediction = 0.0).
Precision counts 2x recall → **false merges are the enemy**.

## 2. Hard rules (violating any of these = disqualification or rejected submission)

1. **No external data.** No APIs, geocoding, business registries, web lookups, LLM APIs (Bedrock/OpenAI/etc.)
   at any stage. Only the provided files. Pretrained open model weights are allowed (see rule 2).
2. **Final model(s): MIT or Apache-2.0 licensed, ≤ 8B parameters.** Record the license + HF revision of every
   model we download in `docs/MODELS.md`. Current allow-list: LightGBM (MIT), `intfloat/multilingual-e5-small|base` (MIT),
   `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0). Anything else → check license first, add to the list.
3. **Country is an open set.** Train = US, India. Test adds **France** (and must be treated generically).
   Never filter, one-hot, or `if country in {"US","India"}`. Country-specific *normalization rules* are fine
   (e.g. a French legal-suffix list), but the pipeline must run for any unseen label.
4. **All files are TSV.** Always: `pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)`
   (or polars equivalent with all columns as Utf8). Write outputs with `\t`, no quoting, no index.
5. **Output rules** (`output/matching_results.tsv`, `output/candidate_pairs.tsv`):
   exactly one row per test S1 id; headers `source1_entity_id` + `matched_entity_ids` / `candidate_entity_ids`;
   comma-separated S2-/S3- ids only, no duplicates, only ids that exist in test; empty string when none;
   **matches ⊆ candidates**; `candidate_pairs.tsv` = the exact set the final model scored (not an earlier blocking pass).
   Always run `python3 utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test`
   before any upload.
6. **Reproducible from `code/business_entity_resolution/` alone.** Scripts, not notebooks, are the source of truth.
   Pinned `requirements.txt`. Fixed seeds. SageMaker jobs must call the same `src/` code.
7. **No copying other teams' public code.** Plagiarism = disqualification. Write our own.
8. Never modify files under `student_resource/` (the official bundle). Data files are git-ignored.

## 3. Data facts

Verified by our EDA (`notes/DATA_CONTEXT.md`, `notes/eda.py`, full output in `notes/eda_output.txt`).

| Fact | Value |
|---|---|
| train S1 / S2 / S3 rows | ~2.21M / ~5.03M / ~5.29M |
| test S1 / S2 / S3 rows | ~1.73M / ~4.89M / ~5.08M |
| test countries | India ~47%, US ~38%, **France ~15% (259k S1)** |
| matches per S1 | mean **3.46**, max 11 (S2 and S3 each contain **duplicates** of the same business) |
| S1 with matches in both S2 and S3 | ~80% |
| S1 singletons | ~5.6% |
| cross-country matches | 0 |
| S2/S3 id linked to >1 S1 | 0 → **strict one-to-one from the S2/S3 side** |
| S2+S3 records per S1, test vs train | ~25% more in test, in every country (≈2.85 vs ≈2.3 per source) |
| postcodes | **effectively absent** (99.99% of matched pairs have none on either side; France ~0.4%). US 5-digit numbers are **house numbers**, not ZIPs |
| native-script names (India) | **23.5% of S2 names, 13% of S3 names** (Devanagari, Bengali, Kannada, Tamil, …) |
| decoys | the nearest non-match is usually the **same business with one change**: house number shifted (6800→6821), or one extra word ("Overseas", "Industries"). In **8%** of matched S1s a non-match scores ≥ the best true match on simple similarity |
| France | only 3 regions / ~18 cities → street + house number carry the signal; S2/S3 abbreviate street types (R/R. ~25%, AV, BD…), replace regions with départements, **drop accents in addresses, add accents in names**; legal forms SARL/SAS/EURL/SASU/SCI incl. dotted (S.A.S.) |
| US vs India distributions | identical to within 0.1 pt → the data comes from one generator; France very likely too |
| data quality | ids/GT consistent, no train/test id overlap; mandated `read_csv` handles quotes + CRLF |

Consequences:
- **No postcode anywhere** in blocking or features. Numbers that matter = **house/plot numbers**.
- **House-number relation** (equal / one missing / digit dropped / small shift / different) is the single most
  important decoy-killer. Same for **token-set difference** (the one extra/missing word).
- **Script handling is mandatory** for India: transliterate every non-Latin name to Latin before any string feature
  (anyascii, ISC). Native-script pairs must be decided by the address + transliterated name.
- Duplicates inside S2 and S3 → **graph features are unusually strong** here (true matches of one S1 look like each other).
- Test is denser → **expect more decoys than in validation**; set the decision layer more conservatively than the
  dev optimum and calibrate that margin with the leaderboard (§8).
- Validator gotchas: no spaces after commas in id lists, **no UTF-8 BOM**, treat unknown ids as a hard error.

## 4. Architecture

```
TSV ──► ingest (parquet cache)
    ──► normalize (per record, country-aware rules, country-agnostic output schema)
    ──► blocking (multi-pass, per country, chunked) ──► candidates  [= candidate_pairs.tsv]
    ──► features (pairwise string + structured + context/competition + graph)
    ──► model: LightGBM (+ optional cross-encoder score as a feature)  ──► calibrated p(match)
    ──► decision layer: one-to-one assignment ──► expected-F0.5 set selection per S1
    ──► matching_results.tsv ──► validator
```

### Repo layout

```
code/business_entity_resolution/
  src/ber/
    config.py            # paths, seeds, all tunables (read from configs/*.yaml)
    io.py                # TSV → parquet cache, TSV writers for submission files
    normalize.py         # name/address normalization, field extraction
    rules/               # abbreviation + legal-suffix dictionaries per language (us_en.py, in_en.py, fr.py, generic.py)
    blocking/
      keys.py            # exact/sorted-key blocks (name core, house_num+street token, phonetic)
      tfidf_knn.py       # char n-gram TF-IDF top-k per country, chunked sparse matmul
      embed_knn.py       # optional: e5 embeddings + FAISS top-k
      merge.py           # union of passes, provenance bitmask, cap per S1
    features/
      pairwise.py        # name/address string similarities
      structured.py      # house-number relation, token-set difference, legal form, script, lengths, missingness
      context.py         # ranks, gaps, competition features (per S1 and per candidate)
      graph.py           # S2↔S3 agreement / cluster consistency within an S1's candidate set
      rarity.py          # token / name frequency (IDF) computed per country on the pool
    model/
      lgbm.py            # train, predict, calibration (isotonic on held-out)
      cross_encoder/     # SageMaker-trainable pair classifier; outputs a score column
    decide/
      assign.py          # one-to-one: each S2/S3 id → best S1 (or none)
      select.py          # expected-F0.5 optimal top-k per S1
    eval/
      split.py           # full-world 5-fold assignment of S1 entities + closed sub-world sampler (§6)
      metric.py          # exact macro F0.5 (mirrors official definition)
      blocking_report.py # recall ceiling, reduction ratio, candidates/S1, by country
      errors.py          # dumps worst FP/FN with raw strings for manual review
    pipeline.py          # CLI: ingest | block | featurize | train | predict | submit | all
  configs/base.yaml
  README.md, requirements.txt
notes/  DATA_CONTEXT.md  EXPERIMENTS.md  SUBMISSIONS.md
docs/   MODELS.md  Documentation_template.md (filled at the end)
```

### Data contracts (parquet, stable column names — change only via PR that updates this section)

- `records_{split}.parquet`: `entity_id, source (1|2|3), country, name_raw, addr_raw, name_script` (dominant Unicode
  script of the raw name), `name_norm` (transliterated, accent-folded, lowercased, abbreviations expanded), `name_core`
  (legal form removed), `legal_form` (canonical or ""), `addr_norm, addr_street` (addr_norm minus house number and
  region/city tokens), `house_num` (primary number as string, "" if none), `addr_nums (list[str])`, `name_tokens (list[str])`
- `candidates_{split}.parquet`: `s1_id, cand_id, country, block_mask (int bitmask of passes), tfidf_name, tfidf_full, knn_rank`
- `features_{split}.parquet`: `s1_id, cand_id, <feature columns>, label (train/val only)`
- `scores_{split}.parquet`: `s1_id, cand_id, p` (calibrated)
- `folds_train.parquet`: `s1_id, fold (0–4), country, n_matches`
- `split ∈ {train, test}`; train artifacts carry the fold via a join on `s1_id`. `--subworld F` restricts to a closed
  sub-world of fraction F (§6) for fast loops.

## 5. What makes us better than the default pipeline (implement these, in this priority)

0. **Decoy-killer features (features/structured.py) — build these first, they matter most on this data.**
   House-number relation as categorical + numeric: `equal`, `one_missing`, `both_missing`, `prefix/digit_dropped`,
   `abs_diff`, `rel_diff`, `different`. Token-set difference: count/IDF-weight of name tokens present on one side only
   (after legal-form removal), same for street tokens; flag "only difference is one extra word". Street-name similarity
   computed **without** the house number and city/region tokens.
1. **Expected-F0.5 set selection (decide/select.py).** Don't use one global threshold. For each S1, sort candidates
   by calibrated p; for k = 0..K compute expected F0.5 ≈ 1.25·ΣTP / (0.25·E[|G|] + k), where ΣTP = Σ top-k p and
   E[|G|] = Σ all p + estimated blocking-miss mass; k = 0 scores Π(1−p) (probability the entity is a singleton).
   Pick the argmax. Tune only calibration, one global shrink factor and `decision_margin` on the full-world OOF predictions. Requires **well-calibrated p**
   (isotonic on held-out folds).
2. **One-to-one structure.** (a) Features: for a pair (s1, c), the similarity gap to the best *other* S1 that c is a
   candidate for; c's rank among all S1s. Compute these from **blocking/string similarities, never from model p** —
   using p would leak labels across folds unless done as a proper second OOF stage. (b) Decision: assign each S2/S3 to
   its single best S1 (by OOF / test p) before set selection.
3. **Graph / cluster features.** Within an S1's candidate set, the true matches look like each other. For each candidate:
   max/mean similarity to the other top candidates, whether an S3 candidate agrees with the top S2 candidate, etc.
4. **Chains and common names.** Name/token frequency per country (from the pool, no external data). Interactions:
   `name_sim × name_rarity`, `name_match & house_num_conflict`, `name_match & street_conflict`. Frequent names must
   be decided by address evidence.
5. **France robustness.** (a) Cross-country validation: train on US → evaluate on India and vice-versa; prefer features
   that transfer. (b) Country-agnostic features (ratios, char n-grams, digits, rarity). (c) French rules in `rules/fr.py`:
   legal forms SARL/SAS/SASU/SA/EURL/SCI/SNC incl. dotted variants (S.A.S. → sas); street types r/r./av/bd/pl/ch/imp/all/rte/fbg/
   qu; **accent folding on both sides**; a **département/region/city stop-list built from the test France records themselves**
   (tokens that are very frequent within the country) so "Gironde" vs "Nouvelle-Aquitaine" is not read as a conflict.
   (d) Optional: synthetic French positives by applying the noise operators measured on train to test-S1 French records
   (inputs only, no labels, no external data — document it openly).
6. **Blocking recall near the ceiling.** Multiple passes, all within country: TF-IDF char n-grams on name (transliterated),
   TF-IDF on street + house number, sorted-token name key, house_num + first street token key, optional e5 kNN
   (multilingual, helps native-script names). No postcode pass. Report recall per country, per script, per pass;
   add a pass for each miss type found.
7. **Cross-encoder (AWS GPU)**, fine-tuned with hard negatives from our own blocking, run only on the top-K per S1 after
   LightGBM; its score becomes a LightGBM feature (stacking, trained out-of-fold).
8. **Error analysis every iteration.** `eval/errors.py` → read 50 worst FP + 50 worst FN, log patterns in EXPERIMENTS.md.

## 6. Validation protocol (the thing everything else depends on)

**Principle: validate in a complete "world", exactly like test.** In test, all S1 entities, all their matches and all
decoys are present at once, and the one-to-one assignment competes across all of them. A split that keeps only some S1s
but all S2/S3 creates orphan records that get falsely assigned; a split that keeps only matched S2/S3 removes the decoys.
Both give misleading scores.

- **Full-world K-fold (the real protocol).** Block the *whole* train set once (all S1, all S2/S3, per country) and cache
  the candidates. Assign S1 entities to 5 folds (seed 42, stratified by country and match count). Train LightGBM on 4
  folds' pairs, predict the 5th → **out-of-fold p for every train pair**. Run the decision layer (one-to-one + set
  selection) over the whole train world with OOF p, then compute macro F0.5 on all S1s (and per fold / per country).
  This mirrors test exactly and never scores in-sample predictions.
- **Fast loop (sub-worlds).** For quick experiments, sample ~10% of S1 entities *together with* their GT matches and the
  decoys whose top-1 blocking neighbour is one of the sampled S1s ("decoy owner" from the cached candidates). This keeps
  the world closed and decoy density realistic. Use for iteration; confirm wins on the full-world OOF before submitting.
- **Density gap.** Test has ~25% more S2/S3 per S1 → validation likely **overstates precision**. Keep a
  `decision_margin` config (extra conservatism in set selection) and set it using leaderboard feedback, not dev.
- Also report **per country**, **per name script (Latin vs native)**, and the **cross-country transfer** scores (§5.5a).
- `eval/metric.py` must mirror the official rule exactly: GT empty & pred empty → 1; GT empty & pred non-empty → 0;
  GT non-empty & pred empty → 0; else F0.5 from set precision/recall.
- Report with every experiment: blocking recall ceiling, candidates/S1, macro F0.5 overall + per country, runtime.

## 7. Scale & engineering

- ~26M records total. The pandas EDA took ~2 h on a 16 GB laptop — the pipeline must not look like that.
  Convert every TSV to parquet once (`io.py`), then use **polars/duckdb** everywhere, parquet caches, `rapidfuzz.process.cdist(..., workers=-1)`
  or vectorized pair scoring, chunked sparse top-k (process S1 in chunks of ~50k per country).
- Never materialize country-wide cross products. Cap candidates per S1 (config `max_cands`), keep top by best pass score.
- Train each LightGBM fold on a **sample** of its training S1 entities (200–400k) with *all* of their candidate pairs —
  more gives diminishing returns. Predict OOF on the full held-out fold.
- Every stage supports `--subworld 0.1` and must finish in < 10 min on it. Full test inference is a
  separate, scheduled run (AWS high-RAM instance if laptops can't hold it).
- Seeds everywhere; log config hash + git commit with every artifact.

## 8. Experiment & submission discipline

- `notes/EXPERIMENTS.md`: one line per run → date, commit, change, blocking recall, OOF macro F0.5 (all / US / India / Latin vs native script), X-country score.
- `notes/SUBMISSIONS.md`: #, date-time IST, commit, what changed, OOF F0.5, public LB. Commit before every upload.
- One change per submission when possible. Reserve 1–2 of the final day's submissions as buffer.
- Final pick = best on **OOF F0.5 and public LB jointly** (private LB decides; don't overfit public).

## 9. Coding-agent instructions

- Read this file + `notes/DATA_CONTEXT.md` first. Stay inside the module you were asked to change; respect §4 contracts.
- Don't add dependencies without pinning them in `requirements.txt`. Never add network calls to the pipeline.
- Every function gets a docstring (the organizers review code). Prefer vectorized code; no Python loops over millions of rows.
- After changes: run the stage on `--subworld 0.1`, report metrics, append to EXPERIMENTS.md.
- Read TSVs only in `io.py`; everything downstream reads the parquet caches.
