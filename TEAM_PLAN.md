# TEAM_PLAN.md — 4 people, 3 days

Read `CLAUDE.md` first. This file says **who owns what, what they hand over, and by when**.
Rule of thumb: people work in parallel against the **data contracts in CLAUDE.md §4**, never against each other's code.

## Roles

Team: **Faiz, Dhanishkaa, Nitish, Chris.** R2–R4 are a first assignment — swap them in the first sync if someone is stronger elsewhere (R4 should be whoever is most comfortable with AWS/GPUs; R3 whoever knows LightGBM best).

| Role | Owner | Owns (modules) | Hands over |
|---|---|---|---|
| **R1 Lead / Eval / Decision** | Faiz | repo skeleton, `config.py`, `io.py`, `eval/*` (folds, sub-world sampler, metric, reports, errors), `decide/*` (one-to-one, expected-F0.5 selection), `pipeline.py`, submissions, final package + documentation | folds + sub-worlds + scorer everyone uses; final TSVs |
| **R2 Normalize / Blocking** | Dhanishkaa | `normalize.py` (incl. **script detection + transliteration**, house-number extraction), `rules/*` (incl. **France rules**), `blocking/*` | `records`, `candidates` artifacts (train + test) + recall report (per country / script / pass) |
| **R3 Features / LightGBM** | Nitish | `features/*` (**decoy-killers first**: house-number relation + token-set difference; then pairwise, rarity, context, graph), `model/lgbm.py` (5-fold OOF), calibration, cross-country validation | `features`, `scores` artifacts (OOF on train), feature importance |
| **R4 Deep models / AWS** | Chris | SageMaker/EC2 setup, `blocking/embed_knn.py` (for R2), `model/cross_encoder/*`, full-scale test runs on the high-RAM box, synthetic France pairs (optional) | e5 kNN candidates; out-of-fold cross-encoder score column; full test inference |

Everyone: log every run in `notes/EXPERIMENTS.md`; do error analysis on your own stage.

## Critical path (what blocks what)

```
R1 split + metric ─┐
R2 records ────────┼─► R2 candidates ─► R3 features ─► R3 scores ─► R1 decision ─► submission
                   │                         ▲
R4 e5 kNN ─────────┘ (extra blocking pass)   └── R4 cross-encoder score (Day 2–3, OOF)
```

Unblock trick for Day 1: R3 and R1 start on **stub data** (a tiny hand-made parquet that follows the contracts) so no one waits.

## Dependencies & handoffs

Artifact names are the `cfg.artifact()` names from CLAUDE.md §4 (e.g. `candidates_train` = `cfg.artifact('candidates', 'train')`).

| Producer → Consumer | Artifact | Needed by (IST) | Until then |
|---|---|---|---|
| Faiz → all | merged `main`: config + `artifact()`, stub data script, contracts, metric, folds | ✅ done 15:30 | — |
| Chris → Dhanishkaa, Faiz, Nitish | `source{1,2,3}_{train,test}`, `gt_train` parquet | 16:30 | work on stub data / a 100k-row slice |
| Dhanishkaa → Faiz | `records_train`, `candidates_train` (+`rank_in_cand`, `block_score`) | 19:00 | Faiz builds decide v2 + errors.py on stub |
| Faiz → Nitish | `folds_train`, `subworld_train` (+ blocking report) | 19:30 (≤30 min after candidates) | Nitish develops features on stub |
| Nitish → Faiz | `scores_train` (OOF, calibrated) + saved model + calibrator | 21:00 | Faiz tunes decide on stub |
| Dhanishkaa + Nitish → Chris | code merged to main for full TEST run | 21:30 | Chris dry-runs the pipeline on the test sub-sample |
| Chris → Faiz | `records/candidates/features/scores_test` | 22:30 | — |
| Faiz → leaderboard | Submission 1 (validator PASS) | 23:30 | — |
| Chris → Dhanishkaa (Day 2) | e5 kNN candidates pass | Sat 13:00 | TF-IDF passes |
| Chris → Nitish (Day 3) | OOF cross-encoder score column | Sun 15:00 | LightGBM without it |

**Shared-file rules**
- `ber/io.py` has two sections: `ingest` (Chris) and the writer (Faiz). Edit only your own section.
- `pipeline.py` belongs to Faiz; others expose functions with the documented signature, and Faiz wires them in.
- `CLAUDE.md` §4 changes need a PR and a ping to the downstream owner.

**Handoff protocol.** When an artifact is ready, the producer posts in the team chat:
- artifact name + split;
- row count;
- config hash + commit;
- location (S3 path / Drive link);
- runtime + peak RAM.

The producer's code must also be on `main` (merged PR) before anyone depends on it for the TEST run.

**Fallbacks** (decided at the 19:00 sync, not at 22:00):
- AWS not ready by 19:00 → the full runs move to the laptop with the most RAM.
- Candidates late past 19:30 → Nitish trains on a 100k-S1 world that Dhanishkaa blocks locally.
- LightGBM not ready by 22:00 → Submission 1 uses Faiz's rule-based fallback scorer (block_score + house_num
  relation + name similarity, threshold tuned on the train sub-world).
- Full test run not finished and validated by 23:15 → skip the Day-1 submission. An unused slot costs nothing;
  a rushed, unvalidated file wastes a slot and can mislead us with a bad score.

**Corrections to the Day-1 prompts.** The Day-1 prompts used `cache/{split}/...` paths; the only valid names are the
`cfg.artifact()` ones:
- ingest writes `source{1,2,3}_{split}` + `gt_train`;
- normalization writes `records_{split}`;
- blocking writes `candidates_{split}` (with `rank_in_cand`, `block_score`);
- features/scores write `features_{split}` / `scores_{split}`.

Always read and write through `cfg.artifact(name, split, subworld)`.

### Status board

| Person | Current task | Blocked on | Next handoff |
|---|---|---|---|
| Faiz | r1/decide-v2 ready for PR (stub-tested): fallback scorer → `scores_fallback`, `errors` report, decide v2 (ef05 exact/approx) + `compare-decide` | real candidates_train (19:00) for blocking report + sub-world; scores_train (21:00) for decide tuning | folds_train + subworld_train → Nitish ≤30 min after candidates; Submission 1 at 23:30 (fallback scorer if LightGBM is late) |
| Dhanishkaa | normalization + blocking v1 | source parquet from Chris (16:30) | records_train + candidates_train by 19:00 |
| Nitish | features v1 + LightGBM on stub | subworld_train from Faiz (~19:30) | OOF scores_train by 21:00 |
| Chris | AWS setup + ingest | — | source{1,2,3}_{train,test} + gt_train by 16:30; full test run by 22:30 |

Each person updates their row at every sync (13:00 / 19:00 / 23:00) and when a handoff lands.

## Timeline (IST)

### Day 1 — Fri 25 Sep: end-to-end baseline + first valid submission
- ✅ EDA done (`notes/DATA_CONTEXT.md`), numbers verified in CLAUDE.md §3.
- **by 16:00** — R1 pushes the repo skeleton + contracts + stub data. R4 (or whoever has the most RAM) converts all TSVs to parquet once and shares them.
- **by 19:00** — R1: exact metric + validator wrapper + fold assignment. R2: normalization v1 (transliteration, accent folding, house numbers) + TF-IDF name blocking on the whole train set, recall report. R4: AWS account, billing alert, high-RAM instance ready. R1 builds the closed sub-world sampler as soon as R2's candidates exist.
- **by 22:00** — R3: feature set v1 (string + house-number relation + token-set difference) + LightGBM, OOF on the sub-world. R1: one-to-one + simple threshold.
- **by 23:30** — **Submission 1**: full-test run of the baseline (R4 runs it on AWS if laptops can't). Validator PASS first.
- Leave ≥2 Day-1 submissions unused unless there's a clear, validated improvement.

### Day 2 — Sat 26 Sep: the differentiators
- R1: expected-F0.5 set selection + calibration → measure gain vs threshold on OOF.
- R2: extra blocking passes (house_num + street token, sorted-token name key, street TF-IDF), R4's e5 kNN pass (helps native-script names); push recall ceiling; France rules v1.
- R3: rarity / chain features, context & competition features, graph features; **cross-country validation** (US→IN, IN→US).
- R4: cross-encoder training on GPU with hard negatives from R2's candidates; out-of-fold scores on the same 5 folds as LightGBM.
- Submissions: ~3–4, each with **one** change. One of them = **France probe** (only France handling changes).
- **by 23:00** — merge day: best config on `main`, full test run, submit.

### Day 3 — Sun 27 Sep: stack, harden, ship
- **by 15:00** — cross-encoder score as LightGBM feature (if OOF gain ≥ +0.3 pt, keep; else drop).
- **by 18:00** — final tuning on OOF only (calibration, shrink factor, max_cands). Freeze features.
- **by 20:00** — **code freeze.** Clean reproduction run from a fresh clone following README exactly.
- **by 21:30** — final submissions (best + one safe backup). Keep 1–2 slots in reserve until then.
- **by 23:00** — package the zip: `output/`, `code/business_entity_resolution/`, filled `Documentation_template.md`. Submit ≥ 45 min before the deadline.

## Working agreements
- **Git:** `main` always runs end-to-end. Branch per person (`r2/blocking-housenum`), small PRs, R1 merges. Never commit data, models > 50MB, or credentials.
- **Contracts:** column changes only via PR that edits CLAUDE.md §4 and pings the downstream owner.
- **Fast loop:** everything runs on `--subworld 0.1` in < 10 min; full-world OOF confirms wins; full test runs only for submissions.
- **Numbers or it didn't happen:** every claim of improvement comes with OOF F0.5 (overall + per country + per script) in EXPERIMENTS.md.
- **Syncs:** 10-min check-ins at 13:00, 19:00, 23:00 each day. Blockers raised immediately in the group chat.
- **AWS:** stop instances when idle; billing alert set; spot where possible; SageMaker jobs call `src/` code, not notebook cells.
- **Documentation as we go:** each owner writes 1 paragraph per module in `docs/methodology_notes.md` — R1 assembles the final document from it on Day 3.

## Definition of done (final package)
- [ ] `validate_submission.py` PASS on the final TSVs
- [ ] fresh clone + README reproduces both TSVs (same ids, same matches)
- [ ] `requirements.txt` pinned; `docs/MODELS.md` lists every model with license + revision
- [ ] no network calls in `src/`; no external data
- [ ] every function has a docstring
- [ ] Documentation covers: methodology, blocking strategy (with recall + reduction ratio numbers), model + features, decision layer, France handling, experiments table
- [ ] `notes/SUBMISSIONS.md` complete
