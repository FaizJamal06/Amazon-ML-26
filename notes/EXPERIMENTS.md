# Experiments

One line per run (CLAUDE.md §8). Scores are OOF macro F0.5 unless stated otherwise.

| date | commit | change | blocking recall | OOF F0.5 all | US | India | Latin script | native script | X-country | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | 2311aab | R1 infra: config, stub data, exact metric, 5 folds, sub-world sampler, blocking report, one-to-one + threshold decide, submission writer, pipeline CLI | n/a (no real candidates yet) | n/a (stub only) | – | – | – | – | – | real train folds: 5 x ~441k S1, per-fold mean matches 3.46 and singletons 5.58% in every country x fold; make_folds 3 s, peak 1.4 GB. Stub end-to-end: decide + submit validator PASS. |
| 2026-09-25 | r1/decide-v2 | **STUB, NOT MEANINGFUL** — fallback scorer + decide v2 sanity run on cache/stub (80 S1, 444 pairs) | stub 1.000 | threshold t=0.8: 0.9864; ef05_approx: 0.9808; ef05_exact: 0.9808 | 0.9750 / 0.9639 (thr / ef05) | 0.9977 | latin 0.9986 | native 1.000 | – | fallback OOF Brier 0.0064; checks the plumbing only (stub labels are easy). fallback-train 30 s cold start; exact ef05 ~41 s per 2.2M S1s (synthetic timing). |
