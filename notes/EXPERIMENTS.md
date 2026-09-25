# Experiments

One line per run (CLAUDE.md §8). Scores are OOF macro F0.5 unless stated otherwise.

| date | commit | change | blocking recall | OOF F0.5 all | US | India | Latin script | native script | X-country | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | 2311aab | R1 infra: config, stub data, exact metric, 5 folds, sub-world sampler, blocking report, one-to-one + threshold decide, submission writer, pipeline CLI | n/a (no real candidates yet) | n/a (stub only) | – | – | – | – | – | real train folds: 5 x ~441k S1, per-fold mean matches 3.46 and singletons 5.58% in every country x fold; make_folds 3 s, peak 1.4 GB. Stub end-to-end: decide + submit validator PASS. |
