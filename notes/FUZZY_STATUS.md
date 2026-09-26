# Fuzzy blocking: status (Sat 26 Sep, branch `r1/fuzzy`, merged to main with fuzzy off / workers 1)

Merged to main with the defaults unchanged. On the laptop: all tests pass, the stub validator passes, and the 20k smoke candidates hash equals main's. The fuzzy on/off eval and the parallel verification are left for the server.
Defaults are unchanged: `blocking.fuzzy: false`, `blocking.workers: 1`.

## Done (committed)

| commit | what |
|---|---|
| af10df8 | `notes/OVERNIGHT_BLOCK.md`, the round-2 summary |
| 820d2a3 | Refactor of `merge.py`: `build_country_index` / `query_passes` / `rerank_chunk`. Output unchanged: 20k smoke hash `42dcbd90e3b89534`, same as `main`. |
| 820d2a3 | Pass E, `blocking/fuzzy.py` (block_mask bit 4, flag `blocking.fuzzy`, default false). See below. |
| 820d2a3 | `normalize.py`: `strip_handle` / `is_handle_expr` / `strip_handle_expr` for handles and domains. Used only by pass E, so records are unchanged. |
| 820d2a3 | `scipy==1.18.1` pinned (BSD) |
| 104de2d (WIP) | `scripts/eval_fulldensity_blocking.py` (+ `--profile N`), `tests/test_fuzzy.py`, fuzzy `max_df` floor of 50 documents, chunk-parallel block (**untested**) |

Details of pass E:
- char_wb 3-gram TF-IDF on `name_core` + the no-space `name_core` + skeleton, per country;
- 3-grams in more than `fuzzy_max_df` of S1 names are dropped;
- chunked sparse products, with a vectorized numpy top-k per row;
- only runs for records whose best rank_score from passes A–D is below `fuzzy_min_score`.

Tests:
- `tests/test_fuzzy.py` passes: `strip_handle` on the 3 examples + camelCase, top-k vs brute force, and pass E retrieving `REEDPIZZA.COM` / `#salinasguggenheim` / `supremebu1kers.com`.
- `test_blocking`, `test_normalize`, `test_pipeline` and `test_features` passed after the refactor.

## Full-density eval: baseline (current `main` defaults, fuzzy off)

50,000 S2/S3 queries (seed 42) against the full train S1 index per country (India 883k S1, US 1.32M S1).
The report is in `reports/fulldensity_blocking_baseline.md` (git-ignored).

| group | n queries with owner | retrieved at all | recall@1 | recall@2 | recall@3 | recall@5 | S1s kept per unmatched query (k=2) |
|---|---|---|---|---|---|---|---|
| India | 14,687 | 0.899 | 0.856 | 0.875 | 0.882 | 0.889 | 1.78 |
| US | 22,320 | 0.963 | 0.949 | 0.956 | 0.958 | 0.960 | 1.87 |
| Latin script | 34,365 | 0.953 | 0.928 | 0.939 | 0.944 | 0.948 | – |
| native script | 2,642 | 0.735 | 0.704 | 0.717 | 0.722 | 0.728 | – |
| **ALL** | 37,007 | **0.937** | 0.912 | **0.923** | 0.928 | 0.932 | 1.83 |

It agrees with the partial full-train block from the morning (India k=2 0.876). Targets: retrieved ≥ 0.97, k=2 ≥ 0.95, native ≥ 0.90.

**Per pass** (share of owners retrieved by that pass / by that pass only):

| pass | retrieved | only this pass |
|---|---|---|
| name_token | 0.393 | 0.017 |
| addr_key | 0.730 | 0.078 |
| house_num | 0.076 | 0.004 |
| name_street | 0.828 | 0.096 |

**Runtime: 1,476 s total, over the 10-minute target.**

| | index build | query phase |
|---|---|---|
| India | 95 s | 400 s |
| US | 923 s | 24 s |

- Peak memory was 7.7 GB. The laptop had 2.4–7 GB free (IDE, Chrome and other Claude sessions use about 9 GB), so the process paged: 6.5 GB private but only 4.8 GB resident.
- That is why the timings are erratic. Disabling Python's garbage collector did not help, so it's memory, not GC.
- The key-pass indexes are Python dicts of tuples: several GB per country at full size.

## Not done / next steps

1. **Fuzzy on/off comparison on the full-density eval.**
   - Run: `python code/business_entity_resolution/scripts/eval_fulldensity_blocking.py --label fuzzy_on --set blocking.fuzzy=true --profile 2000`
   - It was running when the battery died; there's no result.
   - Compare with the baseline table above; log both in `notes/EXPERIMENTS.md`.
   - Close other apps first, or run on the server.
2. **Profile per pass (ms/record) → full-train / full-test block time on the 16-core, 128 GB server.**
   - The `--profile 2000` run above prints it.
   - Rough numbers so far, sequential: about 1 ms/record for A–D + re-rank when not paging (US: 24 s for 30k queries); index build 1.5–15 min per country depending on memory.
   - So one split (~10M queries) takes about 3 h sequentially: over 1 hour → parallelism needed.
3. **Chunk-parallel block** (in 104de2d, untested):
   - `blocking.workers` N>1 runs phase 1 and phase 2 over chunks with `pool.map`, which keeps order, so the output should be identical.
   - `blocking.pool_start`: `spawn` by default. It works on Windows and Linux; the index is pickled to each worker, so RAM is roughly workers × index (~6–7 GB per country → about 8–12 workers on 128 GB).
   - `fork` is opt-in: polars warns that forked children can deadlock once the parent used its thread pool. Test it on the stub world with a timeout before using it.
   - **To verify on the Linux server:**
     - `tests/test_blocking.py::test_parallel_block_identical` asserts workers=1 == workers=4 with `pool_start=fork` on a stub world. It is skipped on Windows, where fork doesn't exist.
     - If it hangs, polars did not survive fork: use `blocking.pool_start=spawn`.
     - Then hash-compare the 20k smoke world with workers 1 vs N. The reference hash with defaults is `42dcbd90e3b89534`.
4. **Measure each step on the full-density eval and log it** (`notes/EXPERIMENTS.md` has no rows for this branch yet):
   - baseline (above);
   - fuzzy on;
   - tune `fuzzy_min_score` (0.8) and `fuzzy_max_df` (0.002) if recall or runtime needs it.
5. **Then:** run all test files + the stub validator, and merge `r1/fuzzy` to main with fuzzy still OFF.
6. **Longer term:** the key passes loop over rows in Python and hold Python-dict indexes, which is what makes the laptop slow and RAM-heavy. Rewriting the key generation and lookup as polars explode / join / top-k per chunk would be much faster and lighter, and follows CLAUDE.md §9.
