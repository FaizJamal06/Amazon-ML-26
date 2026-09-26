# Overnight blocking, round 2 (Fri 25 → Sat 26 Sep)

Branch `r1/block-stream`, merged to `main` twice:
- `9b6d5cb`: streaming block, name_core name pass, all-numbers address pass;
- `27c7c23`: scale-proof keys, lean IDF, configurable limits.

Both merge messages contain "block-stream".

## TL;DR

- **The 20k smoke world hid a scaling bug.** Session B's full-train block (first merge) has **pair recall 0.718** (US 0.689, India 0.761, native script 0.587). The 20k world reported 0.947.
  - **Cause:** the passes used absolute limits. A token appearing in more than 2,000 records is not a key, and each pass keeps its top 50 S1s. At full size, ordinary words ("spring", "street", "foot", "ankle", "lucky") cross the cap, so easy pairs are never retrieved: same name, same number, same street.
- **The fix is merged and on by default.**
  - Pass D (`name_street`): an exact-name key plus (name token, street token) keys.
  - Address keys choose street tokens without the token cap.
  - Instead of capping single words, both drop any *combined* key shared by more than 50 S1s. Combined keys stay rare at any scale.
  - On the 200k-S1/country world, k=2 recall goes 0.862 → 0.932. On the 20k world it goes 0.947 → 0.955.
- **Full-train check with the fix, India only:** pair recall 0.761 → **0.876** (Latin 0.928, native 0.772). The US half was stopped: it thrashed at a 6.9 GB working set with 0.7 GB free.
- **Memory is fine on a server, tight on this laptop.**
  - Streaming keeps block's peak bounded by chunk size plus one country's indexes.
  - The 200k world peaks at 2.3–2.7 GB and B's full-test block (first merge) peaked at 4.2 GB, but the full-train US indexes with pass D reach about 7 GB.
  - The 12 GB target is met.
  - The laptop itself only has about 2.6 GB free while the IDE, Chrome and other Claude sessions are open.
- **B's `candidates_train` / `candidates_test` from the first merge are stale.** Rerun `block` on train and test with `main` ≥ `27c7c23`, then rebuild features, model and decide.

## 20k smoke world sweep

40k S1, 138k GT pairs, random unmatched records at the full-train density. Pair recall:

| step | commit | ALL | US | India | Latin | native | cands/S1 mean / p95 | block time / peak |
|---|---|---|---|---|---|---|---|---|
| start of round 2 (`main` 5dca8e8) | – | 0.9275 | 0.968 | 0.887 | 0.957 | 0.781 | 8.0 / 20 | 56 s / 4.9 GB |
| 1. streaming block (identical output, same hash) | 6c95838 | 0.9275 | 0.968 | 0.887 | 0.957 | 0.781 | 8.0 / 20 | 86 s / 1.9 GB |
| 2. name pass on name_core + non-legal skeleton | 9bf371d | 0.9277 | 0.968 | 0.888 | 0.957 | 0.782 | 8.0 / 20 | 85–98 s / 1.9–2.3 GB |
| 3. address pass on every number in addr_nums | f322e33 | 0.9467 | 0.972 | 0.921 | 0.967 | 0.846 | 8.1 / 20 | 89–98 s / 1.9–2.3 GB |
| 4. + addr_key_cap 50 + pass D (defaults now) | 586ece4 | **0.9549** | 0.978 | 0.932 | 0.972 | **0.868** | 8.1 / 20 | 50–54 s / 2.0–2.3 GB |

- At step 4, recall at k=1 / 3 / 5 is 0.944 / 0.958 / 0.960.
- The passes themselves, before any cut, reach 0.965.
- The 6,237 remaining k=2 misses: 4,891 were never retrieved by any pass, 1,346 were retrieved but ranked out.
- Timings on this laptop vary by up to 2× with other load.

## Scale test: 200k S1/country world (`cache/smoke200k`)

400k S1, 1.38M GT pairs, 1.87M S2/S3 in total. k=2 only; no uncapped run at this size.

| variant | ALL | US | India | native | block time | peak RAM |
|---|---|---|---|---|---|---|
| defaults at step 3 | 0.862 | 0.891 | 0.833 | 0.659 | 523 s | 4.0 GB |
| + lean IDF (d499ffa) | 0.862 | 0.891 | 0.833 | 0.659 | 793 s* | **2.3 GB** |
| frequency caps ×10 | 0.913 | 0.954 | 0.873 | 0.745 | 2,885 s | 3.7 GB |
| top_k_per_query 200 | 0.878 | 0.918 | 0.838 | 0.658 | 764 s | 5.2 GB |
| addr_key_cap 50 | 0.874 | 0.902 | 0.846 | 0.685 | 468 s | 2.5 GB |
| **addr_key_cap 50 + pass D** | **0.932** | **0.967** | **0.897** | **0.792** | 449 s | 2.7 GB |
| name_pair_keys 1 | failed: MemoryError (machine had ~2.6 GB free); not retried | | | | | |

\* Measured while other jobs were running.

**Diagnosis before the fix** (400 random US misses, defaults):
- 62%: no pass shared any key with the S1.
  - 70% of these: every common name word was over the cap.
  - The rest: domain-style names (`urologypartners.com`) or S2/S3 records without an address number.
- 36%: shared a key but cut by a pass's top 50. For example, 255 S1s tie on "sterling robotics" and the true one ranked 190th.
- 1.5%: retrieved, then ranked out by rank_score.

## Memory

Measured with a per-step working-set probe on one country of the 200k world (1.14M records):
- country records 0.5 GB;
- IDF tables were +1.7 GB transient for `addr_norm` alone. Each column was exploded into (entity_id, token) twice; now it's +0.47 GB, one pass per column, identical dictionaries;
- indexes 0.1 GB; each chunk's re-rank about 1 GB.

**Extrapolation:**
- The steady part grows with the largest country: full-train US is 7.5M records, 6.6× this country.
- That gives about 3–4 GB for records + IDF + indexes, plus about 1 GB per chunk, so **~5 GB peak for full train, and below that for test**. Measured later: about 7 GB for full-train US with pass D.
- B's measured full-test peak was 4.2 GB, with the old IDF code.
- Target ≤ 12 GB: met. No chunk-size or top-k reduction was needed.

**Runtime:**
- The 200k world takes about 450 s (1.87M queries).
- B's full test block (10M queries) finished in under 1 hour.
- Full train (10.3M queries) should take about 40–60 minutes on this laptop; see the full-train check below.

## Full-train check with the fix (`cache/full_r2`, main 27c7c23)

Started 09:33 and stopped at 11:03, after India had finished.

- **India, full train (883k S1, 4.1M S2/S3), k=2, before the per-S1 cap:**
  - pair recall **0.876** (0.761 with the first merge);
  - Latin 0.928, native script 0.772;
  - 7.46M pairs kept.
- **US:** only 9 of about 124 phase-1 chunks were done after about 1 h. The process's working set was 6.9 GB with 0.7 GB free, so it was paging heavily.
  - Real full-train peak for the biggest country is therefore **about 7 GB**, not the ~5 GB extrapolated above. That's still under the 12 GB target on a server, but too much for this laptop while other apps are open.
  - Most of it is the Python index dicts for 1.32M S1 × 4 passes.
- **From here on the measuring stick is `scripts/eval_fulldensity_blocking.py`:** the full S1 index with a sample of queries, instead of full runs.

## What didn't work / open items

- **name_pair_keys** (pairs of frequent name tokens): the only run died with a MemoryError while the machine had about 2.6 GB free. Not retried; pass D covers the same misses better. The code is still there, default off.
- **Session interruption:** the Claude Code process exited around 04:55 IST and killed two background experiments; they were rerun after 09:00.
- **Still missed at scale:**
  - native-script names (0.79 at 200k): the skeleton doesn't bridge every transliteration;
  - names written as domains or handles (`urologypartners.com`, `#salinasguggenheim`): they need a "name with spaces removed" key;
  - S2/S3 records with neither an address number nor a shared rare street word.
- **Python loops:** `keys.py` still loops over rows in Python (her original design; CLAUDE.md §9 prefers vectorized code). It's fast enough now: full block in under an hour, streaming, and it could run chunks in parallel if needed.
- **Decoy density:** the smoke worlds use random unmatched records, not the real decoys, so candidate counts are optimistic. Recall numbers are meaningful.
