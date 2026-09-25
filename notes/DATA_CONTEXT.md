# DATA_CONTEXT — Amazon ML Challenge 2026: Business Entity Resolution

Exploration only (no models, no submissions). All numbers come from `notes/eda.py`, and its full output is saved in `notes/eda_output.txt`.
Deadline: **27 Sep 2026, 11:59 PM IST**.

---

## 1. Folder map

```
ml26/
├── 6ab10eb3b23ba_student_resource.zip      1.09 GB  (original, untouched)
├── .venv/                                  Python 3.12.3 venv
├── notes/
│   ├── DATA_CONTEXT.md                     this file
│   ├── eda.py                              re-runnable EDA (~2 h full; EDA_NROWS=100000 for a 1-min smoke test)
│   └── eda_output.txt                      full EDA output (98 KB)
└── student_resource/                       extracted (skipped __MACOSX/ and .DS_Store)
    ├── README.md                           14 KB  problem statement + submission rules
    ├── Documentation_template.md           2 KB   methodology write-up template (goes in final zip)
    ├── utils/validate_submission.py        14 KB  stdlib-only format validator
    └── dataset/
        ├── train/train_source1.tsv         210 MB   2,206,821 rows
        ├── train/train_source2.tsv         489 MB   5,034,616 rows
        ├── train/train_source3.tsv         504 MB   5,285,603 rows
        ├── train/train_ground_truth.tsv    127 MB   2,206,821 rows
        ├── test/test_source1.tsv           175 MB   1,732,544 rows
        ├── test/test_source2.tsv           509 MB   4,887,273 rows
        └── test/test_source3.tsv           506 MB   5,082,316 rows
```

No sample submission is included in the bundle.

**Environment:** Python 3.12.3. Installed: pandas 3.0.6, numpy 2.5.3, rapidfuzz 3.14.6 (MIT), and pyarrow 25.0.1 (Apache-2.0).
- pyarrow was added because pandas 3 then stores `dtype=str` columns as compact Arrow strings. Without it, the S2/S3 files do not fit comfortably in 16 GB RAM.
- rapidfuzz was used for the nearest-neighbour check in step 6.

**File format:**
- Every file is UTF-8, uses **CRLF** line endings, and is tab-separated.
- Quoting is standard CSV style: fields that contain `"` are wrapped in quotes and the inner quotes are doubled. This affects 4–349 rows per file.
- The mandated `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)` parses everything correctly. Row counts equal `wc -l` minus 1, and `country` has no stray `\r`.
- `keep_default_na=False` matters: addresses contain literal `null`, `NULL`, `N/A` and `NA`.

---

## 2. Documentation vs. the problem statement

**README adds these requirements beyond the brief:**
- **Two output files.**
  - `matching_results.tsv`: scored.
  - `candidate_pairs.tsv`: not scored, but must be in the final zip. It must be the *exact last* candidate set fed to the matcher, not an early blocking pass. The organisers use it to audit blocking recall and reduction ratio. Final matches should be a subset of it.
- **Final zip layout.**
  - `output/{matching_results,candidate_pairs}.tsv`
  - `code/business_entity_resolution/{src/, README.md, requirements.txt}`, runnable end to end with pinned versions
  - the filled-in `Documentation_template.md`
- **Methodology document** must cover: problem analysis/EDA, approach type, blocking keys, candidate count and how recall was protected, name and address features, model type, threshold selection, validation F0.5, typical false positives and false negatives, and code structure.
- **Leaderboards.** The public leaderboard uses a subset of test; the final ranking uses the private remainder. Predict the full test set.
- **Scale.** The test set is ~1.73M S1 entities against ~9.97M S2+S3 records.

**Contradictions and gotchas:**
- The README's own example, `pd.read_csv(..., sep="\t")`, leaves NA parsing on. That would turn `null`/`N/A`/`NA` address strings into NaN. Always use the mandated call.
- README constraint 2 says ids that do not exist in test "will be rejected". The validator's docstring says a nonexistent id "only lowers your score" and is not a rejection. Treat it as a hard rule regardless.

### Rules checked by `utils/validate_submission.py`
Run it from `student_resource/`:
```
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test [--check-ids]
```
**File-level checks:**
1. The file exists and is not empty.
2. The file is valid **UTF-8**. A `UnicodeDecodeError` means FAIL.
3. **Header.** If the header has no TAB but contains a comma, it fails as a CSV. Otherwise, the header split on TAB, then stripped and lower-cased, must equal exactly `["source1_entity_id","matched_entity_ids"]`. For the candidate file it must equal `["source1_entity_id","candidate_entity_ids"]`.
   - A UTF-8 **BOM** would break this check. Write plain `utf-8`, not `utf-8-sig`.

**Row-level checks:**

4. Every non-blank data line contains a TAB. Blank lines are ignored.
5. No `source1_entity_id` appears on more than one row.
6. No id is repeated inside one list.
7. No `S1-` ids appear in a list.
8. Every id starts with `S2-` or `S3-`.
   - Ids are split on `,` and **not stripped**, so `"S2-1, S3-2"` fails. Write lists without spaces.
9. Every S1 id in `test_source1.tsv` has a row, with an empty list meaning no match.
10. There are no rows for S1 ids that are not in test.

**Optional and soft checks:**

11. *(only with `--check-ids`)* Every listed id exists in `test_source2/3.tsv`. This loads several GB into memory.
12. The candidate file goes through the same rules 1–11. If it is missing, you get a **warning**, not a failure.
13. **Warning only:** matched ids that are not in that S1's candidate list.

---

## 3. Key numbers

### Sizes and country mix
| file | rows | US | India | France |
|---|---:|---:|---:|---:|
| train S1 | 2,206,821 | 1,323,633 (60%) | 883,188 (40%) | – |
| train S2 | 5,034,616 | 3,016,817 | 2,017,799 | – |
| train S3 | 5,285,603 | 3,170,056 | 2,115,547 | – |
| test S1 | 1,732,544 | 663,106 (38%) | 809,986 (47%) | 259,452 (15%) |
| test S2 | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 |
| test S3 | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 |

- All `entity_id`s are unique within each file.
- Train and test share **no** ids and no S1 (name, address) pairs.
- The country value set is exactly {US, India} in train and {US, India, France} in test.

### Missing and empty values
- **S1** has no empty fields anywhere, in train or test.
- **S2/S3 empty addresses:** 2.3–3.7%.
- **S2/S3 placeholder tokens** (`null`, `N/A`, `NULL`) inside the address: 2–3% for US and India, 0% for France.
- **S2/S3 names** are never empty.

### Lengths (median chars; name / address)
| country | S1 | S2 | S3 |
|---|---|---|---|
| US | 22 / 34 | 23 / 32 | 23 / 39 |
| India | 27 / 76 | 27 / 68 | 27 / 58 |
| France (test) | 19 / 48 | 20 / 40 | 20 / 41 |

Other shape details:
- US addresses almost always have 3 comma-separated parts.
- India addresses have 3–9 parts, with a median of 5.
- France addresses have 2–3 parts.
- India S3 addresses are often truncated; the 5th-percentile length is 19–20 chars.

### Ground truth (train)
- **Integrity is clean.**
  - Ground-truth S1 ids and `train_source1` ids match exactly.
  - No duplicate rows, no whitespace in id lists, no repeated ids inside a list.
  - All 7,638,365 matched ids exist in S2 or S3.
- **Matches per S1 entity:**

  | # matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10+ |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | % of S1 | **5.58** | 5.40 | 17.00 | 24.05 | 21.94 | 14.59 | 7.47 | 2.90 | 0.85 | 0.19 | 0.03 |

  - The mean is 3.46 and the max is 11.
  - The distribution is **the same for US and India** to within 0.1 percentage points, with singletons at 5.59% vs 5.58%.
- **S2 vs S3 composition:**
  - Matched ids split 48.4% S2 and 51.6% S3.
  - By S1 entity: 80.5% match both S2 and S3, 6.5% match S2 only, 7.4% match S3 only, and 5.6% match nothing.
  - One S1 matches 0–5 S2 records and 0–6 S3 records. So **S2 and S3 are not deduplicated internally**: one business appears several times in the same source.
- **One-to-one from the S2/S3 side:**
  - **No S2/S3 id appears under more than one S1** (0 of 7.64M).
  - Each S2/S3 record belongs to at most one S1 entity. This is a hard assignment constraint we can exploit.
- **S2/S3 records that match nothing:** 26.6% of S2 and 25.4% of S3. The rate is identical for US and India.
- **Cross-country matches: none.** Zero of 3.69M S2 pairs and zero of 3.94M S3 pairs cross countries.

### ⚠ Train → test shift in record density
| | S2 per S1 | S3 per S1 |
|---|---|---|
| train, US and India | 2.28 | 2.40 |
| test, US | 2.82 | 2.93 |
| test, India | 2.86 | 2.97 |
| test, France | 2.71 | 2.82 |

- Test has about **25% more S2/S3 records per S1** than train, in every country.
- Train has about 1.67 matched S2 records per S1. If test has the same matches-per-S1, the unmatched or decoy share in test S2/S3 rises from about 26% to about 41%.
- The other possibility is more matches per S1 in test. We cannot tell which is true. Either way, validation on train will probably be **optimistic about precision**.

---

## 4. Noise catalogue

Measured on 183,242 matched pairs drawn from 50k random matched S1 entities.
- Exact name match (case-insensitive): only 10–11% for India and 19% for US.
- Exact address match: 4–11%.

**Name token_sort_ratio quantiles:**

| quantile | US | India |
|---|---|---|
| 5th | ~52 | 9–10 |
| 25th | 77–81 | 43–64 |
| median | 91–92 | 85–86 |

India's very low 5th percentile comes from the native-script names.

### Name noise
| pattern | examples (S1 → S2/S3) | rate / note |
|---|---|---|
| **Native-script names** (Indic translation or transliteration) | `Aditya Finance Pvt Ltd` → `আদিত্য ফাইন্যান্স প্রাইভেট লিমিটেড`; `Vijay Technology` → `विजय टेक्नोलॉजी`; mixed `क्रिएटिव Services लिमिटेड` | India S2 23.5%, S3 13.3% (S1: 0%). Scripts seen: Devanagari, Bengali, Gujarati, Kannada, Tamil, Telugu, Malayalam. Legal suffixes are phoneticised too (`प्राइवेट लिमिटेड` = Private Limited, `प्रा. लि.` = Pvt. Ltd., `एलएलपी` = LLP) |
| Legal-suffix drop, expand or abbreviate | `Inc` → `Incorporated`; `Limited` → `Ltd`; `Pvt. Ltd.` → `Pvt.`; `Private Limited` → `Private`; `O.D.` → `Service`; `LP` dropped | very common |
| Legal-suffix reordering | `PVT. ASTOR TRADING LTD.`, `Versa Marketing Limited Private`, `P.C. Cardenas Árb0r`, `LLC Moncada …`, `SA PG Finance` | common |
| **Generic words appended or dropped** | `+ Service`, `+ Center`, `+ Holdings`, `+ Ventures`, `+ Services`; `Zr International School Ltd` → `Zr International Ltd Service` | appear in true matches **and** in decoys (see §6) |
| Prefixes and honorifics | `The`, `Shri`, `Sri`, `Dr`, `Mr`, `M/s` | |
| DBA / trade names | `Irinylasolx doing business as Temple Church`, `Fluxmira trading as …`, `Soldelta Labs F/K/A Club des Choeur` | |
| Junk prefixes and suffixes | `-- `, `>> `, `<< `, `"""`, `[Services]`, `(ID: 53810)`, ` - 1524075706` (phone-like) | 1.5–2.9% of matched names start with punctuation |
| Handles and domains as the name | `@nizarmarine`, `wilfordhancock.com`, `creativeconstruction.com` | 4.8–6.1% of matched names |
| Typos (insert, delete, swap) | `Imcdiustries`, `Vntres`, `Connsturctions` | |
| **OCR-style digit-for-letter swaps** | `Hand1ooms`, `Kuna1`, `6arment`, `Advis0rs`, `5umit`, `Árb0r` | |
| **Accent injection** | `Éuropean`, `Sánchez`, `Ínc.`, `límited`, `Prívate`, `Gróup` | 4–7% of US/India S2/S3 names vs 0% in S1 |
| Case | S2 names ALL-CAPS in 15–23% of cases; S3 sometimes all lower-case | |
| Punctuation, spacing, `&`/`and` | `Baker And-Isaac`, `SERVICES-PRIVATE`, double spaces, `,,` | |
| Word-order transposition | `Fiorenze P. D.O., Duncan, M.D.` | |
| Truncation or repetition | `ROBLES, GREER &`, `Hernandez, Cosio and Wilburn,`; `M.D., M.D., P.C. P.C.`, `Service Service` | |

### Address noise
| pattern | examples | rate / note |
|---|---|---|
| **Postcode / PIN** | – | **Essentially absent.** ZIP-like codes appear in ~0% of records, 6-digit PINs in ≤0.2%, French 5-digit postcodes in ~0.4%. 99.99% of matched pairs have no postcode on either side. The 5-digit numbers in US addresses (~10%) are **house numbers** |
| State format differs by source | **US:** S1 and S2 use `NC`; S3 uses `North Carolina`. **India:** S1 uses `Maharashtra`; S2 uses `Maharashtra` or `महाराष्ट्र` (~24% native script); S3 uses `MH` or native script. Spelling variants: `Keralam`/`Kerala`, `Orissa` | systematic per source |
| Component reordering | `CA, Fontana, 9342 Live Oak Avenue`; `MA, 51 COMMONWEALTH AVE, QUINCY` | 7–15% of addresses start with the state or region |
| Street abbreviations | Street→St, Avenue→Ave, Drive→Dr, Court→Ct, Lane→Ln, Cove→Cv, Unit/Apt→`#` | S2/S3 vs S1 |
| Casing | S2 US addresses 90% ALL-CAPS; S2 India 24% | |
| Missing components | Unit dropped; India S3 heavily truncated (`7/3, North Delhi, DL`); empty address in 3–5% of matched records | |
| **House number noise** | leading zeros `00174`, `0091`, `053`; `##` injection `##17`, `No.##68`, `RZ-##6`; an injected spurious prefix `H.NO 717 48`, `Block D-00748 …`, `No. 413 R-565`; digit drop `658`→`58`, `1021`→`021`, `1205`→`120`; ranges `11077-11079` | see §8 |
| City variants | typos `CHARLOTE`, `SEHLDON`, `VCAVILLE`, `Kingsotn CITY`; suffix noise `CITY`, `CCITY`, `CDP`, `Township`, `County`; **city substituted** (`Bombay` for Mumbai, `Salt Lake City` for Murray, `Qinlan` for West Tawakoni, `Ghaziabd`) | |
| Transliteration and spelling | `Bangalore`/`Bengaluru`, `Kolkata`/`Kolkatta`, `Ferozepur`/`Firozpur`, `COIMATORE` | |
| **Landmark phrases** (`Near`, `Nr`, `Opp`, `Behind`) | `Opp.Raymond Show Room`, `Nr Avalon Hotel` | **India only:** 13.5% of S1, 11.5% of S2, 8.7% of S3; ~0% for US and France |
| Care-of | `C/O Samjeeb Kumar …` | India |

---

## 5. France (test only)

- **Volume:** 259,452 S1 (15% of test S1), 703,378 S2 and 731,615 S3 records.
- **Geography is tiny:**
  - 3 regions: Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire.
  - About 18 cities: Bordeaux, Nantes, Lille, Tourcoing, Dunkerque, Roubaix, Calais, Saint-Nazaire, Pessac, La Teste-de-Buch, Mérignac, Lège-Cap-Ferret, Pornic, Saint-Herblain, La Baule-Escoublac, …
  - City and region therefore carry almost no discriminating power. The **street name and house number** carry the signal.
- **S1 style:** `17 Rue de Valmy, Lille, Hauts-de-France`, i.e. number + street, city, region. There are 3 parts in >95% of cases, sometimes rotated. Names are short (median 19 chars, about 3 words), usually `<word(s)> <legal form>`.
- **Legal forms in S1 names:** SARL 28%, SAS 20%, (France) 8%, EURL 6.5%, SA 4.9%, SASU 4.1%, Ets/Établissements 4.1%, Cie 3.6%, SCI 3.2%, Association 1.7%, EI/EIRL 1.6%.
  - S2/S3 add dotted forms (`S.A.S.`, `S.A.R.L.`, `S.A.S.U.`) and SNC (1.1%), plus variants like `Sàrl`, `Groupe`, `Fils`, `Holding`, `Participations`, `Développement`.
- **Street types:**
  - S1 writes them out: `Rue` 66%, `Avenue` 12%, `Boulevard`/`BD`, `Allée`, `Impasse`, `Route`, `Chemin`, `Place`, `Cours`, `Quai`, `bis`/`ter`, `ZA`/`ZI`/`ZAC`.
  - S2/S3 abbreviate: **`R`/`R.` = 24–25%**, plus `AV`/`Av.`, `BD`, `PL`, `CH`, `IMP`, `ALL`, `RTE`, `Q.`, `CRS`, `ESPL`. They also add `N°`/`Nº` and `#` prefixes.
- **Accents:**
  - S1 has accented names in 16% and addresses in 28%.
  - S2/S3 have **more** accented names (24%, from accent injection like `Àzar`, `Frèrês`, `Ínc`) and **fewer** accented addresses (18%, from dropped accents like `MERIGNAC`/`Merignac`, `Lege-cap-ferret`).
  - Fold accents before comparing.
- **Region vs département:** S1 uses the region. S2/S3 use either the region or the **département** (`Gironde`, `Nord`, `Loire-Atlantique`, `Pas-de-Calais`). This mirrors the US state abbreviation ↔ full name mapping.
- **Case:** S2 has ALL-CAPS addresses in 29% (`R DE LA PLANCHE AU GUE`); S3 uses title case with odd lower-casing after hyphens (`Saint-nazaire`, `La Teste-de-buch`).
- **Postcodes:** rare, about 0.4–0.5%. The format is 5 digits: `33000 BORDEAUX`, `44800`, `59200`. False positives come from `BP 80506`, `CS 41195` and leading-zero house numbers (`00223 Rue Faidherbe`).
- **No landmark phrases** (0%). Junk name prefixes such as `"""`, `<<` and `--` appear at 3.7–3.8%, about the same as the other countries.
- **Takeaway:** France looks like the **same synthetic noise generator** applied to French data. That covers suffix drop and reordering, abbreviation, casing, accents, `#`/`N°`/leading-zero house numbers, region↔département, typos, and `F/K/A`. **Country-agnostic features should transfer.** Anything that relies on US/India vocabularies, such as state-abbreviation maps or legal-suffix lists, needs French entries or a data-driven equivalent.

**Test US and India look like train.**
- Every profile rate matches train to within about 0.5 percentage points: lengths, Indic-script share, casing, empty addresses, house-number presence, landmark rate.
- The only differences are the country mix (India now outnumbers US) and the record-density shift in §3.

---

## 6. Hardness of negatives (brute-force nearest neighbour)

Method:
- Queries: 60 singletons and 60 matched S1 entities.
- Candidates: every same-country S2 and S3 record.
- Score: token_sort_ratio on name + address.

**Best-scoring negative per query:**

| | median | range |
|---|---|---|
| singletons | 84–85 | 68–95 |
| matched S1 | ~81 | 65–94 |

**Best true match per matched query:** median 93, range 58–100.

**In 8.3% of matched queries, some negative scores at or above the best true match.**

**Negatives look like deliberate decoys.** A singleton's closest record is usually the same business with one controlled change:
- **A shifted house number:**
  - `6800 Strand Avenue` → `6821 Strand Ave`
  - `2111 Vanda Lane` → `2118 VANDA LANE`
  - `635 Baker Street` → `646 Baker Street`
  - `8620 1185` → `8631 1185`
  - `89-32 88 Street` → `0089-34 88 Street`
  - `Sco No.5` → `SCO NO.1-16`
- **An added or changed name token:**
  - `Vm Business (India) Pvt Ltd` → `Vm Business (India) Overseas Pvt Limited`
  - `Renewable Properties Private Limited` → `Private Renewable Properties Industries Limited`
  - `West Royal Tankers` → `West Royal-Tankers Eastgate Inc`
  - `Anand Food Private Limited` → `Anand Food Limited`

This explains the ~26% of S2/S3 records that match nothing: many of them are decoys built from real S1 entities.

---

## 7. Structured fields

| rate of records with… | US | India | France |
|---|---|---|---|
| US ZIP (standalone 5 digits, or after the state code) | 0.00% | 0.1–0.2% | 0.1% |
| 6-digit PIN | 0.1% | 0.01–0.02% | ~0 |
| any standalone 5-digit number (FR postcode-like) | ~10% (these are **house numbers**) | 0.2–0.8% | 0.4–0.5% |
| house/plot number (component starting with a number) | S1 99.5%, S2/S3 87–90% | 61–66% | S1 99%, S2/S3 84–85% |

**Postcode agreement on matched pairs:** missing on both sides in 99.99% of pairs. Postcode is useless as a key.

**House-number agreement on matched pairs:**
| | agree | conflict | missing one side | missing both |
|---|---|---|---|---|
| US (S2/S3) | 75.5 / 76.8% | 8.0 / 8.0% | 16.0 / 14.7% | 0.6% |
| India (S2/S3) | 47.7 / 46.2% | 9.0 / 8.5% | 12.7 / 14.4% | 30.6 / 30.9% |

**Conflicts in true pairs** come from four kinds of noise:
- a dropped digit (`658` → `58`, `1601` → `601`)
- a lost leading digit (`1021` → `021`)
- ranges (`11047` vs `11077-11079`)
- injected prefixes (`H.NO 717 48`)

**Decoys** instead shift the number by a small amount (+7, +11, +21). A relation feature between the two numbers can separate them: *equal / equal after stripping zeros / one is a digit-subsequence of the other / small numeric shift / unrelated*.

---

## 8. Surprises

1. **There are no postcodes.** The brief expected ZIP/PIN; the data has essentially none.
2. **Strict one-to-one from the S2/S3 side.** Each S2/S3 record is matched to at most one S1, across all 7.64M pairs.
3. **Singletons are rare (5.6%).** The typical S1 entity has 3–4 matches spread over both sources, because S2 and S3 each contain internal duplicates.
4. **Decoy negatives** are near-duplicates of real entities with a shifted house number or an extra name token.
5. **23.5% of India S2 names and 13% of India S3 names are in a native script.** Name similarity on raw strings is near zero for them, and the address has to carry the match.
6. **Test has 25% more S2/S3 records per S1 than train,** in all three countries (§3). This is likely a precision risk.
7. The US vs India match distribution is **identical** to within 0.1 percentage points. The data is synthetic, and the generator is probably the same for France.
8. There are exact-duplicate S2/S3 rows under different ids: 0.3–0.5%.
9. Performance note: pandas 3 `str.contains` runs natively on Arrow strings, but `str.extract` falls back to Python `re` and is about 100× slower on 5M rows.

---

## 9. Implications

### Blocking
- **Block within country, using whatever label the record carries.** There are zero cross-country pairs. Group by the literal string, never a hard-coded `{US, India}`, so France and any other label work automatically.
- **Postcode blocking is not an option.** Use a **union** of cheap blockers:
  - character n-gram TF-IDF on the normalised name, top-k per S1 (sparse matrix product in chunks);
  - an address key of house-number core + first street token;
  - n-gram TF-IDF on the normalised address, for Indic-script names where the name blocker fails.
- **Budget:** true matches average 3.5 and reach 11 per S1. Candidate lists of about 20–50 per S1 across S2+S3 should give high recall. Measure recall on train, since the organisers will audit `candidate_pairs.tsv`.
- **Normalisation before blocking:**
  - casefold and fold accents;
  - strip junk prefixes, handles and domains;
  - map OCR digits (0→o, 1→l, 5→s, 6→g) inside words;
  - strip legal forms (US, India and **French** lists) and honorifics;
  - expand street abbreviations (US and French);
  - drop the state/region component.
- **Scripts:** transliterate Indic scripts to Latin (for example an MIT/Apache transliteration library, or a character map). Otherwise 13–24% of India S2/S3 names are unmatchable by name.

### Features
- **Name:** token_set, token_sort, Jaro-Winkler, and char-n-gram cosine on the normalised name; a legal-form compatibility flag; the **count and identity of unmatched extra tokens**, since decoys add words; a script flag.
- **Address:**
  - the **house-number relation** from §7 (the strongest decoy signal);
  - street-token similarity after abbreviation expansion;
  - city similarity;
  - state equality through a map (abbreviation ↔ full ↔ native script) **learned from train pairs** by co-occurrence. That uses no external data. For France, derive region↔département co-occurrence unsupervised from the test data itself, or ignore the state entirely.
- **Context and competition** (these exploit one-to-one):
  - the candidate's rank and score gap within its S1's list;
  - how strongly this S2/S3 record prefers another S1;
  - how many near-identical siblings it has in the same source.
- **Keep the model country-agnostic:** no country one-hot. France has no labels, so features must mean the same thing in every country. A `source` (S2/S3) indicator is fine because source styles are consistent across countries.

### Thresholding and assignment
- **Enforce one-to-one.** Assign each S2/S3 record to at most one S1 (its best-scoring S1 above threshold). This is free precision.
- **Tune the threshold on per-entity macro F0.5** using an S1-grouped validation split, not pair-level F1.
- **Partial recall still scores well:** 1 of 4 correct with no false positives gives F0.5 = 0.625, and any false positive on a singleton gives 0. Prefer high-precision thresholds.
- **Shade the threshold conservative relative to the validation optimum,** because of the test density shift (§3) and unseen France. One way to calibrate is to validate on train with extra negatives injected to mimic test density.
- **Singletons are only 5.6%.** Predicting empty is rarely right on its own merits; handle them by requiring the decoy checks (house-number and extra-token features) to pass rather than with a special classifier.
