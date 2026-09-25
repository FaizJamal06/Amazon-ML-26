# Submissions

Commit before every upload (CLAUDE.md §8). Limit: 5 per day. The TSVs are not in git; record their sha256
(`sha256sum output/*.tsv` or `certutil -hashfile <file> SHA256`) and where they are stored (Drive/S3 path).

| # | date-time IST | commit | change | OOF F0.5 | public LB | sha256 matching_results.tsv | sha256 candidate_pairs.tsv | stored at |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-25 23:40 IST | 3b48f86 (+ script in this commit) | format probe: all-empty (expected public LB ≈ share of singletons, train 5.6%) | n/a | TODO | 7fd4b11ceda1ccd61de32995b030cb72ac7d6521f5a5f98ecf71f5de45075bfe | b3c13e3f524d8aa43af8196c09214428dc7bb9e4681dc50f499030966b279872 | local output/ (regenerable with scripts/make_empty_submission.py) |
