"""TSV -> parquet cache; TSV writers for the submission files. The ONLY module that reads TSVs.

Contract:
In: student_resource/dataset/{train,test}/*.tsv read with sep='\\t', dtype=str, keep_default_na=False.
Out: parquet caches under cache/; output/matching_results.tsv and output/candidate_pairs.tsv
(tab-separated, no quoting, no index, no BOM, comma-joined ids without spaces).

Owner: Faiz (R1 Lead / Eval / Decision)
"""
