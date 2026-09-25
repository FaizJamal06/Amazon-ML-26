"""Union of blocking passes with a provenance bitmask; caps candidates per S1 at max_cands.

Contract:
In: outputs of keys / tfidf_knn / embed_knn.
Out: candidates_{split}.parquet with s1_id, cand_id, country, block_mask, tfidf_name, tfidf_full, knn_rank.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
