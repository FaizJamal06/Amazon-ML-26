"""Decoy-killers (build first): house-number relation, token-set difference, street similarity
without number/city/region, legal form, script, lengths, missingness.

Contract:
In: candidate pairs joined with normalized records. Out: feature columns keyed by (s1_id, cand_id).

Owner: Nitish (R3 Features / LightGBM)
"""
