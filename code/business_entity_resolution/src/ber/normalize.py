"""Per-record name/address normalization and field extraction (script detection, transliteration,
accent folding, abbreviation expansion, legal-form and house-number extraction).

Contract:
In: raw records from the parquet cache.
Out: records_{split}.parquet with columns entity_id, source (1|2|3), country, name_raw, addr_raw,
name_script, name_norm, name_core, legal_form, addr_norm, addr_street, house_num, addr_nums (list[str]),
name_tokens (list[str]).

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
