"""India rules: legal forms (Pvt, Ltd, LLP...), address markers (H.No, Plot No, Near/Opp, C/O), state codes.

Contract:
Out: dicts/sets consumed by normalize.py.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""

# ── India-specific legal forms ───────────────────────────────────────────────
LEGAL_FORMS: dict[str, str] = {
    # Devanagari transliterated legal suffixes (anyascii output)
    "praaivet limited":    "private limited",
    "praaivet limitedd":   "private limited",
    "pra. li.":            "private limited",
    "elaelapii":           "llp",
    # Common abbreviations
    "pvt":                 "private",
    "pvt.":                "private",
}

# ── India address abbreviations ──────────────────────────────────────────────
ADDR_ABBREVS: dict[str, str] = {
    "rd":     "road",
    "rd.":    "road",
    "st":     "street",
    "st.":    "street",
    "nr":     "near",
    "nr.":    "near",
    "opp":    "opposite",
    "opp.":   "opposite",
    "c/o":    "",
    "h.no":   "",
    "h.no.":  "",
    "hno":    "",
    "dist":   "district",
    "dist.":  "district",
    "nagar":  "nagar",
    "ngr":    "nagar",
    "sec":    "sector",
}

# ── India address markers that precede a house number ────────────────────────
HOUSE_NUM_PREFIXES: set[str] = {
    "plot", "no", "no.", "#", "h.no", "h.no.", "hno", "door", "flat",
    "shop", "building", "bldg", "bldg.",
}

# ── Tokens that follow a structured location, NOT a house number ─────────────
# When these tokens precede a number, the number is NOT a house number.
NON_HOUSE_MARKERS: set[str] = {
    "sector", "block", "phase", "ward", "stage", "lane", "cross", "main",
    "zone", "pocket", "part", "wing", "tower", "floor", "fl",
}
