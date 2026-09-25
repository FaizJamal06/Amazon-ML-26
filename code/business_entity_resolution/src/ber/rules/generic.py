"""Language-agnostic fallback rules applied to every country (incl. unseen labels).

Contract:
Out: dicts/sets consumed by normalize.py.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""

# ── Legal-form canonicalization ──────────────────────────────────────────────
# key = lowercased form found in text  →  value = canonical label.
# Order of matching matters: longer forms are matched first by normalize.py.
LEGAL_FORMS: dict[str, str] = {
    # English (generic)
    "private limited":  "private limited",
    "pvt limited":      "private limited",
    "pvt. limited":     "private limited",
    "pvt ltd":          "private limited",
    "pvt. ltd.":        "private limited",
    "pvt. ltd":         "private limited",
    "pvt ltd.":         "private limited",
    "p ltd":            "private limited",
    "limited":          "limited",
    "ltd":              "limited",
    "ltd.":             "limited",
    "incorporated":     "incorporated",
    "inc":              "incorporated",
    "inc.":             "incorporated",
    "corporation":      "corporation",
    "corp":             "corporation",
    "corp.":            "corporation",
    "company":          "company",
    "co":               "company",
    "co.":              "company",
    "llc":              "llc",
    "l.l.c.":           "llc",
    "l.l.c":            "llc",
    "llp":              "llp",
    "l.l.p.":           "llp",
    "l.l.p":            "llp",
    "lp":               "lp",
    "l.p.":             "lp",
    "pc":               "pc",
    "p.c.":             "pc",
    "pllc":             "pllc",
}

# ── Name abbreviation expansion ──────────────────────────────────────────────
# Applied after lowercasing, before legal-form stripping.
NAME_ABBREVS: dict[str, str] = {
    "pvt":   "private",
    "pvt.":  "private",
    "ltd":   "limited",
    "ltd.":  "limited",
    "corp":  "corporation",
    "corp.": "corporation",
    "inc":   "incorporated",
    "inc.":  "incorporated",
    "co":    "company",
    "co.":   "company",
    "intl":  "international",
    "intl.": "international",
    "natl":  "national",
    "natl.": "national",
    "assn":  "association",
    "assn.": "association",
    "dept":  "department",
    "dept.": "department",
    "mfg":   "manufacturing",
    "mfg.":  "manufacturing",
    "svcs":  "services",
    "svc":   "service",
    "grp":   "group",
    "grp.":  "group",
    "hldgs": "holdings",
    "mgmt":  "management",
    "mgmt.": "management",
}

# ── Address abbreviation expansion ───────────────────────────────────────────
ADDR_ABBREVS: dict[str, str] = {
    "&":     "and",
}

# ── Tokens to strip from names (prefixes / junk) ────────────────────────────
STRIP_PREFIXES: set[str] = {
    "the", "shri", "sri", "smt", "dr", "mr", "mrs", "ms", "m/s",
}

# ── Junk characters to strip from the start/end of names ────────────────────
JUNK_CHARS: str = "-<>\"'`~!@#$%^*+={}[]|\\:;,.?/"

# ── Tokens after which a number is NOT a house number ───────────────────────
NON_HOUSE_MARKERS: set[str] = {
    "sector", "block", "phase", "ward", "stage", "lane", "cross", "main",
    "zone", "pocket", "part", "wing", "tower", "floor", "fl",
    "apt", "apartment", "suite", "ste", "unit", "flat",
}
