"""US English rules: street-type abbreviations, legal forms (LLC, Inc, Corp, PC, PLLC, LP...), state codes.

Contract:
Out: dicts/sets consumed by normalize.py.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""

# ── US-specific legal forms (supplements generic.py) ─────────────────────────
LEGAL_FORMS: dict[str, str] = {
    "professional corporation": "pc",
    "p.c.":          "pc",
    "professional limited liability company": "pllc",
    "pllc":          "pllc",
    "doing business as": "",   # strip DBA marker
    "d/b/a":         "",
    "dba":           "",
    "trading as":    "",
    "t/a":           "",
    "f/k/a":         "",       # formerly known as
    "a/k/a":         "",       # also known as
}

# ── US street-type abbreviation expansion ────────────────────────────────────
ADDR_ABBREVS: dict[str, str] = {
    "st":     "street",
    "st.":    "street",
    "ave":    "avenue",
    "ave.":   "avenue",
    "av":     "avenue",
    "av.":    "avenue",
    "blvd":   "boulevard",
    "blvd.":  "boulevard",
    "dr":     "drive",
    "dr.":    "drive",
    "ct":     "court",
    "ct.":    "court",
    "ln":     "lane",
    "ln.":    "lane",
    "rd":     "road",
    "rd.":    "road",
    "pl":     "place",
    "pl.":    "place",
    "cir":    "circle",
    "cir.":   "circle",
    "pkwy":   "parkway",
    "pkwy.":  "parkway",
    "hwy":    "highway",
    "hwy.":   "highway",
    "trl":    "trail",
    "trl.":   "trail",
    "ter":    "terrace",
    "ter.":   "terrace",
    "cv":     "cove",
    "cv.":    "cove",
    "sq":     "square",
    "sq.":    "square",
    "aly":    "alley",
    "aly.":   "alley",
    "apt":    "apartment",
    "apt.":   "apartment",
    "ste":    "suite",
    "ste.":   "suite",
    "fl":     "floor",
    "fl.":    "floor",
    "bldg":   "building",
    "bldg.":  "building",
    "n":      "north",
    "n.":     "north",
    "s":      "south",
    "s.":     "south",
    "e":      "east",
    "e.":     "east",
    "w":      "west",
    "w.":     "west",
    "ne":     "northeast",
    "nw":     "northwest",
    "se":     "southeast",
    "sw":     "southwest",
}

# ── US address markers that precede a house number ───────────────────────────
HOUSE_NUM_PREFIXES: set[str] = {
    "plot", "no", "no.", "#", "door", "flat", "shop", "building", "bldg", "bldg.",
    "unit", "ste", "ste.", "apt", "apt.", "suite",
}
