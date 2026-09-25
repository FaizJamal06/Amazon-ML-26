"""France rules: legal forms SARL/SAS/SASU/SA/EURL/SCI/SNC incl. dotted variants, street types
r/r./av/bd/pl/ch/imp/all/rte/fbg/qu, N° prefixes.

Contract:
Out: dicts/sets consumed by normalize.py. Region/departement stop-list is built from test France
records themselves, not hard-coded (CLAUDE.md §5.5c).

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""

# ── French legal forms (key=lowered text → value=canonical) ──────────────────
LEGAL_FORMS: dict[str, str] = {
    "societe a responsabilite limitee": "sarl",
    "société à responsabilité limitée": "sarl",
    "sarl":           "sarl",
    "s.a.r.l.":       "sarl",
    "s.a.r.l":        "sarl",
    "sarl.":          "sarl",
    "sàrl":           "sarl",

    "societe par actions simplifiee":   "sas",
    "société par actions simplifiée":   "sas",
    "sas":            "sas",
    "s.a.s.":         "sas",
    "s.a.s":          "sas",

    "sasu":           "sasu",
    "s.a.s.u.":       "sasu",
    "s.a.s.u":        "sasu",

    "societe anonyme": "sa",
    "société anonyme": "sa",
    "sa":             "sa",
    "s.a.":           "sa",
    "s.a":            "sa",

    "eurl":           "eurl",
    "e.u.r.l.":       "eurl",
    "e.u.r.l":        "eurl",

    "sci":            "sci",
    "s.c.i.":         "sci",
    "s.c.i":          "sci",

    "snc":            "snc",
    "s.n.c.":         "snc",
    "s.n.c":          "snc",

    "ei":             "ei",
    "eirl":           "eirl",

    "etablissements": "ets",
    "établissements": "ets",
    "ets":            "ets",
    "ets.":           "ets",

    "cie":            "cie",
    "cie.":           "cie",

    "association":    "association",
    "assoc":          "association",
    "assoc.":         "association",

    "groupe":         "groupe",
    "holding":        "holding",
    "participations": "participations",
    "fils":           "fils",
    "developpement":  "developpement",
    "développement":  "developpement",
}

# ── French street-type abbreviation expansion ────────────────────────────────
ADDR_ABBREVS: dict[str, str] = {
    "r":      "rue",
    "r.":     "rue",
    "av":     "avenue",
    "av.":    "avenue",
    "bd":     "boulevard",
    "bd.":    "boulevard",
    "pl":     "place",
    "pl.":    "place",
    "ch":     "chemin",
    "ch.":    "chemin",
    "imp":    "impasse",
    "imp.":   "impasse",
    "all":    "allee",
    "all.":   "allee",
    "rte":    "route",
    "rte.":   "route",
    "fbg":    "faubourg",
    "fbg.":   "faubourg",
    "qu":     "quai",
    "qu.":    "quai",
    "q.":     "quai",
    "q":      "quai",
    "crs":    "cours",
    "crs.":   "cours",
    "espl":   "esplanade",
    "espl.":  "esplanade",
    "allée":  "allee",
    "allees": "allee",

    # Written-out forms (accent-folded canonical)
    "rue":        "rue",
    "avenue":     "avenue",
    "boulevard":  "boulevard",
    "place":      "place",
    "chemin":     "chemin",
    "impasse":    "impasse",
    "allee":      "allee",
    "route":      "route",
    "faubourg":   "faubourg",
    "quai":       "quai",
    "cours":      "cours",
    "esplanade":  "esplanade",
}

# ── French house-number prefixes ─────────────────────────────────────────────
HOUSE_NUM_PREFIXES: set[str] = {
    "no", "n°", "nº", "#",
}

# ── bis / ter suffix (French house-number modifiers) ─────────────────────────
HOUSE_NUM_SUFFIXES: set[str] = {
    "bis", "ter", "quater",
}
