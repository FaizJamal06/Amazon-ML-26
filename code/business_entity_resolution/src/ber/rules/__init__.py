"""Abbreviation and legal-suffix dictionaries per language/locale.

Contract:
Pure data consumed by normalize.py. Country-specific rules are allowed; the pipeline must
still run for any unseen country label (generic.py is the fallback).

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
from __future__ import annotations

from ber.rules import fr, generic, in_en, us_en

# ── Merged rule lookup by country ────────────────────────────────────────────

_COUNTRY_MODULES: dict[str, object] = {
    "US":     us_en,
    "India":  in_en,
    "France": fr,
}


def _merge_dicts(*dicts: dict) -> dict:
    """Merge multiple dicts left-to-right (later keys win)."""
    out: dict = {}
    for d in dicts:
        out.update(d)
    return out


def legal_forms(country: str) -> dict[str, str]:
    """Return the merged legal-form dict for *country* (generic + country-specific).

    For unseen countries the generic fallback is returned unchanged.
    """
    mod = _COUNTRY_MODULES.get(country)
    extra = getattr(mod, "LEGAL_FORMS", {}) if mod else {}
    return _merge_dicts(generic.LEGAL_FORMS, extra)


def name_abbrevs(country: str) -> dict[str, str]:
    """Name-token abbreviation expansion table for *country*.

    Currently only the generic table is used; country modules may add entries.
    """
    return dict(generic.NAME_ABBREVS)


def addr_abbrevs(country: str) -> dict[str, str]:
    """Address-token abbreviation expansion table for *country* (generic + country-specific)."""
    mod = _COUNTRY_MODULES.get(country)
    extra = getattr(mod, "ADDR_ABBREVS", {}) if mod else {}
    return _merge_dicts(generic.ADDR_ABBREVS, extra)


def house_num_prefixes(country: str) -> set[str]:
    """Tokens that signal the following number is a house/building number."""
    base = {"plot", "no", "no.", "#", "door", "flat", "shop", "building", "bldg", "bldg."}
    mod = _COUNTRY_MODULES.get(country)
    extra = getattr(mod, "HOUSE_NUM_PREFIXES", set()) if mod else set()
    return base | extra


def non_house_markers(country: str) -> set[str]:
    """Tokens after which a number is NOT a house number (sector, block, phase, etc.).

    Applied universally — these are country-generic structural location markers.
    """
    base = {
        "sector", "block", "phase", "ward", "stage", "lane", "cross", "main",
        "zone", "pocket", "part", "wing", "tower", "floor", "fl",
    }
    mod = _COUNTRY_MODULES.get(country)
    extra = getattr(mod, "NON_HOUSE_MARKERS", set()) if mod else set()
    return base | extra


def strip_prefixes() -> set[str]:
    """Name prefixes/honorifics to strip (generic, country-independent)."""
    return set(generic.STRIP_PREFIXES)
