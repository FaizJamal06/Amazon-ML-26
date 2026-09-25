"""Per-record name/address normalization and field extraction (script detection, transliteration,
accent folding, abbreviation expansion, legal-form and house-number extraction).

Contract:
In: raw records from the parquet cache.
Out: records_{split}.parquet with columns entity_id, source (1|2|3), country, name_raw, addr_raw,
name_script, name_norm, name_core, legal_form, addr_norm, addr_street, house_num, addr_nums (list[str]),
name_tokens (list[str]), name_skeleton.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
from __future__ import annotations

import functools
import multiprocessing as mp
import os
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Sequence

import polars as pl
from anyascii import anyascii

from ber import rules
from ber.config import Config

# ═══════════════════════════════════════════════════════════════════════════════
# §1  UNICODE SCRIPT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

# Unicode block ranges for common Indic scripts + Latin/CJK etc.
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0000, 0x024F, "Latin"),       # Basic Latin through Latin Extended-B
    (0x1E00, 0x1EFF, "Latin"),       # Latin Extended Additional
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
]


def _char_script(ch: str) -> str:
    """Return the Unicode script name for a single character."""
    cp = ord(ch)
    for lo, hi, name in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    return "Other"


def detect_script(text: str) -> str:
    """Return the dominant Unicode script of *text* (ignoring digits, punctuation, whitespace).

    Returns 'Latin' if all alphabetic characters are Latin, or the most common
    non-Latin script if any non-Latin alphabetic characters are present.
    """
    counts: Counter[str] = Counter()
    for ch in text:
        if ch.isalpha():
            counts[_char_script(ch)] += 1
    if not counts:
        return "Latin"
    # If any non-Latin script is present, pick the most common one
    non_latin = {k: v for k, v in counts.items() if k != "Latin"}
    if non_latin:
        return max(non_latin, key=non_latin.get)  # type: ignore[arg-type]
    return "Latin"


# ═══════════════════════════════════════════════════════════════════════════════
# §2  TEXT NORMALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fold_accents(text: str) -> str:
    """Strip combining diacritical marks (NFD decomposition → filter combining → NFC)."""
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", stripped)


def transliterate(text: str) -> str:
    """Transliterate any non-Latin string to Latin with anyascii; also folds accents."""
    return fold_accents(anyascii(text))


# Regex: numbers with possible internal slashes/dashes (e.g. 12/3, 45-a, 6800)
_NUM_INSIDE_RE = re.compile(r"\d+-[a-zA-Z]\b|\d[\d/\-a-zA-Z]*\d|\d+")

# Strip punctuation EXCEPT inside numbers like 12/3
_PUNCT_RE = re.compile(r"[^\w\s]")

# Collapse whitespace
_WS_RE = re.compile(r"\s+")

# Junk prefix pattern (leading non-alphanumeric garbage)
_JUNK_PREFIX_RE = re.compile(r'^[\-<>"\'`~!@#$%^*+={}[\]|\\:;,.?/]+\s*')


def _normalize_base(text: str) -> str:
    """Lowercase, fold accents, &→and, strip junk prefix, collapse whitespace.

    Punctuation is stripped except inside numbers like 12/3.
    """
    text = text.strip()
    # Strip junk prefix characters
    text = _JUNK_PREFIX_RE.sub("", text)
    # &→and
    text = text.replace("&", " and ")
    # Fold accents, lowercase
    text = fold_accents(text).lower()
    # Protect numbers with internal punctuation (12/3, 45-a)
    protected: dict[str, str] = {}
    matches = list(_NUM_INSIDE_RE.finditer(text))
    for i, m in reversed(list(enumerate(matches))):
        placeholder = f" __NUM{i}__ "
        protected[placeholder.strip()] = m.group()
        text = text[:m.start()] + placeholder + text[m.end():]
    # Strip remaining punctuation
    text = _PUNCT_RE.sub(" ", text)
    # Restore protected numbers
    for placeholder, original in protected.items():
        text = text.replace(placeholder, original)
    # Collapse whitespace
    text = _WS_RE.sub(" ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# §3  NAME SKELETON (for blocking / dedup of transliteration variants)
# ═══════════════════════════════════════════════════════════════════════════════

# Phonetic mappings for skeleton: sh/s, v/w, ph/f
_SKELETON_MAP: dict[str, str] = {
    "sh": "s", "v": "w", "ph": "f", "ck": "k", "ee": "i", "oo": "u",
    "th": "t", "aa": "a", "gh": "g", "bh": "b", "dh": "d", "kh": "k",
    "chh": "c", "ch": "c",
}
_VOWELS = set("aeiou")


def name_skeleton(name_norm: str) -> str:
    """Build a skeleton key robust to transliteration variants.

    Steps:
    1. Apply phonetic mappings (sh→s, v→w, ph→f, etc.)
    2. Drop vowels after the first letter of each word
    3. Collapse repeated letters
    """
    tokens = name_norm.split()
    result_tokens = []
    for token in tokens:
        # Apply phonetic mappings (longest match first)
        s = token
        for pattern in ("chh",):
            s = s.replace(pattern, _SKELETON_MAP[pattern])
        for pattern, repl in _SKELETON_MAP.items():
            if len(pattern) <= 2:
                s = s.replace(pattern, repl)
        # Drop vowels after the first letter
        if len(s) > 1:
            s = s[0] + "".join(c for c in s[1:] if c not in _VOWELS)
        # Collapse repeated letters
        collapsed = []
        for c in s:
            if not collapsed or c != collapsed[-1]:
                collapsed.append(c)
        result_tokens.append("".join(collapsed))
    return " ".join(result_tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  NAME NORMALIZATION (legal form extraction, abbreviation expansion)
# ═══════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=None)
def _norm_table(kind: str, country: str) -> dict[str, str]:
    """``{_normalize_base(form): canonical}`` for ``rules.<kind>(country)``, built once per (kind, country).

    The rule tables depend only on the country; rebuilding them for every record was ~100 _normalize_base calls
    per record. Callers must not mutate the returned dict.
    """
    return {_normalize_base(form): canonical for form, canonical in getattr(rules, kind)(country).items()}


@functools.lru_cache(maxsize=64)
def _legal_patterns(items: tuple[tuple[str, str], ...]) -> list[tuple[re.Pattern, str]]:
    """(word-boundary pattern, canonical) per legal form, longest form first (stable), compiled once per table."""
    ordered = sorted(items, key=lambda kv: len(kv[0]), reverse=True)
    return [(re.compile(r"(?:^|\s)" + re.escape(form) + r"(?:\s|$)"), canonical) for form, canonical in ordered]


def _expand_tokens(tokens: list[str], table: dict[str, str]) -> list[str]:
    """Expand abbreviations in a list of tokens using the given table."""
    return [table.get(t, t) for t in tokens]


def _extract_legal_form(name_tokens: list[str], legal_table: dict[str, str]) -> tuple[list[str], str]:
    """Remove the legal-form suffix/prefix from name tokens and return (remaining_tokens, canonical_form).

    Matches the longest legal form first. Scans from the end of the token list first,
    then from the beginning (to handle reordered legal suffixes).
    """
    text = " ".join(name_tokens)
    best_form = ""
    best_start = -1
    best_end = -1
    # Longest form first, word-boundary patterns (compiled once per table)
    for pattern, canonical in _legal_patterns(tuple(legal_table.items())):
        m = pattern.search(text)
        if m:
            start = m.start()
            end = m.end()
            # Prefer matches closer to the end (suffix position), and longest match on tie
            if best_form == "" or end > best_end or (end == best_end and start < best_start):
                best_form = canonical
                best_start = start
                best_end = end
    if best_form:
        # Remove the matched legal form from text
        text_before = text[:best_start].strip()
        text_after = text[best_end:].strip()
        remaining = (text_before + " " + text_after).strip()
        core_tokens = remaining.split() if remaining else []
        return core_tokens, best_form
    return list(name_tokens), ""


def normalize_name(raw_name: str, country: str) -> dict[str, str | list[str]]:
    """Normalize a raw business name and extract structured fields.

    Returns a dict with: name_script, name_norm, name_core, legal_form, name_tokens, name_skeleton.
    """
    script = detect_script(raw_name)

    # Transliterate non-Latin names
    if script != "Latin":
        transliterated = transliterate(raw_name)
    else:
        transliterated = raw_name

    # Base normalization: lowercase, accent-fold, &→and, strip junk, collapse whitespace
    norm = _normalize_base(transliterated)

    # Expand name abbreviations
    tokens = norm.split()

    # Strip name prefixes/honorifics
    prefixes = rules.strip_prefixes()
    while tokens and tokens[0] in prefixes:
        tokens = tokens[1:]

    # Expand abbreviations
    name_abbrev_table = _norm_table("name_abbrevs", country)
    tokens = _expand_tokens(tokens, name_abbrev_table)

    name_norm = " ".join(tokens)

    # Extract legal form
    legal_table = _norm_table("legal_forms", country)
    core_tokens, legal_form = _extract_legal_form(tokens, legal_table)
    name_core = " ".join(core_tokens)

    # Build skeleton
    skeleton = name_skeleton(name_norm)

    return {
        "name_script": script,
        "name_norm": name_norm,
        "name_core": name_core,
        "legal_form": legal_form,
        "name_tokens": tokens,
        "name_skeleton": skeleton,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §5  ADDRESS NORMALIZATION (house number extraction, street extraction)
# ═══════════════════════════════════════════════════════════════════════════════

# Regex to find numbers in addresses, including forms like 12/3, 45-a, #23, 00174
_ADDR_NUM_RE = re.compile(
    r"(?:(?:h no|hno|no|n|door no|d no|plot no|shop no|plot|door|flat|shop|bldg|building)\s*)?"
    r"(\d[\d/\-]*[a-zA-Z]?)",
    re.IGNORECASE,
)

# Leading number pattern (address starts with a number)
_LEADING_NUM_RE = re.compile(r"^(\d[\d/\-]*(?:\s*(?:bis|ter)\b|[a-zA-Z]?))\b")

# House-number prefix pattern (explicit marker + number)
_HOUSE_PREFIX_RE = re.compile(
    r"\b(?:h no|hno|no|n|door no|d no|plot no|shop no|plot|door|flat|shop|bldg|building)\s*"
    r"(\d[\d/\-]*(?:\s*(?:bis|ter)\b|[a-zA-Z]?))\b",
    re.IGNORECASE,
)


def _extract_house_num(addr_norm: str, country: str) -> tuple[str, list[str]]:
    """Extract the primary house number and all numbers from a normalized address.

    BUG FIX: numbers following sector/block/phase/ward/stage/lane/cross/main are NOT house numbers.
    Only take numbers after explicit house-number markers (plot/no./door/flat/shop/building/#)
    or leading numbers.

    Returns (house_num, addr_nums) where house_num may be '' if uncertain.
    """
    # Find all numbers in the address
    all_nums: list[str] = []
    # Match numbers, optionally followed by space and bis/ter, or just trailing letters
    for m in re.finditer(r"\d[\d/\-]*(?:\s*(?:bis|ter)\b|[a-zA-Z]?)?", addr_norm):
        num = m.group().lstrip("0") or "0"
        num = re.sub(r"\s+", "", num)
        all_nums.append(num)

    non_house = rules.non_house_markers(country)

    # Strategy 1: Look for explicit house-number prefix markers
    house_num = ""
    house_start = -1
    for m in _HOUSE_PREFIX_RE.finditer(addr_norm):
        num = m.group(1).lstrip("0") or "0"
        num = re.sub(r"\s+", "", num)
        house_num = num
        house_start = m.start()
        break

    # Strategy 2: Leading number (address starts with a number)
    if not house_num:
        m = _LEADING_NUM_RE.match(addr_norm)
        if m:
            num = m.group(1).lstrip("0") or "0"
            num = re.sub(r"\s+", "", num)
            house_num = num
            house_start = m.start()

    # Validate: if the "house number" is actually preceded by a non-house marker, clear it
    if house_num and house_start > 0:
        before = addr_norm[:house_start].strip().split()
        if before and before[-1].lower() in non_house:
            house_num = ""

    return house_num, all_nums


def normalize_address(raw_addr: str, country: str) -> dict[str, str | list[str]]:
    """Normalize a raw business address and extract structured fields.

    Returns a dict with: addr_norm, addr_street, house_num, addr_nums.
    """
    if not raw_addr or raw_addr.strip().lower() in ("", "null", "n/a", "na", "none"):
        return {
            "addr_norm": "",
            "addr_street": "",
            "house_num": "",
            "addr_nums": [],
        }

    # Base normalization (lowercase, accent-fold, strip punct except in numbers)
    norm = _normalize_base(raw_addr)

    # Expand address abbreviations
    addr_table = _norm_table("addr_abbrevs", country)
    tokens = norm.split()
    tokens = _expand_tokens(tokens, addr_table)
    addr_norm = " ".join(tokens)

    # Extract house number
    house_num, addr_nums = _extract_house_num(addr_norm, country)

    # addr_street = addr_norm minus house_num (done later with area token removal)
    # For now, addr_street = addr_norm with house_num token removed
    street_tokens = list(tokens)
    if house_num:
        # Remove the first occurrence of the house number token
        for i, t in enumerate(street_tokens):
            cleaned = t.lstrip("0") or "0"
            if cleaned == house_num or t == house_num:
                street_tokens.pop(i)
                break

    addr_street = " ".join(street_tokens)

    return {
        "addr_norm": addr_norm,
        "addr_street": addr_street,
        "house_num": house_num,
        "addr_nums": addr_nums,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §6  AREA TOKEN REMOVAL (data-driven, not hardcoded city/state lists)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_area_tokens(records: pl.LazyFrame | pl.DataFrame, threshold_frac: float = 0.15) -> dict[str, set[str]]:
    """Compute area tokens per country: tokens whose document frequency exceeds a threshold.

    These are typically city/state/region names that don't help street-level matching.
    Computed from the data itself — no hardcoded city/state/département list.

    Args:
        records: LazyFrame or DataFrame with 'entity_id', 'country', 'addr_norm' columns.
        threshold_frac: fraction of records within a country where a token must appear
                        to be considered an "area token".

    Returns:
        dict mapping country → set of area tokens.
    """
    if isinstance(records, pl.DataFrame):
        records = records.lazy()

    # Get total records per country
    n_recs = records.group_by("country").len().collect()
    n_recs_dict = dict(zip(n_recs["country"], n_recs["len"]))

    # Explode address tokens and count document frequency
    addr_tokens = (
        records
        .select("entity_id", "country", "addr_norm")
        .filter(pl.col("addr_norm") != "")
        .with_columns(pl.col("addr_norm").str.split(" ").alias("tokens"))
        .explode("tokens")
        .filter(pl.col("tokens").str.len_chars() > 1)
        # Exclude numbers
        .filter(~pl.col("tokens").str.contains(r"^\d"))
    )

    token_freq = (
        addr_tokens
        .group_by("country", "tokens")
        .agg(doc_count=pl.col("entity_id").n_unique())
        .collect()
    )

    area_tokens: dict[str, set[str]] = {}
    for country, total in n_recs_dict.items():
        if total == 0:
            area_tokens[country] = set()
            continue
        
        country_tokens = token_freq.filter(pl.col("country") == country)
        valid = country_tokens.filter((pl.col("doc_count") / total) > threshold_frac)
        area_tokens[country] = set(valid["tokens"].to_list())

    return area_tokens


def remove_area_tokens(addr_street: str, area_set: set[str], min_tokens: int = 1) -> str:
    """Remove area tokens from addr_street to get a cleaner street representation.

    If filtering out area tokens leaves fewer than *min_tokens*, fall back to the
    original addr_street so we don't leave addr_street too sparse or empty.
    """
    tokens = addr_street.split()
    filtered = [t for t in tokens if t not in area_set]
    if len(filtered) < min_tokens:
        return addr_street
    return " ".join(filtered)


# ═══════════════════════════════════════════════════════════════════════════════
# §7  VECTORIZED POLARS PIPELINE (build_records)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_row(name_raw: str, addr_raw: str, country: str) -> dict:
    """Normalize a single record (called via map_elements on a struct column)."""
    name_fields = normalize_name(name_raw, country)
    addr_fields = normalize_address(addr_raw, country)
    return {**name_fields, **addr_fields}


def _process_chunk(chunk: pl.DataFrame) -> pl.DataFrame:
    """Normalize a chunk of records.

    Applies per-row normalization using Python UDFs (necessary for regex / anyascii).
    The outer loop uses polars chunking for memory efficiency.
    """
    # We need to apply row-level normalization. Use map_elements on a struct.
    result_rows = []
    for row in chunk.iter_rows(named=True):
        fields = _normalize_row(row["name_raw"], row["addr_raw"], row["country"])
        result_rows.append({
            "entity_id": row["entity_id"],
            "source": row["source"],
            "country": row["country"],
            "name_raw": row["name_raw"],
            "addr_raw": row["addr_raw"],
            **fields,
        })

    schema = {
        "entity_id": pl.String,
        "source": pl.Int8,
        "country": pl.String,
        "name_raw": pl.String,
        "addr_raw": pl.String,
        "name_script": pl.String,
        "name_norm": pl.String,
        "name_core": pl.String,
        "legal_form": pl.String,
        "name_tokens": pl.List(pl.String),
        "name_skeleton": pl.String,
        "addr_norm": pl.String,
        "addr_street": pl.String,
        "house_num": pl.String,
        "addr_nums": pl.List(pl.String),
    }
    return pl.DataFrame(result_rows, schema=schema)


def _normalize_chunk_file(src_path: str, start: int, length: int, out_path: str) -> int:
    """Worker (top-level, spawn-safe): normalize rows [start, start+length) of one source parquet -> ``out_path``.

    Returns the number of records written.
    """
    raw = pl.scan_parquet(src_path).slice(start, length).collect()
    raw = raw.rename({k: v for k, v in {"business_name": "name_raw", "business_address": "addr_raw"}.items()
                      if k in raw.columns})
    if "source" in raw.columns:
        raw = raw.with_columns(pl.col("source").cast(pl.Int8))
    _process_chunk(raw).write_parquet(out_path)
    return raw.height


def remove_area_tokens_expr(area_tokens: dict[str, set[str]]) -> pl.Expr:
    """Vectorized ``remove_area_tokens`` over the ``addr_street`` column, per ``country``.

    Same result as the row version: countries without area tokens keep addr_street unchanged; otherwise the
    whitespace tokens not in the country's area set are joined with single spaces, and if none remain the original
    addr_street is kept.
    """
    out = pl.col("addr_street")
    for country, area in area_tokens.items():  # countries, not rows
        if not area:
            continue
        kept = pl.col("addr_street").str.extract_all(r"\S+").list.eval(
            pl.element().filter(~pl.element().is_in(sorted(area))))
        cleaned = pl.when(kept.list.len() >= 1).then(kept.list.join(" ")).otherwise(pl.col("addr_street"))
        out = pl.when(pl.col("country") == country).then(cleaned).otherwise(out)
    return out


def build_records(cfg: Config, split: str, subworld: bool = False) -> None:
    """Build records_{split}.parquet from source1/2/3 parquet files.

    Reads the ingested source parquets, applies normalization, computes area tokens,
    removes area tokens from addr_street, and writes the result.
    """
    print(f"[normalize] building records_{split}...")

    paths = []
    for src_num in (1, 2, 3):
        path = cfg.artifact(f"source{src_num}", split, subworld)
        if path.exists():
            paths.append(path)
        else:
            print(f"  warning: {path} not found, skipping")

    if not paths:
        out_path = cfg.artifact("records", split, subworld)
        if out_path.exists():
            print(f"  warning: no source parquets found, but {out_path} exists; keeping existing")
            return
        print(f"  warning: no source parquets found for split={split}, skipping")
        return

    tmp_dir = cfg.cache_dir / f"tmp_normalize_{split}{'_sw' if subworld else ''}"
    shutil.rmtree(tmp_dir, ignore_errors=True)  # stale chunks from an earlier run would be globbed into the output
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # One task per slice of each source parquet (<= 100k rows, and >= ~4 tasks per worker so small inputs still use
    # every core); each worker reads its slice and writes its own chunk file. Output does not depend on the slicing.
    workers = int(cfg.get("normalize.workers", 0)) or max(1, (os.cpu_count() or 2) - 1)
    n_rows = {p: pl.scan_parquet(p).select(pl.len()).collect().item() for p in paths}
    chunk_size = max(5_000, min(100_000, -(-sum(n_rows.values()) // (4 * workers))))
    tasks = []
    for path in paths:
        for start in range(0, n_rows[path], chunk_size):
            tasks.append((str(path), start, chunk_size, str(tmp_dir / f"chunk_{len(tasks):05d}.parquet")))
    workers = min(workers, len(tasks))
    t0 = time.time()
    if workers <= 1:
        counts = [_normalize_chunk_file(*t) for t in tasks]
    else:
        # spawn: safe on Windows and avoids fork + polars thread pools; workers import this module afresh
        with mp.get_context("spawn").Pool(workers) as pool:
            counts = pool.starmap(_normalize_chunk_file, tasks)
    chunk_files = [Path(t[3]) for t in tasks]
    print(f"  normalized {sum(counts):,} records across {len(paths)} sources "
          f"({len(tasks)} chunks, {workers} worker(s), {time.time() - t0:.1f}s)")

    # Compute area tokens from the data itself (per country) using lazy scan
    print("  computing area tokens (data-driven)...")
    lazy_records = pl.scan_parquet(chunk_files)
    area_tokens = compute_area_tokens(lazy_records, threshold_frac=0.15)

    for country, tokens in area_tokens.items():
        if tokens:
            print(f"    {country}: {len(tokens)} area tokens (top 10: {sorted(tokens)[:10]})")

    bis_ter_count = lazy_records.filter(
        (pl.col("country") == "France") & (pl.col("addr_norm").str.contains(r"\b(bis|ter)\b"))
    ).select(pl.len()).collect().item()
    if bis_ter_count > 0:
        print(f"  French records with bis/ter: {bis_ter_count:,}")

    # Remove area tokens from addr_street (vectorized) while sinking the chunks into the output
    out_path = cfg.artifact("records", split, subworld)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lazy_records.with_columns(addr_street=remove_area_tokens_expr(area_tokens)).sink_parquet(out_path)

    # Get total count
    total_records = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  wrote {total_records:,} records to {out_path}")

    # Show examples
    example_records = []
    countries = pl.scan_parquet(out_path).select("country").unique().collect()["country"].to_list()
    for country in countries:
        country_recs = pl.scan_parquet(out_path).filter(pl.col("country") == country).head(10).collect()
        example_records.append(country_recs)
    
    _show_examples(pl.concat(example_records), area_tokens)


def _show_examples(records: pl.DataFrame, area_tokens: dict[str, set[str]]) -> None:
    """Show 10 before/after normalization examples per country including native-script and French rows."""
    for country in sorted(records["country"].unique().to_list()):
        print(f"\n  -- Examples: {country} --")
        subset = records.filter(pl.col("country") == country)

        # Try to include native-script examples
        native = subset.filter(pl.col("name_script") != "Latin")
        latin = subset.filter(pl.col("name_script") == "Latin")

        examples = pl.concat([native.head(3), latin.head(7)]) if native.height > 0 else subset.head(10)

        for row in examples.iter_rows(named=True):
            script_tag = row["name_script"][:3] if row["name_script"] else "Lat"
            name_raw = row.get("name_raw", "")[:60]
            name_norm = row.get("name_norm", "")[:50]
            name_core = row.get("name_core", "")[:40]
            legal = row.get("legal_form", "")[:20]
            skeleton = row.get("name_skeleton", "")[:30]
            addr_raw = row.get("addr_raw", "")[:60]
            house = row.get("house_num", "")[:8]
            street = row.get("addr_street", "")[:40]
            try:
                print(f"    [{script_tag}] name_raw={name_raw}")
                print(f"         norm={name_norm}  core={name_core}  legal={legal}  skel={skeleton}")
                print(f"         addr_raw={addr_raw}  house={house}  street={street}")
            except UnicodeEncodeError:
                # Fallback for Windows consoles that can't handle some chars
                print(f"    [{script_tag}] (unicode display error, skipping)")

