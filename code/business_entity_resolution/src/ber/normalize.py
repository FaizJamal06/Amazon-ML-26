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

import math
import re
import unicodedata
from collections import Counter
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
    non_latin = {k: v for k, v in counts.items() if k != "Latin" and k != "Other"}
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
_NUM_INSIDE_RE = re.compile(r"\d[\d/\-a-zA-Z]*\d|\d+")

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
    for i, m in enumerate(_NUM_INSIDE_RE.finditer(text)):
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
    # Sort by length descending to match the longest form first
    for form in sorted(legal_table, key=len, reverse=True):
        # Use word boundary matching
        pattern = re.compile(r"(?:^|\s)" + re.escape(form) + r"(?:\s|$)")
        m = pattern.search(text)
        if m:
            canonical = legal_table[form]
            start = m.start()
            end = m.end()
            # Prefer matches closer to the end (suffix position)
            if best_form == "" or end >= best_end:
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
    name_abbrev_table = rules.name_abbrevs(country)
    tokens = _expand_tokens(tokens, name_abbrev_table)

    name_norm = " ".join(tokens)

    # Extract legal form
    legal_table = rules.legal_forms(country)
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
    r"(?:(?:no\.?|n°|nº|#|h\.?no\.?|plot|door|flat|shop|bldg\.?|building)\s*)?"
    r"(\d[\d/\-]*[a-zA-Z]?)",
    re.IGNORECASE,
)

# Leading number pattern (address starts with a number)
_LEADING_NUM_RE = re.compile(r"^(\d[\d/\-]*[a-zA-Z]?)\b")

# House-number prefix pattern (explicit marker + number)
_HOUSE_PREFIX_RE = re.compile(
    r"\b(?:plot|no\.?|n°|nº|#|h\.?no\.?|door|flat|shop|bldg\.?|building)\s*"
    r"(\d[\d/\-]*[a-zA-Z]?)\b",
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
    for m in re.finditer(r"\d[\d/\-]*[a-zA-Z]?", addr_norm):
        num = m.group().lstrip("0") or "0"
        all_nums.append(num)

    non_house = rules.non_house_markers(country)

    # Strategy 1: Look for explicit house-number prefix markers
    house_num = ""
    for m in _HOUSE_PREFIX_RE.finditer(addr_norm):
        num = m.group(1).lstrip("0") or "0"
        # Check that the marker is actually a house-number marker, not a non-house marker
        prefix_text = addr_norm[:m.start()].lower().split()
        house_num = num
        break

    # Strategy 2: Leading number (address starts with a number)
    if not house_num:
        m = _LEADING_NUM_RE.match(addr_norm)
        if m:
            num = m.group(1).lstrip("0") or "0"
            # Check that the number is not preceded by a non-house marker
            house_num = num

    # Validate: if the "house number" is actually preceded by a non-house marker, clear it
    if house_num:
        # Find the position of this number in the original text and check preceding token
        lower = addr_norm.lower()
        idx = lower.find(house_num.lower())
        if idx > 0:
            before = lower[:idx].strip().split()
            if before and before[-1] in non_house:
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
    addr_table = rules.addr_abbrevs(country)
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

def compute_area_tokens(records: pl.DataFrame, threshold_frac: float = 0.15) -> dict[str, set[str]]:
    """Compute area tokens per country: tokens whose document frequency exceeds a threshold.

    These are typically city/state/region names that don't help street-level matching.
    Computed from the data itself — no hardcoded city/state/département list.

    Args:
        records: DataFrame with 'country', 'addr_norm' columns.
        threshold_frac: fraction of records within a country where a token must appear
                        to be considered an "area token".

    Returns:
        dict mapping country → set of area tokens.
    """
    area_tokens: dict[str, set[str]] = {}

    for country in records["country"].unique().to_list():
        country_recs = records.filter(pl.col("country") == country)
        n_recs = country_recs.height

        # Explode address tokens and count document frequency
        addr_tokens = (
            country_recs
            .select("entity_id", "addr_norm")
            .filter(pl.col("addr_norm") != "")
            .with_columns(
                pl.col("addr_norm").str.split(" ").alias("tokens")
            )
            .explode("tokens")
            .filter(pl.col("tokens").str.len_chars() > 1)
            # Exclude numbers
            .filter(~pl.col("tokens").str.contains(r"^\d"))
        )

        if addr_tokens.height == 0:
            area_tokens[country] = set()
            continue

        token_freq = (
            addr_tokens
            .group_by("tokens")
            .agg(doc_count=pl.col("entity_id").n_unique())
            .with_columns(frac=pl.col("doc_count") / n_recs)
            .filter(pl.col("frac") > threshold_frac)
        )

        area_tokens[country] = set(token_freq["tokens"].to_list())

    return area_tokens


def remove_area_tokens(addr_street: str, area_set: set[str]) -> str:
    """Remove area tokens from addr_street to get a cleaner street representation."""
    tokens = addr_street.split()
    filtered = [t for t in tokens if t not in area_set]
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


def build_records(cfg: Config, split: str, subworld: bool = False) -> None:
    """Build records_{split}.parquet from source1/2/3 parquet files.

    Reads the ingested source parquets, applies normalization, computes area tokens,
    removes area tokens from addr_street, and writes the result.
    """
    print(f"[normalize] building records_{split}...")

    # Read source parquets
    frames = []
    for src_num in (1, 2, 3):
        path = cfg.artifact(f"source{src_num}", split, subworld)
        if path.exists():
            df = pl.read_parquet(path)
            frames.append(df)
        else:
            print(f"  warning: {path} not found, skipping")

    if not frames:
        raise FileNotFoundError(f"No source parquets found for split={split}")

    raw = pl.concat(frames)
    print(f"  {raw.height:,} raw records across {len(frames)} sources")

    # Rename columns to match our internal contract
    col_map = {"business_name": "name_raw", "business_address": "addr_raw"}
    for old, new in col_map.items():
        if old in raw.columns:
            raw = raw.rename({old: new})

    # Ensure source is Int8
    if "source" in raw.columns:
        raw = raw.with_columns(pl.col("source").cast(pl.Int8))

    # Process in chunks for memory efficiency
    chunk_size = 100_000
    n_chunks = math.ceil(raw.height / chunk_size)
    processed_chunks = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, raw.height)
        chunk = raw.slice(start, end - start)
        processed = _process_chunk(chunk)
        processed_chunks.append(processed)
        if (i + 1) % 10 == 0 or i == n_chunks - 1:
            print(f"  normalized {end:,} / {raw.height:,} records")

    records = pl.concat(processed_chunks)

    # Compute area tokens from the data itself (per country)
    print("  computing area tokens (data-driven)...")
    area_tokens = compute_area_tokens(records, threshold_frac=0.15)
    for country, tokens in area_tokens.items():
        if tokens:
            print(f"    {country}: {len(tokens)} area tokens (top 10: {sorted(tokens)[:10]})")

    # Remove area tokens from addr_street
    def _remove_area(addr_street: str, country: str) -> str:
        """Remove area tokens for a specific country."""
        area_set = area_tokens.get(country, set())
        if not area_set:
            return addr_street
        return remove_area_tokens(addr_street, area_set)

    # Apply area token removal row-by-row (polars struct + map)
    addr_street_cleaned = []
    for row in records.select("addr_street", "country").iter_rows():
        addr_street_cleaned.append(_remove_area(row[0], row[1]))
    records = records.with_columns(pl.Series("addr_street", addr_street_cleaned))

    # Show examples
    _show_examples(records, area_tokens)

    # Write output
    out_path = cfg.artifact("records", split, subworld)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records.write_parquet(out_path)
    print(f"  wrote {records.height:,} records to {out_path}")


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

