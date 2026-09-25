"""Decoy-killers (build first): house-number relation, token-set difference, street similarity
without number/city/region, legal form, script, lengths, missingness.

Contract:
In: candidate pairs joined with normalized records. Out: feature columns keyed by (s1_id, cand_id).

Owner: Nitish (R3 Features / LightGBM)
"""
from __future__ import annotations

import polars as pl

from ber.features.rarity import idf_stats, tokens

HN_CLASSES = ["both_missing", "one_missing", "equal", "same_base_diff_suffix", "transposed", "digit_dropped",
              "small_shift", "different"]            # checked in this order; the feature is the index
HN_CONFLICT = ("small_shift", "different")           # the decoy pattern: both present, genuinely different numbers
LEGAL_RELS = ["equal", "one_missing", "both_missing", "conflict"]
MAX_DIGITS = 10  # ponytail: transposition / dropped-digit checks look at the first 10 base digits only
SMALL_SHIFT = 50


def _strip0(x: pl.Expr) -> pl.Expr:
    """Strip leading zeros from a digit string, keeping a lone ``0``."""
    x = x.str.strip_chars_start("0")
    return pl.when(x == "").then(pl.lit("0")).otherwise(x)


def _hn_parts(x: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Split a house number into (cleaned full string, base = leading digits w/o zeros, suffix = rest, alnum only).

    ``"007"`` -> ("007", "7", ""), ``"45-A"`` -> ("45-a", "45", "a"), ``"12/3"`` -> ("12/3", "12", "3").
    The base is null when the string does not start with a digit.
    """
    x = x.fill_null("").str.strip_chars().str.to_lowercase()
    return x, _strip0(x.str.extract(r"^(\d+)")), x.str.replace(r"^\d+", "").str.replace_all(r"[^0-9a-z]", "")


def _swap(s: pl.Expr, i: int) -> pl.Expr:
    """``s`` with the characters at positions ``i`` and ``i+1`` swapped (unchanged when out of range)."""
    return pl.concat_str(s.str.slice(0, i), s.str.slice(i + 1, 1), s.str.slice(i, 1), s.str.slice(i + 2))


def _drop(s: pl.Expr, i: int) -> pl.Expr:
    """``s`` without the character at position ``i``, leading zeros stripped."""
    return _strip0(pl.concat_str(s.str.slice(0, i), s.str.slice(i + 1)))


def house_num_class(a: str | pl.Expr, b: str | pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Fine house-number relation (one of ``HN_CLASSES``, first match wins) + abs_diff and rel_diff of the bases.

    Base = leading digits (leading zeros stripped), suffix = the rest (case-folded, alphanumerics only).
    both_missing > one_missing > equal (base and suffix) > same_base_diff_suffix > transposed (one adjacent swap of
    the base digits) > digit_dropped (one base = the other minus one digit) > small_shift (|diff| <= 50) > different.
    abs_diff / rel_diff (= abs_diff / max base) are null when either base is missing.
    """
    a, b = (pl.col(x) if isinstance(x, str) else x for x in (a, b))
    (fa, ba, sa), (fb, bb, sb) = _hn_parts(a), _hn_parts(b)
    na, nb = ba.cast(pl.Int64, strict=False), bb.cast(pl.Int64, strict=False)
    diff = (na - nb).abs()
    longer = pl.when(ba.str.len_chars() > bb.str.len_chars())
    long_, short = longer.then(ba).otherwise(bb), longer.then(bb).otherwise(ba)
    transposed = (ba.str.len_chars() == bb.str.len_chars()) & pl.any_horizontal(
        [_swap(ba, i) == bb for i in range(MAX_DIGITS - 1)])
    dropped = pl.any_horizontal([_drop(long_, i) == short for i in range(MAX_DIGITS)])
    rel = (pl.when((fa == "") & (fb == "")).then(pl.lit("both_missing"))
           .when((fa == "") | (fb == "")).then(pl.lit("one_missing"))
           .when((fa == fb) | ((ba == bb) & (sa == sb))).then(pl.lit("equal"))
           .when(ba == bb).then(pl.lit("same_base_diff_suffix"))
           .when(transposed).then(pl.lit("transposed"))
           .when(dropped).then(pl.lit("digit_dropped"))
           .when(diff <= SMALL_SHIFT).then(pl.lit("small_shift"))
           .otherwise(pl.lit("different")))
    rel_diff = pl.when(pl.max_horizontal(na, nb) > 0).then(diff / pl.max_horizontal(na, nb))
    return rel, diff, rel_diff.cast(pl.Float32)


def legal_relation(a: str, b: str) -> pl.Expr:
    """Legal-form relation (one of ``LEGAL_RELS``) of two canonical ``legal_form`` columns."""
    a, b = pl.col(a).fill_null(""), pl.col(b).fill_null("")
    return (pl.when((a == "") & (b == "")).then(pl.lit("both_missing"))
            .when((a == "") | (b == "")).then(pl.lit("one_missing"))
            .when(a == b).then(pl.lit("equal")).otherwise(pl.lit("conflict")))


def token_diff(df: pl.DataFrame, a: str, b: str, prefix: str, idf: pl.DataFrame) -> pl.DataFrame:
    """Token-set difference of string columns ``a`` vs ``b`` (needs ``country`` for the IDF lookup).

    Columns: ``{prefix}_only_a/_only_b`` (tokens on one side only), ``{prefix}_extra_idf_a/_b`` (IDF sum of those
    tokens) and ``{prefix}_one_extra`` (1 iff exactly one token differs in total and at least one token is shared).
    """
    t = df.select("country", only_a=tokens(a).list.set_difference(tokens(b)),
                  only_b=tokens(b).list.set_difference(tokens(a)),
                  n_both=tokens(a).list.set_intersection(tokens(b)).list.len())
    na, nb = t["only_a"].list.len(), t["only_b"].list.len()
    return pl.DataFrame({
        f"{prefix}_only_a": na.cast(pl.Int16), f"{prefix}_only_b": nb.cast(pl.Int16),
        f"{prefix}_extra_idf_a": idf_stats(t, "only_a", idf, "x")["x_sum"],
        f"{prefix}_extra_idf_b": idf_stats(t, "only_b", idf, "x")["x_sum"],
        f"{prefix}_one_extra": ((na + nb == 1) & (t["n_both"] > 0)).cast(pl.Int8),
    })


def structured_features(df: pl.DataFrame, name_idf: pl.DataFrame, street_idf: pl.DataFrame) -> pl.DataFrame:
    """Add the decoy-killer columns to a joined pair frame (``a_*`` = S1, ``b_*`` = candidate, plus ``country``).

    House-number class (code into ``HN_CLASSES``) + abs/rel diff + conflict flag, name_core and addr_street
    token-set differences, legal-form relation (code into ``LEGAL_RELS``) and street missingness.
    """
    rel, diff, rel_diff = house_num_class("a_house_num", "b_house_num")
    code = lambda e, vocab: e.replace_strict(vocab, list(range(len(vocab))), return_dtype=pl.Int8)  # noqa: E731
    return pl.concat([df, token_diff(df, "a_name_core", "b_name_core", "name", name_idf),
                      token_diff(df, "a_addr_street", "b_addr_street", "street", street_idf)], how="horizontal") \
        .with_columns(hn_rel=code(rel, HN_CLASSES), hn_abs_diff_log=diff.cast(pl.Float32).log1p(), hn_rel_diff=rel_diff,
                      hn_conflict=rel.is_in(HN_CONFLICT).cast(pl.Int8),
                      legal_rel=code(legal_relation("a_legal_form", "b_legal_form"), LEGAL_RELS),
                      street_missing=((pl.col("a_addr_street").fill_null("") == "").cast(pl.Int8)
                                      + (pl.col("b_addr_street").fill_null("") == "").cast(pl.Int8)))


def house_num_relation(a: str | pl.Expr, b: str | pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """House-number relation of two ``house_num`` columns and the absolute numeric difference.

    Relation: ``equal`` / ``one_missing`` / ``both_missing`` / ``different`` (after stripping leading zeros).
    The difference uses the first digit run of each side; null when either side has no digits.
    """
    a, b = (pl.col(x) if isinstance(x, str) else x for x in (a, b))
    a, b = a.fill_null("").str.strip_chars_start("0"), b.fill_null("").str.strip_chars_start("0")
    rel = (pl.when((a == "") & (b == "")).then(pl.lit("both_missing"))
           .when((a == "") | (b == "")).then(pl.lit("one_missing"))
           .when(a == b).then(pl.lit("equal")).otherwise(pl.lit("different")))
    num = lambda x: x.str.extract(r"(\d+)").cast(pl.Int64, strict=False)  # noqa: E731
    return rel, (num(a) - num(b)).abs()


def legal_conflict(a: str, b: str) -> pl.Expr:
    """1.0 when both sides carry a legal form and they differ, else 0.0."""
    return ((pl.col(a) != "") & (pl.col(b) != "") & (pl.col(a) != pl.col(b))).cast(pl.Float32)
