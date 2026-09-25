"""Write tiny §4-conformant stub parquet files to cache/stub/ so downstream owners can start before real data exists.

Usage (from the repo root):  python code/business_entity_resolution/scripts/make_stub_data.py [--n-s1 40]

Writes, for split in {train, test}:
  records_{split}.parquet     §4 records contract (S1 + S2 + S3, all countries of that split)
  candidates_{split}.parquet  §4 candidates contract, 3-8 candidates per S1 (true matches, near-duplicate decoys,
                              fillers from the same country), incl. rank_in_cand and block_score
  gt_{split}.parquet          long ground truth (s1_id, match_id); train only
Train has US + India; test adds France (unseen country). Deterministic (seed 42). The normalised columns are produced
by a toy normaliser here, NOT by ber.normalize — they only need to have the right schema and plausible values.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.config import load_config  # noqa: E402

WORLD = {
    "US": dict(a=["Golden", "Prime", "Summit", "River", "Oak", "Liberty", "Apex", "Harbor", "Cedar", "Blue"],
               b=["Dental", "Cleaning", "Consulting", "Bakery", "Auto", "Clinic", "Logistics", "Realty"],
               legal=["LLC", "Inc", "Corp", "PC"], extra=["Holdings", "Center", "Services", "Group"],
               streets=["Main Street", "Oak Avenue", "Maple Drive", "Cedar Lane", "Lake Road"],
               cities=[("Austin", "TX", "Texas"), ("Raleigh", "NC", "North Carolina"), ("Columbus", "OH", "Ohio")],
               abbr={"Street": "ST", "Avenue": "AVE", "Drive": "DR", "Lane": "LN", "Road": "RD"}),
    "India": dict(a=["Shree", "Sai", "Global", "Royal", "Om", "Aditya", "Vijay", "Lotus", "Star", "Ganesh"],
                  b=["Infotech", "Traders", "Finance", "Exports", "Builders", "Foods", "Pharma", "Textiles"],
                  legal=["Private Limited", "Pvt Ltd", "LLP", "Limited"], extra=["Overseas", "Industries", "India", "Ventures"],
                  streets=["MG Road", "Station Road", "Nehru Nagar", "Sector 4", "Gandhi Marg"],
                  cities=[("Pune", "MH", "Maharashtra"), ("Delhi", "DL", "Delhi"), ("Bangalore", "KA", "Karnataka")],
                  abbr={"Road": "RD", "Nagar": "NGR", "Sector": "SEC", "Marg": "MARG"}),
    "France": dict(a=["Club", "Ecole", "Amicale", "Maison", "Comité", "Union", "Cercle", "Atelier", "Foyer", "Espace"],
                   b=["Sportive", "Musique", "Parents", "Santé", "Loisirs", "Culture", "Jeunesse", "Nature"],
                   legal=["SARL", "SAS", "EURL", "SCI"], extra=["Groupe", "France", "Participations", "Fils"],
                   streets=["Rue de la Paix", "Avenue Jean Jaurès", "Boulevard Victor Hugo", "Rue Nationale", "Place Carnot"],
                   cities=[("Bordeaux", "Gironde", "Nouvelle-Aquitaine"), ("Lille", "Nord", "Hauts-de-France"),
                           ("Nantes", "Loire-Atlantique", "Pays de la Loire")],
                   abbr={"Rue": "R", "Avenue": "AV", "Boulevard": "BD", "Place": "PL"}),
}
DEVANAGARI = {"Shree": "श्री", "Sai": "साई", "Global": "ग्लोबल", "Royal": "रॉयल", "Om": "ओम", "Aditya": "आदित्य",
              "Vijay": "विजय", "Lotus": "लोटस", "Star": "स्टार", "Ganesh": "गणेश", "Infotech": "इंफोटेक",
              "Traders": "ट्रेडर्स", "Finance": "फाइनेंस", "Exports": "एक्सपोर्ट्स", "Builders": "बिल्डर्स",
              "Foods": "फूड्स", "Pharma": "फार्मा", "Textiles": "टेक्सटाइल्स", "Private": "प्राइवेट", "Limited": "लिमिटेड"}
LEGAL_CANON = {"llc": "llc", "inc": "inc", "corp": "corp", "pc": "pc", "private limited": "private limited",
               "pvt ltd": "private limited", "llp": "llp", "limited": "limited", "ltd": "limited", "sarl": "sarl",
               "sas": "sas", "eurl": "eurl", "sci": "sci"}
EXPAND = {"st": "street", "ave": "avenue", "dr": "drive", "ln": "lane", "rd": "road", "ngr": "nagar", "sec": "sector",
          "r": "rue", "av": "avenue", "bd": "boulevard", "pl": "place"}


def fold(s: str) -> str:
    """Toy normaliser: strip accents, lowercase, keep alphanumerics, collapse spaces."""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def make_entity(rng, country, used):
    """Draw a new base business (unique name core) for ``country``."""
    w = WORLD[country]
    while True:
        core = f"{rng.choice(w['a'])} {rng.choice(w['a'])} {rng.choice(w['b'])}"
        if core not in used and len(set(core.split())) == 3:
            used.add(core)
            break
    city = w["cities"][rng.integers(len(w["cities"]))]
    return dict(country=country, core=core, legal=str(rng.choice(w["legal"])), house=int(rng.integers(1, 9999)),
                street=str(rng.choice(w["streets"])), city=city[0], region_short=city[1], region=city[2])


def render(e, style, rng):
    """Render an entity as (name_raw, addr_raw, transliterated name) in the style of source 1, 2 or 3."""
    w = WORLD[e["country"]]
    name, street, house, region = f"{e['core']} {e['legal']}", e["street"], str(e["house"]), e["region"]
    translit = name
    if style == 2:                                   # S2: upper case, abbreviated street, leading zeros, maybe native
        street = " ".join(w["abbr"].get(t, t) for t in street.split())
        house = house.zfill(5) if rng.random() < 0.3 else house
        name = name.upper() if rng.random() < 0.5 else e["core"]
        if e["country"] == "India" and rng.random() < 0.3:
            name = " ".join(DEVANAGARI.get(t, t) for t in f"{e['core']} Private Limited".split())
            translit = f"{e['core']} Private Limited"
        street = street.upper()
    elif style == 3:                                 # S3: short region, legal moved to the front, maybe no house number
        region = e["region_short"]
        name = f"{e['legal']} {e['core']}" if rng.random() < 0.4 else name
        house = "" if rng.random() < 0.15 else house
    addr = ", ".join(x for x in [f"{house} {street}".strip(), e["city"], region] if x)
    return name, addr, translit


def record_row(eid, source, e, name, addr, translit):
    """Build one §4 records row with toy-normalised columns."""
    script = "Devanagari" if re.search(r"[ऀ-ॿ]", name) else "Latin"
    norm = fold(translit if script != "Latin" else name)
    legal = next((LEGAL_CANON[k] for k in sorted(LEGAL_CANON, key=len, reverse=True) if re.search(rf"\b{k}\b", norm)), "")
    core = " ".join(t for t in norm.split() if t not in {"llc", "inc", "corp", "pc", "private", "pvt", "ltd", "limited",
                                                            "llp", "sarl", "sas", "eurl", "sci"})
    addr_norm = " ".join(EXPAND.get(t, t) for t in fold(addr).split())
    nums = [n.lstrip("0") or "0" for n in re.findall(r"\d+", addr)]
    return dict(entity_id=eid, source=source, country=e["country"], name_raw=name, addr_raw=addr, name_script=script,
                name_norm=norm, name_core=core, legal_form=legal, addr_norm=addr_norm,
                addr_street=" ".join(EXPAND.get(t, t) for t in fold(e["street"]).split()),
                house_num=nums[0] if nums else "", addr_nums=nums, name_tokens=norm.split())


def build_split(split, countries, n_s1, rng):
    """Generate records, candidates and GT for one split."""
    records, gt, owner = [], [], {}
    ids = iter(rng.choice(np.arange(10**6, 10**9), size=100_000, replace=False))
    new_id = lambda src: f"S{src}-{next(ids)}"  # noqa: E731
    for country in countries:
        used = set()
        for i in range(n_s1):
            e = make_entity(rng, country, used)
            s1 = new_id(1)
            records.append(record_row(s1, 1, e, *render(e, 1, rng)))
            singleton = i < max(1, n_s1 // 12)
            for _ in range(0 if singleton else int(rng.integers(1, 6))):     # true matches, 1-5
                src = int(rng.integers(2, 4))
                mid = new_id(src)
                records.append(record_row(mid, src, e, *render(e, src, rng)))
                gt.append((s1, mid))
            for _ in range(int(rng.integers(1, 3)) if singleton or rng.random() < 0.4 else 0):  # decoys
                d = dict(e)
                if rng.random() < 0.5:
                    d["house"] = e["house"] + int(rng.integers(3, 25))
                else:
                    d["core"] = f"{e['core']} {rng.choice(WORLD[country]['extra'])}"
                src = int(rng.integers(2, 4))
                records.append(record_row(new_id(src), src, d, *render(d, src, rng)))
            owner[s1] = country
        for _ in range(n_s1 // 2):                                           # unrelated fillers
            e = make_entity(rng, country, used)
            src = int(rng.integers(2, 4))
            records.append(record_row(new_id(src), src, e, *render(e, src, rng)))
    rec = pl.DataFrame(records, schema_overrides={"source": pl.Int8})
    gt_df = pl.DataFrame(gt, schema=["s1_id", "match_id"], orient="row")
    return rec, build_candidates(rec, gt_df, rng), gt_df


def build_candidates(rec, gt, rng):
    """Per S1: its true matches + the closest same-country records up to a random total in [3, 8]."""
    rows = []
    s1 = rec.filter(pl.col("source") == 1)
    pool = rec.filter(pl.col("source") > 1)
    true = {(a, b) for a, b in gt.iter_rows()}
    for r in s1.iter_rows(named=True):
        cands = pool.filter(pl.col("country") == r["country"])
        full = f"{r['name_norm']} {r['addr_norm']}"
        scored = [(c["entity_id"], fuzz.token_set_ratio(r["name_norm"], c["name_norm"]) / 100,
                   fuzz.token_set_ratio(full, f"{c['name_norm']} {c['addr_norm']}") / 100,
                   r["house_num"] != "" and r["house_num"] == c["house_num"]
                   and r["addr_street"].split()[:1] == c["addr_street"].split()[:1])
                  for c in cands.iter_rows(named=True)]
        must = [s for s in scored if (r["entity_id"], s[0]) in true]
        rest = sorted((s for s in scored if (r["entity_id"], s[0]) not in true), key=lambda s: -max(s[1], s[2]))
        n = max(int(rng.integers(3, 9)), len(must))
        for cid, tn, tf, key in must + rest[: max(0, min(8, n) - len(must))]:
            mask = (1 if tn >= 0.5 else 0) | 2 | (4 if key else 0)
            rows.append((r["entity_id"], cid, r["country"], mask, tn, tf, max(tn, tf)))
    cand = pl.DataFrame(rows, schema=["s1_id", "cand_id", "country", "block_mask", "tfidf_name", "tfidf_full",
                                      "block_score"], orient="row")
    return cand.with_columns(
        knn_rank=pl.col("block_score").rank("ordinal", descending=True).over("s1_id").cast(pl.Int32),
        rank_in_cand=pl.col("block_score").rank("ordinal", descending=True).over("cand_id").cast(pl.Int32),
        block_mask=pl.col("block_mask").cast(pl.Int32),
    ).select("s1_id", "cand_id", "country", "block_mask", "tfidf_name", "tfidf_full", "knn_rank", "rank_in_cand",
             "block_score").sort("s1_id", "knn_rank")


def main():
    """Generate and write the stub files, then print a short summary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-s1", type=int, default=40, help="S1 entities per country (default 40)")
    args = ap.parse_args()
    cfg = load_config()
    out = cfg.cache_dir / "stub"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    for split, countries in [("train", ["US", "India"]), ("test", ["US", "India", "France"])]:
        rec, cand, gt = build_split(split, countries, args.n_s1, rng)
        rec.write_parquet(out / f"records_{split}.parquet")
        cand.write_parquet(out / f"candidates_{split}.parquet")
        if split == "train":
            gt.write_parquet(out / f"gt_{split}.parquet")
        per_s1 = cand.group_by("s1_id").len()["len"]
        print(f"{split}: {rec.height} records ({rec['source'].value_counts().sort('source').rows()}), "
              f"{cand.height} candidate pairs, {per_s1.min()}-{per_s1.max()} per S1, {gt.height} GT pairs, "
              f"countries={sorted(rec['country'].unique())}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
