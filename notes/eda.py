"""EDA for Amazon ML Challenge 2026 — Business Entity Resolution. Exploration only: no models, no submissions.

Re-run from ml26/:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python notes/eda.py > notes/eda_output.txt

Loads one S2/S3 file at a time (each is ~5M rows) so it fits in 16 GB RAM. Full run took ~2 h on a 16 GB laptop (mostly the brute-force NN scan; memory-bound). Smoke test: EDA_NROWS=300000 (numbers then meaningless).
Regexes avoid look-arounds so pandas can run them on pyarrow strings (RE2).
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

DATA = Path(__file__).resolve().parents[1] / "student_resource" / "dataset"
SEED = 0
NROWS = int(os.environ["EDA_NROWS"]) if "EDA_NROWS" in os.environ else None  # smoke-test knob
N_PAIR_S1 = 50_000 if NROWS is None else 2_000  # S1 entities whose matched pairs we pull for pair-level stats
N_NN = 60            # singleton / matched S1 queries for the brute-force nearest-neighbour check
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 50)

T0 = time.time()


def h(title):
    print(f"\n{'=' * 110}\n## {title}   [t={time.time() - T0:.0f}s]\n{'=' * 110}", flush=True)


def load(split, name):
    return pd.read_csv(DATA / split / f"{split}_{name}.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=NROWS)


# ---------------------------------------------------------------- patterns
PLACEHOLDERS = {"null", "n/a", "na", "none", "-", "nan", "--"}
INDIC = r"[ऀ-෿]"                         # Devanagari .. Sinhala blocks
ACCENT = r"[À-ÖØ-öø-ÿŒœ]"
JUNK_PREFIX = r"^[^\w\"'(ऀ-෿]"            # name starts with punctuation like >> -- << @ *
HANDLE = r"(?i)(^@|\.(com|in|net|org|fr|co)\b|^www\.)"
NEAR = r"(?i)\b(near|nr|opp|opposite|behind|beside|next to|in front of)\b"
ZIP_US = r"(?:^|,|\b[A-Z]{2})\s*(\d{5})(?:-\d{4})?\s*(?:,|$)"   # a 5-digit component, or after 'NC'
PIN6 = r"(?:^|[^\d/-])([1-9]\d{5})(?:$|[^\d/-])"                 # contiguous 6 digits
FIVE = r"(?:^|[^\d/-])(\d{5})(?:$|[^\d/-])"                      # any standalone 5-digit number
# first integer of the first comma-component that starts with a (house/plot/door) number; leading zeros dropped
HOUSE = (r"(?i)(?:^|,)\s*(?:(?:h\.?\s*no|kh\.?\s*no|plot\s*no|flat\s*no|door\s*no|shop\s*no|no|#)\.?\s*[-:#.]*\s*)?"
         r"#*\s*(?:[a-z]{1,2}\s?-\s?)?0*(\d+)(?:[a-z]?(?:[\s,/-]|$))")
PLACEHOLDER_TOKEN = r"(?i)(?:^|,)\s*(null|n/a|none|nan)\s*(?:,|$)"


def postcode(addr, country):
    """Country-specific postcode guess; unknown countries fall back to a standalone 5-digit number."""
    zip_, pin, five = (addr.str.extract(p, expand=False) for p in (ZIP_US, PIN6, FIVE))
    return zip_.where(country == "US", pin.where(country == "India", five))


def prep(s):
    """lowercase, strip punctuation, sort tokens -> fuzz.ratio == token_sort_ratio, but precomputed once."""
    return " ".join(sorted(utils.default_process(s).split()))


# ---------------------------------------------------------------- profiling
SUMMARY = []


def profile(df, label, train_countries=None):
    h(f"PROFILE {label}")
    print(f"rows={len(df):,}  unique ids={df.entity_id.nunique():,}  prefixes={df.entity_id.str[:3].value_counts().to_dict()}")
    print("country counts:", df.country.value_counts().to_dict(), " raw values:", [repr(c) for c in df.country.unique()])
    dup = df.duplicated(["business_name", "business_address", "country"]).sum()
    print(f"exact duplicate (name,address,country) rows under different ids: {dup:,} ({dup / len(df):.2%})")

    name, addr = df.business_name, df.business_address
    f = pd.DataFrame({
        "country": df.country,
        "name_len": name.str.len(), "addr_len": addr.str.len(),
        "name_words": name.str.count(r"\S+"), "addr_parts": addr.str.count(",") + (addr.str.strip() != ""),
        "name_empty": name.str.strip() == "", "addr_empty": addr.str.strip() == "",
        "name_placeholder": name.str.strip().str.lower().isin(PLACEHOLDERS),
        "addr_placeholder_tok": addr.str.contains(PLACEHOLDER_TOKEN),
        "name_indic": name.str.contains(INDIC), "addr_indic": addr.str.contains(INDIC),
        "name_accent": name.str.contains(ACCENT), "addr_accent": addr.str.contains(ACCENT),
        "name_upper": (name == name.str.upper()) & name.str.contains(r"[A-Za-z]"),
        "addr_upper": (addr == addr.str.upper()) & addr.str.contains(r"[A-Za-z]"),
        "name_junk_prefix": name.str.contains(JUNK_PREFIX), "name_handle": name.str.contains(HANDLE),
        "addr_near": addr.str.contains(NEAR),
        "addr_zip_us": addr.str.contains(ZIP_US), "addr_pin6": addr.str.contains(PIN6),
        "addr_5digit": addr.str.contains(FIVE), "addr_house_no": addr.str.contains(HOUSE),
        "addr_any_digit": addr.str.contains(r"\d"),
    })
    flags = [c for c in f.columns if f[c].dtype == bool]
    print("\nmissing / empty rate per column (overall):")
    for col in ["entity_id", "business_name", "business_address", "country"]:
        print(f"  {col:17s} empty={(df[col].str.strip() == '').mean():.3%}")
    print("\nrates per country (fraction of rows):")
    print(f.groupby("country")[flags].mean().T.round(4).to_string())
    print("\nlength distributions per country (chars / words / comma-parts):")
    print(f.groupby("country")[["name_len", "addr_len", "name_words", "addr_parts"]]
          .describe(percentiles=[.05, .25, .5, .75, .95]).T.round(1).to_string())

    for c, g in df.groupby("country"):
        print(f"\n-- 10 random rows, country={c}")
        print(g.sample(min(10, len(g)), random_state=SEED).to_string(index=False))
        last_tok = g.business_name.str.lower().str.extract(r"([\w.&]+)\W*$", expand=False)
        print(f"\n   top name last-tokens ({c}):", last_tok.value_counts().head(20).to_dict())
        last_part = g.business_address.str.extract(r"([^,]*)$", expand=False).str.strip()
        print(f"   top address last-components ({c}):", last_part.value_counts().head(20).to_dict())
        first_part = g.business_address.str.extract(r"^([^,]*)", expand=False).str.strip()
        print(f"   first address component is a region (== some last-component in top-60):",
              round(first_part.isin(set(last_part.value_counts().head(60).index)).mean(), 4))

    for c, g in f.groupby("country"):
        SUMMARY.append({"file": label, "country": c, "rows": len(g),
                        "name_len_med": g.name_len.median(), "addr_len_med": g.addr_len.median(),
                        **{k: round(g[k].mean(), 4) for k in
                           ["addr_empty", "name_indic", "addr_indic", "name_accent", "name_upper",
                            "addr_upper", "name_junk_prefix", "addr_house_no", "addr_zip_us", "addr_pin6", "addr_5digit"]}})

    if train_countries is not None:
        for c in sorted(set(df.country) - set(train_countries)):
            new_country_detail(df[df.country == c], label, c)


FR_LEGAL = {"SARL": r"s\.?a\.?r\.?l", "SAS": r"s\.?a\.?s", "SASU": r"s\.?a\.?s\.?u", "SA": r"s\.?a",
            "EURL": r"e\.?u\.?r\.?l", "SCI": r"s\.?c\.?i", "SNC": r"s\.?n\.?c", "SCOP": r"scop",
            "SELARL": r"selarl", "SCM": r"scm", "GIE": r"gie", "EI/EIRL": r"eirl|ei", "Association": r"association|asso",
            "Groupe": r"groupe", "(France)": r"\(france\)", "Cie/Compagnie": r"cie|compagnie", "Ets/Établissements": r"ets|établissements|etablissements"}
FR_STREET = {"rue": r"rue", "r": r"r", "avenue": r"avenue", "av/ave": r"av|ave", "boulevard": r"boulevard",
             "bd/bld/blvd": r"bd|bld|blvd", "place": r"place", "pl": r"pl", "route": r"route", "rte": r"rte",
             "chemin": r"chemin", "ch/chem": r"ch|chem", "allée": r"allée|allee", "all": r"all", "impasse": r"impasse",
             "imp": r"imp", "quai": r"quai", "cours": r"cours", "crs": r"crs", "faubourg/fbg": r"faubourg|fbg|fg",
             "square/sq": r"square|sq", "bis/ter": r"bis|ter", "lieu-dit": r"lieu-dit|ld", "ZI/ZA/ZAC": r"zi|za|zac",
             "cedex": r"cedex", "BP": r"bp"}


def new_country_detail(g, label, c):
    h(f"NEW (not in train) COUNTRY DETAIL: {c} in {label}  rows={len(g):,}")
    print(g.sample(min(20, len(g)), random_state=SEED).to_string(index=False))
    for title, d, col in [("legal-form tokens in NAME", FR_LEGAL, g.business_name),
                          ("street-type tokens in ADDRESS", FR_STREET, g.business_address)]:
        rates = {k: round(col.str.contains(rf"(?i)(?:^|[\s,(]){p}\.?(?:$|[\s,)])").mean(), 4) for k, p in d.items()}
        print(f"\n{title}:", dict(sorted(rates.items(), key=lambda kv: -kv[1])))
    print(f"\naccented chars: name={g.business_name.str.contains(ACCENT).mean():.3f} "
          f"addr={g.business_address.str.contains(ACCENT).mean():.3f}  "
          f"| ALL-CAPS 'RUE': {g.business_address.str.contains(r'RUE').mean():.3f}")
    five = g.business_address.str.extract(FIVE, expand=False)
    print(f"standalone 5-digit (postcode-like) rate: {five.notna().mean():.4f}; examples:",
          g.business_address[five.notna()].head(8).tolist())
    print("top address last-components:", g.business_address.str.extract(r"([^,]*)$", expand=False).str.strip().value_counts().head(25).to_dict())
    parts = g.business_address.str.split(",")
    print("top 2nd-to-last components (city?):", parts.str[-2].str.strip().value_counts().head(25).to_dict())


# ---------------------------------------------------------------- nearest-neighbour hardness check
def nn_scan(queries, df, src):
    """For each query S1 (same country only) brute-force the best token_sort_ratio(name+address) in df."""
    out = []
    for c, q in queries.groupby("country"):
        g = df[df.country == c]
        ids = g.entity_id.to_numpy()
        choices = [prep(n + " " + a) for n, a in zip(g.business_name.tolist(), g.business_address.tolist())]
        qs = [prep(n + " " + a) for n, a in zip(q.business_name, q.business_address)]
        for i in range(0, len(qs), 20):
            sc = process.cdist(qs[i:i + 20], choices, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
            for j, row in enumerate(sc):
                r = q.iloc[i + j]
                truth = r.truth
                is_true = np.isin(ids, list(truth)) if truth else np.zeros(len(ids), bool)
                neg = np.where(is_true, 0, row)
                bi = int(neg.argmax())
                best_true = int(row[is_true].max()) if is_true.any() else None
                out.append({"s1": r.entity_id, "src": src, "country": c, "n_true_here": int(is_true.sum()),
                            "best_neg_score": int(neg[bi]), "best_neg_id": ids[bi],
                            "best_true_score": best_true,
                            "negs_scoring_>=_best_true": int((neg >= best_true).sum()) if best_true is not None else None})
    return pd.DataFrame(out)


# ================================================================ TRAIN
h("TRAIN: source1 + ground truth")
s1 = load("train", "source1")
TRAIN_COUNTRIES = sorted(s1.country.unique())
profile(s1, "train_source1")

gt = load("train", "ground_truth")
print("\nground truth rows:", f"{len(gt):,}", "| columns:", list(gt.columns))
raw = gt.matched_entity_ids
gt["n"] = np.where(raw.str.strip() == "", 0, raw.str.count(",") + 1)
print("rows with whitespace around ids / empty list items:",
      int(raw.str.contains(r"\s").sum()), int(raw.str.contains(r"^,|,,|,$").sum()))
print("duplicate source1_entity_id rows in GT:", int(gt.source1_entity_id.duplicated().sum()))
s1_ids, gt_ids = set(s1.entity_id), set(gt.source1_entity_id)
print(f"GT S1 ids not in train_source1: {len(gt_ids - s1_ids)} | train_source1 ids not in GT: {len(s1_ids - gt_ids)}")

pairs = (gt.loc[gt.n > 0, ["source1_entity_id", "matched_entity_ids"]]
         .assign(mid=lambda d: d.matched_entity_ids.str.split(",")).explode("mid")[["source1_entity_id", "mid"]])
pairs["mid"] = pairs.mid.astype(str).str.strip()
pairs["src"] = pairs.mid.str[:2]
print(f"total matched pairs: {len(pairs):,} | by prefix: {pairs.src.value_counts().to_dict()}")
print("duplicate id inside one S1's list:", int(pairs.duplicated(["source1_entity_id", "mid"]).sum()))
gt = gt.merge(s1[["entity_id", "country"]], left_on="source1_entity_id", right_on="entity_id", how="left")
gt = gt.drop(columns=["entity_id", "matched_entity_ids"])

h("GROUND TRUTH: matches per S1 entity")
cap = gt.n.clip(upper=10)
dist = pd.DataFrame({"count": cap.value_counts().sort_index(), "pct": (cap.value_counts(normalize=True).sort_index() * 100).round(2)})
dist.index = [f"{i}{'+' if i == 10 else ''}" for i in dist.index]
print(dist.to_string())
print(f"\nsingletons: {(gt.n == 0).mean():.2%}  | mean matches={gt.n.mean():.2f}  max={gt.n.max()}")
print("\nper country (% of that country's S1):")
print((pd.crosstab(cap, gt.country, normalize="columns") * 100).round(2).to_string())
print("\nsingleton % per country:", (gt.groupby("country").n.apply(lambda s: (s == 0).mean() * 100)).round(2).to_dict())
print("S1 count per country:", gt.country.value_counts().to_dict())

h("GROUND TRUTH: S2 vs S3 composition")
comp = pairs.groupby(["source1_entity_id", "src"]).size().unstack(fill_value=0)
comp = comp.reindex(gt.source1_entity_id, fill_value=0)
comp["country"] = gt.set_index("source1_entity_id").country
print("matched-id share by source:", (pairs.src.value_counts(normalize=True) * 100).round(2).to_dict())
kind = np.select([(comp.S2 > 0) & (comp.S3 > 0), comp.S2 > 0, comp.S3 > 0], ["S2+S3", "S2 only", "S3 only"], "none")
print("\nS1 entities by which sources they match (% per country):")
print((pd.crosstab(kind, comp.country, normalize="columns") * 100).round(2).to_string())
print("\n#S2 matches per S1 (among S1 with >=1 match):")
m = comp[(comp.S2 + comp.S3) > 0]
print(pd.crosstab(m.S2.clip(upper=6), m.country, normalize="columns").mul(100).round(2).to_string())
print("\n#S3 matches per S1 (among S1 with >=1 match):")
print(pd.crosstab(m.S3.clip(upper=6), m.country, normalize="columns").mul(100).round(2).to_string())
print("\ntop (n_S2, n_S3) combos:")
print(m.groupby(["S2", "S3"]).size().sort_values(ascending=False).head(15).to_string())

h("GROUND TRUTH: is matching one-to-one from the S2/S3 side?")
vc = pairs.mid.value_counts()
multi = vc[vc > 1]
print(f"distinct matched ids: {len(vc):,} | ids appearing under >1 S1 entity: {len(multi):,} "
      f"({len(multi) / len(vc):.4%}); max S1s per id: {vc.max()}")
if len(multi):
    print("examples:", multi.head(5).to_dict())
    ex = pairs[pairs.mid.isin(multi.index[:3])].merge(s1, left_on="source1_entity_id", right_on="entity_id")
    print(ex[["mid", "source1_entity_id", "business_name", "business_address"]].to_string(index=False))
matched_set = set(vc.index)
del vc, comp, m, kind

# sample S1 entities for pair-level stats and the NN check
rng = np.random.default_rng(SEED)
matched_s1 = gt.loc[gt.n > 0, "source1_entity_id"].to_numpy()
pair_s1 = set(rng.choice(matched_s1, N_PAIR_S1, replace=False))
sub_pairs = pairs[pairs.source1_entity_id.isin(pair_s1)]
truth = pairs.groupby("source1_entity_id").mid.agg(set)
nn_q = pd.concat([
    s1[s1.entity_id.isin(gt.loc[gt.n == 0, "source1_entity_id"])].groupby("country").sample(N_NN // 2, random_state=SEED),
    s1[s1.entity_id.isin(list(pair_s1))].groupby("country").sample(N_NN // 2, random_state=SEED)])
nn_q["truth"] = nn_q.entity_id.map(truth).apply(lambda x: x if isinstance(x, set) else set())
del truth

matched_rows, nn_rows, nn_negs, all_ids, unmatched_stats = [], [], [], {}, []
for src in ["source2", "source3"]:
    df = load("train", src)
    profile(df, f"train_{src}")
    all_ids[src] = set(df.entity_id)
    um = ~df.entity_id.isin(matched_set)
    unmatched_stats.append(pd.DataFrame({"src": src, "country": df.country, "unmatched": um}))
    # cross-country check over ALL pairs of this source
    cc = pairs[pairs.src == ("S2" if src == "source2" else "S3")].merge(
        df[["entity_id", "country"]], left_on="mid", right_on="entity_id", how="left")
    cc = cc.merge(s1[["entity_id", "country"]], left_on="source1_entity_id", right_on="entity_id", suffixes=("_m", "_s1"))
    print(f"\n[{src}] matched ids missing from file: {cc.country_m.isna().sum():,} | "
          f"cross-country pairs: {(cc.country_m != cc.country_s1).sum():,} of {len(cc):,}")
    del cc
    matched_rows.append(sub_pairs.merge(df, left_on="mid", right_on="entity_id").drop(columns="entity_id"))
    print(f"[{src}] running NN hardness scan ...", flush=True)
    nn_rows.append(nn_scan(nn_q, df, src))
    nn_negs.append(df[df.entity_id.isin(set(nn_rows[-1].best_neg_id))])
    del df

h("GROUND TRUTH: S2/S3 records that match nothing")
us = pd.concat(unmatched_stats)
print((us.groupby(["src", "country"]).unmatched.mean() * 100).round(2).to_string())
print("overall:", (us.groupby("src").unmatched.mean() * 100).round(2).to_dict())
del us
print(f"\nall matched ids exist in S2/S3: {matched_set <= (all_ids['source2'] | all_ids['source3'])}; "
      f"S2 ids with S3 prefix etc.: {sum(not i.startswith('S2-') for i in all_ids['source2'])}, "
      f"{sum(not i.startswith('S3-') for i in all_ids['source3'])}")
print("id overlap S1/S2/S3 numeric parts (just curiosity):",
      len({i[3:] for i in all_ids['source2']} & {i[3:] for i in all_ids['source3']}))
del all_ids, pairs

# ---------------------------------------------------------------- pair-level noise stats
h(f"MATCHED PAIRS: noise statistics on {N_PAIR_S1:,} random matched S1 entities")
pr = pd.concat(matched_rows).merge(s1, left_on="source1_entity_id", right_on="entity_id", suffixes=("_m", "_s1"))
pr = pr.rename(columns={"business_name_m": "m_name", "business_address_m": "m_addr", "business_name_s1": "name",
                        "business_address_s1": "addr", "country_s1": "country"})
norm = lambda s: s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
pr["name_exact"] = norm(pr.name) == norm(pr.m_name)
pr["addr_exact"] = norm(pr.addr) == norm(pr.m_addr)
pr["name_tsr"] = [fuzz.token_sort_ratio(a, b, processor=utils.default_process) for a, b in zip(pr.name, pr.m_name)]
pr["addr_tsr"] = [fuzz.token_sort_ratio(a, b, processor=utils.default_process) for a, b in zip(pr.addr, pr.m_addr)]
pr["name_tset"] = [fuzz.token_set_ratio(a, b, processor=utils.default_process) for a, b in zip(pr.name, pr.m_name)]
pr["m_name_indic"] = pr.m_name.str.contains(INDIC)
pr["m_addr_indic"] = pr.m_addr.str.contains(INDIC)
pr["m_name_upper"] = (pr.m_name == pr.m_name.str.upper()) & pr.m_name.str.contains(r"[A-Za-z]")
pr["m_addr_upper"] = (pr.m_addr == pr.m_addr.str.upper()) & pr.m_addr.str.contains(r"[A-Za-z]")
pr["m_junk_prefix"] = pr.m_name.str.contains(JUNK_PREFIX)
pr["m_handle"] = pr.m_name.str.contains(HANDLE)
pr["m_addr_empty"] = pr.m_addr.str.strip() == ""
pr["m_near"] = pr.m_addr.str.contains(NEAR)
pr["s1_near"] = pr.addr.str.contains(NEAR)
pr["src"] = pr.mid.str[:2]
flags = ["name_exact", "addr_exact", "m_name_indic", "m_addr_indic", "m_name_upper", "m_addr_upper",
         "m_junk_prefix", "m_handle", "m_addr_empty", "s1_near", "m_near"]
print(f"pairs sampled: {len(pr):,}")
print(pr.groupby(["country", "src"])[flags].mean().T.round(4).to_string())
print("\nname token_sort_ratio / token_set_ratio / address token_sort_ratio quantiles:")
print(pr.groupby(["country", "src"])[["name_tsr", "name_tset", "addr_tsr"]]
      .quantile([.05, .1, .25, .5, .75]).unstack(level=[0, 1]).round(0).to_string())

# structured field agreement
for col, fn in [("postcode", lambda a, c: postcode(a, c)), ("house_no", lambda a, c: a.str.extract(HOUSE, expand=False))]:
    a, b = fn(pr.addr, pr.country), fn(pr.m_addr, pr.country)
    status = np.select([a.notna() & b.notna() & (a == b), a.notna() & b.notna(), a.notna() | b.notna()],
                       ["agree", "conflict", "missing one side"], "missing both")
    print(f"\n{col} agreement on matched pairs (% per country/src):")
    print((pd.crosstab(status, [pr.country, pr.src], normalize="columns") * 100).round(2).to_string())
    pr[f"{col}_status"] = status
print("\nexamples of house_no conflicts:")
print(pr[pr.house_no_status == "conflict"].sample(frac=1, random_state=SEED).head(10)[["addr", "m_addr"]].to_string(index=False))

h("25 MATCHED PAIRS SIDE BY SIDE (mixed US/India, S2/S3)")
ex = pr.sample(frac=1, random_state=SEED).groupby(["country", "src"]).head(7).head(25)
for r in ex.itertuples():
    print(f"[{r.country:5s}] S1  {r.name!r:60s} | {r.addr!r}")
    print(f"        {r.src}  {r.m_name!r:60s} | {r.m_addr!r}   (name_tsr={r.name_tsr:.0f}, addr_tsr={r.addr_tsr:.0f})\n")

h("NEAREST NEIGHBOUR HARDNESS (brute force, same country, token_sort_ratio on name+address)")
nn = pd.concat(nn_rows)
nn["kind"] = np.where(nn.s1.isin(nn_q.loc[nn_q.truth.map(len) == 0, "entity_id"]), "singleton", "matched")
best = nn.groupby(["s1", "kind", "country"]).agg(best_neg=("best_neg_score", "max"), best_true=("best_true_score", "max"),
                                                  negs_above=("negs_scoring_>=_best_true", "min")).reset_index()
print("best NEGATIVE score per query S1 (quantiles):")
print(best.groupby(["kind", "country"]).best_neg.describe().round(1).to_string())
print("\nbest TRUE-match score for matched queries:")
print(best[best.kind == "matched"].groupby("country").best_true.describe().round(1).to_string())
mq = best[best.kind == "matched"]
print(f"\nmatched queries where some negative scores >= best true match: {(mq.best_neg >= mq.best_true).mean():.2%}")
negs = pd.concat(nn_negs).set_index("entity_id")
s1i = s1.set_index("entity_id")
print("\n10 S1 SINGLETONS and their most similar S2/S3 record:")
for s in best[best.kind == "singleton"].groupby("country").head(5).s1:
    r = s1i.loc[s]
    print(f"[{r.country}] S1 {s}: {r.business_name!r} | {r.business_address!r}")
    for x in nn[nn.s1 == s].itertuples():
        n = negs.loc[x.best_neg_id]
        print(f"      best {x.src} ({x.best_neg_score}): {n.business_name!r} | {n.business_address!r}")
del s1i, negs

# ================================================================ TEST
for src in ["source1", "source2", "source3"]:
    df = load("test", src)
    profile(df, f"test_{src}", train_countries=TRAIN_COUNTRIES)
    del df

h("TRAIN vs TEST comparison (per file x country)")
print(pd.DataFrame(SUMMARY).set_index(["file", "country"]).to_string())
print(f"\ndone in {time.time() - T0:.0f}s")
