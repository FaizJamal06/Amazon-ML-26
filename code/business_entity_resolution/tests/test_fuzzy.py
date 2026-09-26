"""Tests for pass E (ber.blocking.fuzzy) and the handle/domain helpers of ber.normalize. Run directly or with pytest."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.blocking.fuzzy import build_fuzzy_index, query_fuzzy, row_top_k  # noqa: E402
from ber.normalize import is_handle_expr, strip_handle, strip_handle_expr  # noqa: E402


def test_strip_handle():
    """Handles and domains become plain words; ordinary names are untouched."""
    assert strip_handle("#salinasguggenheim") == "salinasguggenheim"
    assert strip_handle("REEDPIZZA.COM") == "REEDPIZZA"
    assert strip_handle("supremebu1kers.com") == "supremebu1kers"
    assert strip_handle("ReedPizza") == "Reed Pizza"
    assert strip_handle("Lucky Bistro LLC") == "Lucky Bistro LLC"
    df = pl.DataFrame({"name_raw": ["#salinasguggenheim", "REEDPIZZA.COM", "supremebu1kers.com", "ReedPizza",
                                    "Lucky Bistro LLC"]})
    out = df.select(h=is_handle_expr(), c=strip_handle_expr())
    assert out["h"].to_list() == [True, True, True, True, False]
    assert out["c"].to_list()[:4] == ["salinasguggenheim", "reedpizza", "supremebu1kers", "reed pizza"]


def test_row_top_k_matches_brute_force():
    """Vectorized per-row top-k == sorting each row, with ties broken by column."""
    m = sp.random(40, 30, density=0.3, format="csr", random_state=0, dtype=np.float32)
    m.data = np.round(m.data, 1)                       # create ties
    r, c, v = row_top_k(m, 3)
    got = sorted(zip(r.tolist(), c.tolist()))
    want = []
    for i in range(m.shape[0]):  # brute force reference, 40 rows
        row = m.getrow(i)
        items = sorted(zip(row.indices.tolist(), row.data.tolist()), key=lambda x: (-x[1], x[0]))[:3]
        want += [(i, j) for j, _ in items]
    assert got == sorted(want)


def test_fuzzy_retrieves_handles_and_typos():
    """Queries written as handles / domains / with an OCR digit find the right S1 first."""
    s1 = pl.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c", "S1-d", "S1-e"],
                       "name_raw": ["Reed Pizza LLC", "Reed Pasta Inc", "Supreme Bulkers", "Salinas Guggenheim LLC",
                                    "Pizza Palace"],
                       "name_core": ["reed pizza", "reed pasta", "supreme bulkers", "salinas guggenheim", "pizza palace"],
                       "name_skeleton": ["rd pz", "rd pst", "sprm blkrs", "slns gggnhm", "pz plc"]})
    q = pl.DataFrame({"entity_id": ["S2-1", "S2-2", "S3-3"],
                      "name_raw": ["REEDPIZZA.COM", "#salinasguggenheim", "supremebu1kers.com"],
                      "name_core": ["reedpizza com", "salinasguggenheim", "supremebu1kers com"],
                      "name_skeleton": ["rdpz cm", "slnsgggnhm", "sprmb1krs cm"]})
    ix = build_fuzzy_index(s1, max_df=1.0)
    hits = query_fuzzy(ix, q, top_k=1, min_cos=0.0)
    assert dict(hits.select("cand_id", "s1_id").rows()) == {"S2-1": "S1-a", "S2-2": "S1-d", "S3-3": "S1-c"}
    assert hits["score"].is_between(0, 1.0001).all()


if __name__ == "__main__":
    test_strip_handle()
    test_row_top_k_matches_brute_force()
    test_fuzzy_retrieves_handles_and_typos()
    print("fuzzy tests passed")
