#!/usr/bin/env python3
"""Check that every committed data file agrees with the manuscript.

Fails loudly on the class of defect this guards against: a stale file copied
forward from an earlier version of the study, carrying substance labels that no
longer match the corrected SUB1 mapping.

Run from the repository root:  python src/17_check_consistency.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

H2H = "data/p1_plan_eval_17substance_gbt_vs_logistic.csv"
PARQUET = "data/teds_d_analysis_2015_2022.parquet"
# The manuscript itself ships in the Overleaf package, not in this repository.
# Check 4 compares the generated table bodies against it when a copy is available:
# point PAPER_TEX at one, e.g.
#   PAPER_TEX=~/overleaf/paper.tex python src/17_check_consistency.py
PAPER = os.environ.get("PAPER_TEX", "paper.tex")

# Table II writes some names differently from Table III; both are intentional.
LBL2 = {"Non_rx_methadone": "Non rx methadone", "Other_opioids": "Other opioids",
        "Other_amphetamines": "Other amphetamines",
        "Other_hallucinogens": "Other hallucinogens",
        "Other_stimulants": "Other stimulants",
        "Other_tranquilizers": "Other tranquilizers",
        "Other_sedatives_hypnotics": "Other sedatives hypnotics", "OTC": "OTC"}

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def check_table_bodies():
    """Compare the generated table bodies against a copy of the manuscript."""
    paper = re.sub(r"[\s,&\\]+", "", open(PAPER, encoding="utf-8").read())
    for body in ("tables/table2_body.tex", "tables/table3_body.tex"):
        rows = [r for r in open(body, encoding="utf-8").read().strip().split("\n")
                if "multicolumn" not in r]
        bad = [r.split("&")[0].strip() for r in rows
               if re.sub(r"[\s,&\\]+", "", r.rsplit(chr(92) * 2, 1)[0]) not in paper]
        check(not bad, f"{body}: all {len(rows)} rows found in {PAPER}"
                       + ("" if not bad else f" (missing {bad})"))


def main():
    h = pd.read_csv(H2H)
    truth = set(h.substance)
    print(f"reference: {len(truth)} substances in {H2H}")

    # 1. the analysis table itself (may be distributed separately: it is ~39 MB)
    df = None
    if os.path.exists(PARQUET):
        df = pd.read_parquet(PARQUET, columns=["subname", "y"])
        check(set(df.subname) == truth, "parquet substances match the results file")
        check(len(df) == 6_679_648, f"parquet has 6,679,648 rows (got {len(df):,})")
    else:
        print(f"  skip   {PARQUET} absent; checks 1 and 5 need it "
              "(run src/00 then src/01, or copy the file into data/)")

    # 2. every committed CSV that names substances
    import glob
    for path in sorted(glob.glob("data/*.csv")):
        d = pd.read_csv(path)
        col = next((c for c in ("subname", "substance") if c in d.columns), None)
        if col is None:
            continue
        got = set(d[col].dropna())
        # the categorical-verdict file covers the eleven lower-volume substances only
        expect_subset = "verdicts" in path
        ok = got <= truth if expect_subset else got == truth
        check(ok, f"{path}: substances {'subset of' if expect_subset else '=='} reference"
                  + ("" if ok else f" (extra {sorted(got - truth)}, missing {sorted(truth - got)})"))

    # 3. class sizes reconcile with the cohort and with Table II
    CLS = "data/teds_d_class_sizes_2015_2022.csv"
    if os.path.exists(CLS):
        cls = pd.read_csv(CLS)
        check(cls.n.sum() == 6_679_648, f"class sizes sum to the cohort ({cls.n.sum():,})")
        grp = h.set_index("substance").group
        check(all(cls.set_index("subname").group[s] == grp[s] for s in truth),
              "big6/small grouping agrees with the results file")
    else:
        print(f"  skip   {CLS} absent; regenerate it with src/16_build_tables.py")

    # 4. generated table bodies appear verbatim in the manuscript, when it is available
    if not os.path.exists(PAPER):
        print(f"  skip   {PAPER} absent; the manuscript ships in the Overleaf package. "
              "Set PAPER_TEX=<path> to run this check.")
    else:
        check_table_bodies()

    # 5. Table II counts equal a fresh groupby
    if df is None:
        print()
        if fails:
            print(f"{len(fails)} CHECK(S) FAILED")
            sys.exit(1)
        print("all runnable checks passed (parquet-dependent checks skipped)")
        return
    g = df.groupby("subname").agg(n=("y", "size"), dr=("y", "mean"))
    t2 = {}
    for line in open("tables/table2_body.tex", encoding="utf-8").read().strip().split("\n"):
        p = [x.strip() for x in line.replace(chr(92) * 2, "").split("&")]
        t2[p[0]] = (int(p[1].replace(",", "")), float(p[3]))
    bad = [s for s in truth
           if t2[LBL2.get(s, s)][0] != g.n[s] or abs(t2[LBL2.get(s, s)][1] - g.dr[s]) > 5e-4]
    check(not bad, f"Table II counts and dropout rates recompute ({len(truth)} rows)"
                   + ("" if not bad else f" (bad {bad})"))


    # 6. the overview schematic (Fig. 1) agrees with the data it summarizes
    OVSVG = "figures/fig1_overview.svg"
    if os.path.exists(OVSVG):
        svg = open(OVSVG, encoding="utf-8").read()
        big = h[h.group == "BIG6"]
        tiny = h[h.train_n < 5000]
        want = [f"{len(df) / 1e6:.2f}M episodes", f"{len(truth)} substances",
                f"{df.y.mean() * 100:.1f}% dropout",
                f"{int(h.train_n.min())} to {h.train_n.max() / 1e6:.2f}M training episodes",
                f"substances, {big.train_n.sum() / 1e6:.2f}M episodes",
                f"{h.gbt_pooled.mean():.3f} pooled vs "
                f"{h.gbt_single.mean():.3f} specific",
                f"{tiny.gbt_delta.mean():+.3f} avg under 5k episodes",
                f"{tiny.gbt_delta.max():+.3f} lowest-volume substance",
                f"SD {h.gbt_pooled.std(ddof=1):.3f} vs "
                f"{h.gbt_single.std(ddof=1):.3f} across {len(truth)} substances"]
        missing = [w for w in want if w not in svg]
        check(not missing, f"{OVSVG}: all {len(want)} displayed values recompute"
                           + ("" if not missing else f" (missing {missing})"))

    print()
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
