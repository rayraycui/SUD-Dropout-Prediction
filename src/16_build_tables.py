#!/usr/bin/env python3
"""Regenerate the manuscript's Table II and Table III bodies from the committed CSVs.

Writes LaTeX table bodies to tables/table2_body.tex and tables/table3_body.tex so the
numbers in the manuscript can be checked against the data without manual transcription.

Run from the repository root:  python src/16_build_tables.py
"""
import os
import sys

import pandas as pd, numpy as np

PARQUET = "data/teds_d_analysis_2015_2022.parquet"
H2H = "data/p1_plan_eval_17substance_gbt_vs_logistic.csv"

LBL = {"Non_rx_methadone": "Non rx methadone",   # Table II convention
        "Other_opioids": "Other opioids",
       "Other_amphetamines": "Other amphetamines", "Other_hallucinogens": "Other hallucinogens",
       "Other_stimulants": "Other stimulants", "Other_tranquilizers": "Other tranquilizers",
       "Other_sedatives_hypnotics": "Other sedatives hypnotics", "OTC": "OTC"}


LBL3 = {"Non_rx_methadone": "Non-Rx methadone",          # Table III conventions
        "Other_sedatives_hypnotics": "Other sedatives/hypnotics",
        "OTC": "OTC medications"}



def require_parquet():
    """The analysis table is large, so it may be distributed separately."""
    if not os.path.exists(PARQUET):
        sys.exit(
            f"missing {PARQUET}\n"
            "This step needs the analysis table, which is ~39 MB and may be shipped\n"
            "separately from the code archive. Either:\n"
            "  * place teds_d_analysis_2015_2022.parquet in data/, or\n"
            "  * rebuild it:  python src/00_download_teds.py && "
            "python src/01_build_analysis_table.py"
        )


def main():
    require_parquet()
    df = pd.read_parquet(PARQUET)
    n = len(df)
    print(f"cohort N = {n:,}  dropout = {df.y.mean():.3f}")

    # ---- Table II: per-substance cohort counts -------------------------------
    g = (df.groupby("subname")
           .agg(n=("y", "size"), dr=("y", "mean"))
           .sort_values("n", ascending=False))
    NL2 = chr(92) * 2
    rows = [f"{LBL.get(s, s.replace('_', ' '))} & {int(r.n):,} & "
            f"{int(r.n) / n * 100:.1f} & {r.dr:.3f} " + NL2
            for s, r in g.iterrows()]
    os.makedirs("tables", exist_ok=True)
    with open("tables/table2_body.tex", "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"tables/table2_body.tex: {len(rows)} substance rows")

    # ---- per-substance class sizes ------------------------------------------
    cls = g.reset_index().rename(columns={"dr": "dropout"})
    cls["dropout"] = cls["dropout"].round(3)
    big6 = set(pd.read_csv(H2H).query("group == 'BIG6'").substance)
    cls["is_big6"] = cls.subname.isin(big6)
    cls["group"] = cls.is_big6.map({True: "BIG6", False: "small"})
    cls.to_csv("data/teds_d_class_sizes_2015_2022.csv", index=False)
    print(f"data/teds_d_class_sizes_2015_2022.csv: {len(cls)} substances, "
          f"{cls.n.sum():,} episodes")
    assert cls.n.sum() == n, "class sizes must sum to the cohort"
    assert set(cls.subname) == big6 | set(pd.read_csv(H2H).substance), \
        "class-size substances must match the results file exactly"

    # ---- Table III: AUROC under both strategies and both learners -----------
    # Matches the manuscript layout: Substance | Test n | GBT single | GBT pooled |
    # LR single | LR pooled, split by whether the substance is in the pooled
    # training set, bold marking the row maximum.
    h = pd.read_csv(H2H).sort_values("test_n", ascending=False)
    cols = ["gbt_single", "gbt_pooled", "lr_single", "lr_pooled"]
    NL = chr(92) * 2          # a LaTeX row terminator
    BS = chr(92)              # a single backslash
    out = []
    for grp, header in [("BIG6", "In the pooled training set"),
                        ("small", "Not in the pooled training set")]:
        pre = BS + "addlinespace " if grp == "small" else ""
        out.append(pre + BS + "multicolumn{6}{@{}l}{" + BS + "textit{" + header + "}} " + NL)
        for r in h[h.group == grp].itertuples():
            vals = [getattr(r, c) for c in cols]
            best = max(vals)
            cells = [(BS + "textbf{" + f"{v:.3f}" + "}") if v == best else f"{v:.3f}"
                     for v in vals]
            label = LBL3.get(r.substance, LBL.get(r.substance, r.substance.replace("_", " ")))
            out.append(f"{label} & {int(r.test_n):,} & " + " & ".join(cells) + " " + NL)
    with open("tables/table3_body.tex", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"tables/table3_body.tex: "
          f"{sum(1 for o in out if 'multicolumn' not in o)} substance rows")

    # ---- headline values quoted in the text ---------------------------------
    w = h.test_n.values
    ew = {c: np.average(h[c], weights=w) for c in cols}
    print(f"episode-weighted over {h.test_n.sum():,} test episodes:")
    for c in cols:
        print(f"   {c:11s} {ew[c]:.4f}")
    print(f"   cost of pooling  {ew['gbt_single'] - ew['gbt_pooled']:.4f}")
    print(f"   GBT-LR gap: single {(h.gbt_single - h.lr_single).mean():.4f}, "
          f"pooled {(h.gbt_pooled - h.lr_pooled).mean():.4f}")
    print(f"   best value is a GBT column in "
          f"{sum(1 for r in h.itertuples() if max(getattr(r, c) for c in cols) in (r.gbt_single, r.gbt_pooled))}"
          f" of {len(h)} rows")


if __name__ == "__main__":
    main()
