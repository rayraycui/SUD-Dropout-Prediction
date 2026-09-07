#!/usr/bin/env python3
"""Regenerate Fig. 1, the study-overview schematic.

The diagram's layout lives in assets/fig0_overview_template.svg; every number in it
is a placeholder filled from the committed data here, so the figure cannot drift
away from the results it summarizes. Re-running this script on unchanged data
reproduces figures/fig0_overview.svg byte for byte.

Writes figures/fig0_overview.svg, and figures/fig0_overview.pdf (the figure the
manuscript includes)
when an SVG converter is available.

Run from the repository root:  python src/18_overview_figure.py
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

PARQUET = "data/teds_d_analysis_2015_2022.parquet"
H2H = "data/p1_plan_eval_17substance_gbt_vs_logistic.csv"
TEMPLATE = "assets/fig1_overview_template.svg"
OUT_SVG = "figures/fig1_overview.svg"
OUT_PDF = "figures/fig1_overview.pdf"


def facts():
    """Read every displayed quantity from the committed data."""
    if not os.path.exists(PARQUET):
        sys.exit(
            f"missing {PARQUET}\n"
            "This step needs the analysis table, which is ~39 MB and may be shipped\n"
            "separately from the code archive. Either place it in data/, or rebuild:\n"
            "  python src/00_download_teds.py && python src/01_build_analysis_table.py"
        )
    df = pd.read_parquet(PARQUET, columns=["subname", "y"])
    h = pd.read_csv(H2H)
    big = h[h.group == "BIG6"]
    tiny = h[h.train_n < 5000]
    return {
        "n_m": len(df) / 1e6,
        "n_sub": h.substance.nunique(),
        "dropout_pct": df.y.mean() * 100,
        "train_min": int(h.train_n.min()),
        "train_max_m": h.train_n.max() / 1e6,
        "pool_m": big.train_n.sum() / 1e6,
        "auroc_pooled": h.gbt_pooled.mean(),
        "auroc_single": h.gbt_single.mean(),
        "tiny_mean": tiny.gbt_delta.mean(),
        "tiny_max": tiny.gbt_delta.max(),
        "sd_pooled": h.gbt_pooled.std(ddof=1),
        "sd_single": h.gbt_single.std(ddof=1),
    }


def main():
    f = facts()
    os.makedirs("figures", exist_ok=True)
    with open(TEMPLATE, encoding="utf-8") as fh:
        svg = fh.read().format(**f)
    with open(OUT_SVG, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(f"wrote {OUT_SVG}")
    for k in ("n_m", "n_sub", "dropout_pct", "auroc_pooled", "auroc_single",
              "tiny_mean", "tiny_max", "sd_pooled", "sd_single"):
        print(f"    {k:14s} {f[k]}")

    # LaTeX includes the PDF; convert if a converter is installed.
    for cmd in (["rsvg-convert", "-f", "pdf", "-o", OUT_PDF, OUT_SVG],
                ["cairosvg", OUT_SVG, "-o", OUT_PDF],
                ["inkscape", OUT_SVG, "--export-type=pdf",
                 f"--export-filename={OUT_PDF}"]):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"wrote {OUT_PDF} via {cmd[0]}")
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    print(f"note: no SVG converter found, so {OUT_PDF} is unchanged.\n"
          "      Install rsvg-convert, cairosvg or inkscape to regenerate it.")


if __name__ == "__main__":
    main()
