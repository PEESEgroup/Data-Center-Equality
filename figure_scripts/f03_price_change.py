# -*- coding: utf-8 -*-
"""Horizontal scatter plots of DC-induced electricity price changes with CIs.

Reads table3 (absolute $/MWh + CI) and table4 (percentage % + CI) from
r3_summary/ and produces two SVG figures saved to figures/01-2/.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import proplot as pplt

# ======================== config ========================
TABLE3_PATH = Path("./results/r3_summary/table3_zone_impact_2025.csv")
TABLE4_PATH = Path("./results/r3_summary/table4_zone_relative_2025.csv")
OUT_DIR     = Path("./figures/01-2")
PCT_THRESH  = 1.0          # only show zones with dP_pct > this (%)
ABS_THRESH  = 0.5          # only show zones with dP_dollar_MWh > this ($/MWh)
A4_WIDTH_IN = 8.27
FIG_WIDTH   = A4_WIDTH_IN / 3.0
FONTSIZE_PT = 7

# ======================== load data ========================
def load_data() -> pd.DataFrame:
    t3 = pd.read_csv(TABLE3_PATH)
    t4 = pd.read_csv(TABLE4_PATH)

    df = t3.merge(
        t4[["Region", "Zone", "dP_pct", "dP_pct_CI_lo", "dP_pct_CI_hi"]],
        on=["Region", "Zone"],
    )
    df["label"] = df["Region"] + ":" + df["Zone"]
    return df


# ======================== plot ========================
def plot_scatterx(
    means: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    labels: list[str],
    xlabel: str,
    outfile: Path,
    color: str,
    marker: str,
    markercolor: str,
):
    pplt.rc.update(fontsize=FONTSIZE_PT)
    n = means.size
    height = max(1.8, 0.14 * n)

    fig, ax = pplt.subplots(figsize=(FIG_WIDTH, height))

    shade = np.vstack([ci_lo, ci_hi])
    kw = dict(
        shadedata=shade,
        marker=marker,
        markersize=12,
        markercolors=markercolor,
        linewidth=0.8,
        color=color,
        label="mean",
        shadelabel="CI",
        barzorder=0,
        boxmarker=False,
    )
    ax.scatterx(means, **kw)

    ax.format(
        yticklabels=labels,
        ylocator=np.arange(0, n),
        ytickminor=False,
        xlabel=xlabel,
        ylim=(-0.5, n + 1),
    )

    xlo, xhi = np.nanmin(ci_lo), np.nanmax(ci_hi)
    span = xhi - xlo
    if np.isfinite(span) and span > 0:
        ax.set_xlim(xlo - 0.1 * span, xhi + 0.1 * span)

    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("0.2")

    fig.savefig(outfile, dpi=300, bbox_inches="tight", transparent=True)
    print(f"  saved {outfile}")


# ======================== main ========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # absolute: sort and filter independently
    df_abs = df.sort_values("dP_dollar_MWh", ascending=False).reset_index(drop=True)
    df_abs = df_abs[df_abs["dP_dollar_MWh"] > ABS_THRESH].reset_index(drop=True)
    plot_scatterx(
        means=df_abs["dP_dollar_MWh"].to_numpy(),
        ci_lo=df_abs["dP_CI_lo"].to_numpy(),
        ci_hi=df_abs["dP_CI_hi"].to_numpy(),
        labels=df_abs["label"].tolist(),
        xlabel="Absolute Price Increase ($/MWh)",
        outfile=OUT_DIR / "fig_abs_2025.svg",
        color="#019092",
        marker="x",
        markercolor="blue",
    )

    # percentage: sort and filter independently
    df_pct = df.sort_values("dP_pct", ascending=False).reset_index(drop=True)
    df_pct = df_pct[df_pct["dP_pct"] > PCT_THRESH].reset_index(drop=True)
    plot_scatterx(
        means=df_pct["dP_pct"].to_numpy(),
        ci_lo=df_pct["dP_pct_CI_lo"].to_numpy(),
        ci_hi=df_pct["dP_pct_CI_hi"].to_numpy(),
        labels=df_pct["label"].tolist(),
        xlabel="Percentage Price Increase (%)",
        outfile=OUT_DIR / "fig_pct_2025.svg",
        color="#5DBFE9",
        marker="o",
        markercolor="blue",
    )


if __name__ == "__main__":
    main()
