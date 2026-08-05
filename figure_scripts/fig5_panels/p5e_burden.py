#!/usr/bin/env python3
"""Figure 5e -- who pays the energy bill and who gets the return.

Panel d encodes the energy-burden increase as colour; this panel puts it on an
axis, because the relationship is the paper's equity claim and a colour ramp
cannot be read quantitatively.

WHAT IS PLOTTED, AND WHAT IS NOT.  The return ratio has the subsidy in its
denominator, so any variable correlated with subsidy will correlate with the
ratio for purely mechanical reasons.  Two candidate relationships were tested
against a permutation null in which labor income is shuffled across states,
destroying every real relationship but preserving the ratio's construction:

    capacity      vs ratio   observed -0.42, null mean -0.54 [-0.71, -0.32]
                             -> INSIDE the null band, mechanical, NOT PLOTTED
    energy burden vs ratio   observed -0.75, null mean -0.46 [-0.67, -0.24]
                             -> outside the null band, plotted here

Using the level rather than the ratio, capacity in fact runs the other way:
Spearman(capacity, labor income) = +0.53, P = 0.005.  The panel therefore makes
no claim about capacity and return.
"""
import os
import sys
from pathlib import Path

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as pplt   # noqa: F401
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (R06, AUD, _P, FS_ANNOT, FS_TICK, EMP, HALF_W, PANEL_W, PANEL_W2, PANEL_W3, footnote, label_beside, place_labels, save, tidy)  # noqa: E402

LABEL = ["VA", "OK", "TX", "OH", "IL", "MN", "CA", "GA", "AZ", "MD", "WA"]


def load():
    d = pd.read_csv(AUD / "labor_income_cumulative.csv")
    abbr = pd.read_csv(R06 / "panel_state_year.csv")[
        ["state_abbr", "state_name"]].drop_duplicates()
    d = d.merge(abbr, left_on="state", right_on="state_name", how="left")
    d = d.rename(columns={"state_abbr": "st", "subsidy_M": "subsidy",
                          "li_cumulative_M": "li",
                          "ratio_cumulative": "ratio"})[
        ["st", "subsidy", "li", "ratio"]].dropna()
    eb = pd.read_csv(EMP.parent / "rider" / "compare" / "state_poverty_share_2025.csv")
    eb = eb.rename(columns={"state_abbr": "st"})
    d = d.merge(eb[["st", "share_diff_gt6_state"]], on="st", how="left")
    d["eb"] = d.share_diff_gt6_state * 100
    pan = pd.read_csv(R06 / "panel_state_year.csv")
    d["cap"] = d.st.map(pan[pan.year == 2024].set_index("state_abbr")["dc_gw"])
    kept = d.dropna(subset=["eb"]).query("ratio > 0").reset_index(drop=True)
    # The states with no energy-burden series have no horizontal position, but they
    # do have a return.  Dropping them outright left the shaded band empty and made
    # it look as though every state cleared break-even, when in fact BOTH states that
    # do not are in this group.  They are returned so the panel can show them in a
    # margin strip at their own vertical position.
    missing = d[d.eb.isna()].reset_index(drop=True)
    return kept, missing


def draw(height=2.35, large=False):
    d, miss = load()
    rho, pv = stats.spearmanr(d.eb, d.ratio)
    fit = stats.linregress(d.eb, np.log10(d.ratio))

    k = 1.9 if large else 1.0
    fig, ax = plt.subplots(figsize=(PANEL_W2 * k, height * k), dpi=600)
    ax.axhspan(0.01, 1, color="0.92", zorder=0)
    ax.axhline(1, color="0.35", lw=0.9, ls="--", zorder=2)

    base = mcm.get_cmap("Blues")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "Blues_trim", base(np.linspace(0.28, 0.95, 256)))
    norm = mcolors.LogNorm(max(d.subsidy.min(), 0.5), d.subsidy.max())
    cs = np.sqrt(d.cap.to_numpy(float))
    sizes = 16 + 150 * cs / max(cs.max(), 1e-9)
    ax.scatter(d.eb, d.ratio, s=sizes, c=cmap(norm(d.subsidy.clip(lower=0.5))),
               zorder=5, edgecolor="white", linewidth=0.4)

    xs = np.linspace(d.eb.min(), d.eb.max(), 40)
    ax.plot(xs, 10 ** (fit.intercept + fit.slope * xs), color="0.25", lw=1.1,
            zorder=4)

    if large:
        label_beside(ax, d.eb, d.ratio, d.st, sizes=sizes, size=7.0)

    ax.set_yscale("log")
    # A margin strip for the states with no burden series was tried and removed:
    # placing them on the panel says they are absent from the relationship, when
    # what is true is that the relationship cannot be measured for them.  Their
    # identity is carried by the grey fill in the fiscal panel and by the caption.
    ax.set_xlabel("Increase in share of households with energy burden > 6% (pp)",
                  fontsize=FS_TICK, labelpad=2)
    ax.set_ylabel("Labor income gain per dollar\nof incentive, 2020 to 2024",
                  fontsize=FS_TICK, labelpad=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    tidy(ax)
    ax.text(0.03, 0.05, "below break-even", transform=ax.transAxes, fontsize=FS_ANNOT,
            color="0.45", va="bottom")
    ax.text(0.72, 0.96,
            f"n = {len(d)} units\nrank correlation {rho:.2f}, {_P(pv)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_ANNOT, color="0.25")

    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.015, aspect=16)
    cb.set_label("Cumulative incentive ($M)", fontsize=FS_ANNOT, labelpad=2)
    cb.ax.tick_params(labelsize=5.3, width=0.4, length=2)
    cb.outline.set_linewidth(0)

    if os.environ.get("CHECK"):
        import matplotlib.pyplot as _p
        pass  # debug savefig removed for release
    save(fig, "f05e_burden_labeled" if large else "f05e_burden")
    return d, rho, pv


if __name__ == "__main__":
    draw(large=True)
    d, rho, pv = draw()
    print(f"n = {len(d)}  Spearman(eb, ratio) = {rho:.3f}, P = {pv:.4g}")
    print(f"below break-even: {(d.ratio < 1).sum()}  ({', '.join(d[d.ratio<1].st)})")
