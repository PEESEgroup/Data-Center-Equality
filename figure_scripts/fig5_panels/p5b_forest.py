#!/usr/bin/env python3
"""Causal effect of data center capacity on employment, by sector.

Replaces an earlier panel of per-state OLS slopes, whose values ran to 2.4
million jobs per GW and carried no causal reading.

Intervals are 95% Anderson-Rubin sets, not clustered-t intervals: the two
disagree for data center employment, where t gives P = 0.059 while the AR set
excludes zero, and the Supplementary Information privileges AR throughout.
Filled markers mark sets that exclude zero.

SYMMETRIC-LOG VERTICAL AXIS.  The estimates span 1,309 for nonresidential
building to 15,468 for construction, and telecommunications runs negative, so
neither a linear nor a plain logarithmic axis holds them in one frame at one
column width.  A horizontal layout with per-block axes was legible but needed a
full page column.

NAICS 51 is absent because its share-exogeneity pre-trend is rejected; NAICS
5415 by author decision.  Nonresidential building lies inside construction, and
the two data center rows are two measurements of one quantity, so no interval
may be added to another.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt   # noqa: F401  (sets the shared rc style)
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (C_CONSTR, C_INFO, FS_ANNOT, FS_LABEL, PANEL_W, PANEL_W2, PANEL_W3, iv_table,
                    save, tidy)  # noqa: E402

ROWS = [
    # The observed series is dropped: it is the same quantity as the in-state
    # one measured without the spillover correction, and panel 1 is devoted to
    # exactly that distinction.  Four categories also read at 61 mm; five did not.
    # Labels are kept to ~8 characters on each line: at 61 mm across four
    # categories each gets about 15 mm, and "Construction" set on one line
    # collided with "Nonres." next to it.  The NAICS code carries the precision
    # the shortened word gives up, and the caption spells the sectors out.
    ("emp_5182_cleaned", "Data\ncenters\n(5182)",   C_INFO),
    ("emp_23",           "Constr.\n(23)",           C_CONSTR),
    ("emp_2362",         "Nonres.\n(2362)",         C_CONSTR),
    ("emp_517",          "Telecom\n(517)",          C_CONSTR),
]
LT = 1000.0          # symlog linear threshold


def draw(height=2.05):
    d = iv_table()
    fig, ax = plt.subplots(figsize=(PANEL_W3, height), dpi=600)
    ax.set_yscale("symlog", linthreshy=LT)
    ax.axhline(0, color="0.45", lw=0.7, zorder=1)

    for x, (oc, lab, colour) in enumerate(ROWS):
        r = d.loc[oc]
        ax.plot([x, x], [r.ar_lo, r.ar_hi], color=colour, lw=1.3, zorder=3)
        for yy in (r.ar_lo, r.ar_hi):
            ax.plot([x - 0.16, x + 0.16], [yy, yy], color=colour, lw=1.3, zorder=3)
        ax.scatter(x, r.beta, s=20, zorder=5, linewidths=1.0,
                   facecolor=colour if r.excl0 else "white", edgecolor=colour)
        ax.text(x + 0.24, r.beta, f"{r.beta:,.0f}".replace("-", "−"),
                ha="left", va="center", fontsize=FS_ANNOT,
                fontweight="bold" if r.excl0 else "normal", color="0.1")

    ax.set_xticks(range(len(ROWS)))
    ax.set_xticklabels([l for _, l, _ in ROWS], fontsize=FS_ANNOT)
    ax.set_xlim(-0.55, len(ROWS) - 0.25)
    # proplot adds minor ticks by default; on a categorical axis they mark
    # nothing and one of them fell to the right of the last category.
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", which="major", length=0)
    ax.set_ylim(-11000, 70000)
    ax.set_ylabel("Causal effect (jobs per GW)", fontsize=FS_LABEL, labelpad=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "0" if v == 0 else f"{v:,.0f}".replace("-", "−")))
    tidy(ax)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.text(0.97, 0.03, f"n = {int(d.loc['emp_5182','n'])} state-years",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_ANNOT,
            color="0.3")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc="0.25", mec="0.25", ms=3.2),
        Line2D([], [], marker="o", ls="none", mfc="white", mec="0.25", mew=0.9,
               ms=3.2)],
        labels=["95% AR set excludes zero", "contains zero"],
        loc="upper left", frameon=False, fontsize=FS_ANNOT, handletextpad=0.3,
        borderpad=0.1, labelspacing=0.25)
    save(fig, "f05b_forest")
    import os
    if os.environ.get("CHECK"):
        pass  # debug savefig removed for release
    return d


if __name__ == "__main__":
    d = draw()
    for oc, lab, _ in ROWS:
        r = d.loc[oc]
        print(f"  {lab.replace(chr(10),' '):28s} {r.beta:9,.0f}  "
              f"AR [{r.ar_lo:,.0f}, {r.ar_hi:,.0f}]  excl0={r.excl0}")
