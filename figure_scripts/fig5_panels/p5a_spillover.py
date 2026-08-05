#!/usr/bin/env python3
"""Figure 5a -- how much of a state's data-processing employment is its own.

DESCRIPTIVE. No causal claim is made or implied here.  The panel exists to
motivate the series used in every later panel: observed NAICS 5182 employment
in a state reflects both in-state data center activity and spillover from data
centers elsewhere, and the two cannot be told apart in the raw series.

lambda_s = E_direct / (E_direct + E_spill/M) is the share driven in state,
from the PwC IMPLAN decomposition (SI section 6.1).  Plotting it against the
state's own 2024 capacity shows the problem plainly: the least-capacity quartile
gets 41% of its data-processing employment from other states' data centers.
That is why the analysis cleans the series before instrumenting it.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import proplot as pplt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (_P, FS_ANNOT, FS_TICK, C_INFO, HALF_W, PANEL_W, PANEL_W2, PANEL_W3, R06, footnote, label_beside, place_labels, save, tidy)  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "employment"))
from analysis_universe import ANALYSIS_UNITS  # noqa: E402

LABEL = ["VA", "TX", "OR", "WY", "DC"]   # a legible subset; the cluster near 2-9 GW cannot carry more


def load():
    p = pd.read_csv(R06 / "panel_state_year.csv")
    d = p[(p.year == 2024) & (p.state_abbr.isin(ANALYSIS_UNITS))]
    return d[["state_abbr", "dc_gw", "local_share"]].dropna().reset_index(drop=True)


def draw(height=2.05, large=False):
    d = load()
    rho, pv = stats.spearmanr(d.dc_gw, d.local_share)
    import matplotlib.pyplot as plt
    w = PANEL_W3 * (2.6 if large else 1.0)
    fig, ax = plt.subplots(figsize=(w, height * (2.6 if large else 1.0)), dpi=600)

    x = np.maximum(d.dc_gw.to_numpy(float), 3e-4)   # DC and WV sit near zero
    ax.scatter(x, d.local_share, s=17, alpha=0.85, color=C_INFO,
               edgecolor="white", linewidth=0.35, zorder=3)

    # National mean and the two capacity-quartile medians.  Region is NOT a
    # useful encoding here: an ANOVA of lambda on Census region gives F = 0.58,
    # P = 0.63, and region explains 3.7% of the variance.  Colouring by region
    # would suggest a regional structure the data do not have.
    mean_lam = float(d.local_share.mean())
    lo = d[d.dc_gw <= d.dc_gw.quantile(.25)].local_share.median()
    hi = d[d.dc_gw >= d.dc_gw.quantile(.75)].local_share.median()
    for yv, ls, col, lab in ((hi, ":", "0.45", "top quartile"),
                             (lo, ":", "0.45", "bottom quartile"),
                             (mean_lam, "--", C_INFO, "mean")):
        ax.axhline(yv, color=col, lw=0.85, ls=ls, zorder=2)
        ax.text(0.012, yv + 0.006, f"{lab} {yv:.2f}",
                transform=ax.get_yaxis_transform(), va="bottom", ha="left",
                fontsize=FS_ANNOT, color=col)

    if large:
        # Every unit named, at 2.6x the manuscript size.  The manuscript panel
        # carries no state labels: at 61 mm they overlap the markers and each
        # other whatever the placement algorithm does.
        label_beside(ax, np.maximum(d.dc_gw, 3e-4), d.local_share, d.state_abbr,
                     sizes=17, size=7.0)

    ax.set_xscale("log")
    ax.set_xlabel("Data center capacity, 2024 (GW)", fontsize=FS_TICK, labelpad=2)
    ax.set_ylabel("Share of data center employment\ndriven by in-state data centers",
                  fontsize=FS_TICK, labelpad=2)
    ax.set_ylim(0.40, 0.86)
    tidy(ax)
    ax.text(0.975, 0.05, f"n = {len(d)} units\nrank correlation {rho:.2f}, {_P(pv)}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=FS_ANNOT,
            color="0.3", linespacing=1.4)
    if os.environ.get("CHECK"):
        import matplotlib.pyplot as _p
        pass  # debug savefig removed for release
    save(fig, "f05a_spillover_labeled" if large else "f05a_spillover")
    return d, rho, pv, lo, hi


if __name__ == "__main__":
    draw(large=True)
    d, rho, pv, lo, hi = draw()
    print(f"n={len(d)}  rho={rho:.4f}  p={pv:.3g}")
    print(f"lambda: min {d.local_share.min():.3f} max {d.local_share.max():.3f} "
          f"median {d.local_share.median():.3f}")
    print(f"quartile medians: lowest {lo:.4f}  highest {hi:.4f}  "
          f"-> lowest quartile imports {100*(1-lo):.0f}% of its 5182 employment")
