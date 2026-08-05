#!/usr/bin/env python3
"""Figure 5d -- incentive spending against the labor income beside it.

No filter is applied that would remove a counterexample to the pattern the
text describes:

  d_tli > 0        NOT applied: it would drop Nebraska, the only state whose
                   labor income fell
  y-axis floor     NOT applied: it would hide Oklahoma, whose 0.14 ratio the
                   text discusses
  subsidy >= 1M    applied: Florida at $0.056M and Kansas at $0.005M give
                   ratios of 53,538 and 22,216, which are division-by-almost-
                   zero artefacts, not returns

West Virginia and Kentucky are not in the incentive data at all and so cannot
appear on this panel.

LOG-LOG.  On these axes the 45-degree line is simultaneously the break-even
line and the line of proportionality, so two facts that look contradictory sit
in one picture: most states are above break-even,
AND the fitted relationship is far flatter than proportional.  The elasticity is
0.17 on the states with a programme above $1M, which is decisively below one
(t = -6.4 against the null of proportionality).
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as pplt   # noqa: F401
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (R06, AUD, FS_ANNOT, FS_TICK, C_CONSTR, C_INFO, EMP, HALF_W, PANEL_W, PANEL_W2, PANEL_W3, footnote,
                    label_beside, place_labels, save, tidy)  # noqa: E402

LABEL = ["VA", "OK", "TX", "CA", "WA", "GA", "IL", "AZ", "MD"]
MIN_SUB = 1.0


def load_extra():
    """Energy burden and capacity, for the colour and size encodings."""
    eb = pd.read_csv(EMP.parent / "rider" / "compare" / "state_poverty_share_2025.csv")
    eb = eb.rename(columns={"state_abbr": "st"})[["st", "share_diff_gt6_state"]]
    eb["eb"] = eb.share_diff_gt6_state * 100
    pan = pd.read_csv(R06 / "panel_state_year.csv")
    cap = pan[pan.year == 2024].set_index("state_abbr")["dc_gw"].rename("cap")
    return eb[["st", "eb"]], cap


def load():
    """Cumulative labor income, 2020-2024, against cumulative incentives.

    The earlier numerator was a single year's flow (jobs added by 2024 valued
    at 2024 earnings) divided by a five-year outlay.  22_labor_income_cumulative.py
    computes the quantity the manuscript actually describes, summing each year's
    additional wage bill over the window, and reproduces the earlier value first
    as a guard.
    """
    d = pd.read_csv(AUD / "labor_income_cumulative.csv")
    abbr = pd.read_csv(R06 / "panel_state_year.csv")[
        ["state_abbr", "state_name"]].drop_duplicates()
    d = d.merge(abbr, left_on="state", right_on="state_name", how="left")
    d = d.rename(columns={"state_abbr": "st", "subsidy_M": "subsidy",
                          "li_cumulative_M": "li"})[["st", "subsidy", "li"]].dropna()
    d["ratio"] = d.li / d.subsidy
    eb, cap = load_extra()
    d = d.merge(eb, on="st", how="left")
    d["cap"] = d.st.map(cap)
    return d.reset_index(drop=True)


def draw(height=2.35, large=False):
    d = load()
    main = d[(d.subsidy >= MIN_SUB) & (d.li > 0)]
    small = d[(d.subsidy < MIN_SUB) & (d.li > 0)]
    neg = d[d.li <= 0]
    fit = stats.linregress(np.log(main.subsidy), np.log(main.li))
    t_vs1 = (fit.slope - 1) / fit.stderr

    # 1.9 was not enough room for 28 state labels: several ended up nearer a
    # neighbour's marker than their own.  The oversize version exists only to be
    # read, so it can afford the extra canvas.
    k = 2.9 if large else 1.0
    fig, ax = plt.subplots(figsize=(PANEL_W2 * k, height * k), dpi=600)
    lim = (0.5, max(d.subsidy.max(), main.li.max()) * 2.2)
    ax.plot(lim, lim, ls="--", lw=0.9, color="0.4", zorder=2)
    xs = np.geomspace(main.subsidy.min(), main.subsidy.max(), 50)
    ax.plot(xs, np.exp(fit.intercept) * xs ** fit.slope, color=C_CONSTR, lw=1.2,
            zorder=3)
    # Colour carries the energy-burden increase and size carries capacity, the
    # two state characteristics the paper's equity argument turns on.  The
    # the earlier figure used this encoding and it is kept, so the panel still reads
    # as part of the same figure.
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    base = mcm.get_cmap("OrRd")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "OrRd_trim", base(np.linspace(0.25, 0.98, 256)))
    norm = mcolors.Normalize(0, np.nanmax(d.eb) * 1.05)
    # Colour carries the energy-burden increase.  nan_to_num(..., nan=0.0) used to
    # send the states with NO burden series to the bottom of the scale, painting
    # "unknown" in the colour of "no increase" -- six of the fitted states, a quarter
    # of the panel.  Missing is now its own flat grey, declared in the legend, and
    # the sub-$1M states are coloured like everyone else: being left out of the fit
    # says nothing about their energy burden.
    C_UNKNOWN = "0.72"
    def _fill(df):
        # Built with a loop, not a masked assignment: cmap() returns RGBA tuples and
        # numpy reads a list of them as a 2-D array, which will not broadcast into a
        # boolean-indexed object array.
        return [C_UNKNOWN if np.isnan(v) else cmap(norm(v))
                for v in df.eb.to_numpy(float)]
    def _size(df):
        cs = np.sqrt(np.nan_to_num(df.cap.to_numpy(float), nan=0.0))
        return 14 + 150 * cs / max(np.sqrt(np.nan_to_num(d.cap.to_numpy(float), nan=0.0)).max(), 1e-9)
    sizes = _size(main)          # still referenced by the labelling code below
    ax.scatter(main.subsidy, main.li, s=sizes, c=_fill(main),
               zorder=4, edgecolor="white", linewidth=0.4, alpha=0.92)
    # Excluded from the fit, but still observed: same fill, a grey ring instead of
    # the white one, so the reader can tell "not fitted" from "not measured".
    ax.scatter(small.subsidy.clip(lower=0.5), small.li, s=_size(small),
               c=_fill(small), zorder=4, edgecolor="0.35", linewidth=0.8,
               alpha=0.92)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", markerfacecolor=C_UNKNOWN,
               markeredgecolor="white", markeredgewidth=0.4, markersize=4.2,
               label="burden not measured"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="white",
               markeredgecolor="0.35", markeredgewidth=0.8, markersize=4.2,
               label="outside the fit"),
    ], loc="lower right", fontsize=FS_ANNOT - 1.0, frameon=False,
        handletextpad=0.35, borderpad=0.1, labelspacing=0.25)

    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.015, aspect=16)
    cb.set_label("Increase in share of households\nwith energy burden > 6% (pp)",
                 fontsize=FS_ANNOT, labelpad=2)
    cb.ax.tick_params(labelsize=5.3, width=0.4, length=2)
    cb.outline.set_linewidth(0)

    # Manual offsets, not adjustText: on these log axes with a colorbar the
    # optimiser pushed the figure to 336,526 x 354,455 pixels before failing.
    if large:
        # A single fixed offset was used here before; it stacked labels on top
        # of one another wherever two states sat close together.  label_beside
        # keeps the same point-offset idea but steps through candidate offsets
        # until the label clears everything already placed, and needs no leader.
        lab = d[d.li > 0]
        lab_sizes = np.where(lab.subsidy >= 1.0, sizes.mean(), 14.0)
        label_beside(ax, lab.subsidy.clip(lower=0.5), lab.li, lab.st,
                     sizes=lab_sizes, size=7.0)

    ax.set_xscale("log"); ax.set_yscale("log")
    # States spending under $1M are clipped to the 0.5 floor; starting the axis
    # at that same floor cut their markers in half against the spine.
    ax.set_xlim(lim[0] * 0.62, lim[1]); ax.set_ylim(main.li.min() * 0.45, main.li.max() * 2.4)
    # Nebraska has NEGATIVE cumulative labor income, so it has no position on a log
    # axis and was being computed and then silently not drawn.  It is the only state
    # that lost, which makes it the one observation a reader most needs to see; an
    # empty spot where the worst case belongs reads as if there were no worst case.
    # It is marked at the floor of the axis as off-scale, with the sign in the label,
    # so the panel shows that it exists and shows that it cannot be placed.
    for _, r in neg.iterrows():
        y0 = ax.get_ylim()[0]
        ax.scatter([r.subsidy], [y0], marker="v", s=22, facecolor="none",
                   edgecolor="0.35", linewidth=0.8, clip_on=False, zorder=6)
        ax.annotate(f"{r.st}, below axis\n(labor income fell)",
                    (r.subsidy, y0), textcoords="offset points", xytext=(-6, 1),
                    fontsize=FS_ANNOT - 0.6, color="0.35", va="bottom", ha="right",
                    annotation_clip=False)
    ax.set_xlabel("Cumulative state incentives, 2020 to 2024 ($M)",
                  fontsize=FS_TICK, labelpad=2)
    ax.set_ylabel("Cumulative labor income gain,\n2020 to 2024 ($M)",
                  fontsize=FS_TICK, labelpad=2)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    tidy(ax)
    ax.text(0.035, 0.965,
            f"n = {len(main)} units\nelasticity {fit.slope:.2f}",
            transform=ax.transAxes, va="top", fontsize=FS_ANNOT, color=C_CONSTR,
            linespacing=1.4)
    ax.annotate("break-even", xy=(0.845, 0.90), xycoords="axes fraction",
                fontsize=FS_ANNOT, color="0.4", rotation=34, ha="center")
    if os.environ.get("CHECK"):
        import matplotlib.pyplot as _p
        pass  # debug savefig removed for release
    save(fig, "f05d_fiscal_labeled" if large else "f05d_fiscal")
    return d, main, fit, t_vs1


if __name__ == "__main__":
    draw(large=True)
    d, main, fit, t1 = draw()
    print(f"n total {len(d)}, in fit {len(main)}")
    print(f"elasticity {fit.slope:.4f} (SE {fit.stderr:.4f}), R2 {fit.rvalue**2:.3f}, "
          f"p vs 0 = {fit.pvalue:.3f}, t vs 1 = {t1:.2f}")
    print(f"above break-even: {(d.ratio > 1).sum()} of {len(d)}")
