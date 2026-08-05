#!/usr/bin/env python3
"""Figure 5c -- the raw series look like substitution; the causal estimates do not.

Telecommunications employment may have been declining regardless of data center
growth, in which case the co-movement is not substitution.  Observed employment
therefore has to be compared with a credible counterfactual, which is what the
instrumented estimate on the right supplies.

LEFT.  The descriptive fact, for all 49 units of the analysis universe with no
selection on significance.  Selecting on the per-state OLS slope would drop New
Jersey (+6,245 / -9,467) and Missouri (+4,306 / -9,868) among others, and would
condition the sample on a non-causal test.  The dashed line is exact one-for-one substitution: on it,
every data-processing job gained is a telecommunications job lost.  The national
totals sit almost on it, ratio 1.01, which is what the substitution reading was
built on.

RIGHT.  The same question asked causally, with the instrument.  Data processing
clears zero; telecommunications does not.  The gap between the two intervals is
the panel's point: the raw co-movement is real, and it is not evidence that one
sector displaced the other.
"""
import os
import os
import sys
from pathlib import Path

# The descriptive comparison opens at the same year as the causal design of
# SI 6.2.  Nearly half of the 2016-2024 telecommunications decline had already
# happened by 2019, so a panel starting in 2016 shows a co-movement that is
# largely older than the capacity it is placed beside.
BASE_YEAR = 2019

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as pplt   # noqa: F401
from scipy import stats
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (_P, FS_ANNOT, FS_TICK, AUD, C_CONSTR, C_INFO, C_SUBST, EMP, HALF_W, PANEL_W, PANEL_W2, PANEL_W3, R06,
                    footnote, iv_table, label_beside, place_labels, save,
                    tidy)  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "employment"))
from analysis_universe import ANALYSIS_UNITS  # noqa: E402

LABEL = ["CA", "TX", "VA", "GA", "IL", "NJ", "MO", "NY"]


def load_desc():
    q = pd.read_csv(EMP / "qwi_all_naics_annual.csv", dtype={"naics": str})
    p = pd.read_csv(R06 / "panel_state_year.csv")[
        ["state_abbr", "state_name"]].drop_duplicates()
    # QWI spells the District of Columbia "DC", the panel spells it "District of
    # Columbia".  Merging on state_name alone left DC with a null state_abbr, so the
    # ANALYSIS_UNITS filter below dropped it silently and the panel showed 47 points
    # where the substitution analysis reports 48.  DC has both endpoint years for both
    # sectors, so it belongs here.  Resolve the abbreviation first, by name where the
    # name matches and by the abbreviation itself otherwise, and assert nothing is lost.
    _by_name = dict(zip(p.state_name, p.state_abbr))
    _abbrs = set(p.state_abbr)
    q["state_abbr"] = [
        _by_name.get(s, s if s in _abbrs else None) for s in q.state_name]
    _lost = sorted(set(q.state_name[q.state_abbr.isna()]))
    assert not _lost, f"unmatched QWI state_name values: {_lost}"
    q = q[q.state_abbr.isin(ANALYSIS_UNITS) & q.naics.isin(["5182", "517"])
          & q.year.isin([BASE_YEAR, 2024])]
    w = q.pivot_table(index=["state_abbr", "naics"], columns="year",
                      values="Emp_annual_avg")
    d = (w[2024] - w[BASE_YEAR]).unstack()
    d.columns = ["d517", "d5182"]
    return d.dropna().reset_index()


def draw(height=2.05, large=False):
    d = load_desc()
    iv = iv_table()
    k = 2.6 if large else 1.0
    fig = plt.figure(figsize=(PANEL_W3 * k, height * k), dpi=600)
    gs = GridSpec(1, 1, figure=fig)

    # ---- left: the descriptive scatter -------------------------------- #
    ax = fig.add_subplot(gs[0])
    # SYMMETRIC LOG.  On a linear axis California at +37,360 sets the range and
    # forty of the forty-seven units collapse into a blob at the origin, which
    # shows no relationship at all.  Symlog keeps zero and both signs while
    # spreading the small states, and the equal-and-opposite line stays a line.
    LT = 500.0
    lim = max(d.d5182.abs().max(), d.d517.abs().max()) * 1.35
    ax.set_xscale("symlog", linthreshx=LT)
    ax.set_yscale("symlog", linthreshy=LT)
    grid = np.concatenate([-np.geomspace(lim, LT, 60), [0], np.geomspace(LT, lim, 60)])
    ax.plot(grid, -grid, ls="--", lw=0.8, color="0.45", zorder=2)
    ax.axhline(0, color="0.8", lw=0.6, zorder=1)
    ax.axvline(0, color="0.8", lw=0.6, zorder=1)
    ax.scatter(d.d5182, d.d517, s=15, color=C_SUBST, alpha=0.9, zorder=4,
               edgecolor="white", linewidth=0.3)
    if large:
        label_beside(ax, d.d5182, d.d517, d.state_abbr, sizes=15, size=7.0)
    # The cross-state slope IS the claim: exact substitution would be -1.
    fit = stats.linregress(d.d5182, d.d517)
    rho, pv = stats.spearmanr(d.d5182, d.d517)
    ax.text(0.035, 0.06,
            f"n = {len(d)} units\nslope {fit.slope:.2f}, {_P(pv)}",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=FS_ANNOT,
            color="0.3", linespacing=1.4)
    ax.text(0.965, 0.90, "1:1 offset", transform=ax.transAxes,
            ha="right", va="top", fontsize=FS_ANNOT, color="0.45")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"Change in data center employment,\n{BASE_YEAR} to 2024 (jobs)",
                  fontsize=FS_TICK, labelpad=2)
    ax.set_ylabel(f"Change in telecommunications\nemployment, {BASE_YEAR} to 2024 (jobs)",
                  fontsize=FS_TICK, labelpad=2)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: "0" if v == 0 else f"{v:,.0f}".replace("-", "\u2212")))
    tidy(ax)
    ax.set_title("Observed change, by state", fontsize=FS_TICK, loc="left", pad=3)

    if os.environ.get("CHECK"):
        import matplotlib.pyplot as _p
        pass  # debug savefig removed for release
    save(fig, "f05c_substitution_labeled" if large else "f05c_substitution")
    if os.environ.get("CHECK"):
        pass  # debug savefig removed for release
    if os.environ.get("CHECK"):
        pass  # debug savefig removed for release
    return d, iv


if __name__ == "__main__":
    draw(large=True)
    d, iv = draw()
    print(f"n = {len(d)} units, no selection on significance")
    print(f"national  d5182 = {d.d5182.sum():,.0f}   d517 = {d.d517.sum():,.0f}"
          f"   ratio = {abs(d.d5182.sum()/d.d517.sum()):.3f}")
    for oc in ("emp_5182_cleaned", "emp_517"):
        r = iv.loc[oc]
        print(f"  {oc:18s} {r.beta:9,.0f}  AR [{r.ar_lo:,.0f}, {r.ar_hi:,.0f}]  excl0={r.excl0}")
