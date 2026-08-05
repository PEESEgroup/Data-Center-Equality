#!/usr/bin/env python3
"""
f11_waterfall.py — Self-generation incentive: purchase price vs LCOE
====================================================================
Horizontal stacked bars for each ISO's purchase price (wholesale +
transmission), plus horizontal error-bar ranges for three generation
technologies.

Usage:
    python f11_waterfall.py
"""

import numpy as np
import pandas as pd
import proplot as pplt
import matplotlib.patches as mpatches
from pathlib import Path

TABLE3 = Path("./results/r3_summary/table3_zone_impact_2025.csv")
TABLE4 = Path("./results/r3_summary/table4_zone_relative_2025.csv")
TABLE5 = Path("./results/r3_summary/table5_zone_projected_2030.csv")
FIG_DIR = Path("./figures/06")

FONTSIZE = 7
A4W = 8.27
pplt.rc.update({"font.size": FONTSIZE})

ISO_ORDER = ["PJM", "ERCOT", "MISO", "CAISO"]
ZONES = {"PJM": "DOM", "ERCOT": "North", "CAISO": "PGE", "MISO": "LRZ 8, 9, 10"}
ZONE_FULL = {
    "PJM": "PJM:\nDOM",
    "ERCOT": "ERCOT:\nNorth",
    "CAISO": "CAISO:\nPGE",
    "MISO": "MISO:\nLRZ 8–10",
}

TSR_MWH = {"PJM": 9.73, "ERCOT": 9.53, "CAISO": 14.0, "MISO": 7.9}

LCOE = {
    "Natural Gas\n(Combined Cycle)": (48, 107),
    "Solar +\nBattery Storage":      (60, 120),
    "Small Modular\nReactor":        (89, 200),
}
LCOE_COLORS = ["#888888", "#f0b753", "#b07aa1"]
LCOE_MARKERS = ["o", "s", "D"]

C_PW  = "#4c78a8"
C_TSR = "#f58518"


def _barh(ax, y, w, h, left=0.0, **kw):
    """Proplot barh wrapper — ensures numpy types."""
    return ax.barh(np.atleast_1d(y), np.atleast_1d(w), np.float64(h),
                   left=np.atleast_1d(left), **kw)


def load_pw():
    t3 = pd.read_csv(TABLE3)
    t4 = pd.read_csv(TABLE4)
    t5 = pd.read_csv(TABLE5)
    pw = {}
    for iso, zone in ZONES.items():
        pb = t4.loc[(t4.Region == iso) & (t4.Zone == zone), "P_base_dollar_MWh"].iloc[0]
        d3 = t3.loc[(t3.Region == iso) & (t3.Zone == zone), "dP_dollar_MWh"].iloc[0]
        d5 = t5.loc[(t5.Region == iso) & (t5.Zone == zone), "dP_dollar_MWh"].iloc[0]
        pw[iso] = pb + d3 + d5
    return pw


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pw_dict = load_pw()

    n_iso = len(ISO_ORDER)
    n_lcoe = len(LCOE)

    fig, ax = pplt.subplots(figwidth=A4W * 0.3, figheight=3.0)

    bh = 0.7

    # ── ISO purchase price bars (compact group at top) ──
    iso_sp = 0.9
    iso_ys = np.arange(n_iso)[::-1] * iso_sp + n_lcoe * 0.85 + 0.4
    for i, iso in enumerate(ISO_ORDER):
        y = iso_ys[i]
        pw = pw_dict[iso]
        tsr = TSR_MWH[iso]
        total = pw + tsr
        _barh(ax, y, pw, bh, color=C_PW, ec="none")
        _barh(ax, y, tsr, bh, left=pw, color=C_TSR, ec="none")

    # ── LCOE error bars (compact group at bottom) ──
    lcoe_sp = 0.85
    lcoe_ys = np.arange(n_lcoe)[::-1] * lcoe_sp
    for j, ((name, (lo, hi)), clr, mk) in enumerate(
            zip(LCOE.items(), LCOE_COLORS, LCOE_MARKERS)):
        y = lcoe_ys[j]
        mid = (lo + hi) / 2.0
        ax.errorbar(mid, y, xerr=[[mid - lo], [hi - mid]],
                    fmt=mk, color=clr, ms=5, capsize=3, capthick=0.8,
                    elinewidth=1.2, zorder=5)

    # ── Vertical dashed lines from LCOE lo/hi up to ISO bars ──
    y_top = iso_ys[0] + bh / 2
    y_bot = lcoe_ys[-1] - bh / 2
    for (name, (lo, hi)), clr in zip(LCOE.items(), LCOE_COLORS):
        ax.plot([lo, lo], [y_bot, y_top], color=clr, lw=0.6, ls="--",
                alpha=0.5, zorder=1)
        ax.plot([hi, hi], [y_bot, y_top], color=clr, lw=0.6, ls="--",
                alpha=0.5, zorder=1)

    # ── Y-axis labels ──
    all_ys = list(iso_ys) + list(lcoe_ys)
    all_labels = [ZONE_FULL[iso] for iso in ISO_ORDER] + list(LCOE.keys())
    ax.set_yticks(all_ys)
    ax.set_yticklabels(all_labels, fontsize=7)

    ax.format(xlim=(0, 215), grid=False,
              xlabel="$/MWh")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("0.2")

    # ── Legend ──
    handles = [
        mpatches.Patch(fc=C_PW,  ec="none", label="Wholesale Price"),
        mpatches.Patch(fc=C_TSR, ec="none", label="Transmission"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              frameon=False, handlelength=1.0, handletextpad=0.4,
              ncols=1)

    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)

    out_svg = FIG_DIR / "selfgen.svg"
    out_png = FIG_DIR / "selfgen.png"
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")
    pplt.close(fig)


if __name__ == "__main__":
    main()
