# -*- coding: utf-8 -*-
"""Cross-sectional scatter: grid metrics vs DC price impact.

Three panels explaining why ERCOT/PJM/CAISO show larger effects
while SPP/NYISO/ISONE do not.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import proplot as pplt
from scipy import stats as sp_stats

# ======================== config ========================
REGION_DATA = Path("./r3_cross_sectional/region_level_data.csv")
OUT_DIR     = Path("./figures_revised/01-2")
A4_WIDTH_IN = 8.27
FIG_WIDTH   = A4_WIDTH_IN / 3.0
FIG_HEIGHT  = FIG_WIDTH * 0.85 * (2 / 3)
FONTSIZE_PT = 7

X_OPTIONS = [
    {"col": "summer_extreme_margin_earliest",
     "label": "Extreme Reserve Margin",
     "tag": "ext_margin",
     "pct_fmt": True,
     "stats_xy": (0.97, 0.97), "stats_ha": "right", "stats_va": "top"},
    {"col": "summer_demand_90_latest",
     "label": "Peak Demand Under Extreme Conditions (GW)",
     "tag": "peak_demand",
     "pct_fmt": False,
     "stats_xy": (0.97, 0.03), "stats_ha": "right", "stats_va": "bottom"},
    {"col": "lt_lolh_2026_earliest",
     "label": "Projected Loss-of-Load Hours, 2026",
     "tag": "lolh_2026",
     "pct_fmt": False,
     "stats_xy": (0.97, 0.03), "stats_ha": "right", "stats_va": "bottom"},
]

Y_COL   = "dP_pct_max"
Y_LABEL = "Max Price Increase (%)"
Y_TAG   = "pct"

LABEL_COL = "region"
NICE_LABELS = {
    "CAISO": "CAISO", "ERCOT": "ERCOT", "MISO": "MISO",
    "PJM": "PJM", "ISONE": "ISO-NE", "NYISO": "NYISO", "SPP": "SPP",
    "Cities:NWPP-US": "NW Cities", "Cities:SERC-E": "SE-E Cities",
    "Cities:SERC-FP": "FL Cities", "Cities:SERC-SE": "SE Cities",
    "Cities:SRSG": "SW Cities",
}

INSIG_REGIONS = {"ISONE", "NYISO", "SPP"}
INSIG_NICE    = {"ISO-NE", "NYISO", "SPP"}

C_POS   = "#4477AA"
C_NEG   = "#CC6677"
C_INSIG = "#AAAAAA"

LABEL_NUDGE = {
    "ext_margin": {
        "CAISO":       ( 0,  4, "center", "bottom"),
        "ERCOT":       ( 4,  3, "left",   "bottom"),
        "PJM":         ( 0, -6, "center", "top"),
        "NW Cities":   ( 4,  0, "left",   "center"),
        "MISO":        (-4, -3, "right",  "top"),
        "SE Cities":   ( 4,  3, "left",   "bottom"),
        "SW Cities":   (-4, -3, "right",  "top"),
        "NYISO":       (-5,  4, "right",  "bottom"),
        "SE-E Cities": ( 0,  5, "center", "bottom"),
        "ISO-NE":      ( 4,  4, "left",   "bottom"),
        "FL Cities":   ( 4,  3, "left",   "bottom"),
        "SPP":         (-4,  4, "right",  "bottom"),
    },
    "peak_demand": {
        "CAISO":       ( 0,  4, "center", "bottom"),
        "ERCOT":       ( 4,  3, "left",   "bottom"),
        "NW Cities":   (-4,  3, "right",  "bottom"),
        "PJM":         (-4,  3, "right",  "bottom"),
        "MISO":        ( 4,  3, "left",   "bottom"),
        "SW Cities":   ( 0,  4, "center", "bottom"),
        "SE Cities":   ( 4, -3, "left",   "top"),
        "ISO-NE":      (-4,  4, "right",  "bottom"),
        "NYISO":       ( 4,  4, "left",   "bottom"),
        "SPP":         ( 4,  3, "left",   "bottom"),
        "FL Cities":   ( 4, -4, "left",   "top"),
        "SE-E Cities": (-4,  4, "right",  "bottom"),
    },
    "lolh_2026": {
        "CAISO":       (-4,  3, "right",  "bottom"),
        "ERCOT":       ( 4,  3, "left",   "bottom"),
        "NW Cities":   ( 4, -4, "left",   "top"),
        "PJM":         (-4,  0, "right",  "center"),
        "MISO":        ( 4,  0, "left",   "center"),
        "SW Cities":   ( 4,  3, "left",   "bottom"),
        "SE Cities":   (-4, -4, "right",  "top"),
        "ISO-NE":      ( 5,  5, "left",   "bottom"),
        "NYISO":       (-5,  5, "right",  "bottom"),
        "SPP":         ( 5, -4, "left",   "top"),
        "FL Cities":   (-5, -4, "right",  "top"),
        "SE-E Cities": ( 4,  3, "left",   "bottom"),
    },
}
DEFAULT_NUDGE = (4, 3, "left", "bottom")


def _stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _plot_one(df, xopt):
    from matplotlib.ticker import PercentFormatter

    x_col = xopt["col"]
    x_label = xopt["label"]
    x_tag = xopt["tag"]
    pct_fmt = xopt["pct_fmt"]

    pplt.rc.update(fontsize=FONTSIZE_PT)
    valid = df[[Y_COL, x_col, "nice_label", "region"]].dropna()
    x = valid[x_col].to_numpy()
    y = valid[Y_COL].to_numpy()
    labels = valid["nice_label"].tolist()
    is_insig = valid["region"].isin(INSIG_REGIONS).to_numpy()
    nudge_map = LABEL_NUDGE.get(x_tag, {})

    slope, intercept, r_val, p_val, _ = sp_stats.linregress(x, y)
    positive = r_val >= 0
    c_main = C_POS if positive else C_NEG

    fig, ax = pplt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.scatter(x[~is_insig], y[~is_insig], s=28, color=c_main, zorder=5,
               edgecolors="white", linewidths=0.4)
    ax.scatter(x[is_insig], y[is_insig], s=28, facecolors="none",
               edgecolors=C_INSIG, linewidths=1.2, zorder=5)

    x_pad = (x.max() - x.min()) * 0.05
    x_fit = np.linspace(x.min() - x_pad, x.max() + x_pad, 50)
    ax.plot(x_fit, slope * x_fit + intercept, color=c_main, lw=1.0,
            ls="--", zorder=4, alpha=0.7)

    for xi, yi, lab in zip(x, y, labels):
        dx, dy, ha, va = nudge_map.get(lab, DEFAULT_NUDGE)
        clr = C_INSIG if lab in INSIG_NICE else "0.2"
        ax.annotate(lab, (xi, yi), fontsize=5, ha=ha, va=va,
                    xytext=(dx, dy), textcoords="offset points", color=clr)

    rho, rho_p = sp_stats.spearmanr(x, y)
    ann = (f"Pearson = {r_val:+.3f}{_stars(p_val)}\n"
           f"Spearman = {rho:+.3f}{_stars(rho_p)}")
    sxy = xopt.get("stats_xy", (0.97, 0.97))
    sha = xopt.get("stats_ha", "right")
    sva = xopt.get("stats_va", "top")
    ax.annotate(ann, xy=sxy, xycoords="axes fraction",
                fontsize=5.5, ha=sha, va=sva,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="0.7", lw=0.4, alpha=0.9))

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    ax.set_xlim(x.min() - x_range * 0.15, x.max() + x_range * 0.15)
    ax.set_ylim(-y_range * 0.14, y.max() + y_range * 0.15)

    if pct_fmt:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    ax.format(xlabel=x_label, ylabel=Y_LABEL,
              xminorlocator="null", yminorlocator="null")
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("0.3")
    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)

    p = OUT_DIR / f"fig_cross_{Y_TAG}_{x_tag}.svg"
    fig.savefig(p, dpi=300, bbox_inches="tight", transparent=True)
    print(f"  saved {p}")
    pplt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REGION_DATA)
    df["nice_label"] = df[LABEL_COL].map(NICE_LABELS).fillna(df[LABEL_COL])

    for xopt in X_OPTIONS:
        _plot_one(df, xopt)


if __name__ == "__main__":
    main()
