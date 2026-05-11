#!/usr/bin/env python3
"""
f10_theta_sensitivity.py — ΔP/P vs θ (new generation coverage ratio)
=====================================================================
Shows how increasing θ (fraction of DC load growth covered by new generation)
reduces wholesale electricity price increases for the most impacted zones.

Usage:
    python f10_theta_sensitivity.py
"""

import numpy as np
import pandas as pd
import proplot as pplt
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
TABLE5_PATH = Path("./results/r3_summary/table5_zone_projected_2030.csv")
FIG_DIR = Path("./figures_revised/06")

FONTSIZE_PT = 7
A4_WIDTH_IN = 8.27
FIG_WIDTH = A4_WIDTH_IN / 4.0

pplt.rc.update({"font.size": FONTSIZE_PT})

ZONES_TO_PLOT = [
    ("PJM", "DOM"),
    ("CAISO", "PGE"),
    ("ERCOT", "North"),
    ("MISO", "LRZ 8, 9, 10"),
]

ZONE_COLORS = {
    ("PJM", "DOM"): "#1f77b4",
    ("CAISO", "PGE"): "#e6550d",
    ("ERCOT", "North"): "#2ca02c",
    ("MISO", "LRZ 8, 9, 10"): "#9467bd",
}

ZONE_LABELS = {
    ("PJM", "DOM"): "PJM: DOM",
    ("CAISO", "PGE"): "CAISO: PGE",
    ("ERCOT", "North"): "ERCOT: North",
    ("MISO", "LRZ 8, 9, 10"): "MISO: LRZ 8–10",
}

THRESHOLDS = [5, 10, 20]


# =============================================================================
# Main
# =============================================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TABLE5_PATH)

    theta = np.linspace(0, 1, 500)

    fig, ax = pplt.subplots(figwidth=FIG_WIDTH, figheight=3.0)

    # draw CI bands first (back to front by peak magnitude) then lines on top
    zone_data = []
    for iso, zone in ZONES_TO_PLOT:
        row = df[(df["Region"] == iso) & (df["Zone"] == zone)].iloc[0]
        ddc = row["dDC_2025_2030_GW"]
        p_base = row["P_base_dollar_MWh"]
        beta = row["beta"]
        beta_lo = row["dP_CI_lo"] / ddc
        beta_hi = row["dP_CI_hi"] / ddc

        dp_mean = beta * (1 - theta) * ddc / p_base * 100
        dp_lo = beta_lo * (1 - theta) * ddc / p_base * 100
        dp_hi = beta_hi * (1 - theta) * ddc / p_base * 100

        zone_data.append((iso, zone, dp_mean, dp_lo, dp_hi))

    zone_data.sort(key=lambda x: x[2][0], reverse=True)
    for iso, zone, dp_mean, dp_lo, dp_hi in zone_data:
        color = ZONE_COLORS[(iso, zone)]
        ax.fill_between(theta * 100, dp_lo, dp_hi,
                        color=color, alpha=0.12, zorder=2, linewidth=0)

    for iso, zone, dp_mean, dp_lo, dp_hi in zone_data:
        color = ZONE_COLORS[(iso, zone)]
        label = ZONE_LABELS[(iso, zone)]
        ax.plot(theta * 100, dp_mean,
                color=color, linewidth=1.2, label=label, zorder=3)

    for thr in THRESHOLDS:
        ax.axhline(thr, color="0.5", linestyle="--", linewidth=0.6, zorder=1)
        ax.text(101, thr, f"{thr}%", va="center", ha="left",
                fontsize=6, color="0.5")

    ax.format(
        xlim=(0, 100),
        ylim=(0, None),
        xlabel="Equivalent Grid-Connected\nCapacity Reduction (%)",
        ylabel="Wholesale Price\nPercentage Increase (%)",
        grid=False,
    )

    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("0.2")

    ax.legend(loc="upper right", fontsize=6, frameon=False, ncols=1)

    out_svg = FIG_DIR / "theta_sensitivity.svg"
    out_png = FIG_DIR / "theta_sensitivity.png"
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")
    pplt.close(fig)


if __name__ == "__main__":
    main()
