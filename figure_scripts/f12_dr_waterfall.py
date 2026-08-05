#!/usr/bin/env python3
"""
f12_dr_waterfall.py — DR incentive waterfall ($/kW-yr per kW enrolled DR)
=========================================================================
Vertical waterfall for each ISO: DR compensation + transmission savings
+ wholesale savings − GPU opportunity cost = net benefit.
Unit: $/kW-yr per kW of enrolled DR capacity.
(1)(2) are fixed (independent of curtailment hours), (3)(4) vary with hours.
ERCOT shown twice: base + scarcity pricing (20h × $2,000/MWh).

Usage:
    python f12_dr_waterfall.py
"""

import numpy as np
import proplot as pplt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

FIG_DIR = Path("./figures/06")

FONTSIZE = 7
A4W = 8.27
pplt.rc.update({"font.size": FONTSIZE})

ISO_ORDER = ["PJM", "ERCOT", "CAISO", "MISO"]

DR_RATE = {"PJM": 122, "ERCOT": 66, "CAISO": 200, "MISO": 79}
TSR_KW_YR = {"PJM": 68.2, "ERCOT": 66.8, "CAISO": 98.0, "MISO": 55.5}
PW_MWH = {"PJM": 52.56, "ERCOT": 54.18, "CAISO": 58.80, "MISO": 35.73}

GPU_PER_KW = 0.603

SCENARIOS = {
    "low":  {"hours": 22, "gpu_rate": 2.5},
    "high": {"hours": 87, "gpu_rate": 3.3},
}

ERCOT_SCARCITY_HOURS = 20
ERCOT_SCARCITY_PRICE = 2000  # $/MWh (RTC+B RTSWCAP)
ERCOT_NORMAL_PRICE = 54.18   # $/MWh (same as PW_MWH["ERCOT"])

C_DR  = "#1b4332"
C_TSR = "#40916c"
C_PW  = "#95d5b2"
C_GPU = "#2b5c8a"
COMP_COLORS = [C_DR, C_TSR, C_PW, C_GPU]
COMP_LABELS = [
    "DR Compensation",
    "Transmission Savings",
    "Wholesale Savings",
    "GPU Opportunity Cost",
]

GROUP_ORDER = ["PJM", "ERCOT", "ERCOT_S", "MISO", "CAISO"]
GROUP_LABELS = ["PJM", "ERCOT", "ERCOT\n(+scarcity)", "MISO", "CAISO"]


def _bar(ax, x, h, w, bottom=0.0, **kw):
    return ax.bar(np.atleast_1d(x), np.atleast_1d(h), np.float64(w),
                  bottom=np.atleast_1d(bottom), **kw)


def compute_dr(hours, gpu_rate):
    gpu_cost = hours * GPU_PER_KW * gpu_rate

    results = {}
    for iso in ISO_ORDER:
        dr_comp = DR_RATE[iso]
        tsr_save = TSR_KW_YR[iso]
        pw_save = hours * PW_MWH[iso] / 1000
        results[iso] = [dr_comp, tsr_save, pw_save, gpu_cost]

    # ERCOT scarcity: same (1)(2)(4), different (3)
    scarcity_h = min(hours, ERCOT_SCARCITY_HOURS)
    normal_h = max(0, hours - ERCOT_SCARCITY_HOURS)
    pw_save_s = (scarcity_h * ERCOT_SCARCITY_PRICE
                 + normal_h * ERCOT_NORMAL_PRICE) / 1000
    results["ERCOT_S"] = [
        results["ERCOT"][0],
        results["ERCOT"][1],
        pw_save_s,
        gpu_cost,
    ]
    return results


def plot_waterfall(dr, tag, suffix):
    n_comp = 4
    bw = 0.25
    comp_sp = 0.30
    group_gap = 0.50

    group_xs = {}
    x_cursor = 0
    for grp in GROUP_ORDER:
        xs = [x_cursor + j * comp_sp for j in range(n_comp)]
        group_xs[grp] = xs
        x_cursor += n_comp * comp_sp + group_gap

    fw = A4W / 4
    fig, ax = pplt.subplots(figwidth=fw, figheight=3.0)

    for grp in GROUP_ORDER:
        vals = dr[grp]
        xs = group_xs[grp]

        cumsum = 0
        for j in range(3):
            _bar(ax, xs[j], vals[j], bw, bottom=cumsum,
                 color=COMP_COLORS[j], ec="none")
            cumsum += vals[j]

        subtotal = cumsum
        gpu = vals[3]
        net = subtotal - gpu

        _bar(ax, xs[3], gpu, bw, bottom=net,
             color=COMP_COLORS[3], ec="none")

        for j in range(2):
            y_conn = sum(vals[:j + 1])
            ax.plot([xs[j] + bw / 2, xs[j + 1] - bw / 2],
                    [y_conn, y_conn],
                    color="0.6", lw=0.5, ls="--", zorder=1)
        ax.plot([xs[2] + bw / 2, xs[3] - bw / 2],
                [subtotal, subtotal],
                color="0.6", lw=0.5, ls="--", zorder=1)

        sign = "+" if net >= 0 else ""
        color_net = "#1b4332" if net >= 0 else "#2b5c8a"
        arrow_x = xs[3] - bw / 2 - 0.08
        ax.plot(arrow_x, net, marker=">", color=color_net,
                ms=3, zorder=5, clip_on=False)
        ax.text(arrow_x, net - 6,
                f"{sign}${net:.1f}", fontsize=5.5,
                ha="center", va="top", color=color_net)

    iso_ticks = [np.mean(group_xs[grp]) for grp in GROUP_ORDER]
    ax.set_xticks(iso_ticks)
    ax.set_xticklabels(GROUP_LABELS, fontsize=7, rotation=90, ha="center")
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", which="both", length=0, pad=3)

    all_nets = [sum(dr[g][:3]) - dr[g][3] for g in GROUP_ORDER]
    ylo = min(0, min(all_nets) - 25)
    ax.format(ylim=(ylo, None), grid=False, ylabel="$/kW-yr")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("0.2")

    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)

    handles = [
        mpatches.Patch(fc=c, ec="none", label=l)
        for c, l in zip(COMP_COLORS, COMP_LABELS)
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              frameon=False, ncols=1)

    for fmt in ("svg", "png"):
        out = FIG_DIR / f"dr_waterfall_{suffix}.{fmt}"
        dpi = 300 if fmt == "svg" else 600
        fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
        print(f"Saved: {out}")
    pplt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for tag, params in SCENARIOS.items():
        dr = compute_dr(**params)
        plot_waterfall(dr, tag, tag)


if __name__ == "__main__":
    main()
