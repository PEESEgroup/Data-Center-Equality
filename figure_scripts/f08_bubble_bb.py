"""
州级气泡图：宽带覆盖率 vs 能源贫困差值。
颜色 = GDP per capita，大小 = DC 容量。
拟合线标注公式和 p 值。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import proplot as pplt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import matplotlib as mpl
from scipy import stats

# ===== 配置 =====
DATA_DIR = Path("./rider/compare")
OUTDIR   = Path("./figures_revised/04")
YEARS    = [2025, 2030]

COL_X    = "broadband_pct"
COL_Y    = "share_diff_gt6_state"
COL_GDP  = "Nominal_GDP_per_capita_2024_USD"
COL_ABBR = "abbr"

COLORS_CMAP = ["#004b4b", "#7ed0cf", "#fdfcf2", "#d2ab3a", "#6b3b00"]
CMAP = LinearSegmentedColormap.from_list("gold_teal", COLORS_CMAP)

SIZE_MIN, SIZE_MAX = 50, 1500
FIG_W = 8.27 / 2.4
FIG_H = 2.5

pplt.rc.update({"font.size": 8})


def draw_bubble(year: int):
    col_cap = f"Capacity_{year}"
    path = DATA_DIR / f"merged_states_{year}.xlsx"
    df = pd.read_excel(path)
    df = df.dropna(subset=[COL_X, COL_Y, COL_GDP, col_cap])

    x_raw = df[COL_X].to_numpy(dtype=float)
    x   = x_raw * 100  # 转为百分比
    y   = df[COL_Y].to_numpy(dtype=float)
    gdp = df[COL_GDP].to_numpy(dtype=float)
    cap = df[col_cap].to_numpy(dtype=float)

    cap_min, cap_max = float(cap.min()), float(cap.max())
    cap_norm = (cap - cap_min) / (cap_max - cap_min + 1e-9)
    sizes = SIZE_MIN + cap_norm * (SIZE_MAX - SIZE_MIN)

    # --- 线性回归 ---
    slope, intercept, r, p, se = stats.linregress(x, y)
    r2 = r ** 2
    print(f"  {year}: slope={slope:.6f}, intercept={intercept:.4f}, R²={r2:.4f}, p={p:.4e}")

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    # --- 绘图 ---
    fig, ax = pplt.subplots(figwidth=FIG_W, figheight=FIG_H)

    ax.scatter(x, y, s=sizes, c=gdp, cmap=CMAP, alpha=0.8,
               edgecolors="none", smin=SIZE_MIN, smax=SIZE_MAX)

    ax.plot(x_line, y_line, color="gray5", linestyle="--", linewidth=1)

    sign = "+" if intercept >= 0 else "−"
    abs_int = abs(intercept)
    formula = f"y = {slope:.4f}x {sign} {abs_int:.4f}"
    annot = f"{formula}\np = {p:.2e}"
    ax.text(0.98, 0.98, annot, transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color="gray5",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    ax.format(
        xlabel="Broadband Coverage (%)",
        ylabel="Increased Population in\nEnergy Burden (%)",
        grid=False,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlim(x.min() - 2, x.max() + 2)
    ax.set_ylim(y.min() - 0.005, y.max() + 0.005)

    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")

    vmin_g, vmax_g = float(gdp.min()), float(gdp.max())

    out = OUTDIR / f"f08_bubble_bb_{year}.svg"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    pplt.close(fig)
    print(f"  -> {out}")

    # --- 图例 ---
    fig_leg, ax_leg = pplt.subplots(figwidth=FIG_W, figheight=1.8)
    ax_leg.spines["top"].set_visible(False)
    ax_leg.spines["right"].set_visible(False)
    ax_leg.spines["bottom"].set_visible(False)
    ax_leg.spines["left"].set_visible(False)
    ax_leg.format(xlim=(0, 1), ylim=(0, 1), xlocator=[], ylocator=[], grid=False)
    ax_leg.set_facecolor("none")
    fig_leg.patch.set_facecolor("none")

    norm_g = mpl.colors.Normalize(vmin=vmin_g, vmax=vmax_g)
    sm = mpl.cm.ScalarMappable(norm=norm_g, cmap=CMAP)
    sm.set_array([])
    cbar = fig_leg.colorbar(sm, loc="top", pad=0.2, length=0.3,
                            label="Nominal GDP Per Capita (USD)")
    cbar.set_ticks([vmin_g, vmax_g])
    cbar.ax.set_xticklabels([f"{vmin_g:,.0f}", f"{vmax_g:,.0f}"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)

    def _cap_to_size(val):
        n = (val - cap_min) / (cap_max - cap_min + 1e-9)
        return SIZE_MIN + n * (SIZE_MAX - SIZE_MIN)

    cap_levels = [cap_min, 3.0, 6.0, 9.0]
    xs_demo = [0.15, 0.3, 0.45, 0.6]
    for xpos, val in zip(xs_demo, cap_levels):
        ax_leg.scatter(xpos, 0.3, s=_cap_to_size(val),
                       facecolor="gray5", alpha=0.25, edgecolor="gray5")
        ax_leg.text(xpos, 0.02, f"{val:.2f}", ha="center", va="bottom")
    ax_leg.text(0.4, 0.55, "Data Center Capacity (GW)", ha="center", va="center")

    out_leg = OUTDIR / f"f08_bubble_bb_{year}_legend.svg"
    fig_leg.savefig(out_leg, dpi=300, bbox_inches="tight")
    pplt.close(fig_leg)
    print(f"  -> {out_leg}")


def draw_bubble_labeled(year: int):
    col_cap = f"Capacity_{year}"
    path = DATA_DIR / f"merged_states_{year}.xlsx"
    df = pd.read_excel(path)
    df = df.dropna(subset=[COL_X, COL_Y, COL_GDP, col_cap, COL_ABBR])

    x_raw = df[COL_X].to_numpy(dtype=float)
    x     = x_raw * 100
    y     = df[COL_Y].to_numpy(dtype=float)
    gdp   = df[COL_GDP].to_numpy(dtype=float)
    cap   = df[col_cap].to_numpy(dtype=float)
    abbrs = df[COL_ABBR].to_numpy()

    cap_min, cap_max = float(cap.min()), float(cap.max())
    cap_norm = (cap - cap_min) / (cap_max - cap_min + 1e-9)
    sizes = SIZE_MIN + cap_norm * (SIZE_MAX - SIZE_MIN)

    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig_w_lg = 8.27 / 1.3
    fig_h_lg = 4.0
    fig, ax = pplt.subplots(figwidth=fig_w_lg, figheight=fig_h_lg)

    ax.scatter(x, y, s=sizes, c=gdp, cmap=CMAP, alpha=0.8,
               edgecolors="none", smin=SIZE_MIN, smax=SIZE_MAX, zorder=3)

    for xi, yi, lab in zip(x, y, abbrs):
        ax.text(xi, yi, str(lab), ha="center", va="center",
                fontsize=6, color="black", zorder=4)

    ax.plot(x_line, y_line, color="gray5", linestyle="--", linewidth=1, zorder=2)

    sign = "+" if intercept >= 0 else "−"
    formula = f"y = {slope:.4f}x {sign} {abs(intercept):.4f}"
    annot = f"{formula}\np = {p:.2e}"
    ax.text(0.98, 0.98, annot, transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="gray5",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    ax.format(
        xlabel="Broadband Coverage (%)",
        ylabel="Increased Population in Energy Burden (%)",
        grid=False,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlim(x.min() - 2, x.max() + 2)
    ax.set_ylim(y.min() - 0.005, y.max() + 0.005)

    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")

    vmin_g, vmax_g = float(gdp.min()), float(gdp.max())
    norm_g = mpl.colors.Normalize(vmin=vmin_g, vmax=vmax_g)
    sm = mpl.cm.ScalarMappable(norm=norm_g, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Nominal GDP per capita (2024 USD)")

    out = OUTDIR / f"f08_bubble_bb_labeled_{year}.svg"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    pplt.close(fig)
    print(f"  -> {out}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        print(f"绘制 {y} ...")
        draw_bubble(y)
        draw_bubble_labeled(y)
    print("完成!")


if __name__ == "__main__":
    main()
