"""
RUCC scatter, bar chart and legend.
x = energy-poverty difference (log10), y = RUCC code, colour = rural-urban class.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import proplot as pplt

DATA_DIR = Path("./rider/rural")
OUTDIR   = Path("./figures/04")
YEARS    = [2025, 2030]

VALID_VALS = {28, 36, 44}
VALS_SORTED = sorted(VALID_VALS)

pplt.rc.update({"font.size": 7})
COLORS = pplt.Colormap("nuuk")(np.linspace(0.3, 0.9, 3))
COLOR_MAP = dict(zip(VALS_SORTED, COLORS))


def draw_scatter(year: int):
    df = pd.read_excel(DATA_DIR / f"county_rucc_{year}.xlsx")
    df = df[["log_share_diff_gt6", "RUCC_2023", "rucc_group"]].dropna()
    df = df[df["rucc_group"].isin(VALID_VALS)].copy()

    fig_width = 8.27 / 4.0 * 1.5
    fig, ax = pplt.subplots(figsize=(fig_width, 1.8), dpi=600)

    for val in VALS_SORTED:
        sub = df[df["rucc_group"] == val]
        ax.scatter(sub["log_share_diff_gt6"], sub["RUCC_2023"],
                   s=20, color=COLOR_MAP[val], alpha=0.7, edgecolor="none", label=f"{val}")

    grouped = df.groupby("rucc_group")["log_share_diff_gt6"]
    means = grouped.mean()
    q95 = grouped.quantile(0.95)

    print(f"  {year} means: {dict(10**(means.loc[VALS_SORTED]))}")
    print(f"  {year} q95:   {dict(10**(q95.loc[VALS_SORTED]))}")

    label_y = {28: 9.2, 36: 9.2, 44: 9.2}
    for i, val in enumerate(VALS_SORTED):
        c = COLOR_MAP[val]
        m_x = means.loc[val]
        q_x = q95.loc[val]
        ax.vlines(m_x, 0.5, 9.5, color=c, lw=1.2, linestyle="-")
        ax.vlines(q_x, 0.5, 9.5, color=c, lw=1.0, linestyle="--")
        m_pct = 10 ** m_x * 100
        q_pct = 10 ** q_x * 100
        y_pos = label_y[val] - i * 0.55
        ax.text(m_x, y_pos, f"{m_pct:.1f}%", ha="center", va="bottom",
                fontsize=5, color=c, fontweight="bold")
        ax.text(q_x, y_pos, f"{q_pct:.1f}%", ha="center", va="bottom",
                fontsize=5, color=c, style="italic")

    x = df["log_share_diff_gt6"].to_numpy()
    ax.format(
        xlabel="Increased Population in Energy Burden (%)",
        ylabel="Rural Urban Continuum Codes",
        grid=False,
        xlim=(x.min() - 0.5, x.max() + 0.1),
        xlocator=(-5, -4, -3, -2, -1),
        xtickminor=False,
        xticklabels=["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1", "10"],
        ylim=(0.5, 9.5),
        yticks=range(1, 10),
        yminorlocator=1,
    )

    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTDIR / f"f08_scatter_rucc_{year}.svg"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    pplt.close(fig)
    print(f"  -> {out}")


def draw_barh():
    rucc_codes = np.arange(1, 10)
    values = np.array([44, 44, 36, 36, 36, 28, 28, 28, 28])
    bar_colors = [COLOR_MAP[v] for v in values]

    fig_width = 8.27 / 4 * 0.5
    fig, ax = pplt.subplots(figsize=(fig_width, 1.8), dpi=600)

    ax.barh(rucc_codes, values, color=bar_colors, alpha=0.7)
    ax.set_yticks([])
    ax.set_xlabel("AI Usage (%)")
    ax.grid(False)
    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTDIR / "f08_rucc_barh.svg"
    fig.savefig(out, bbox_inches="tight")
    pplt.close(fig)
    print(f"  -> {out}")


def draw_legend():
    fig_width = 8.27 / 4 * 1.5
    fig, ax = pplt.subplots(figsize=(fig_width, 0.6), dpi=600)

    handles = []
    for v in VALS_SORTED:
        h = ax.scatter([], [], s=20, color=COLOR_MAP[v], marker="o")
        handles.append(h)

    ax.legend(handles, ["Rural", "Suburban", "Urban"],
              ncols=3, frame=False, loc="c")
    ax.axis("off")
    fig.patch.set_facecolor("none")

    out = OUTDIR / "f08_rucc_legend.svg"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    pplt.close(fig)
    print(f"  -> {out}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        print(f"scatter {y} ...")
        draw_scatter(y)
    print("bar chart ...")
    draw_barh()
    print("legend ...")
    draw_legend()
    print("done!")


if __name__ == "__main__":
    main()
