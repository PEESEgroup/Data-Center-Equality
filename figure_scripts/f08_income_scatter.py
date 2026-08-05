"""
Income versus energy-burden change: scatter, bar chart and legend.
x = burden_diff_%, y = income_per_unit, colour = AMI bracket.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import proplot as pplt

DATA_DIR = Path("./rider/income")
OUTDIR   = Path("./figures/04")
YEARS    = [2025, 2030]

AMI_ORDER = ["0-30%", "30-60%", "60-80%", "80-100%", "100-150%", "150%+"]

ppt_rc = {"font.size": 7}
pplt.rc.update(ppt_rc)


def draw_scatter(year: int):
    df = pd.read_excel(DATA_DIR / f"income_ami150_{year}.xlsx")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["burden_diff_%", "income_per_unit"])

    x = df["burden_diff_%"].to_numpy(dtype=float)
    y = df["income_per_unit"].to_numpy(dtype=float)
    df["AMI150"] = df["AMI150"].astype(str)
    cats = [c for c in AMI_ORDER if c in df["AMI150"].unique()]

    fig_width = 8.27 / 4 * 1.5
    fig, ax = pplt.subplots(figsize=(fig_width, 1.8), dpi=600)

    colors = pplt.get_colors("tokyo", len(cats))
    for color, cat in zip(colors, cats):
        sub = df[df["AMI150"] == cat]
        ax.scatter(sub["burden_diff_%"], sub["income_per_unit"],
                   s=2, color=color, edgecolor="none", alpha=0.5)

    bins_edge = [0, 30000, 70000, 100000, 200000]
    ax.format(
        xlabel="Extra Energy Burden Due to Data Centers (%)",
        ylabel="Annual Income ($)",
        grid=False,
        xlim=(0, x.max() + 0.5),
        xlocator=np.arange(0, int(x.max()) + 2, 2),
        xtickminor=False,
        ylim=(y.min() - 5000, 200000),
        ylocator=bins_edge,
        yformatter="sci",
        ytickminor=False,
    )

    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTDIR / f"f08_income_scatter_{year}.png"
    fig.savefig(out, dpi=1200, bbox_inches="tight", transparent=True)
    pplt.close(fig)
    print(f"  -> {out}")


def draw_barh():
    bins = [(0, 30000), (30000, 69999), (70000, 99999), (100000, 200000)]
    values = np.array([28., 35., 43., 55.])
    y_centers = np.array([(lo + hi) / 2 for lo, hi in bins])

    colors = pplt.get_colors("tokyo", len(values))

    fig_width = 8.27 / 4 * 0.5
    fig, ax = pplt.subplots(figsize=(fig_width, 1.8), dpi=600)

    ax.barh(y_centers, values, width=20000, absolute_width=True,
            color=colors, edgecolor="none", alpha=0.7)
    ax.format(xlabel="AI Usage (%)", ylim=(-5000, 200000),
              grid=False, yformatter="sci")
    ax.set_yticks([])

    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTDIR / "f08_income_barh.svg"
    fig.savefig(out, bbox_inches="tight")
    pplt.close(fig)
    print(f"  -> {out}")


def draw_legend():
    colors = pplt.get_colors("tokyo", len(AMI_ORDER))

    fig_width = 8.27 / 4 * 1.5
    fig, ax = pplt.subplots(figsize=(fig_width, 1.8), dpi=600)

    handles = []
    for color, label in zip(colors, AMI_ORDER):
        h = ax.scatter([], [], s=20, color=color, edgecolor="none")
        handles.append(h)

    ax.legend(
        handles, AMI_ORDER,
        loc="center", ncols=3, frame=False,
        title="Area Median Income (AMI)",
        handletextpad=0.6, columnspacing=1.2, alpha=0.5,
    )

    ax.set_axis_off()
    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")

    out = OUTDIR / "f08_income_legend.svg"
    fig.savefig(out, bbox_inches="tight")
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
    print("done")


if __name__ == "__main__":
    main()
