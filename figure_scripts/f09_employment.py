"""
Figure 9: Employment Effects of Data Centers
(a) State-level OLS forest — heterogeneity & spillover
(b) National Panel FE — employment per GW (4 specs)
(c) Construction Panel FE — spillover causality (3 NAICS)
(d) Substitution — NAICS 5182 vs 517
(e) Fiscal return — Δ labor income vs subsidy
"""
from pathlib import Path
import numpy as np
import pandas as pd
import proplot as pplt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

DATA_DIR = Path("./employment")
OUTDIR   = Path("./figures_revised/05")
HALF_W   = 8.27 * 0.9 / 2.0
PANEL_H  = 1.8

FIPS_TO_ST = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT",
    10: "DE", 11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID",
    17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA",
    23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS",
    29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH", 34: "NJ",
    35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK",
    41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY",
}

pplt.rc.update({"font.size": 7})


# ===================== (a) Forest Plot =====================

REGIONS = [
    ("Northeast",         ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA"]),
    ("South Atlantic",    ["DE", "MD", "VA", "WV", "NC", "SC", "GA", "FL"]),
    ("East South Central",  ["KY", "TN", "AL", "MS"]),
    ("West South Central",  ["AR", "LA", "OK", "TX"]),
    ("East North Central",  ["OH", "IN", "IL", "MI", "WI"]),
    ("West North Central",  ["MN", "IA", "MO", "ND", "SD", "NE", "KS"]),
    ("Mountain",          ["MT", "WY", "ID", "CO", "NM", "UT", "AZ", "NV"]),
    ("Pacific",           ["WA", "OR", "CA", "AK", "HI"]),
]
REGION_COLORS = {
    "Northeast":         "#4e79a7",
    "South Atlantic":    "#e15759",
    "East South Central":  "#f1ce63",
    "West South Central":  "#f28e2b",
    "East North Central":  "#59a14f",
    "West North Central":  "#8cd17d",
    "Mountain":          "#b07aa1",
    "Pacific":           "#76b7b2",
}


def _load_ols():
    df = pd.read_excel(
        DATA_DIR / "02_ols_employment_dc_capacity.xlsx",
        sheet_name="Raw OLS by Capacity",
        header=None, skiprows=2,
    )
    df = df.iloc[:, :12]
    df.columns = ["st", "name", "n", "cap_min", "cap_max",
                  "beta", "se", "ci_lo", "ci_hi", "intercept", "r2", "p"]
    df = df.dropna(subset=["st"])
    df = df[df["st"].str.match(r"^[A-Z]{2}$")].copy()
    for c in ("cap_max", "beta", "se", "ci_lo", "ci_hi", "p"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["beta"])
    st_to_reg = {}
    for reg, sts in REGIONS:
        for s in sts:
            st_to_reg[s] = reg
    df["region"] = df["st"].map(st_to_reg)
    df["sig"] = df["p"] < 0.05

    ordered = []
    for is_sig in [True, False]:
        for reg_name, reg_states in REGIONS:
            for st in reg_states:
                mask = (df["st"] == st) & (df["sig"] == is_sig)
                if mask.any():
                    ordered.append(df[mask].iloc[0])
    return pd.DataFrame(ordered).reset_index(drop=True)


def _draw_a(df_plot, scale):
    n = len(df_plot)
    n_sig = int(df_plot["sig"].sum())

    fig, ax = pplt.subplots(figsize=(HALF_W * 2 + 0.6, 2.0), dpi=600)
    x = np.arange(n)

    # --- right Y: DC capacity (log, drawn first → behind) ---
    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    cap = np.maximum(df_plot["cap_max"].to_numpy(dtype=float), 1e-4)
    cap_floor = 1e-4
    ax2.vlines(x, cap_floor, cap, linewidth=4, color="lightgray", alpha=0.35)

    ax2.set_yscale("log")
    ax2.set_ylim(cap_floor, cap.max() * 3)
    ax2.set_ylabel("DC Capacity (GW)")
    ax2.tick_params(labelsize=5, which="both")
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # --- left Y: forest plot (employment, drawn on top) ---
    for i, (_, row) in enumerate(df_plot.iterrows()):
        c = REGION_COLORS[row["region"]]
        fc = c if row["sig"] else "none"
        ax.plot([x[i], x[i]], [row["ci_lo"], row["ci_hi"]],
                color=c, lw=0.8, zorder=3)
        ax.scatter(x[i], row["beta"], s=25, facecolor=fc, edgecolor=c,
                   linewidths=0.8, zorder=5)

    if n_sig < n:
        ax.axvline(n_sig - 0.5, color="gray", lw=0.5, ls=":", zorder=1)

    ax.axhline(0, color="gray", lw=0.5, ls="--", zorder=1)
    pooled = 1811.0
    ax.axhline(pooled, color="steelblue", lw=0.8, ls=":", zorder=2)
    ax.text(n - 0.3, pooled, f" Average: {pooled:,.0f}",
            color="steelblue", va="bottom", ha="right")

    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=1000)

    ax.set_xlim(-0.8, n + 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["st"].tolist(), rotation=45, ha="right")
    ax.format(ylabel="Direct Data Center\nEmployment (jobs/GW)", grid=False,
              xtickminor=False, ytickminor=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("none")

    handles = [
        ax.scatter([], [], s=20, facecolor=REGION_COLORS[r], edgecolor=REGION_COLORS[r],
                   linewidths=0.5)
        for r, _ in REGIONS
    ]
    ax.legend(handles, [r for r, _ in REGIONS],
              ncols=4, frame=False, loc="t")

    fig.patch.set_facecolor("none")

    out = OUTDIR / f"f09a_forest_{scale}.svg"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    pplt.close(fig)
    print(f"  -> {out}")


def draw_panel_a():
    df_plot = _load_ols()
    _draw_a(df_plot, "linear")
    _draw_a(df_plot, "symlog")


# ===================== (b) National Multiplier =====================
def draw_panel_b():
    df = pd.read_excel(
        DATA_DIR / "02_ols_employment_dc_capacity.xlsx",
        sheet_name="Panel FE", header=None, skiprows=2,
    )
    df = df.iloc[:4, :8]
    df.columns = ["spec", "xvar", "beta", "se", "ci_lo", "ci_hi", "p", "r2"]
    for c in ("beta", "ci_lo", "ci_hi", "p"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    label_map = {
        0: "Direct State\n(with spillover)",
        1: "Direct Local",
        2: "Total State\n(multiplier + spillover)",
        3: "Total Local\n(multiplier effects)",
    }
    df["label"] = df.index.map(label_map)
    df = df.sort_values("beta", ascending=True).reset_index(drop=True)

    fig, ax = pplt.subplots(figsize=(HALF_W * 2 / 3, PANEL_H), dpi=600)

    y = np.arange(4)
    colors = ["#377eb8", "#4daf4a", "#ff7f00", "#e41a1c"]

    xmax = float(df["ci_hi"].max())
    for i in range(4):
        row = df.iloc[i]
        sig = "***" if row["p"] < 0.01 else "**" if row["p"] < 0.05 else "*" if row["p"] < 0.10 else ""
        ax.barh(y[i], row["beta"], width=0.6, absolute_width=True,
                color=colors[i], alpha=0.7, edgecolor="none")
        ax.plot([row["ci_lo"], row["ci_hi"]], [y[i], y[i]],
                color="k", lw=0.8, zorder=5)
        ax.scatter([row["ci_lo"], row["ci_hi"]], [y[i], y[i]],
                   s=8, color="k", zorder=5, marker="|")
        ax.text(row["ci_hi"] + xmax * 0.02, y[i],
                f'{row["beta"]:,.0f}{sig}', va="center")

    ax.set_xlim(None, xmax * 1.25)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist())
    ax.format(xlabel="Employment per GW (jobs)", grid=False,
              ytickminor=False, xtickminor=False)
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")

    out = OUTDIR / "f09b_multiplier.svg"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    pplt.close(fig)
    print(f"  -> {out}")


# ===================== (c) Construction =====================
def draw_panel_c():
    df = pd.read_excel(
        DATA_DIR / "04_construction_analysis.xlsx",
        sheet_name="Panel FE", header=None, skiprows=2,
    )
    df = df.iloc[:3, :8]
    df.columns = ["spec", "xvar", "beta", "se", "ci_lo", "ci_hi", "p", "r2"]
    for c in ("beta", "ci_lo", "ci_hi", "p"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    labels = [
        "Construction\n(NAICS 23)",
        "Nonresidential\n(NAICS 2362)",
        "Computer Design\n(NAICS 5415)",
    ]

    fig, ax = pplt.subplots(figsize=(HALF_W * 2 / 3, PANEL_H), dpi=600)

    y = np.arange(3)
    colors_c = ["#4daf4a" if p < 0.05 else "#999999" for p in df["p"]]

    xmax = 0
    for i in range(3):
        row = df.iloc[i]
        sig = "***" if row["p"] < 0.01 else "**" if row["p"] < 0.05 else "*" if row["p"] < 0.10 else ""
        ax.barh(y[i], row["beta"], width=0.5, absolute_width=True,
                color=colors_c[i], alpha=0.7, edgecolor="none")
        ax.plot([row["ci_lo"], row["ci_hi"]], [y[i], y[i]],
                color="k", lw=0.8, zorder=5)
        ax.scatter([row["ci_lo"], row["ci_hi"]], [y[i], y[i]],
                   s=8, color="k", zorder=5, marker="|")
        xmax = max(xmax, row["ci_hi"])
        ax.text(row["ci_hi"] + xmax * 0.02, y[i],
                f'{row["beta"]:,.0f}{sig}', va="center")

    ax.set_xlim(None, xmax * 1.25)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.format(xlabel="Employment per GW (jobs)", grid=False,
              ytickminor=False, xtickminor=False)
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")

    out = OUTDIR / "f09c_construction.svg"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    pplt.close(fig)
    print(f"  -> {out}")


# ===================== (d) Substitution =====================
def _load_substitution_data(sig_states):
    qwi = pd.read_csv(DATA_DIR / "qwi_all_naics_annual.csv")
    qwi["st"] = qwi["state"].map(FIPS_TO_ST)
    qwi = qwi[qwi["st"].isin(sig_states)].copy()
    qwi["naics"] = qwi["naics"].astype(str)

    rows = []
    for st in sig_states:
        sub = qwi[qwi["st"] == st]
        for naics in ["5182", "517"]:
            s = sub[sub["naics"] == naics]
            e16 = s.loc[s["year"] == 2016, "Emp_annual_avg"]
            e24 = s.loc[s["year"] == 2024, "Emp_annual_avg"]
            if e16.empty or e24.empty:
                continue
            rows.append({"st": st, "naics": naics,
                         "delta": float(e24.iloc[0]) - float(e16.iloc[0])})

    df = pd.DataFrame(rows)
    d5182 = df[df["naics"] == "5182"].set_index("st")["delta"].rename("d5182")
    d517 = df[df["naics"] == "517"].set_index("st")["delta"].rename("d517")
    out = pd.concat([d5182, d517], axis=1).dropna()
    out = out.reset_index()

    st_to_reg = {}
    for reg, sts in REGIONS:
        for s in sts:
            st_to_reg[s] = reg
    out["region"] = out["st"].map(st_to_reg)
    reg_order = [r for r, _ in REGIONS]
    out["reg_idx"] = out["region"].map(lambda r: reg_order.index(r))
    out = out.sort_values(["reg_idx", "d5182"], ascending=[True, True]).reset_index(drop=True)
    return out


def draw_panel_d():
    ols = _load_ols()
    sig_states = ols[ols["sig"]]["st"].tolist()

    df = _load_substitution_data(sig_states)
    n = len(df)

    # Region boundaries for horizontal separator lines
    reg_boundaries = []
    prev_reg = None
    for i, row in df.iterrows():
        if prev_reg is not None and row["region"] != prev_reg:
            reg_boundaries.append(i - 0.5)
        prev_reg = row["region"]

    fig, ax = pplt.subplots(figsize=(HALF_W * 2 / 3, PANEL_H * 2), dpi=600)

    y = np.arange(n)
    bh = 0.6

    ax.barh(y, df["d5182"].to_numpy(), width=bh, absolute_width=True,
            color="#377eb8", alpha=0.7, edgecolor="none",
            label="NAICS 5182 (Data Center)")
    ax.barh(y, df["d517"].to_numpy(), width=bh, absolute_width=True,
            color="#e41a1c", alpha=0.7, edgecolor="none",
            label="NAICS 517 (Telecom)")

    for b in reg_boundaries:
        ax.axhline(b, color="gray", lw=0.3, ls=":", alpha=0.5)

    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["st"].tolist())
    ax.set_ylim(-0.8, n - 0.2)
    ax.format(xlabel="Employment Change 2016–2024", grid=False,
              ytickminor=False, xtickminor=False)
    ax.legend(ncols=2, frame=False, loc="b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")

    out = OUTDIR / "f09d_substitution.svg"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    pplt.close(fig)
    print(f"  -> {out}")


# ===================== (e) Fiscal Return =====================
def _load_fiscal():
    import matplotlib.cm as mcm
    from matplotlib.colors import LinearSegmentedColormap

    df = pd.read_excel(
        DATA_DIR / "05_labor_income_vs_subsidy.xlsx",
        sheet_name="Incremental LI vs Subsidy",
        header=None, skiprows=2,
    )
    df = df.iloc[:, :14]
    df.columns = ["st", "name", "emp20", "emp24", "d_raw", "ls", "d_clean",
                  "earn", "li_mult", "d_dli", "d_tli", "subsidy", "ratio", "net"]
    df = df.dropna(subset=["st"])
    for c in ("d_tli", "subsidy", "ratio"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["subsidy", "d_tli"])

    eb = pd.read_csv(Path("./rider/compare/state_poverty_share_2025.csv"))
    eb = eb.rename(columns={"state_abbr": "st"})
    df = df.merge(eb[["st", "share_diff_gt6_state"]], on="st", how="left")
    df["eb_pct"] = df["share_diff_gt6_state"] * 100

    df = df[(df["subsidy"] >= 1) & (df["d_tli"] > 0)].copy()

    gdp_df = pd.read_excel(Path("./rider/compare/merged_states_2025.xlsx"))
    gdp_map = dict(zip(gdp_df["abbr"], gdp_df["Nominal_GDP_per_capita_2024_USD"]))
    df["gdp"] = df["st"].map(gdp_map)
    df = df.dropna(subset=["gdp"])

    base = mcm.get_cmap("OrRd")
    cmap = LinearSegmentedColormap.from_list("OrRd_trim", base(np.linspace(0.2, 1.0, 256)))
    vmin_eb, vmax_eb = 0, float(np.nanmax(df["eb_pct"])) * 1.05
    norm_eb = mcolors.Normalize(vmin=vmin_eb, vmax=vmax_eb)
    return df, cmap, norm_eb


def _draw_fiscal(df, cmap, norm_eb, labeled, figsize, out_path):
    import matplotlib.cm as mcm
    import matplotlib.pyplot as plt

    SIZE_MIN, SIZE_MAX = 20, 500
    gdp = df["gdp"].to_numpy(dtype=float)
    gdp_norm = (gdp - gdp.min()) / (gdp.max() - gdp.min() + 1e-9)
    sizes = SIZE_MIN + gdp_norm * (SIZE_MAX - SIZE_MIN)

    fig, ax = plt.subplots(figsize=figsize, dpi=600)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    colors = cmap(norm_eb(np.nan_to_num(df["eb_pct"].to_numpy(), nan=0)))
    ax.scatter(df["subsidy"].to_numpy(), df["d_tli"].to_numpy(),
               s=sizes, c=colors, edgecolor="none", zorder=5, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")

    xmin = df["subsidy"].min() * 0.3
    xmax = df["subsidy"].max() * 2.5
    ymin = df["d_tli"].min() * 0.3
    ymax = df["d_tli"].max() * 2.0
    ax.plot([xmin, xmax], [xmin, xmax], color="gray", ls="--", lw=0.8, zorder=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if labeled:
        from adjustText import adjust_text
        texts = []
        for _, row in df.iterrows():
            texts.append(ax.text(row["subsidy"], row["d_tli"], row["st"],
                                 va="center", ha="left", fontsize=7))
        adjust_text(texts, ax=ax)

        sm = mcm.ScalarMappable(norm=norm_eb, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("Energy Burden\nIncrease (%)", fontsize=7)
        cb.ax.tick_params(labelsize=7)

    ax.set_xlabel("Cumulative Tax Incentive\n2020–2024 ($M)", fontsize=7)
    ax.set_ylabel("Labor Income\nIncrease ($M)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.tick_params(which="minor", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  -> {out_path}")

    if not labeled:
        fig_leg, ax_leg = plt.subplots(figsize=(figsize[0], 0.7), dpi=300)
        ax_leg.set_facecolor("none")
        fig_leg.patch.set_facecolor("none")
        ax_leg.axis("off")

        sm = mcm.ScalarMappable(norm=norm_eb, cmap=cmap)
        cb = fig_leg.colorbar(sm, ax=ax_leg, orientation="horizontal",
                              fraction=0.8, pad=0.05, aspect=30)
        cb.outline.set_linewidth(0)
        cb.ax.tick_params(labelsize=7, width=0.5)
        cb.set_label("Energy Burden Increase (%)", fontsize=7)

        gdp_levels = [75000, 90000, 105000]
        s_levels = [SIZE_MIN + (g - gdp.min()) / (gdp.max() - gdp.min() + 1e-9) * (SIZE_MAX - SIZE_MIN)
                    for g in gdp_levels]
        x_start = 0.15
        for g, s in zip(gdp_levels, s_levels):
            ax_leg.scatter(x_start, 0.85, s=s, facecolor="gray", edgecolor="none",
                           transform=ax_leg.transAxes, clip_on=False)
            label = f"${g/1000:.0f}k"
            ax_leg.text(x_start, 0.55, label, ha="center", va="top",
                        transform=ax_leg.transAxes, fontsize=6)
            x_start += 0.12

        leg_path = out_path.parent / (out_path.stem + "_legend.svg")
        fig_leg.savefig(leg_path, dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig_leg)
        print(f"  -> {leg_path}")


def draw_panel_e():
    df, cmap, norm_eb = _load_fiscal()
    _draw_fiscal(df, cmap, norm_eb, labeled=False,
                 figsize=(HALF_W * 2 / 3, PANEL_H),
                 out_path=OUTDIR / "f09e_fiscal.svg")
    _draw_fiscal(df, cmap, norm_eb, labeled=True,
                 figsize=(HALF_W * 2, PANEL_H * 2),
                 out_path=OUTDIR / "f09e_fiscal_labeled.svg")


# ===================== main =====================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("(a) Forest plot ...")
    draw_panel_a()
    print("(b) National multiplier ...")
    draw_panel_b()
    print("(c) Construction ...")
    draw_panel_c()
    print("(d) Substitution ...")
    draw_panel_d()
    print("(e) Fiscal return ...")
    draw_panel_e()
    print("Done!")


if __name__ == "__main__":
    main()
