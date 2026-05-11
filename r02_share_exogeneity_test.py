#!/usr/bin/env python3
"""
Bartik IV Share Exogeneity Test: National Pooled Version
========================================================
Pools all zones across all ISOs into a single cross-sectional regression
for each year-pair, with ISO fixed effects:

    ΔP_{i,τ} = α_τ + δ_{ISO(i)} + γ_τ · s_{i,0} + ε_{i,τ}

This provides substantially more statistical power than per-ISO tests
(~50-80 obs per year-pair vs. 3-17 per ISO).

Why this is the correct test:
    - The Bartik share s_{i,0} = DC_{i,2019} / DC_{national,2019} uses the
      national total as denominator, so shares are comparable across ISOs.
    - Share exogeneity must hold across ALL zones nationally, not just within
      one ISO, because the IV is constructed at the national level.
    - ISO FE absorb cross-ISO differences in price levels and trends.

Outputs:
    1. Event-study plot (γ_τ with 95% CI) — the main figure for the paper
    2. Regression table for all year-pairs
    3. Balance test (correlation of s_{i,0} with zone baseline characteristics)

Usage:
    python r03_share_exogeneity_national.py \
        --dc-file ./tables/datacenter_sum.xlsx \
        --price-file ./tables/annual_price.xlsx \
        --out-dir ./r3_share_exogeneity
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import proplot as pplt
    HAS_PROPLOT = True
except ImportError:
    import matplotlib.pyplot as plt
    HAS_PROPLOT = False
    print("WARNING: proplot not found, falling back to matplotlib.")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DC_FILE = "./tables/dc_cumulative_by_iso.xlsx"
DEFAULT_PRICE_FILE = "./tables/yearly_price.xlsx"
DEFAULT_OUT_DIR = "./r2_share_exogeneity"

# City-specific files (single-sheet, wide format: Year × City)
DEFAULT_CITY_DC_FILE = "./tables_city/dc_cumulative_by_city.xlsx"
DEFAULT_CITY_PRICE_FILE = "./tables_city/yearly_price.xlsx"

SHARE_BASE_YEAR = 2019
STUDY_START_YEAR = 2019
NATIONAL_SHEET = "National"

# ISOs to include in the pooled analysis (override with --isos)
DEFAULT_ISOS = ["PJM", "ERCOT", "CAISO", "MISO", "ISONE", "NYISO", "SPP"]


# =============================================================================
# Data Loading
# =============================================================================

def load_dc_capacity(dc_file, iso):
    """Load annual DC capacity for one ISO. Returns [year, zone, dc_cum]."""
    xls = pd.ExcelFile(dc_file)
    df = pd.read_excel(xls, sheet_name=iso)

    year_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("year",) or str(c).strip().lower().startswith("year"):
            year_col = c
            break
    if year_col is None:
        year_col = df.columns[0]

    df = df.rename(columns={year_col: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()

    zone_cols = [c for c in df.columns if c != "year"]
    for c in zone_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    long = df.melt(id_vars="year", var_name="zone", value_name="dc_cum")
    long = long.dropna(subset=["dc_cum"])
    long["iso"] = iso
    return long.sort_values(["zone", "year"]).reset_index(drop=True)


def load_national_dc(dc_file):
    """Load national total DC capacity. Returns [year, dc_national]."""
    xls = pd.ExcelFile(dc_file)
    df = pd.read_excel(xls, sheet_name=NATIONAL_SHEET)

    year_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("year",) or str(c).strip().lower().startswith("year"):
            year_col = c
            break
    if year_col is None:
        year_col = df.columns[0]

    df = df.rename(columns={year_col: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    nat_col = None
    for c in df.columns:
        if str(c).strip().lower() == "national":
            nat_col = c
            break
    if nat_col is None:
        raise KeyError(f"'National' column not found. Available: {df.columns.tolist()}")

    out = df[["year", nat_col]].rename(columns={nat_col: "dc_national"}).copy()
    out["dc_national"] = pd.to_numeric(out["dc_national"], errors="coerce")
    return out.dropna().sort_values("year").reset_index(drop=True)


def load_annual_prices(price_file, iso):
    """Load annual average prices for one ISO. Returns [year, zone, price]."""
    xls = pd.ExcelFile(price_file)
    df = pd.read_excel(xls, sheet_name=iso)

    year_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("year",) or str(c).strip().lower().startswith("year"):
            year_col = c
            break
    if year_col is None:
        year_col = df.columns[0]

    df = df.rename(columns={year_col: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()

    zone_cols = [c for c in df.columns if c != "year"]
    for c in zone_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    long = df.melt(id_vars="year", var_name="zone", value_name="price")
    long = long.dropna(subset=["price"])
    long["iso"] = iso
    return long.sort_values(["zone", "year"]).reset_index(drop=True)


def load_city_dc(city_dc_file, sheet="Sheet1"):
    """
    Load city-level DC capacity from a single-sheet file (Year × City).
    Skips the 'National' column (already loaded from ISO file).
    Returns [year, zone, dc_cum] with iso='City'.
    """
    df = pd.read_excel(city_dc_file, sheet_name=sheet)

    year_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("year",) or str(c).strip().lower().startswith("year"):
            year_col = c
            break
    if year_col is None:
        year_col = df.columns[0]

    df = df.rename(columns={year_col: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()

    # Drop 'National' column if present — it's already in the ISO national sheet
    zone_cols = [c for c in df.columns
                 if c != "year" and str(c).strip().lower() != "national"]
    for c in zone_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    long = df[["year"] + zone_cols].melt(id_vars="year", var_name="zone", value_name="dc_cum")
    long = long.dropna(subset=["dc_cum"])
    long["iso"] = "City"
    return long.sort_values(["zone", "year"]).reset_index(drop=True)


def load_city_prices(city_price_file, sheet="Sheet1"):
    """
    Load city-level annual prices from a single-sheet file (Year × City).
    Cities with missing prices for certain years are naturally handled:
    dropna removes NaN entries, so only year-pairs where BOTH years have
    valid prices will participate in the ΔP regression.
    Returns [year, zone, price] with iso='City'.
    """
    df = pd.read_excel(city_price_file, sheet_name=sheet)

    year_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("year",) or str(c).strip().lower().startswith("year"):
            year_col = c
            break
    if year_col is None:
        year_col = df.columns[0]

    df = df.rename(columns={year_col: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()

    zone_cols = [c for c in df.columns if c != "year"]
    for c in zone_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    long = df.melt(id_vars="year", var_name="zone", value_name="price")
    long = long.dropna(subset=["price"])  # Drop city×year with missing prices
    long["iso"] = "City"

    # Report coverage
    n_cities = long["zone"].nunique()
    year_counts = long.groupby("zone")["year"].count()
    print(f"    City prices loaded: {n_cities} cities")
    print(f"    Year coverage per city: min={year_counts.min()}, max={year_counts.max()}, "
          f"median={year_counts.median():.0f}")
    sparse = year_counts[year_counts < year_counts.max()]
    if not sparse.empty:
        print(f"    Cities with incomplete years: {dict(sparse)}")

    return long.sort_values(["zone", "year"]).reset_index(drop=True)


# =============================================================================
# Core Analysis
# =============================================================================

def load_and_pool_all(dc_file, price_file, isos, city_dc_file=None, city_price_file=None):
    """
    Load DC capacity and prices for all ISOs + cities, pool into national datasets.
    Compute Bartik shares using national denominator.
    """
    dc_national = load_national_dc(dc_file)
    nat_base = dc_national[dc_national["year"] == SHARE_BASE_YEAR]["dc_national"]
    if nat_base.empty:
        raise ValueError(f"No national DC data for base year {SHARE_BASE_YEAR}")
    dc_nat_base = float(nat_base.iloc[0])

    all_dc = []
    all_prices = []
    skipped = []

    # Load ISO data
    for iso in isos:
        try:
            dc = load_dc_capacity(dc_file, iso)
            pr = load_annual_prices(price_file, iso)
            all_dc.append(dc)
            all_prices.append(pr)
        except Exception as e:
            print(f"  WARNING: Skipping {iso}: {e}")
            skipped.append(iso)

    # Load city data (if provided)
    if city_dc_file and city_price_file:
        city_dc_path = Path(city_dc_file)
        city_price_path = Path(city_price_file)
        if city_dc_path.exists() and city_price_path.exists():
            try:
                print(f"  Loading city DC capacity from {city_dc_path.name}...")
                city_dc = load_city_dc(city_dc_file)
                print(f"  Loading city prices from {city_price_path.name}...")
                city_pr = load_city_prices(city_price_file)
                all_dc.append(city_dc)
                all_prices.append(city_pr)
                print(f"  ✓ City data loaded: {city_dc['zone'].nunique()} cities")
            except Exception as e:
                print(f"  WARNING: Failed to load city data: {e}")
        else:
            if not city_dc_path.exists():
                print(f"  WARNING: City DC file not found: {city_dc_path}")
            if not city_price_path.exists():
                print(f"  WARNING: City price file not found: {city_price_path}")

    if not all_dc:
        raise RuntimeError("No data loaded successfully.")

    dc_all = pd.concat(all_dc, ignore_index=True)
    prices_all = pd.concat(all_prices, ignore_index=True)

    # Create unique zone identifiers: "ISO:zone"
    dc_all["zone_id"] = dc_all["iso"] + ":" + dc_all["zone"]
    prices_all["zone_id"] = prices_all["iso"] + ":" + prices_all["zone"]

    # Compute shares: s_{i,0} = DC_{i, base_year} / DC_{national, base_year}
    dc_base = dc_all[dc_all["year"] == SHARE_BASE_YEAR][["zone_id", "iso", "zone", "dc_cum"]].copy()
    dc_base = dc_base.rename(columns={"dc_cum": "dc_base"})
    dc_base["share"] = dc_base["dc_base"] / dc_nat_base

    loaded_isos = sorted(set(dc_all["iso"]))
    print(f"\n  Loaded ISOs: {loaded_isos}")
    if skipped:
        print(f"  Skipped ISOs: {skipped}")
    print(f"  Total zones (national): {dc_base['zone_id'].nunique()}")
    print(f"  National DC at {SHARE_BASE_YEAR}: {dc_nat_base:,.1f} MW")
    print(f"  Share sum (all ISOs): {dc_base['share'].sum():.4f}")
    print(f"  Share range: [{dc_base['share'].min():.6f}, {dc_base['share'].max():.6f}]")

    return dc_all, prices_all, dc_base, dc_national


def run_pooled_pretrend(prices_all, shares):
    """
    National pooled pre-trend regression with ISO fixed effects:
        ΔP_{i,τ} = α + δ_{ISO(i)} + γ · s_{i,0} + ε_{i}

    Returns DataFrame with [year_pair, gamma, se, pval, ci_lower, ci_upper, ...].
    """
    all_years = sorted(prices_all["year"].unique())
    print(f"\n  Available price years: {all_years}")

    results = []

    for i in range(len(all_years) - 1):
        y1, y2 = int(all_years[i]), int(all_years[i + 1])

        p1 = prices_all[prices_all["year"] == y1][["zone_id", "iso", "price"]].rename(
            columns={"price": "p1"})
        p2 = prices_all[prices_all["year"] == y2][["zone_id", "iso", "price"]].rename(
            columns={"price": "p2"})

        merged = p1.merge(p2, on=["zone_id", "iso"], how="inner")
        merged["delta_p"] = merged["p2"] - merged["p1"]

        reg_data = merged.merge(shares[["zone_id", "share"]], on="zone_id", how="inner")

        if len(reg_data) < 5:
            print(f"  WARNING: Only {len(reg_data)} obs for {y1}-{y2}, skipping.")
            continue

        # Build design matrix: constant + share + ISO dummies
        reg_data = reg_data.reset_index(drop=True)
        iso_dummies = pd.get_dummies(reg_data["iso"], prefix="iso", drop_first=True, dtype=float)
        X = pd.concat([pd.Series(1.0, index=reg_data.index, name="const"),
                        reg_data[["share"]],
                        iso_dummies], axis=1)
        y = reg_data["delta_p"]

        model = sm.OLS(y, X).fit(cov_type="HC1")

        gamma = model.params["share"]
        se = model.bse["share"]
        t_stat = model.tvalues["share"]
        pval = model.pvalues["share"]
        ci = model.conf_int(alpha=0.05).loc["share"]

        is_pre = (y2 <= STUDY_START_YEAR)
        n_isos = reg_data["iso"].nunique()

        results.append({
            "year_pair": f"{y1}-{y2}",
            "year_start": y1,
            "year_end": y2,
            "gamma": gamma,
            "se": se,
            "t_stat": t_stat,
            "pval": pval,
            "ci_lower": ci.iloc[0],
            "ci_upper": ci.iloc[1],
            "n_obs": len(reg_data),
            "n_isos": n_isos,
            "r_squared": model.rsquared,
            "is_pre_period": is_pre,
        })

        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        label = "PRE " if is_pre else "POST"
        print(f"  {label} {y1}-{y2}: γ={gamma:>10.2f}  SE={se:>8.2f}  "
              f"p={pval:.4f}{sig:>4s}  N={len(reg_data)}, ISOs={n_isos}")

    return pd.DataFrame(results)


def run_balance_test(shares, dc_all, prices_all):
    """
    National pooled balance test with ISO FE:
    Regress baseline zone characteristics on share, controlling for ISO FE.
    """
    s = shares[["zone_id", "iso", "share"]].copy()

    # Base-year DC level
    dc_base = dc_all[dc_all["year"] == SHARE_BASE_YEAR][["zone_id", "dc_cum"]].rename(
        columns={"dc_cum": "dc_level"})

    # Base-year price level
    price_base = prices_all[prices_all["year"] == SHARE_BASE_YEAR][["zone_id", "price"]].rename(
        columns={"price": "price_level"})

    # Pre-period average price
    price_pre = prices_all[prices_all["year"] <= SHARE_BASE_YEAR].groupby("zone_id")["price"].agg(
        price_mean_pre="mean", price_std_pre="std"
    ).reset_index()

    balance = s.merge(dc_base, on="zone_id", how="left") \
               .merge(price_base, on="zone_id", how="left") \
               .merge(price_pre, on="zone_id", how="left")

    test_vars = ["price_level", "price_mean_pre", "price_std_pre"]
    results = []

    print(f"\n  Balance Test (national pooled, with ISO FE):")
    print(f"  {'Variable':<25s} {'γ':>10s} {'SE':>10s} {'p-value':>10s} {'N':>5s}")
    print(f"  {'-'*62}")

    for var in test_vars:
        if var not in balance.columns:
            continue
        valid = balance[["share", "iso", var]].dropna()
        if len(valid) < 5:
            continue

        valid = valid.reset_index(drop=True)
        iso_dummies = pd.get_dummies(valid["iso"], prefix="iso", drop_first=True, dtype=float)
        X = pd.concat([pd.Series(1.0, index=valid.index, name="const"),
                        valid[["share"]],
                        iso_dummies], axis=1)
        y = valid[var]
        model = sm.OLS(y, X).fit(cov_type="HC1")

        gamma = model.params["share"]
        se_val = model.bse["share"]
        pval = model.pvalues["share"]

        results.append({
            "variable": var,
            "gamma": gamma,
            "se": se_val,
            "pval": pval,
            "n_obs": len(valid),
        })

        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {var:<25s} {gamma:>10.2f} {se_val:>10.2f} {pval:>10.4f} {len(valid):>5d} {sig}")

    return pd.DataFrame(results)


# =============================================================================
# Plotting
# =============================================================================

def plot_event_study(result_df, out_path, title="National Pooled"):
    """Plot γ_τ with 95% CI. Pre-period in blue, post-period in red."""
    df = result_df.sort_values("year_start").reset_index(drop=True)

    x = np.arange(len(df))
    labels = df["year_pair"].tolist()
    pre_mask = df["is_pre_period"].values
    post_mask = ~pre_mask

    yerr = np.array([df["gamma"] - df["ci_lower"], df["ci_upper"] - df["gamma"]])

    # Find boundary position
    boundary = x[pre_mask].max() + 0.5 if pre_mask.any() else -0.5

    if HAS_PROPLOT:
        fig, ax = pplt.subplots(figwidth=7, figheight=4.2)

        # Shade pre-period
        if pre_mask.any():
            ax.axvspan(x[pre_mask].min() - 0.4, x[pre_mask].max() + 0.4,
                       color="blue9", alpha=0.04, zorder=0)

        ax.axvline(boundary, color="k", ls="--", lw=0.8, alpha=0.4)
        ax.axhline(0, color="k", ls="-", lw=0.6, alpha=0.3)

        if pre_mask.any():
            ax.errorbar(np.asarray(x[pre_mask]), np.asarray(df.loc[pre_mask, "gamma"]),
                        yerr=yerr[:, pre_mask],
                        fmt="o", color="steelblue", capsize=4, capthick=1.5,
                        markersize=7, elinewidth=1.5, label="Pre-study period", zorder=5)

        if post_mask.any():
            ax.errorbar(np.asarray(x[post_mask]), np.asarray(df.loc[post_mask, "gamma"]),
                        yerr=yerr[:, post_mask],
                        fmt="s", color="firebrick", capsize=4, capthick=1.5,
                        markersize=7, elinewidth=1.5, label="Study period", zorder=5)

        # Annotate N for each point
        for idx in range(len(df)):
            ax.text(x[idx], df.loc[idx, "ci_lower"] - (df["ci_upper"].max() - df["ci_lower"].min()) * 0.06,
                    f"n={df.loc[idx, 'n_obs']}", ha="center", va="top", fontsize=7, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.format(
            xlabel="Year Pair",
            ylabel=r"$/MWh per unit share",
            title=f"Share Exogeneity Pre-Trend Test ({title})",
        )
        ax.legend(loc="upper left", fontsize=9)

        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        pplt.close(fig)

    else:
        fig, ax = plt.subplots(figsize=(7, 4.2))

        if pre_mask.any():
            ax.axvspan(x[pre_mask].min() - 0.4, x[pre_mask].max() + 0.4,
                       color="lightblue", alpha=0.15)
        ax.axvline(boundary, color="k", ls="--", lw=0.8, alpha=0.4)
        ax.axhline(0, color="k", ls="-", lw=0.6, alpha=0.3)

        if pre_mask.any():
            ax.errorbar(x[pre_mask], df.loc[pre_mask, "gamma"],
                        yerr=yerr[:, pre_mask],
                        fmt="o", color="steelblue", capsize=4, capthick=1.5,
                        markersize=7, elinewidth=1.5, label="Pre-study period")
        if post_mask.any():
            ax.errorbar(x[post_mask], df.loc[post_mask, "gamma"],
                        yerr=yerr[:, post_mask],
                        fmt="s", color="firebrick", capsize=4, capthick=1.5,
                        markersize=7, elinewidth=1.5, label="Study period")

        for idx in range(len(df)):
            ax.text(x[idx], df.loc[idx, "ci_lower"] - (df["ci_upper"].max() - df["ci_lower"].min()) * 0.06,
                    f"n={df.loc[idx, 'n_obs']}", ha="center", va="top", fontsize=7, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Year Pair")
        ax.set_ylabel(r"$\gamma_{\tau}$ ($/MWh per unit share)")
        ax.set_title(f"Share Exogeneity Pre-Trend Test ({title})")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved → {out_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bartik IV Share Exogeneity Test — National Pooled Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python r03_share_exogeneity_national.py
    python r03_share_exogeneity_national.py --isos PJM ERCOT CAISO MISO
    python r03_share_exogeneity_national.py --dc-file ./my_dc.xlsx --price-file ./my_prices.xlsx
    python r03_share_exogeneity_national.py --list-isos
        """
    )
    global SHARE_BASE_YEAR, STUDY_START_YEAR
    parser.add_argument("--isos", nargs="+", default=DEFAULT_ISOS,
                        help=f"ISOs to include. (default: {DEFAULT_ISOS})")
    parser.add_argument("--dc-file", default=DEFAULT_DC_FILE,
                        help=f"ISO DC capacity Excel file. (default: {DEFAULT_DC_FILE})")
    parser.add_argument("--price-file", default=DEFAULT_PRICE_FILE,
                        help=f"ISO annual price Excel file. (default: {DEFAULT_PRICE_FILE})")
    parser.add_argument("--city-dc-file", default=DEFAULT_CITY_DC_FILE,
                        help=f"City DC capacity file. Use 'none' to skip cities. (default: {DEFAULT_CITY_DC_FILE})")
    parser.add_argument("--city-price-file", default=DEFAULT_CITY_PRICE_FILE,
                        help=f"City annual price file. Use 'none' to skip cities. (default: {DEFAULT_CITY_PRICE_FILE})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Output directory. (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--base-year", type=int, default=SHARE_BASE_YEAR,
                        help=f"Base year for shares. (default: {SHARE_BASE_YEAR})")
    parser.add_argument("--list-isos", action="store_true",
                        help="List available sheets and exit.")

    args = parser.parse_args()

    dc_file = Path(args.dc_file)
    price_file = Path(args.price_file)
    out_dir = Path(args.out_dir)
    SHARE_BASE_YEAR = args.base_year
    STUDY_START_YEAR = SHARE_BASE_YEAR + 1

    # Handle city files — 'none' means skip
    city_dc_file = None if args.city_dc_file.lower() == "none" else args.city_dc_file
    city_price_file = None if args.city_price_file.lower() == "none" else args.city_price_file

    if not dc_file.exists():
        print(f"ERROR: {dc_file} not found"); sys.exit(1)
    if args.list_isos:
        xls = pd.ExcelFile(dc_file)
        print(f"Available sheets: {xls.sheet_names}"); sys.exit(0)
    if not price_file.exists():
        print(f"ERROR: {price_file} not found"); sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    include_cities = city_dc_file is not None and city_price_file is not None

    print(f"{'='*70}")
    print(f"  Bartik IV Share Exogeneity Test — National Pooled")
    print(f"  Base year: {SHARE_BASE_YEAR}  |  Study starts: {STUDY_START_YEAR}")
    print(f"  ISOs: {args.isos}")
    print(f"  Include cities: {'Yes' if include_cities else 'No'}")
    print(f"{'='*70}")

    # ---- Load & pool ----
    print("\n[1] Loading and pooling data across ISOs" +
          (" + cities..." if include_cities else "..."))
    dc_all, prices_all, shares, dc_national = load_and_pool_all(
        dc_file, price_file, args.isos,
        city_dc_file=city_dc_file,
        city_price_file=city_price_file)

    shares_path = out_dir / "shares_national.csv"
    shares.to_csv(shares_path, index=False)
    print(f"  Saved shares → {shares_path.name}")

    # ---- National pooled pre-trend ----
    print(f"\n[2] National pooled pre-trend regressions (with ISO FE)...")
    pooled_df = run_pooled_pretrend(prices_all, shares)

    if pooled_df.empty:
        print("  ERROR: No valid year-pairs. Exiting."); sys.exit(1)

    pooled_path = out_dir / "pretrend_national_pooled.csv"
    pooled_df.to_csv(pooled_path, index=False)
    print(f"\n  Saved → {pooled_path.name}")

    # Assessment
    pre_results = pooled_df[pooled_df["is_pre_period"]]
    if not pre_results.empty:
        any_sig = (pre_results["pval"] < 0.05).any()
        print(f"\n  {'='*50}")
        if any_sig:
            sig_pairs = pre_results[pre_results["pval"] < 0.05]["year_pair"].tolist()
            print(f"  ⚠ Significant pre-trends: {sig_pairs}")
        else:
            print(f"  ✓ No significant pre-trends (all pre-period p > 0.05)")
            print(f"  Share exogeneity assumption is supported.")
        print(f"  {'='*50}")

    # ---- Balance test ----
    print(f"\n[3] Balance test (national pooled, with ISO FE)...")
    balance_df = run_balance_test(shares, dc_all, prices_all)
    if not balance_df.empty:
        balance_path = out_dir / "balance_test_national.csv"
        balance_df.to_csv(balance_path, index=False)
        print(f"  Saved → {balance_path.name}")

    # ---- Plots ----
    print(f"\n[4] Generating plots...")
    for ext in ["png"]:
        plot_path = out_dir / f"pretrend_event_study_national.{ext}"
        plot_event_study(pooled_df, plot_path, title="National Pooled, ISO FE")

    print(f"\n{'='*70}")
    print(f"  ✓ All done. Results in: {out_dir.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()