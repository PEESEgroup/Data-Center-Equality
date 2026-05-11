#!/usr/bin/env python3
"""
Bartik IV 2SLS Regression: Causal Effect of DC Capacity on Wholesale Prices
============================================================================
Implements the Bartik shift-share instrumental variable strategy with
leave-one-ISO-out correction (following Greenstone, Mas & Nguyen 2020):

    Z_{it}^{Bartik} = s_{i,0} × (G_t^{-ISO} - G_{base}^{-ISO})

where s_{i,0} = DC_{i,2019} / DC_{national,2019} is the base-year zone share,
and G_t^{-ISO} = G_national_t - DC_{ISO,t} is the national DC capacity at
time t EXCLUDING the entire ISO that zone i belongs to.

The leave-one-ISO-out correction removes the entire ISO (rather than just a
single zone) from the national aggregate, addressing spatial correlation
among zones within the same electricity market.

First stage:
    DC_{it} = π_1·Z_{it} + π_2·HDD² + π_3·CDD² + π_4·GasPrice
              + zone_FE + year_FE + month×dow_FE + ν_{it}

Second stage:
    P_{it} = β_1·DC_hat_{it} + β_2·HDD² + β_3·CDD² + β_4·GasPrice
             + zone_FE + year_FE + month×dow_FE + ε_{it}

Reuses the same data pipeline as r01_panel_regression.py (daily panel, per-ISO).

Usage:
    python r04_bartik_iv.py --iso PJM
    python r04_bartik_iv.py --iso PJM ERCOT CAISO
    python r04_bartik_iv.py --iso ALL 
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS, IVLIML
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_PATHS = {
    "t1": "./tables/price.xlsx",
    "t2": "./tables/temperature_filled.xlsx",
    "t3": "./tables/fuel_marginal.xlsx",
    "t5": "./tables/datacenter_sum.xlsx",       # Zone-level DC (ISO sheets)
    "t6": "./tables/capacity_by_iso.xlsx",    # Renewable capacity by ISO (monthly)
}

# File containing national DC totals (may be the same file as t5, different sheet)
NATIONAL_DC_FILE = "./tables/dc_cumulative_by_iso.xlsx"
NATIONAL_SHEET = "National"

OUTPUT_DIR = Path("./r3_bartik_iv")

START_DATE = None
END_DATE = pd.Timestamp("2025-12-31")

BASE_HEAT = 65.0
BASE_COOL = 65.0
HDD_SCALE = 100
CDD_SCALE = 100

PRICE_WINSOR_LOWER = 0.01
PRICE_WINSOR_UPPER = 0.99
FUEL_WINSOR_LOWER = 0.01
FUEL_WINSOR_UPPER = 0.99
DC_THRESH = 0.000

SHARE_BASE_YEAR = 2020
ISO_BASE_YEAR = {"ERCOT": 2022}

# ---------------------------------------------------------------------------
# Robustness: winsorization levels for reduced-form sensitivity checks
# ---------------------------------------------------------------------------
ROBUSTNESS_WINSOR_LEVELS = [
    None,                # default (1%, 99%)
]


# =============================================================================
# Data Loading (reused from r01)
# =============================================================================

def _open_xls(path):
    path = Path(path)
    assert path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}, f"Expected Excel: {path}"
    return pd.ExcelFile(path)

def _norm(s):
    return "".join(str(s).upper().replace("-", "").replace("_", "").split())

def _pick_sheet(xls, iso):
    target = _norm(iso)
    for name in xls.sheet_names:
        if _norm(name) == target:
            return name
    for name in xls.sheet_names:
        if target in _norm(name):
            return name
    raise KeyError(f"Sheet not found for '{iso}'; available: {xls.sheet_names}")

def _find_ts_col(cols):
    keys = ["timestamp", "time", "datetime", "date", "hour", "interval"]
    for c in cols:
        if any(k in str(c).lower() for k in keys):
            return c
    return cols[0]

def _to_datetime(series):
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(pd.to_numeric(series, errors="coerce"),
                            unit="d", origin="1899-12-30", errors="coerce")
    return dt

def _coerce_numeric(df, except_cols=("ts",)):
    num_cols = [c for c in df.columns if c not in except_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df


def read_table1_prices(path, iso):
    xls = _open_xls(path)
    sheet = _pick_sheet(xls, iso)
    df = pd.read_excel(xls, sheet_name=sheet)
    ts_col = _find_ts_col(df.columns)
    df = df.rename(columns={ts_col: "ts"})
    df["ts"] = _to_datetime(df["ts"])
    df = _coerce_numeric(df, except_cols=("ts",)).dropna(subset=["ts"])
    out = df.melt(id_vars="ts", var_name="zone", value_name="lmp_da")
    out["iso"] = iso
    return out[["iso", "zone", "ts", "lmp_da"]].sort_values(["zone", "ts"])

def read_table2_temperature(path, iso):
    xls = _open_xls(path)
    sheet = _pick_sheet(xls, iso)
    df = pd.read_excel(xls, sheet_name=sheet)
    ts_col = _find_ts_col(df.columns)
    df = df.rename(columns={ts_col: "ts"}).copy()
    df["ts"] = _to_datetime(df["ts"])
    df = _coerce_numeric(df, except_cols=("ts",)).dropna(subset=["ts"])
    out = df.melt(id_vars="ts", var_name="zone", value_name="temperature")
    out["iso"] = iso
    return out[["iso", "zone", "ts", "temperature"]].sort_values(["zone", "ts"])

def read_table3_fuelmix(path, iso):
    xls = _open_xls(path)
    sheet = _pick_sheet(xls, iso)
    header = pd.read_excel(xls, sheet_name=sheet, nrows=0)
    cols = list(header.columns)
    ts_col = _find_ts_col(cols)
    price_col = "gas_price"
    df = pd.read_excel(xls, sheet_name=sheet, usecols=[ts_col, price_col])
    df.columns = ["ts", "gas_price"]
    df["ts"] = _to_datetime(df["ts"])
    df["gas_price"] = pd.to_numeric(df["gas_price"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df["iso"] = iso
    return df[["iso", "ts", "gas_price"]]

def read_table5_dc_capacity(path, iso):
    xls = _open_xls(path)
    sheet = _pick_sheet(xls, iso)
    df = pd.read_excel(xls, sheet_name=sheet)
    year_col = None
    for c in df.columns:
        if str(c).strip().lower().startswith("year"):
            year_col = c; break
    if year_col is None:
        year_col = df.columns[0]
    df = df.rename(columns={year_col: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"].ge(2020)].dropna(subset=["year"]).copy()
    zone_cols = [c for c in df.columns if c != "year"]
    df[zone_cols] = df[zone_cols].apply(pd.to_numeric, errors="coerce")
    out = df.melt(id_vars="year", var_name="zone", value_name="dc_cum").dropna(subset=["dc_cum"])
    out["iso"] = iso
    return out[["iso", "zone", "year", "dc_cum"]].sort_values(["zone", "year"])


def read_table6_renewable_capacity(path, iso):
    """
    Read monthly renewable energy installed capacity from Excel.
    Each ISO is stored in a separate sheet.
    The data format has columns: Month, Renewable_MW, Gas_MW, Nuclear_MW, ...
    Returns a DataFrame with columns: [iso, year_month, renewable_gw]
    where renewable_gw = Renewable_MW / 1000.
    """
    xls = _open_xls(path)
    sheet = _pick_sheet(xls, iso)
    df = pd.read_excel(xls, sheet_name=sheet)

    # Find the Month column
    month_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("month", "date", "year_month", "yearmonth"):
            month_col = c; break
    if month_col is None:
        month_col = df.columns[0]

    # Find the Renewable_MW column
    renew_col = None
    for c in df.columns:
        cl = str(c).strip().lower().replace(" ", "_")
        if cl in ("renewable_mw", "renewablemw", "renewable", "renew_mw"):
            renew_col = c; break
    if renew_col is None:
        # Try partial match
        for c in df.columns:
            if "renew" in str(c).lower():
                renew_col = c; break
    if renew_col is None:
        raise KeyError(f"Renewable capacity column not found for {iso}. "
                       f"Available columns: {df.columns.tolist()}")

    out = df[[month_col, renew_col]].copy()
    out.columns = ["month_str", "renewable_mw"]
    out["renewable_mw"] = pd.to_numeric(out["renewable_mw"], errors="coerce")

    # Parse month: expected format like "2020-01" or similar
    out["year_month"] = pd.to_datetime(out["month_str"], errors="coerce").dt.to_period("M")
    out = out.dropna(subset=["year_month", "renewable_mw"]).copy()

    # Convert MW → GW
    out["renewable_gw"] = out["renewable_mw"] / 1000.0

    out["iso"] = iso
    print(f"  Renewable capacity loaded: {len(out)} months, "
          f"range [{out['renewable_gw'].min():.2f}, {out['renewable_gw'].max():.2f}] GW")
    return out[["iso", "year_month", "renewable_gw"]].reset_index(drop=True)


def load_all_tables(paths, iso):
    print(f"  Loading prices, temperature, fuel, DC capacity, renewable capacity...")
    result = {
        "prices": read_table1_prices(paths["t1"], iso),
        "temperature": read_table2_temperature(paths["t2"], iso),
        "fuelmix": read_table3_fuelmix(paths["t3"], iso),
        "dc_capacity": read_table5_dc_capacity(paths["t5"], iso),
    }
    # Load renewable capacity (optional — skip if file missing or sheet absent)
    if "t6" in paths and Path(paths["t6"]).exists():
        try:
            result["renewable"] = read_table6_renewable_capacity(paths["t6"], iso)
        except Exception as e:
            print(f"  WARNING: Could not load renewable capacity for {iso}: {e}")
            result["renewable"] = None
    else:
        result["renewable"] = None
    return result


# =============================================================================
# National DC Capacity & Bartik IV Construction
# =============================================================================

def load_national_dc(national_dc_file):
    """Load national total DC capacity from the 'National' sheet."""
    xls = pd.ExcelFile(national_dc_file)
    df = pd.read_excel(xls, sheet_name=NATIONAL_SHEET)

    year_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("year",) or str(c).strip().lower().startswith("year"):
            year_col = c; break
    if year_col is None:
        year_col = df.columns[0]

    df = df.rename(columns={year_col: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    nat_col = None
    for c in df.columns:
        if str(c).strip().lower() == "national":
            nat_col = c; break
    if nat_col is None:
        raise KeyError(f"'National' column not found. Available: {df.columns.tolist()}")

    out = df[["year", nat_col]].rename(columns={nat_col: "dc_national"}).copy()
    out["dc_national"] = pd.to_numeric(out["dc_national"], errors="coerce")
    return out.dropna().sort_values("year").reset_index(drop=True)


def compute_zone_shares(dc_zone_df, dc_national_df, iso_tag):
    """
    Compute s_{i,0} = DC_{i, 2019} / DC_{national, 2019} for each zone.
    Returns DataFrame: [zone, share].
    """
    # Zone-level capacity at base year
    zone_base = dc_zone_df[dc_zone_df["year"] == SHARE_BASE_YEAR-1][["zone", "dc_cum"]].copy()
    if zone_base.empty:
        # Try loading base year from the full DC file (which may include years < 2020)
        raise ValueError(f"No DC data for base year {SHARE_BASE_YEAR} in zone data. "
                         f"Ensure the DC file includes {SHARE_BASE_YEAR}.")

    # National total at base year
    nat_row = dc_national_df[dc_national_df["year"] == SHARE_BASE_YEAR-1]
    if nat_row.empty:
        raise ValueError(f"No national DC data for {SHARE_BASE_YEAR-1}")
    dc_nat_base = float(nat_row["dc_national"].iloc[0])

    zone_base["share"] = zone_base["dc_cum"] / dc_nat_base
    print(f"  Shares computed: {len(zone_base)} zones, "
          f"range [{zone_base['share'].min():.6f}, {zone_base['share'].max():.6f}], "
          f"sum={zone_base['share'].sum():.4f}")

    return zone_base[["zone", "share"]].reset_index(drop=True), dc_nat_base


def load_dc_with_base_year(dc_file, iso):
    """
    Load DC capacity including pre-2020 years (for base year share computation).
    The main read_table5 filters to >=2020; this version keeps all years.
    """
    xls = _open_xls(dc_file)
    sheet = _pick_sheet(xls, iso)
    df = pd.read_excel(xls, sheet_name=sheet)
    year_col = None
    for c in df.columns:
        if str(c).strip().lower().startswith("year"):
            year_col = c; break
    if year_col is None:
        year_col = df.columns[0]
    df = df.rename(columns={year_col: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()  # Keep ALL years, including 2019
    zone_cols = [c for c in df.columns if c != "year"]
    df[zone_cols] = df[zone_cols].apply(pd.to_numeric, errors="coerce")
    out = df.melt(id_vars="year", var_name="zone", value_name="dc_cum").dropna(subset=["dc_cum"])
    out["iso"] = iso
    return out[["iso", "zone", "year", "dc_cum"]].sort_values(["zone", "year"])


def interpolate_national_dc_daily(dc_national_df, date_range):
    """
    Interpolate national DC capacity from annual to daily, using the same
    linear interpolation logic as zone-level dc_local.
    Returns a Series indexed by date with daily G_t values.
    """
    # Build annual target/prev table
    nat = dc_national_df.sort_values("year").copy()
    nat["dc_prev"] = nat["dc_national"].shift(1)
    # For the first available year, assume prev = 0 or same
    nat["dc_prev"] = nat["dc_prev"].fillna(0.0)

    # Create daily records
    daily_records = []
    for _, row in nat.iterrows():
        yr = int(row["year"])
        target = row["dc_national"]
        prev = row["dc_prev"]
        y_start = pd.Timestamp(f"{yr}-01-01")
        y_end = pd.Timestamp(f"{yr}-12-31")
        total_days = (y_end - y_start).days

        dates = pd.date_range(y_start, y_end, freq="D")
        for d in dates:
            elapsed = (d - y_start).days
            frac = elapsed / total_days if total_days > 0 else 1.0
            frac = min(max(frac, 0.0), 1.0)
            g_t = prev + (target - prev) * frac
            daily_records.append({"date": d, "dc_national_daily": g_t})

    daily_df = pd.DataFrame(daily_records)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    return daily_df


def build_bartik_iv(panel_df, shares_df, dc_national_daily, dc_nat_base, dc_zone_cum_df):
    """
    Construct daily Bartik IV with leave-one-ISO-out national growth:
        Z_{it} = s_{i,0} × (G_t^{-ISO} - G_{base}^{-ISO})

    where G_t^{-ISO} = G_national_t - DC_{ISO,t} excludes the ENTIRE ISO
    (all zones within the same ISO) from the national total.

    Rationale (Greenstone, Mas & Nguyen 2020):
        Zones within the same ISO share the same electricity market and are
        subject to highly correlated demand/supply shocks.  A simple
        leave-one-zone-out still leaves the correlated sibling zones in the
        national aggregate, so the mechanical endogeneity is not fully purged.
        Removing the whole ISO eliminates this spatial-correlation channel.

    dc_zone_cum_df must be the CUMULATIVE capacity file (dc_cumulative_by_iso.xlsx),
    NOT the incremental file (datacenter_sum.xlsx).
    """
    df = panel_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Merge shares
    df = df.merge(shares_df, on="zone", how="left")
    df["share"] = df["share"].fillna(0.0)

    # Merge national daily DC
    dc_national_daily = dc_national_daily.copy()
    dc_national_daily["date"] = pd.to_datetime(dc_national_daily["date"])
    df = df.merge(dc_national_daily, on="date", how="left")

    # --- Aggregate zone-level cumulative DC to ISO level, then interpolate to daily ---
    df["year"] = df["date"].dt.year

    # Sum all zones within this ISO to get ISO-level cumulative capacity per year
    iso_cum = (dc_zone_cum_df
               .groupby(["iso", "year"], as_index=False)["dc_cum"]
               .sum()
               .rename(columns={"dc_cum": "dc_iso_cum"})
               .sort_values(["iso", "year"]))
    iso_cum["dc_iso_cum_prev"] = (iso_cum
                                   .groupby("iso")["dc_iso_cum"]
                                   .shift(1)
                                   .fillna(0.0))

    # Each row in df belongs to one ISO; merge ISO-year totals
    # (df already has an "iso" column from the panel construction)
    iso_tag = dc_zone_cum_df["iso"].iloc[0]  # current ISO being processed
    iso_cum_this = iso_cum[iso_cum["iso"] == iso_tag][["year", "dc_iso_cum", "dc_iso_cum_prev"]]
    df = df.merge(iso_cum_this, on="year", how="left")
    df["dc_iso_cum"] = df["dc_iso_cum"].fillna(0.0)
    df["dc_iso_cum_prev"] = df["dc_iso_cum_prev"].fillna(0.0)

    # Within-year interpolation for ISO-level cumulative capacity
    year_start = pd.to_datetime(df["year"].astype(str) + "-01-01")
    year_end = pd.to_datetime(df["year"].astype(str) + "-12-31")
    d0 = df["date"].dt.normalize()
    elapsed = (d0 - year_start).dt.days.astype("int64")
    total = (year_end - year_start).dt.days.astype("int64")
    frac = (elapsed / total).clip(0.0, 1.0)
    df["dc_iso_daily"] = df["dc_iso_cum_prev"] + (df["dc_iso_cum"] - df["dc_iso_cum_prev"]) * frac

    # Leave-one-ISO-out: G_t^{-ISO} = G_national_t - DC_{ISO,t}
    df["g_leave_iso_out"] = df["dc_national_daily"] - df["dc_iso_daily"]

    # Base value (end of SHARE_BASE_YEAR - 1) for leave-one-ISO-out:
    #   G_{base}^{-ISO} = G_national_{base} - DC_{ISO,base}
    base_yr = SHARE_BASE_YEAR - 1
    iso_base_row = iso_cum_this[iso_cum_this["year"] == base_yr]
    if iso_base_row.empty:
        # Fallback: if no data for base_yr-1, try base_yr itself
        iso_base_row = iso_cum_this[iso_cum_this["year"] == SHARE_BASE_YEAR]
    dc_iso_base = float(iso_base_row["dc_iso_cum"].iloc[0]) if not iso_base_row.empty else 0.0
    df["g_base_loo_iso"] = dc_nat_base - dc_iso_base

    # Bartik IV: s_{i,0} × (G_t^{-ISO} - G_{base}^{-ISO})
    df["bartik_iv"] = df["share"] * (df["g_leave_iso_out"] - df["g_base_loo_iso"])
    df["bartik_iv"] = df["bartik_iv"] / 1000.0  # MW → GW

    print(f"  Bartik IV (leave-one-ISO-out) range: "
          f"[{df['bartik_iv'].min():.6f}, {df['bartik_iv'].max():.6f}]")
    print(f"  Bartik IV mean: {df['bartik_iv'].mean():.6f}")
    print(f"  ISO-level DC removed from national: "
          f"{dc_iso_base:.1f} MW (base), {df['dc_iso_daily'].max():.1f} MW (max daily)")

    # Clean up temp columns
    df = df.drop(columns=["dc_iso_cum", "dc_iso_cum_prev", "dc_iso_daily",
                           "g_leave_iso_out", "g_base_loo_iso",
                           "national_dc_growth"], errors="ignore")

    return df


# =============================================================================
# Data Processing (reused from r01)
# =============================================================================

def aggregate_to_daily(data_hour):
    data = {}
    for key in ["prices", "temperature"]:
        df = data_hour[key].copy()
        df["ts"] = pd.to_datetime(df["ts"])
        df["date"] = df["ts"].dt.date
        num_cols = df.select_dtypes(include="number").columns
        daily_df = df.groupby(["iso", "zone", "date"])[num_cols].mean().reset_index()
        data[key] = daily_df
    df = data_hour["fuelmix"].copy()
    df["date"] = pd.to_datetime(df["ts"])
    num_cols = df.select_dtypes(include="number").columns
    fuelmix_daily = df.groupby(["iso", "date"])[num_cols].mean().reset_index()
    fuelmix_daily["gas_price"] = fuelmix_daily["gas_price"].astype(float)
    data["fuelmix"] = fuelmix_daily
    data["dc_capacity"] = data_hour["dc_capacity"].copy()
    # Pass through renewable capacity (already monthly, no aggregation needed)
    data["renewable"] = data_hour.get("renewable", None)
    return data

def compute_degree_days(temp_df):
    temp_df = temp_df.copy()
    T = pd.to_numeric(temp_df["temperature"], errors="coerce")
    temp_df["HDD"] = ((BASE_HEAT - T).clip(lower=0) / HDD_SCALE).astype(float)
    temp_df["CDD"] = ((T - BASE_COOL).clip(lower=0) / CDD_SCALE).astype(float)
    temp_df["HDD2"] = temp_df["HDD"] ** 2
    temp_df["CDD2"] = temp_df["CDD"] ** 2
    return temp_df

def interpolate_dc_capacity(df, dc):
    """Same as r01: interpolate zone-level DC capacity to daily."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    keys_local = df[["iso", "zone", "year"]].drop_duplicates()
    dc_y = dc[(dc["year"] >= 2020) & (dc["year"] <= 2025)].copy()
    dc_local_raw = dc_y[["iso", "zone", "year", "dc_cum"]].copy()
    dc_2020_25 = keys_local.merge(dc_local_raw, on=["iso", "zone", "year"], how="left") \
                           .sort_values(["iso", "zone", "year"])

    dc_local_year = dc_2020_25.rename(columns={"dc_cum": "dc_local_target"}) \
                              [["iso", "zone", "year", "dc_local_target"]] \
                              .sort_values(["iso", "zone", "year"])
    dc_local_year["dc_local_prev"] = dc_local_year.groupby(["iso", "zone"])["dc_local_target"] \
                                                   .shift(1).fillna(0.0)
    dc_local_year = dc_local_year.fillna(0.0)

    dc_iso_year = dc_2020_25.groupby(["iso", "year"], as_index=False)["dc_cum"].sum() \
                            .rename(columns={"dc_cum": "dc_iso_target"}) \
                            .sort_values(["iso", "year"])
    dc_iso_year["dc_iso_prev"] = dc_iso_year.groupby("iso")["dc_iso_target"].shift(1).fillna(0.0)
    dc_iso_year = dc_iso_year.fillna(0.0)

    df = df.merge(dc_local_year, on=["iso", "zone", "year"], how="left") \
           .merge(dc_iso_year, on=["iso", "year"], how="left")
    for col in ["dc_local_target", "dc_local_prev", "dc_iso_target", "dc_iso_prev"]:
        df[col] = df[col].fillna(0.0)
    for cur, prev in [("dc_local_target", "dc_local_prev"), ("dc_iso_target", "dc_iso_prev")]:
        df[cur] = df[cur].fillna(df[prev])
        df[prev] = df[prev].fillna(0.0)

    year_start = pd.to_datetime(df["year"].astype(str) + "-01-01")
    year_end = pd.to_datetime(df["year"].astype(str) + "-12-31")
    d0 = df["date"].dt.normalize()
    elapsed = (d0 - year_start).dt.days.astype("int64")
    total = (year_end - year_start).dt.days.astype("int64")
    frac = (elapsed / total).clip(0.0, 1.0)

    df["dc_local"] = df["dc_local_prev"] + (df["dc_local_target"] - df["dc_local_prev"]) * frac
    df["dc_iso_total"] = df["dc_iso_prev"] + (df["dc_iso_target"] - df["dc_iso_prev"]) * frac
    df["dc_external"] = (df["dc_iso_total"] - df["dc_local"]).clip(lower=0.0)
    df["dc_local"] = df["dc_local"].astype(float) / 1000.0
    df[["dc_iso_total", "dc_external"]] = df[["dc_iso_total", "dc_external"]].astype(float) / 1000.0 / 10.0
    return df

def winsorize_by_group(d, cols, ql, qh, by="zone"):
    g = d.groupby(by)
    lo = g[cols].transform(lambda s: s.quantile(ql))
    hi = g[cols].transform(lambda s: s.quantile(qh))
    keep = ((d[cols] >= lo) & (d[cols] <= hi)).all(axis=1)
    return d.loc[keep].copy()


# =============================================================================
# Panel Construction (modified from r01 to include Bartik IV)
# =============================================================================

def build_panel(data, iso_tag, shares_df, dc_national_daily, dc_nat_base, dc_zone_cum_df,
                price_winsor_override=None):
    """
    Build the regression-ready panel for one ISO.
    Same as r01, but also constructs the Bartik IV column.

    Parameters
    ----------
    price_winsor_override : tuple or None
        If given, (lower, upper) quantiles to override default winsorization.
    """
    prices = data["prices"].query("iso == @iso_tag").copy()
    temps = data["temperature"].query("iso == @iso_tag")[
        ["iso", "zone", "date", "temperature", "HDD", "CDD", "HDD2", "CDD2"]].copy()
    mix = data["fuelmix"].query("iso == @iso_tag")[["iso", "date", "gas_price"]].copy()
    dc = data["dc_capacity"].query("iso == @iso_tag").copy()

    for d in [prices, temps, mix]:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")

    prices["price_diff"] = prices["lmp_da"]

    df = prices.merge(temps, on=["iso", "zone", "date"], how="left") \
               .merge(mix, on=["iso", "date"], how="left")
    
    df = interpolate_dc_capacity(df, dc)

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

    # Time variables
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek

    # Add Bartik IV
    df = build_bartik_iv(df, shares_df, dc_national_daily, dc_nat_base, dc_zone_cum_df)

    # Merge renewable energy installed capacity (monthly → daily)
    renewable_df = data.get("renewable", None)
    has_renewable = False
    if renewable_df is not None and len(renewable_df) > 0:
        df["year_month"] = df["date"].dt.to_period("M")
        df = df.merge(renewable_df[["year_month", "renewable_gw"]],
                      on="year_month", how="left")
        df["renewable_gw"] = df["renewable_gw"].ffill().bfill()  # fill any gaps
        if df["renewable_gw"].notna().any():
            has_renewable = True
            print(f"  Renewable capacity merged: "
                  f"range [{df['renewable_gw'].min():.2f}, {df['renewable_gw'].max():.2f}] GW")
        df = df.drop(columns=["year_month"], errors="ignore")
    else:
        print(f"  NOTE: No renewable capacity data — running without renewable_gw control")

    # Select columns
    need_cols = [
        "price_diff", "dc_local", "dc_external", "dc_iso_total",
        "temperature", "HDD", "CDD", "HDD2", "CDD2",
        "gas_price",
        "zone", "month", "year", "dow", "date",
        "bartik_iv", "share",
    ]
    if has_renewable:
        need_cols.append("renewable_gw")
    reg = df[need_cols].dropna().copy()
    print(f"  Sample: {len(reg):,} rows, {reg['zone'].nunique()} zones")

    # Filter trivial DC zones
    reg["dc_local"] = pd.to_numeric(reg["dc_local"], errors="coerce").fillna(0.0)
    zone_max = reg.groupby("zone")["dc_local"].apply(lambda s: s.abs().max())
    keep_zones = sorted(zone_max[zone_max > DC_THRESH].index.tolist())
    drop_zones = sorted(zone_max[zone_max <= DC_THRESH].index.tolist())
    if drop_zones:
        print(f"  Dropped zones: {drop_zones}")
    print(f"  Retained zones: {keep_zones}")

    # Winsorize — use override if provided
    pw_lo = price_winsor_override[0] if price_winsor_override else PRICE_WINSOR_LOWER
    pw_hi = price_winsor_override[1] if price_winsor_override else PRICE_WINSOR_UPPER
    reg1 = winsorize_by_group(reg.loc[reg["zone"].isin(keep_zones)].copy(),
                              ["price_diff"], pw_lo, pw_hi, "zone")
    reg2 = winsorize_by_group(reg1, ["gas_price"], FUEL_WINSOR_LOWER, FUEL_WINSOR_UPPER, "zone")
    print(f"  After winsorization [{pw_lo:.3f}, {pw_hi:.3f}]: {len(reg2):,} rows")

    return reg, reg2


# =============================================================================
# Regression: OLS + IV2SLS
# =============================================================================

def run_ols(panel_df):
    """Run the OLS panel FE regression (same as r01) for comparison."""
    renew_term = " + renewable_gw" if "renewable_gw" in panel_df.columns else ""
    formula = (
        f"price_diff ~ 1 + HDD2 + CDD2 + gas_price{renew_term} + dc_local "
        "+ C(year) + C(month):C(dow) + EntityEffects"
    )
    panel = panel_df.set_index(["zone", "date"]).sort_index()
    res = PanelOLS.from_formula(formula, data=panel).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=7
    )
    return res


def run_iv2sls(panel_df):
    """
    Run Bartik IV 2SLS regression.

    Since IV2SLS does not have built-in EntityEffects, zone FE are included
    explicitly as C(zone) dummies. With ~17 zones this is computationally fine.

    Formula:
        price_diff ~ HDD2 + CDD2 + gas_price + C(zone) + C(year) + C(month):C(dow)
                     + [dc_local ~ bartik_iv]
    """
    renew_term = " + renewable_gw" if "renewable_gw" in panel_df.columns else ""
    formula = (
        f"price_diff ~ 1 + HDD2 + CDD2 + gas_price{renew_term} "
        "+ C(zone) + C(year) + C(month):C(dow) "
        "+ [dc_local ~ bartik_iv]"
    )
    res = IV2SLS.from_formula(formula, data=panel_df).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=7
    )
    return res


def run_first_stage_explicit(panel_df):
    """
    Run the first stage explicitly as OLS for detailed diagnostics:
        dc_local ~ bartik_iv + HDD2 + CDD2 + gas_price + C(zone) + C(year) + C(month):C(dow)

    Reports the F-statistic on the excluded instrument (bartik_iv).
    """
    renew_term = " + renewable_gw" if "renewable_gw" in panel_df.columns else ""
    formula = (
        f"dc_local ~ 1 + bartik_iv + HDD2 + CDD2 + gas_price{renew_term} "
        "+ C(year) + C(month):C(dow) + EntityEffects"
    )
    panel = panel_df.set_index(["zone", "date"]).sort_index()
    res = PanelOLS.from_formula(formula, data=panel).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=7
    )
    return res


def run_reduced_form(panel_df):
    """
    Reduced form regression: regress outcome DIRECTLY on the instrument.

        price ~ bartik_iv + controls + FE

    This is ALWAYS consistently estimated regardless of instrument strength.
    If π_RF is significant, we have evidence that the instrument affects
    the outcome through the endogenous variable, even if 2SLS is unreliable.

    The ratio π_RF / π_FS = β_IV (Wald estimator), so the reduced-form
    coefficient divided by the first-stage coefficient gives the causal effect.
    """
    renew_term = " + renewable_gw" if "renewable_gw" in panel_df.columns else ""
    formula = (
        f"price_diff ~ 1 + bartik_iv + HDD2 + CDD2 + gas_price{renew_term} "
        "+ C(year) + C(month):C(dow) + EntityEffects"
    )
    panel = panel_df.set_index(["zone", "date"]).sort_index()
    res = PanelOLS.from_formula(formula, data=panel).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=7
    )
    return res



# =============================================================================
# Output
# =============================================================================

def result_to_df(res):
    df = pd.DataFrame({
        "coef": res.params, "std_err": res.std_errors,
        "t_stat": res.tstats, "pval": res.pvalues,
    })
    ci = res.conf_int(level=0.95)
    ci.columns = ["ci_lower", "ci_upper"]
    return pd.concat([df, ci], axis=1)


def print_comparison(iso_tag, res_ols, res_iv, res_first):
    """Print OLS vs IV comparison table."""
    print(f"\n{'='*75}")
    print(f"  RESULTS FOR {iso_tag}: OLS vs. Bartik IV")
    print(f"{'='*75}")

    # First stage diagnostics
    if "bartik_iv" in res_first.params.index:
        pi1 = res_first.params["bartik_iv"]
        se1 = res_first.std_errors["bartik_iv"]
        t1 = res_first.tstats["bartik_iv"]
        p1 = res_first.pvalues["bartik_iv"]
        f_stat = t1 ** 2  # For single instrument, F = t²
        print(f"\n  FIRST STAGE:")
        print(f"    π_1 (bartik_iv):  {pi1:.6f}  (SE={se1:.6f}, t={t1:.2f}, p={p1:.4f})")
        print(f"    First-stage F:    {f_stat:.2f}  {'✓ > 10' if f_stat > 10 else '⚠ WEAK INSTRUMENT'}")
        print(f"    R² (within):      {res_first.rsquared:.4f}")

    # Second stage comparison
    print(f"\n  SECOND STAGE (dc_local coefficient):")
    print(f"  {'Method':<20s} {'Coef':>10s} {'SE':>10s} {'t':>8s} {'p':>8s} {'95% CI':>24s}")
    print(f"  {'-'*80}")

    pairs = [("OLS (Panel FE)", res_ols)]
    if res_iv is not None:
        pairs.append(("Bartik IV (2SLS)", res_iv))
    for label, res in pairs:
        var = "dc_local"
        if var in res.params.index:
            coef = res.params[var]
            se = res.std_errors[var]
            t = res.tstats[var]
            p = res.pvalues[var]
            ci = res.conf_int(level=0.95).loc[var]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {label:<20s} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {p:>8.4f} "
                  f"[{ci.iloc[0]:>9.4f}, {ci.iloc[1]:>9.4f}] {sig}")

    # --- Enhanced diagnostics: decompose why IV differs from OLS ---
    if res_iv is not None and "dc_local" in res_ols.params.index and "dc_local" in res_iv.params.index:
        ols_coef = res_ols.params["dc_local"]
        ols_se = res_ols.std_errors["dc_local"]
        iv_coef = res_iv.params["dc_local"]
        iv_se = res_iv.std_errors["dc_local"]

        print(f"\n  DIAGNOSTIC DECOMPOSITION:")
        print(f"    SE inflation (IV / OLS):        {iv_se / ols_se:>10.2f}x")

        # Check if IV CI includes OLS point estimate
        iv_ci = res_iv.conf_int(level=0.95).loc["dc_local"]
        ols_in_iv_ci = iv_ci.iloc[0] <= ols_coef <= iv_ci.iloc[1]
        print(f"    OLS coef within IV 95% CI:      {'Yes' if ols_in_iv_ci else 'No'}")

        if abs(iv_coef) < abs(ols_coef) * 0.5:
            print(f"\n  → IV coefficient much smaller: OLS may be upward-biased")
        elif iv_coef > ols_coef:
            print(f"\n  → IV > OLS: consistent with downward OLS bias from reverse causality")
        else:
            print(f"\n  → IV ≈ OLS: no evidence of substantial endogeneity bias")

    print(f"{'='*75}\n")



def run_rf_robustness(iso_tag, data, shares_df, dc_national_daily,
                      dc_nat_base, dc_zone_cum_df, res_first_baseline):
    """
    Reduced-form robustness analysis — runs for ANY ISO when IV is strong
    but 2SLS is not significant. Sweeps across winsorization levels to test
    whether the causal channel (instrument → price) is robust.

    This is the "merged first+second stage" test: if reduced form is
    significant, the exogenous component of DC growth causally affects prices,
    even though 2SLS lacks power to pin down the per-GW coefficient precisely.
    """
    print(f"\n{'#'*70}")
    print(f"# {iso_tag}: REDUCED-FORM ROBUSTNESS")
    print(f"{'#'*70}")

    results_table = []
    fs_coef = float(res_first_baseline.params.get("bartik_iv", 0))

    for i, winsor in enumerate(ROBUSTNESS_WINSOR_LEVELS):
        pct_label = f"{winsor[0]*100:.0f}%/{winsor[1]*100:.0f}%" if winsor else "1%/99%"
        print(f"\n  --- Winsorization: {pct_label} ---")
        try:
            reg_raw_rf, panel = build_panel(data, iso_tag, shares_df,
                                            dc_national_daily, dc_nat_base, dc_zone_cum_df,
                                            price_winsor_override=winsor)
            if len(panel) == 0:
                print(f"    SKIP: empty panel"); continue

            res_ols = run_ols(panel)
            res_iv = run_iv2sls(panel)
            res_rf = run_reduced_form(panel)

            row = {"winsor": pct_label, "N": len(panel)}

            # OLS
            if "dc_local" in res_ols.params.index:
                row["ols_coef"] = float(res_ols.params["dc_local"])
                row["ols_p"] = float(res_ols.pvalues["dc_local"])

            # IV
            if "dc_local" in res_iv.params.index:
                row["iv_coef"] = float(res_iv.params["dc_local"])
                row["iv_p"] = float(res_iv.pvalues["dc_local"])

            # Reduced form + Wald ratio
            if "bartik_iv" in res_rf.params.index:
                rf_coef = float(res_rf.params["bartik_iv"])
                row["rf_coef"] = rf_coef
                row["rf_p"] = float(res_rf.pvalues["bartik_iv"])
                row["wald"] = rf_coef / fs_coef if fs_coef != 0 else np.nan

            results_table.append(row)

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    # Print summary table
    if results_table:
        print(f"\n{'='*105}")
        print(f"  {iso_tag} REDUCED-FORM ROBUSTNESS TABLE")
        print(f"{'='*105}")
        print(f"  {'Winsor':<12s} {'N':>6s} "
              f"{'OLS β':>8s} {'OLS p':>7s} "
              f"{'IV β':>8s} {'IV p':>7s} "
              f"{'RF β':>8s} {'RF p':>7s} {'RF':>4s} "
              f"{'Wald':>8s}")
        print(f"  {'-'*101}")
        for r in results_table:
            rf_sig = ""
            if "rf_p" in r:
                if r["rf_p"] < 0.01: rf_sig = "***"
                elif r["rf_p"] < 0.05: rf_sig = "**"
                elif r["rf_p"] < 0.1: rf_sig = "*"
            print(f"  {r['winsor']:<12s} {r.get('N',''):>6,} "
                  f"{r.get('ols_coef', 0):>8.4f} {r.get('ols_p', 1):>7.4f} "
                  f"{r.get('iv_coef', 0):>8.4f} {r.get('iv_p', 1):>7.4f} "
                  f"{r.get('rf_coef', 0):>8.4f} {r.get('rf_p', 1):>7.4f} "
                  f"{rf_sig:>4s} {r.get('wald', 0):>8.4f}")
        print(f"{'='*105}")

        # Interpretation
        any_rf_sig = any(r.get("rf_p", 1) < 0.05 for r in results_table)
        all_rf_sig = all(r.get("rf_p", 1) < 0.10 for r in results_table)
        n_zones = results_table[0].get("N", 0) // 365  # rough zone-years

        print(f"\n  Wald ratio = RF / first-stage = implied causal effect ($/MWh per GW)")
        if all_rf_sig:
            print(f"  Reduced form is significant across ALL winsorization levels.")
        elif any_rf_sig:
            print(f"Reduced form is significant in SOME specifications.")
        else:
            print(f"Reduced form is not significant in any specification.")

        # Save
        rob_dir = OUTPUT_DIR / iso_tag
        rob_dir.mkdir(parents=True, exist_ok=True)
        rob_df = pd.DataFrame(results_table)
        rob_df.to_excel(rob_dir / f"rf_robustness_{iso_tag}.xlsx", index=False)
        rob_df.to_csv(rob_dir / f"rf_robustness_{iso_tag}.csv", index=False)
        print(f"  Saved to {rob_dir}")

    return results_table


def save_results(iso_tag, res_ols, res_iv, res_first, panel_df, reg_raw):
    """Save all results to ISO-specific subfolder. res_iv can be None."""
    iso_dir = OUTPUT_DIR / iso_tag
    iso_dir.mkdir(parents=True, exist_ok=True)
    sheet_tag = iso_tag[:31]  # Excel sheet name limit

    # 1. Full coefficient tables
    save_pairs = [(res_ols, "ols"), (res_first, "first_stage")]
    if res_iv is not None:
        save_pairs.append((res_iv, "iv2sls"))
    for res, name in save_pairs:
        rdf = result_to_df(res)
        rdf.to_excel(iso_dir / f"coefficients_{name}.xlsx", sheet_name=sheet_tag)
    print(f"  Saved coefficient tables")

    # 2. Key summary JSON
    summary = {"iso": iso_tag}

    if "bartik_iv" in res_first.params.index:
        t1 = float(res_first.tstats["bartik_iv"])
        summary["first_stage"] = {
            "pi_1": float(res_first.params["bartik_iv"]),
            "se": float(res_first.std_errors["bartik_iv"]),
            "t_stat": t1,
            "pval": float(res_first.pvalues["bartik_iv"]),
            "F_stat": t1 ** 2,
            "F_above_10": t1 ** 2 > 10,
            "r_squared_within": float(res_first.rsquared),
        }

    json_pairs = [("ols", res_ols)]
    if res_iv is not None:
        json_pairs.append(("iv2sls", res_iv))
    for label, res in json_pairs:
        if "dc_local" in res.params.index:
            ci = res.conf_int(level=0.95).loc["dc_local"]
            summary[label] = {
                "dc_local_coef": float(res.params["dc_local"]),
                "dc_local_se": float(res.std_errors["dc_local"]),
                "dc_local_pval": float(res.pvalues["dc_local"]),
                "dc_local_ci_lower": float(ci.iloc[0]),
                "dc_local_ci_upper": float(ci.iloc[1]),
            }

    if res_iv is None:
        summary["iv_skipped"] = "Weak instrument or IV SE too inflated (>5x OLS SE)"

    with open(iso_dir / f"summary_{iso_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary JSON")

    # 3. Full model summaries
    with open(iso_dir / f"model_summaries_{iso_tag}.txt", "w") as f:
        f.write(f"{'='*80}\nBartik IV Results for {iso_tag}\n{'='*80}\n\n")
        f.write("FIRST STAGE: dc_local ~ bartik_iv + controls + FE\n")
        f.write(f"{'-'*80}\n{res_first.summary}\n\n")
        if res_iv is not None:
            f.write("SECOND STAGE (IV2SLS): price ~ dc_local_hat + controls + FE\n")
            f.write(f"{'-'*80}\n{res_iv.summary}\n\n")
        else:
            f.write("SECOND STAGE (IV2SLS): SKIPPED — IV SE too inflated or weak instrument\n\n")
        f.write("OLS COMPARISON: price ~ dc_local + controls + FE\n")
        f.write(f"{'-'*80}\n{res_ols.summary}\n\n")
    print(f"  Saved model summaries")

    # 4. Panel data with Bartik IV (for reproducibility)
    panel_df.to_csv(iso_dir / f"panel_with_bartik_{iso_tag}.csv", index=False)
    print(f"  Saved panel data")

    # ================= 新增：保存未缩尾的均价 =================
    zone_mean_df = (
        pd.to_numeric(reg_raw["price_diff"], errors="coerce")
        .groupby(reg_raw["zone"]).mean()
        .rename("avg_price")
        .reset_index()
        .sort_values("zone")
    )
    zone_mean_df["avg_price"] = zone_mean_df["avg_price"].round(6)
    
    out_file = iso_dir / "zone_price_diff_means.xlsx"
    if out_file.exists():
        with pd.ExcelWriter(out_file, engine="openpyxl", mode="a",
                            if_sheet_exists="replace") as writer:
            zone_mean_df.to_excel(writer, sheet_name=sheet_tag, index=False)
    else:
        with pd.ExcelWriter(out_file, engine="openpyxl", mode="w") as writer:
            zone_mean_df.to_excel(writer, sheet_name=sheet_tag, index=False)


# =============================================================================
# Main
# =============================================================================

def run_iso(iso_tag, data, shares_df, dc_national_daily, dc_nat_base, dc_zone_cum_df):
    """Full pipeline for one ISO with conditional inference strategy."""
    print(f"\n{'#'*70}")
    print(f"# {iso_tag}: Bartik IV 2SLS (leave-one-ISO-out)")
    print(f"{'#'*70}")

    # [1] Build panel
    print("\n[1] Building panel with Bartik IV...")
    reg_raw, panel_df = build_panel(data, iso_tag, shares_df, dc_national_daily, dc_nat_base, dc_zone_cum_df)
    if len(panel_df) == 0:
        print(f"  WARNING: No data for {iso_tag}. Skipping."); return

    # [2] First stage — gate all subsequent analysis
    print("\n[2] Running first stage...")
    res_first = run_first_stage_explicit(panel_df)

    f_stat = None
    if "bartik_iv" in res_first.params.index:
        t1 = float(res_first.tstats["bartik_iv"])
        f_stat = t1 ** 2
        p_fs = float(res_first.pvalues["bartik_iv"])
        print(f"  π_1 = {res_first.params['bartik_iv']:.6f}  "
              f"(t={t1:.2f}, F={f_stat:.1f}, p={p_fs:.4f})")

    if f_stat is None or f_stat < 10:
        # ── WEAK INSTRUMENT: stop here ──
        print(f"\n  ⚠ WEAK INSTRUMENT (F={(f_stat or 0):.1f} < 10)")
        print(f"    Bartik IV is not informative for {iso_tag}.")
        print(f"    Possible causes: too few zones ({panel_df['zone'].nunique()}), "
              f"insufficient cross-sectional variation in shares.")
        print(f"    Skipping 2SLS and reduced-form analysis.")
        print(f"\n  Saving first-stage diagnostics only...")
        iso_dir = OUTPUT_DIR / iso_tag
        iso_dir.mkdir(parents=True, exist_ok=True)
        rdf = result_to_df(res_first)
        rdf.to_excel(iso_dir / "coefficients_first_stage.xlsx", sheet_name=iso_tag[:31])
        print(f"\n  ✓ {iso_tag} done (weak instrument — first stage only).\n")
        return

    # ── STRONG INSTRUMENT: proceed with full analysis ──
    print(f"  ✓ Strong instrument (F={f_stat:.1f})")

    # [3] OLS
    print("\n[3] Running OLS panel FE...")
    res_ols = run_ols(panel_df)

    # [4] IV 2SLS
    print("\n[4] Running Bartik IV 2SLS...")
    res_iv = run_iv2sls(panel_df)

    # Check if IV SE is too inflated relative to OLS
    iv_noisy = False
    if "dc_local" in res_ols.params.index and "dc_local" in res_iv.params.index:
        se_ols = res_ols.std_errors["dc_local"]
        se_iv = res_iv.std_errors["dc_local"]
        if se_iv > 10 * se_ols:
            iv_noisy = True
            print(f"\n  ⚠ IV SE ({se_iv:.4f}) is {se_iv/se_ols:.1f}x OLS SE ({se_ols:.4f})")
            print(f"    IV estimates too noisy — reporting but skipping further diagnostics.")

    # Print comparison (always report IV results)
    print_comparison(iso_tag, res_ols, res_iv, res_first)

    # Skip reduced-form if IV is too noisy
    if iv_noisy:
        print(f"\n  Skipping reduced-form robustness (IV SE too inflated).")
        print("\n[5] Saving results...")
        save_results(iso_tag, res_ols, res_iv, res_first, panel_df)
        print(f"\n  ✓ {iso_tag} done (IV reported but noisy).\n")
        return

    # Check IV significance
    iv_sig = False
    if "dc_local" in res_iv.params.index:
        iv_p = float(res_iv.pvalues["dc_local"])
        iv_sig = iv_p < 0.10

    if iv_sig:
        # ── IV SIGNIFICANT: full causal evidence established ──
        print(f"\n  IV is significant (p={iv_p:.4f}): direct causal evidence.")
        # Still report reduced form as supplementary
        res_rf = run_reduced_form(panel_df)
        if "bartik_iv" in res_rf.params.index:
            rf_coef = res_rf.params["bartik_iv"]
            fs_coef = res_first.params["bartik_iv"]
            print(f"  Reduced form (supplementary): π_RF={rf_coef:.4f}, "
                  f"Wald ratio={rf_coef/fs_coef:.4f}")
    else:
        # ── IV NOT SIGNIFICANT: run reduced-form robustness ──
        print(f"\n  IV is not significant (p={iv_p:.4f}) despite strong first stage.")
        print(f"    Running reduced-form robustness to test causal channel...")
        run_rf_robustness(iso_tag, data, shares_df,
                          dc_national_daily, dc_nat_base, dc_zone_cum_df,
                          res_first)

    # [5] Save
    print("\n[5] Saving results...")
    save_results(iso_tag, res_ols, res_iv, res_first, panel_df, reg_raw)
    print(f"\n  ✓ {iso_tag} done.\n")


def get_available_isos():
    try:
        return pd.ExcelFile(DATA_PATHS["t1"]).sheet_names
    except Exception:
        return []


def main():
    global SHARE_BASE_YEAR, OUTPUT_DIR, START_DATE

    parser = argparse.ArgumentParser(
        description="Bartik IV 2SLS: DC Capacity → Wholesale Prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python r04_bartik_iv.py --iso PJM
    python r04_bartik_iv.py --iso PJM ERCOT CAISO MISO
    python r04_bartik_iv.py --iso ALL
    python r04_bartik_iv.py --list-isos
        """
    )
    parser.add_argument("--iso", nargs="+", default=["PJM"])
    parser.add_argument("--national-dc-file", default=NATIONAL_DC_FILE,
                        help=f"File with National DC sheet. (default: {NATIONAL_DC_FILE})")
    parser.add_argument("--renewable-file", default=DATA_PATHS.get("t6", ""),
                        help="Excel file with monthly renewable capacity by ISO. "
                             "(default: %(default)s)")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--base-year", type=int, default=SHARE_BASE_YEAR)
    parser.add_argument("--list-isos", action="store_true")

    args = parser.parse_args()
    SHARE_BASE_YEAR = args.base_year
    START_DATE = pd.Timestamp(f"{SHARE_BASE_YEAR}-01-01")
    OUTPUT_DIR_local = Path(args.out_dir)
    OUTPUT_DIR = OUTPUT_DIR_local

    # Override renewable capacity file path if specified
    if args.renewable_file:
        DATA_PATHS["t6"] = args.renewable_file

    if args.list_isos:
        print(f"Available ISOs: {get_available_isos()}"); sys.exit(0)

    # Validate files
    for k, p in DATA_PATHS.items():
        if not Path(p).exists():
            if k == "t6":
                print(f"WARNING: Renewable capacity file {p} not found — will run without it")
            else:
                print(f"ERROR: {p} not found"); sys.exit(1)
    if not Path(args.national_dc_file).exists():
        print(f"ERROR: {args.national_dc_file} not found"); sys.exit(1)

    iso_list = get_available_isos() if args.iso == ["ALL"] else args.iso

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  Bartik IV 2SLS Regression")
    print(f"  Base year: {SHARE_BASE_YEAR}")
    print(f"  ISOs: {iso_list}")
    print(f"  Renewable capacity file: {DATA_PATHS.get('t6', 'N/A')}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"{'='*70}")

    # Load national DC (shared across all ISOs)
    print("\n[0] Loading national DC capacity...")
    dc_national_df = load_national_dc(args.national_dc_file)
    dc_national_daily = interpolate_national_dc_daily(dc_national_df,
                                                      pd.date_range(START_DATE, END_DATE))
    nat_base_row = dc_national_df[dc_national_df["year"] == SHARE_BASE_YEAR - 1]
    if nat_base_row.empty:
        nat_base_row = dc_national_df[dc_national_df["year"] == SHARE_BASE_YEAR]
    dc_nat_base = float(nat_base_row["dc_national"].iloc[0])
    print(f"  Growth baseline (end of {SHARE_BASE_YEAR - 1}): {dc_nat_base:,.1f} MW")

    # Process each ISO
    default_by = args.base_year
    for iso_tag in iso_list:
        try:
            iso_by = ISO_BASE_YEAR.get(iso_tag, default_by)
            if iso_by != SHARE_BASE_YEAR:
                SHARE_BASE_YEAR = iso_by
                START_DATE = pd.Timestamp(f"{iso_by}-01-01")
                nat_base_row = dc_national_df[dc_national_df["year"] == iso_by - 1]
                if nat_base_row.empty:
                    nat_base_row = dc_national_df[dc_national_df["year"] == iso_by]
                dc_nat_base = float(nat_base_row["dc_national"].iloc[0])
                dc_national_daily = interpolate_national_dc_daily(
                    dc_national_df, pd.date_range(START_DATE, END_DATE))
                print(f"\n  [Base year for {iso_tag}: {iso_by}]")

            # Load data
            print(f"\n[Loading data for {iso_tag}]")
            data_hour = load_all_tables(DATA_PATHS, iso=iso_tag)
            print(f"  Aggregating to daily...")
            data = aggregate_to_daily(data_hour)
            data["temperature"] = compute_degree_days(data["temperature"])

            # Load CUMULATIVE DC (for shares + leave-one-ISO-out Bartik IV)
            # This is dc_cumulative_by_iso.xlsx, NOT datacenter_sum.xlsx
            print(f"  Computing zone shares from cumulative DC file...")
            dc_full = load_dc_with_base_year(args.national_dc_file, iso_tag)
            shares_df, _ = compute_zone_shares(dc_full, dc_national_df, iso_tag)

            # Run pipeline (pass dc_full for leave-one-ISO-out construction)
            run_iso(iso_tag, data, shares_df, dc_national_daily, dc_nat_base, dc_full)

        except Exception as e:
            print(f"\n  ERROR processing {iso_tag}: {e}")
            import traceback; traceback.print_exc()
            print(f"  Skipping.\n")
        finally:
            if SHARE_BASE_YEAR != default_by:
                SHARE_BASE_YEAR = default_by
                START_DATE = pd.Timestamp(f"{default_by}-01-01")
                nat_base_row = dc_national_df[dc_national_df["year"] == default_by - 1]
                if nat_base_row.empty:
                    nat_base_row = dc_national_df[dc_national_df["year"] == default_by]
                dc_nat_base = float(nat_base_row["dc_national"].iloc[0])
                dc_national_daily = interpolate_national_dc_daily(
                    dc_national_df, pd.date_range(START_DATE, END_DATE))

    print(f"\n{'='*70}")
    print(f"All done. Results: {OUTPUT_DIR.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
