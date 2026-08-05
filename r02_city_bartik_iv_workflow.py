#!/usr/bin/env python3
"""
Bartik IV 2SLS Regression: Non-ISO Cities
==========================================
Adapts the Bartik IV strategy for non-ISO cities.

Key differences from the ISO version (r02_bartik_iv_workflow.py):
    1. Fuel price is city-specific (merged on zone+date, not ISO-level)
    2. Control variables use zone-interacted slopes: C(zone):HDD2, C(zone):CDD2,
       C(zone):gas_price — because cities span diverse climate/market zones
    3. National DC totals read from the city DC file's "National" column
    4. All cities pooled into one panel (no --iso argument needed)
    5. dc_local is the endogenous variable; bartik_iv is the instrument

Model:
    First stage:
        DC_{it} = π_1·Z_bartik + C(zone):HDD² + C(zone):CDD² + C(zone):GasPrice
                  + city_FE + year_FE + month×dow_FE + ν
    Second stage:
        P_{it} = β_1·DC_hat + C(zone):HDD² + C(zone):CDD² + C(zone):GasPrice
                 + city_FE + year_FE + month×dow_FE + ε

Usage:
    python r02_city_bartik_iv_workflow.py
    python r02_city_bartik_iv_workflow.py --base-year 2020
    python r02_city_bartik_iv_workflow.py --city-dc-cum ./tables_city/city_dc_cumulative.xlsx
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

# City data tables (single-sheet each)
CITY_DATA_PATHS = {
    "t1": "./tables_city/city_prices.xlsx",    # Hourly day-ahead prices by city
    "t2": "./tables_city/city_temp.xlsx",       # Hourly temperature by city
    "t3": "./tables_city/city_fuel.xlsx",       # Fuel hub price by city
    "t5": "./tables_city/city_dc.xlsx",         # Annual DC incremental capacity by city
}

# City cumulative DC file (Year × City, includes "National" column)
# Used for share computation — contains absolute cumulative values
CITY_DC_CUM_FILE = "./tables_city/dc_cumulative_by_city.xlsx"

SHEET_NAME = "Sheet1"

OUTPUT_DIR = Path("./results/r3_bartik_iv/city")

START_DATE = None  # Set dynamically from --base-year
END_DATE = pd.Timestamp("2025-12-31")

BASE_HEAT = 65.0
BASE_COOL = 65.0
HDD_SCALE = 100
CDD_SCALE = 100

PRICE_WINSOR_LOWER = 0.10
PRICE_WINSOR_UPPER = 0.90
FUEL_WINSOR_LOWER = 0.10
FUEL_WINSOR_UPPER = 0.90
DC_THRESH = 0.000

SHARE_BASE_YEAR = 2020

# Robustness: winsorization levels for reduced-form sensitivity checks
ROBUSTNESS_WINSOR_LEVELS = [
    None,                # default
]


# =============================================================================
# Data Reading (from city panel regression)
# =============================================================================

def _open_xls(path):
    path = Path(path)
    assert path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}, f"Expected Excel: {path}"
    return pd.ExcelFile(path)

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


def read_city_prices(path, sheet=SHEET_NAME):
    xls = _open_xls(path)
    df = pd.read_excel(xls, sheet_name=sheet)
    ts_col = _find_ts_col(df.columns)
    df = df.rename(columns={ts_col: "ts"})
    df["ts"] = _to_datetime(df["ts"])
    df = _coerce_numeric(df, except_cols=("ts",)).dropna(subset=["ts"])
    out = df.melt(id_vars="ts", var_name="zone", value_name="lmp_da")
    out["iso"] = sheet
    return out[["iso", "zone", "ts", "lmp_da"]].sort_values(["zone", "ts"])

def read_city_temperature(path, sheet=SHEET_NAME):
    xls = _open_xls(path)
    df = pd.read_excel(xls, sheet_name=sheet)
    ts_col = _find_ts_col(df.columns)
    df = df.rename(columns={ts_col: "ts"}).copy()
    df["ts"] = _to_datetime(df["ts"])
    df = _coerce_numeric(df, except_cols=("ts",)).dropna(subset=["ts"])
    out = df.melt(id_vars="ts", var_name="zone", value_name="temperature")
    out["iso"] = sheet
    return out[["iso", "zone", "ts", "temperature"]].sort_values(["zone", "ts"])

def read_city_fuel(path, sheet=SHEET_NAME):
    """City-level fuel hub price — merged at (zone, date), not ISO-level."""
    xls = _open_xls(path)
    df = pd.read_excel(xls, sheet_name=sheet)
    ts_col = _find_ts_col(df.columns)
    df = df.rename(columns={ts_col: "ts"}).copy()
    df["ts"] = _to_datetime(df["ts"])
    df = _coerce_numeric(df, except_cols=("ts",)).dropna(subset=["ts"])
    out = df.melt(id_vars="ts", var_name="zone", value_name="gas_price")
    out["iso"] = sheet
    return out[["iso", "zone", "ts", "gas_price"]].sort_values(["zone", "ts"])

def read_city_dc_capacity(path, sheet=SHEET_NAME):
    """Read city DC incremental capacity (used for dc_local interpolation)."""
    xls = _open_xls(path)
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
    out["iso"] = sheet
    return out[["iso", "zone", "year", "dc_cum"]].sort_values(["zone", "year"])

def load_all_city_tables(paths):
    print(f"  Loading city prices, temperature, fuel, DC capacity...")
    return {
        "prices": read_city_prices(paths["t1"]),
        "temperature": read_city_temperature(paths["t2"]),
        "fuel": read_city_fuel(paths["t3"]),
        "dc_capacity": read_city_dc_capacity(paths["t5"]),
    }


# =============================================================================
# National DC & Bartik IV (adapted for city cumulative file)
# =============================================================================

def load_national_dc_from_city_file(city_dc_cum_file, sheet=SHEET_NAME):
    """
    Load national DC totals from the city cumulative DC file.
    This file has columns: Year, National, Phoenix, Atlanta, ...
    """
    df = pd.read_excel(city_dc_cum_file, sheet_name=sheet)

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


def load_city_dc_cumulative(city_dc_cum_file, sheet=SHEET_NAME):
    """
    Load city-level cumulative DC capacity (all years, including base year).
    Excludes the 'National' column.
    Returns [zone, year, dc_cum] with iso='City'.
    """
    df = pd.read_excel(city_dc_cum_file, sheet_name=sheet)

    year_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("year",) or str(c).strip().lower().startswith("year"):
            year_col = c; break
    if year_col is None:
        year_col = df.columns[0]

    df = df.rename(columns={year_col: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()

    zone_cols = [c for c in df.columns
                 if c != "year" and str(c).strip().lower() != "national"]
    for c in zone_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df[["year"] + zone_cols].melt(id_vars="year", var_name="zone", value_name="dc_cum")
    out = out.dropna(subset=["dc_cum"])
    out["iso"] = "City"
    return out[["iso", "zone", "year", "dc_cum"]].sort_values(["zone", "year"]).reset_index(drop=True)


def compute_city_shares(dc_city_cum_df, dc_national_df):
    """Compute s_{i,0} = DC_{i, base_year} / DC_{national, base_year} for each city."""
    zone_base = dc_city_cum_df[dc_city_cum_df["year"] == SHARE_BASE_YEAR-1][["zone", "dc_cum"]].copy()
    if zone_base.empty:
        raise ValueError(f"No city DC data for base year {SHARE_BASE_YEAR-1}")

    nat_row = dc_national_df[dc_national_df["year"] == SHARE_BASE_YEAR-1]
    if nat_row.empty:
        raise ValueError(f"No national DC data for {SHARE_BASE_YEAR-1}")
    dc_nat_base = float(nat_row["dc_national"].iloc[0])

    zone_base["share"] = zone_base["dc_cum"] / dc_nat_base
    print(f"  City shares: {len(zone_base)} cities, "
          f"range [{zone_base['share'].min():.6f}, {zone_base['share'].max():.6f}], "
          f"sum={zone_base['share'].sum():.4f}")

    return zone_base[["zone", "share"]].reset_index(drop=True), dc_nat_base


def interpolate_national_dc_daily(dc_national_df):
    """Interpolate national DC capacity from annual to daily."""
    nat = dc_national_df.sort_values("year").copy()
    nat["dc_prev"] = nat["dc_national"].shift(1).fillna(0.0)

    daily_records = []
    for _, row in nat.iterrows():
        yr = int(row["year"])
        target = row["dc_national"]
        prev = row["dc_prev"]
        y_start = pd.Timestamp(f"{yr}-01-01")
        y_end = pd.Timestamp(f"{yr}-12-31")
        total_days = (y_end - y_start).days

        for d in pd.date_range(y_start, y_end, freq="D"):
            elapsed = (d - y_start).days
            frac = min(max(elapsed / total_days, 0.0), 1.0) if total_days > 0 else 1.0
            g_t = prev + (target - prev) * frac
            daily_records.append({"date": d, "dc_national_daily": g_t})

    daily_df = pd.DataFrame(daily_records)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    return daily_df


def build_bartik_iv(panel_df, shares_df, dc_national_daily, dc_nat_base, dc_city_cum_df):
    """
    Construct daily Bartik IV with leave-one-out:
        Z_{it} = s_{i,0} × (G_t^{-i} - G_{base}^{-i})
    where G_t^{-i} = G_national_t - DC_{i,t} excludes city i.
    
    dc_city_cum_df must be the CUMULATIVE capacity file, not incremental.
    """
    df = panel_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.merge(shares_df, on="zone", how="left")
    df["share"] = df["share"].fillna(0.0)

    dc_national_daily = dc_national_daily.copy()
    dc_national_daily["date"] = pd.to_datetime(dc_national_daily["date"])
    df = df.merge(dc_national_daily, on="date", how="left")

    # Interpolate city cumulative DC to daily (for leave-one-out)
    df["year"] = df["date"].dt.year
    zone_cum = dc_city_cum_df[["zone", "year", "dc_cum"]].copy().sort_values(["zone", "year"])
    zone_cum["dc_cum_prev"] = zone_cum.groupby("zone")["dc_cum"].shift(1).fillna(0.0)

    df = df.merge(zone_cum[["zone", "year", "dc_cum", "dc_cum_prev"]],
                  on=["zone", "year"], how="left")
    df["dc_cum"] = df["dc_cum"].fillna(0.0)
    df["dc_cum_prev"] = df["dc_cum_prev"].fillna(0.0)

    year_start = pd.to_datetime(df["year"].astype(str) + "-01-01")
    year_end = pd.to_datetime(df["year"].astype(str) + "-12-31")
    d0 = df["date"].dt.normalize()
    elapsed = (d0 - year_start).dt.days.astype("int64")
    total = (year_end - year_start).dt.days.astype("int64")
    frac = (elapsed / total).clip(0.0, 1.0)
    df["dc_zone_daily"] = df["dc_cum_prev"] + (df["dc_cum"] - df["dc_cum_prev"]) * frac

    # Leave-one-out
    df["g_leave_one_out"] = df["dc_national_daily"] - df["dc_zone_daily"]

    # Base value: end of (SHARE_BASE_YEAR - 1), matching dc_nat_base
    base_yr = SHARE_BASE_YEAR - 1
    dc_base_by_zone = dc_city_cum_df[dc_city_cum_df["year"] == base_yr][["zone", "dc_cum"]].copy()
    dc_base_by_zone = dc_base_by_zone.rename(columns={"dc_cum": "dc_zone_base"})
    df = df.merge(dc_base_by_zone, on="zone", how="left")
    df["dc_zone_base"] = df["dc_zone_base"].fillna(0.0)
    df["g_base_loo"] = dc_nat_base - df["dc_zone_base"]

    # Bartik IV
    df["bartik_iv"] = df["share"] * (df["g_leave_one_out"] - df["g_base_loo"])
    df["bartik_iv"] = df["bartik_iv"] / 1000.0

    print(f"  Bartik IV (leave-one-out) range: [{df['bartik_iv'].min():.6f}, {df['bartik_iv'].max():.6f}]")

    df = df.drop(columns=["dc_cum", "dc_cum_prev", "dc_zone_daily",
                           "g_leave_one_out", "dc_zone_base", "g_base_loo"], errors="ignore")
    return df


# =============================================================================
# Data Processing (from city panel regression)
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
    # Fuel: city-level (iso, zone, date)
    df = data_hour["fuel"].copy()
    df["date"] = pd.to_datetime(df["ts"])
    num_cols = df.select_dtypes(include="number").columns
    fuel_daily = df.groupby(["iso", "zone", "date"])[num_cols].mean().reset_index()
    fuel_daily["gas_price"] = fuel_daily["gas_price"].astype(float)
    data["fuel"] = fuel_daily
    data["dc_capacity"] = data_hour["dc_capacity"].copy()
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
    """Interpolate city-level DC capacity to daily (same logic as ISO version)."""
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
# Panel Construction
# =============================================================================

def build_city_panel(data, shares_df, dc_national_daily, dc_nat_base, dc_city_cum_df,
                     price_winsor_override=None):
    """Build the regression-ready panel for all cities, including leave-one-out Bartik IV.
    
    Parameters
    ----------
    price_winsor_override : tuple or None
        If given, (lower, upper) quantiles to override default winsorization.
    """
    prices = data["prices"].copy()
    temps = data["temperature"][
        ["iso", "zone", "date", "temperature", "HDD", "CDD", "HDD2", "CDD2"]].copy()
    fuel = data["fuel"][["iso", "zone", "date", "gas_price"]].copy()
    dc = data["dc_capacity"].copy()

    for d in [prices, temps, fuel]:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")

    prices["price_diff"] = prices["lmp_da"]

    # Merge — fuel at city level (iso, zone, date)
    df = prices.merge(temps, on=["iso", "zone", "date"], how="left") \
               .merge(fuel, on=["iso", "zone", "date"], how="left")

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

    # Interpolate DC capacity
    df = interpolate_dc_capacity(df, dc)

    # Time variables
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek

    # Add Bartik IV
    df = build_bartik_iv(df, shares_df, dc_national_daily, dc_nat_base, dc_city_cum_df)

    zone_to_region = {
        # ── SRSG (Southwest Reserve Sharing Group) — Arizona ──
        "Phoenix":        "SRSG",
        "Mesa":           "SRSG",
        "Goodyear":       "SRSG",
        "Chandler":       "SRSG",
        # ── NWPP-US+RMRG (Northwest Power Pool + Rocky Mountain) ──
        # Oregon, Washington, Nevada (north)
        "Hillsboro":      "NWPP-USpRMRG",   # OR
        "Boardman":       "NWPP-USpRMRG",   # OR
        "Seattle":        "NWPP-USpRMRG",   # WA
        "Tukwila":        "NWPP-USpRMRG",   # WA
        "Quincy":         "NWPP-USpRMRG",   # WA
        "Sparks":         "NWPP-USpRMRG",   # NV (Reno area, NV Energy → NWPP)
        "Reno":           "NWPP-USpRMRG",   # NV
        "Las Vegas":      "NWPP-USpRMRG",   # NV (NV Energy)
        # ── SERC ──
        "Atlanta":        "SERC",
        "Huntsville":     "SERC",   # AL
        "Lithia Springs": "SERC",   # GA (Atlanta metro)
        "Charlotte":      "SERC",    # NC
        "Moncks Corner":  "SERC",    # SC
        "Miami":          "SERC",   # FL
    }
    df["region"] = df["zone"].map(zone_to_region).fillna("Other")

    # Select columns
    need_cols = [
        "price_diff", "dc_local", "dc_external", "dc_iso_total",
        "temperature", "HDD", "CDD", "HDD2", "CDD2",
        "gas_price", "zone", "region", "month", "year", "dow", "date",
        "bartik_iv", "share",
    ]
    reg = df[need_cols].dropna().copy()
    print(f"  Sample: {len(reg):,} rows, {reg['zone'].nunique()} cities")

    # Filter trivial DC cities
    reg["dc_local"] = pd.to_numeric(reg["dc_local"], errors="coerce").fillna(0.0)
    zone_max = reg.groupby("zone")["dc_local"].apply(lambda s: s.abs().max())
    keep_zones = sorted(zone_max[zone_max > DC_THRESH].index.tolist())
    drop_zones = sorted(zone_max[zone_max <= DC_THRESH].index.tolist())
    if drop_zones:
        print(f"  Dropped cities: {drop_zones}")
    print(f"  Retained cities: {keep_zones}")

    # Winsorize
    pw_lo = price_winsor_override[0] if price_winsor_override else PRICE_WINSOR_LOWER
    pw_hi = price_winsor_override[1] if price_winsor_override else PRICE_WINSOR_UPPER
    reg1 = winsorize_by_group(reg.loc[reg["zone"].isin(keep_zones)].copy(),
                              ["price_diff"], pw_lo, pw_hi, "zone")
    reg2 = winsorize_by_group(reg1, ["gas_price"], FUEL_WINSOR_LOWER, FUEL_WINSOR_UPPER, "zone")
    print(f"  After winsorization: {len(reg2):,} rows")

    return reg, reg2


# =============================================================================
# Regression: OLS + IV2SLS (city-specific formulas)
# =============================================================================

def run_ols(panel_df):
    """OLS with region-interacted controls to balance heterogeneity and degrees of freedom."""
    formula = (
        "price_diff ~ C(region):HDD2 + C(region):CDD2 + C(region):gas_price "
        "+ dc_local + C(year) + C(month):C(dow) + C(zone)"
    )
    panel_df = panel_df.copy()
    panel_df["date"] = pd.to_datetime(panel_df["date"])
    panel = panel_df.set_index(["zone", "date"]).sort_index()
    panel = panel.assign(zone=lambda d: d.index.get_level_values("zone"))

    res = PanelOLS.from_formula(formula, data=panel).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=7
    )
    return res


def run_iv2sls(panel_df):
    """
    Bartik IV 2SLS with region-interacted controls.
    Use C(zone) for city fixed effects intercepts, and C(region) for climate/fuel slopes.
    """
    formula = (
        "price_diff ~ C(zone) + C(region):HDD2 + C(region):CDD2 + C(region):gas_price "
        "+ C(year) + C(month):C(dow) "
        "+ [dc_local ~ bartik_iv]"
    )
    res = IV2SLS.from_formula(formula, data=panel_df).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=7
    )
    return res


def run_first_stage_explicit(panel_df):
    """First stage with region-interacted controls for diagnostics."""
    formula = (
        "dc_local ~ bartik_iv + C(region):HDD2 + C(region):CDD2 + C(region):gas_price "
        "+ C(year) + C(month):C(dow) + C(zone)"
    )
    panel_df = panel_df.copy()
    panel_df["date"] = pd.to_datetime(panel_df["date"])
    panel = panel_df.set_index(["zone", "date"]).sort_index()
    panel = panel.assign(zone=lambda d: d.index.get_level_values("zone"))

    res = PanelOLS.from_formula(formula, data=panel).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=7
    )
    return res


def run_reduced_form(panel_df):
    """
    Reduced form: regress price DIRECTLY on the instrument.
    Always consistent regardless of instrument strength.
    Wald ratio = RF coef / first-stage coef = implied causal effect.
    """
    formula = (
        "price_diff ~ bartik_iv + C(region):HDD2 + C(region):CDD2 + C(region):gas_price "
        "+ C(year) + C(month):C(dow) + C(zone)"
    )
    panel_df = panel_df.copy()
    panel_df["date"] = pd.to_datetime(panel_df["date"])
    panel = panel_df.set_index(["zone", "date"]).sort_index()
    panel = panel.assign(zone=lambda d: d.index.get_level_values("zone"))

    res = PanelOLS.from_formula(formula, data=panel).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=7
    )
    return res



def run_rf_robustness(data, shares_df, dc_national_daily,
                      dc_nat_base, dc_city_cum_df, res_first_baseline):
    """
    Reduced-form robustness: sweep winsorization levels to test whether
    the causal channel (instrument → price) is robust for non-ISO cities.
    """
    print(f"\n{'#'*70}")
    print(f"# NON-ISO CITIES: REDUCED-FORM ROBUSTNESS")
    print(f"{'#'*70}")

    results_table = []
    fs_coef = float(res_first_baseline.params.get("bartik_iv", 0))

    for i, winsor in enumerate(ROBUSTNESS_WINSOR_LEVELS):
        pct_label = f"{winsor[0]*100:.0f}%/{winsor[1]*100:.0f}%" if winsor else "default"
        print(f"\n  --- Winsorization: {pct_label} ---")
        try:
            reg_raw_rf, panel = build_city_panel(data, shares_df,
                                                 dc_national_daily, dc_nat_base, dc_city_cum_df,
                                                 price_winsor_override=winsor)
            if len(panel) == 0:
                print(f"    SKIP: empty panel"); continue

            res_ols = run_ols(panel)
            res_iv = run_iv2sls(panel)
            res_rf = run_reduced_form(panel)

            row = {"winsor": pct_label, "N": len(panel)}

            if "dc_local" in res_ols.params.index:
                row["ols_coef"] = float(res_ols.params["dc_local"])
                row["ols_p"] = float(res_ols.pvalues["dc_local"])
            if "dc_local" in res_iv.params.index:
                row["iv_coef"] = float(res_iv.params["dc_local"])
                row["iv_p"] = float(res_iv.pvalues["dc_local"])
            if "bartik_iv" in res_rf.params.index:
                rf_coef = float(res_rf.params["bartik_iv"])
                row["rf_coef"] = rf_coef
                row["rf_p"] = float(res_rf.pvalues["bartik_iv"])
                row["wald"] = rf_coef / fs_coef if fs_coef != 0 else np.nan

            results_table.append(row)

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    if results_table:
        print(f"\n{'='*105}")
        print(f"  CITIES REDUCED-FORM ROBUSTNESS TABLE")
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

        any_rf_sig = any(r.get("rf_p", 1) < 0.05 for r in results_table)
        all_rf_sig = all(r.get("rf_p", 1) < 0.10 for r in results_table)

        print(f"\n  Wald ratio = RF / first-stage = implied causal effect ($/MWh per GW)")
        if all_rf_sig:
            print(f"  ✓ Reduced form significant across ALL winsorization levels.")
        elif any_rf_sig:
            print(f"  ~ Reduced form significant in SOME specifications.")
        else:
            print(f"  ✗ Reduced form not significant in any specification.")

        rob_dir = OUTPUT_DIR / "cities"
        rob_dir.mkdir(parents=True, exist_ok=True)
        rob_df = pd.DataFrame(results_table)
        rob_df.to_excel(rob_dir / "rf_robustness_cities.xlsx", index=False)
        rob_df.to_csv(rob_dir / "rf_robustness_cities.csv", index=False)
        print(f"  Saved to {rob_dir}")

    return results_table


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


def print_comparison(res_ols, res_iv, res_first):
    """Print OLS vs IV comparison with diagnostic decomposition. res_iv can be None."""
    print(f"\n{'='*75}")
    print(f"  RESULTS FOR NON-ISO CITIES: OLS vs. Bartik IV")
    print(f"{'='*75}")

    if "bartik_iv" in res_first.params.index:
        pi1 = res_first.params["bartik_iv"]
        se1 = res_first.std_errors["bartik_iv"]
        t1 = res_first.tstats["bartik_iv"]
        p1 = res_first.pvalues["bartik_iv"]
        f_stat = t1 ** 2
        print(f"\n  FIRST STAGE:")
        print(f"    π_1 (bartik_iv):  {pi1:.6f}  (SE={se1:.6f}, t={t1:.2f}, p={p1:.4f})")
        print(f"    First-stage F:    {f_stat:.2f}  {'✓ > 10' if f_stat > 10 else '⚠ WEAK INSTRUMENT'}")
        print(f"    R² (within):      {res_first.rsquared:.4f}")

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

    # Enhanced diagnostics
    if res_iv is not None and "dc_local" in res_ols.params.index and "dc_local" in res_iv.params.index:
        ols_coef = res_ols.params["dc_local"]
        ols_se = res_ols.std_errors["dc_local"]
        iv_coef = res_iv.params["dc_local"]
        iv_se = res_iv.std_errors["dc_local"]

        print(f"\n  DIAGNOSTIC DECOMPOSITION:")
        if ols_coef != 0:
            print(f"    Coefficient change (IV - OLS):  {iv_coef - ols_coef:>+10.4f} "
                  f"({(iv_coef - ols_coef)/abs(ols_coef)*100:>+.1f}%)")
        print(f"    SE inflation (IV / OLS):        {iv_se / ols_se:>10.2f}x")

        iv_ci = res_iv.conf_int(level=0.95).loc["dc_local"]
        ols_in_iv_ci = iv_ci.iloc[0] <= ols_coef <= iv_ci.iloc[1]
        print(f"    OLS coef within IV 95% CI:      {'Yes' if ols_in_iv_ci else 'No'}")

        if iv_se / ols_se > 2.0:
            print(f"\n  → SE inflated >{iv_se/ols_se:.1f}x: "
                  f"likely a PRECISION problem, not absence of effect")
        elif abs(iv_coef) < abs(ols_coef) * 0.5:
            print(f"\n  → IV coefficient much smaller: OLS may be upward-biased")
        elif iv_coef > ols_coef:
            print(f"\n  → IV > OLS: downward OLS bias (reverse causality)")
        else:
            print(f"\n  → IV ≈ OLS: no substantial endogeneity bias")

    elif res_iv is None:
        print(f"\n  → IV skipped or discarded (weak instrument or SE too inflated).")

    print(f"{'='*75}\n")


def save_results(res_ols, res_iv, res_first, panel_df, reg_raw):
    """Save all results. res_iv can be None."""
    out_dir = OUTPUT_DIR / "cities"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = "cities"
    sheet_tag = tag[:31]

    # 1. Coefficient tables
    save_pairs = [(res_ols, "ols"), (res_first, "first_stage")]
    if res_iv is not None:
        save_pairs.append((res_iv, "iv2sls"))
    for res, name in save_pairs:
        rdf = result_to_df(res)
        rdf.to_excel(out_dir / f"coefficients_{name}.xlsx", sheet_name=sheet_tag)
    print(f"  Saved coefficient tables")

    # 2. Summary JSON
    summary = {"panel": "non-ISO cities"}
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
        summary["iv_skipped"] = "Weak instrument (F<10) or IV SE too inflated (>5x OLS SE)"

    with open(out_dir / "summary_cities.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary JSON")

    # 3. Model summaries
    with open(out_dir / "model_summaries_cities.txt", "w") as f:
        f.write(f"{'='*80}\nBartik IV Results for Non-ISO Cities\n{'='*80}\n\n")
        f.write("  Bartik IV uses leave-one-out national growth.\n\n")
        f.write(f"FIRST STAGE\n{'-'*80}\n{res_first.summary}\n\n")
        if res_iv is not None:
            f.write(f"SECOND STAGE (IV2SLS)\n{'-'*80}\n{res_iv.summary}\n\n")
        else:
            f.write("SECOND STAGE (IV2SLS): SKIPPED — weak instrument or SE too inflated\n\n")
        f.write(f"OLS COMPARISON\n{'-'*80}\n{res_ols.summary}\n\n")
    print(f"  Saved model summaries")

    # 4. Panel data
    panel_df.to_csv(out_dir / "panel_with_bartik_cities.csv", index=False)
    print(f"  Saved panel data")

    # ================= Save each city's untrimmed mean price =================
    zone_mean_df = (
        pd.to_numeric(reg_raw["price_diff"], errors="coerce")
        .groupby(reg_raw["zone"]).mean()
        .rename("avg_price")
        .reset_index()
        .sort_values("zone")
    )
    zone_mean_df["avg_price"] = zone_mean_df["avg_price"].round(6)

    out_file = out_dir / "zone_price_diff_means.xlsx"
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

def main():
    global SHARE_BASE_YEAR, OUTPUT_DIR, START_DATE

    parser = argparse.ArgumentParser(
        description="Bartik IV 2SLS for non-ISO cities (leave-one-out)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python r02_city_bartik_iv_workflow.py
    python r02_city_bartik_iv_workflow.py --base-year 2020
    python r02_city_bartik_iv_workflow.py --city-dc-cum ./my_cumulative.xlsx
        """
    )
    parser.add_argument("--city-dc-cum", default=CITY_DC_CUM_FILE,
                        help=f"City cumulative DC file (has National column). "
                             f"(default: {CITY_DC_CUM_FILE})")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--base-year", type=int, default=SHARE_BASE_YEAR)

    args = parser.parse_args()
    SHARE_BASE_YEAR = args.base_year
    START_DATE = pd.Timestamp(f"{SHARE_BASE_YEAR}-01-01")
    OUTPUT_DIR = Path(args.out_dir)

    # Validate files
    for k, p in CITY_DATA_PATHS.items():
        if not Path(p).exists():
            print(f"ERROR: {p} not found"); sys.exit(1)
    if not Path(args.city_dc_cum).exists():
        print(f"ERROR: {args.city_dc_cum} not found"); sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  Bartik IV 2SLS — Non-ISO Cities (leave-one-out)")
    print(f"  Base year: {SHARE_BASE_YEAR}")
    print(f"  Study period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"{'='*70}")

    # Load national DC from city cumulative file
    print("\n[0] Loading national DC capacity...")
    dc_national_df = load_national_dc_from_city_file(args.city_dc_cum)
    dc_national_daily = interpolate_national_dc_daily(dc_national_df)

    nat_base_row = dc_national_df[dc_national_df["year"] == SHARE_BASE_YEAR - 1]
    if nat_base_row.empty:
        nat_base_row = dc_national_df[dc_national_df["year"] == SHARE_BASE_YEAR]
    dc_nat_base = float(nat_base_row["dc_national"].iloc[0])
    print(f"  Growth baseline (end of {SHARE_BASE_YEAR - 1}): {dc_nat_base:,.1f} MW")

    # Compute city shares
    print("\n[1] Computing city shares...")
    dc_city_cum = load_city_dc_cumulative(args.city_dc_cum)
    shares_df, _ = compute_city_shares(dc_city_cum, dc_national_df)

    # Load city data
    print("\n[2] Loading city data...")
    data_hour = load_all_city_tables(CITY_DATA_PATHS)
    print("  Aggregating to daily...")
    data = aggregate_to_daily(data_hour)
    data["temperature"] = compute_degree_days(data["temperature"])

    # Build panel (pass dc_city_cum for leave-one-out)
    print("\n[3] Building panel with Bartik IV (leave-one-out)...")
    reg_raw, panel_df = build_city_panel(data, shares_df, dc_national_daily, dc_nat_base, dc_city_cum)

    if len(panel_df) == 0:
        print("  ERROR: No data. Exiting."); sys.exit(1)

    # Diagnostic: share table
    diag = panel_df.groupby("zone").agg(
        share=("share", "first"),
        dc_start=("dc_local", lambda s: s.iloc[0]),
        dc_end=("dc_local", lambda s: s.iloc[-1]),
        bartik_min=("bartik_iv", "min"),
        bartik_max=("bartik_iv", "max"),
    ).sort_values("share", ascending=False).round(6)
    print(f"\n  Share diagnostic:")
    print(diag.to_string())

    # [4] First stage
    print("\n[4] Running first stage...")
    res_first = run_first_stage_explicit(panel_df)

    f_stat = 0.0
    if "bartik_iv" in res_first.params.index:
        t1 = float(res_first.tstats["bartik_iv"])
        f_stat = t1 ** 2
        print(f"  π_1 = {res_first.params['bartik_iv']:.6f}  "
              f"(t={t1:.2f}, F={f_stat:.1f}, p={res_first.pvalues['bartik_iv']:.4f})")

    if f_stat < 10:
        print(f"\n  ⚠ WEAK INSTRUMENT (F={f_stat:.1f} < 10)")
        print(f"    Skipping IV 2SLS. Reporting OLS only.")
        print("\n[5] Running OLS panel FE...")
        res_ols = run_ols(panel_df)
        print_comparison(res_ols, None, res_first)
        print("[6] Saving results...")
        save_results(res_ols, None, res_first, panel_df, reg_raw)
        print(f"\n  ✓ Non-ISO cities done (weak IV — OLS only).")
        print(f"{'='*70}")
        return

    # [5] OLS
    print(f"  ✓ Strong instrument (F={f_stat:.1f})")
    print("\n[5] Running OLS panel FE...")
    res_ols = run_ols(panel_df)

    # [6] IV 2SLS
    print("\n[6] Running Bartik IV 2SLS...")
    res_iv = run_iv2sls(panel_df)

    # SE ratio check
    iv_noisy = False
    if "dc_local" in res_ols.params.index and "dc_local" in res_iv.params.index:
        se_ols = res_ols.std_errors["dc_local"]
        se_iv = res_iv.std_errors["dc_local"]
        if se_iv > 10 * se_ols:
            iv_noisy = True
            print(f"\n  ⚠ IV SE ({se_iv:.4f}) is {se_iv/se_ols:.1f}x OLS SE ({se_ols:.4f})")
            print(f"    IV reported but too noisy for further diagnostics.")

    # Always print comparison (report IV even if noisy)
    print_comparison(res_ols, res_iv, res_first)

    # If IV is too noisy, skip further diagnostics
    if iv_noisy:
        print("  Skipping reduced-form robustness (IV SE too inflated).")
        print("\n[7] Saving results...")
        save_results(res_ols, res_iv, res_first, panel_df, reg_raw)
        print(f"\n  ✓ Non-ISO cities done (IV reported but noisy).")
        print(f"{'='*70}")
        return

    # Check IV significance
    iv_sig = False
    if "dc_local" in res_iv.params.index:
        iv_p = float(res_iv.pvalues["dc_local"])
        iv_sig = iv_p < 0.10

    if iv_sig:
        # ── IV SIGNIFICANT: full causal evidence ──
        print(f"\n  ✓ IV is significant (p={iv_p:.4f}): direct causal evidence.")
        res_rf = run_reduced_form(panel_df)
        if "bartik_iv" in res_rf.params.index:
            rf_coef = res_rf.params["bartik_iv"]
            fs_coef = res_first.params["bartik_iv"]
            print(f"  Reduced form (supplementary): π_RF={rf_coef:.4f}, "
                  f"Wald ratio={rf_coef/fs_coef:.4f}")
    else:
        # ── IV NOT SIGNIFICANT: run reduced-form robustness ──
        print(f"\n  ⚠ IV is not significant (p={iv_p:.4f}) despite strong first stage.")
        print(f"    Running reduced-form robustness to test causal channel...")
        run_rf_robustness(data, shares_df, dc_national_daily,
                          dc_nat_base, dc_city_cum, res_first)

    # [7] Save
    print("\n[7] Saving results...")
    save_results(res_ols, res_iv, res_first, panel_df, reg_raw)

    print(f"\n  ✓ Non-ISO cities done.")
    print(f"  Results: {OUTPUT_DIR.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
