#!/usr/bin/env python3
"""
Bartik IV Results Summary (v3)
==============================
Reads all region results from r3_bartik_iv/ and produces:
  1. Regression summary table (paper-ready, all regions)
  2. Per-GW price impact (OLS vs causal estimate, with 95% CI)
  3. Zone-level total price increase by 2025 (every load zone / city)
  4. Zone-level RELATIVE price increase by 2025 (as % of 2019 price)
     Requires --iso-price and --city-price Excel files containing annual
     average prices 2015-2020, with ISO workbooks having one sheet per ISO
     and the city workbook having a single sheet.

Causal estimate logic:
  - IV significant (p<0.10) + non-noisy → use 2SLS β and CI directly
  - IV not significant BUT Reduced Form significant → use Wald ratio
    (RF_coef / FS_coef) with delta-method CI
  - Weak IV / noisy / nothing significant → exclude from Tables 2 & 3

For Wald CI when only RF is significant:
    |t_rf|  = Φ^{-1}(1 - p_rf/2)          (back-compute from saved p-value)
    se_rf   = |rf_coef| / |t_rf|
    se_wald = se_rf / |fs_coef|            (delta method, FS treated as fixed)
    CI      = wald ± 1.96 × se_wald

Usage:
    python r02_summary.py
    python r02_summary.py --input-dir ./r3_bartik_iv --out-dir ./r3_summary \
        --iso-price ./data/iso_annual_prices.xlsx \
        --city-price ./data/city_annual_prices.xlsx \
        --base-year 2019
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
INPUT_DIR = Path("./results/r3_bartik_iv")
OUTPUT_DIR = Path("./results/r3_summary")
SIGNIFICANCE_LEVEL = 0.10
IV_SE_INFLATION_CUTOFF = 10   # SE_IV > 10× SE_OLS → noisy
BASE_YEAR = 2019              # Baseline year for relative-growth calculation


# =============================================================================
# Data loaders
# =============================================================================

def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_ci(block, key_lo="dc_local_ci_lower", key_hi="dc_local_ci_upper",
           key_list="dc_local_ci"):
    """Extract CI, handling both ISO format (separate keys) and City format (list)."""
    lo = block.get(key_lo)
    hi = block.get(key_hi)
    if lo is not None and hi is not None:
        return float(lo), float(hi)
    ci = block.get(key_list)
    if isinstance(ci, list) and len(ci) == 2:
        return float(ci[0]), float(ci[1])
    return np.nan, np.nan


def _norm_zone(name):
    """Normalize zone names for fuzzy matching (case/whitespace-insensitive)."""
    if name is None:
        return ""
    return str(name).strip().upper().replace(" ", "").replace("_", "").replace("-", "")


def _read_long_price_xlsx(path):
    """
    Read a long-format price xlsx with columns `zone` and `price_diff_mean`.
    Returns {normalized_zone: price} or {} on failure.
    """
    if not path.exists():
        return {}
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"  WARNING: failed to read {path}: {e}")
        return {}

    if df.empty:
        return {}

    # Find the zone and price columns (case-insensitive)
    zone_col = None
    price_col = None
    for c in df.columns:
        cn = str(c).strip().lower()
        if cn == "zone":
            zone_col = c
        elif cn in ("price_diff_mean", "price", "price_mean","avg_price"):
            price_col = c
    if zone_col is None or price_col is None:
        print(f"  WARNING: {path.name} missing 'zone' or 'price_diff_mean' column")
        return {}

    prices = {}
    for _, row in df.iterrows():
        zone = row[zone_col]
        val = row[price_col]
        if pd.isna(zone) or pd.isna(val):
            continue
        try:
            prices[_norm_zone(zone)] = float(val)
        except (TypeError, ValueError):
            continue
    return prices


def load_iso_price_tables(panel_root):
    """
    Load ISO base prices from r1_panel_result/<ISO>/zone_price_diff_means.xlsx.
    Returns {iso_normalized_name: {zone: price}}.
    """
    out = {}
    root = Path(panel_root)
    if not root.exists():
        print(f"  WARNING: ISO price root not found: {root}")
        return out

    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.upper() == "CITY":
            continue
        path = sub / "zone_price_diff_means.xlsx"
        if not path.exists():
            continue
        prices = _read_long_price_xlsx(path)
        if prices:
            out[_norm_zone(sub.name)] = prices
    return out


def load_city_price_table(panel_root):
    """
    Load city base prices from r1_panel_result/CITY/cities/city_price_means.xlsx.
    Returns {city_normalized_name: price}.
    """
    root = Path(panel_root)
    # Try both CITY and city (case sometimes matters on Linux)
    for city_dir_name in ("CITY", "city"):
        path = root / city_dir_name / "cities" / "zone_price_diff_means.xlsx"
        if path.exists():
            return _read_long_price_xlsx(path)
    print(f"  WARNING: city price file not found under {root}")
    return {}


def lookup_base_price(iso_tables, city_table, region_name, zone_name, is_city=False):
    """
    Look up the base-year price for a given (region, zone).

    City: zone name is looked up directly in the flat city_table.
    ISO:  region_name selects the ISO sub-dict, then zone_name is the key.

    Matching is case/whitespace-insensitive. Returns float or NaN.
    """
    target_zone = _norm_zone(zone_name)

    if is_city:
        if not city_table:
            return np.nan
        return city_table.get(target_zone, np.nan)

    if not iso_tables:
        return np.nan
    target_region = _norm_zone(region_name)
    sub = iso_tables.get(target_region)
    if sub is None:
        # Fuzzy fallback: folder name contains or is contained by region
        for k, v in iso_tables.items():
            if target_region in k or k in target_region:
                sub = v
                break
    if sub is None:
        return np.nan
    return sub.get(target_zone, np.nan)


def load_rf_robustness(region_dir, region_tag, is_city=False):
    """
    Load reduced-form robustness results.
    Returns the BASELINE row (first winsorization level) or None.
    """
    if is_city:
        path = region_dir / "rf_robustness_cities.xlsx"
        if not path.exists():
            path = region_dir / "rf_robustness_cities.csv"
    else:
        path = region_dir / f"rf_robustness_{region_tag}.xlsx"
        if not path.exists():
            path = region_dir / f"rf_robustness_{region_tag}.csv"

    if not path.exists():
        return None

    try:
        if path.suffix == ".xlsx":
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        if df.empty:
            return None
        # Return baseline row (first row = default winsorization)
        return df.iloc[0].to_dict()
    except Exception:
        return None


# =============================================================================
# Classify region workflow status
# =============================================================================

def classify_region(summary):
    """
    Returns:
        'weak_iv'   : F<10 or iv_skipped
        'iv_noisy'  : IV SE > 10× OLS SE
        'iv_sig'    : IV significant (p<0.10) and non-noisy
        'iv_insig'  : IV not significant, non-noisy (may have RF robustness)
        'ols_only'  : only OLS (weak IV in city workflow)
    """
    if "iv_skipped" in summary:
        return "weak_iv"

    ols = summary.get("ols", {})
    iv = summary.get("iv2sls", {})
    fs = summary.get("first_stage", {})

    if not iv:
        return "ols_only" if ols else "weak_iv"

    if fs.get("F_stat", 0) < 10:
        return "weak_iv"

    ols_se = ols.get("dc_local_se")
    iv_se = iv.get("dc_local_se")
    if ols_se and iv_se and ols_se > 0 and iv_se / ols_se > IV_SE_INFLATION_CUTOFF:
        return "iv_noisy"

    if iv.get("dc_local_pval", 1.0) < SIGNIFICANCE_LEVEL:
        return "iv_sig"
    return "iv_insig"


# =============================================================================
# Compute the "best causal estimate" for each region
# =============================================================================

def compute_causal_estimate(region_info):
    """
    Determine the best available causal estimate for a region.

    Returns dict:
        {
            source: 'iv2sls' | 'wald' | None,
            beta: float,
            se: float,
            p: float,
            ci_lo: float,
            ci_hi: float,
        }
    or None if no credible causal estimate.
    """
    s = region_info["summary"]
    status = region_info["status"]

    if s is None or status in ("weak_iv", "ols_only"):
        return None

    # ── Case 1: IV significant and non-noisy → use 2SLS directly ──
    if status == "iv_sig":
        iv = s["iv2sls"]
        ci_lo, ci_hi = get_ci(iv)
        return {
            "source": "iv2sls",
            "beta": iv["dc_local_coef"],
            "se": iv["dc_local_se"],
            "p": iv["dc_local_pval"],
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }

    # ── Case 2: IV noisy → no reliable IV estimate ──
    if status == "iv_noisy":
        return None

    # ── Case 3: IV not significant → check reduced form ──
    if status == "iv_insig":
        is_city = region_info["type"] == "City"
        rf_row = load_rf_robustness(
            region_info["dir"], region_info["region"], is_city=is_city
        )
        if rf_row is None:
            return None

        rf_coef = rf_row.get("rf_coef")
        rf_p = rf_row.get("rf_p")
        wald = rf_row.get("wald")
        fs = s.get("first_stage", {})
        fs_coef = fs.get("pi_1", 0)

        # RF must be significant
        if rf_p is None or rf_p >= SIGNIFICANCE_LEVEL:
            return None
        if fs_coef == 0 or rf_coef is None:
            return None

        # Back-compute RF standard error from p-value
        #   |t_rf| = Φ^{-1}(1 - p_rf/2)
        #   se_rf  = |rf_coef| / |t_rf|
        t_rf_abs = sp_stats.norm.ppf(1 - rf_p / 2)
        if t_rf_abs <= 0:
            return None
        rf_se = abs(rf_coef) / t_rf_abs

        # Wald ratio and delta-method SE
        wald_beta = rf_coef / fs_coef  # same as 2SLS asymptotically
        wald_se = rf_se / abs(fs_coef)
        wald_ci_lo = wald_beta - 1.96 * wald_se
        wald_ci_hi = wald_beta + 1.96 * wald_se

        return {
            "source": "wald",
            "beta": wald_beta,
            "se": wald_se,
            "p": rf_p,  # report RF p-value as significance indicator
            "ci_lo": wald_ci_lo,
            "ci_hi": wald_ci_hi,
            "rf_coef": rf_coef,
            "rf_se": rf_se,
            "rf_p": rf_p,
        }

    return None


# =============================================================================
# Zone-level DC growth computation
# =============================================================================

def compute_zone_dc_growth(panel_path):
    """
    Read panel CSV and compute per-zone DC growth.

    dc_end_2025 is linearly EXTRAPOLATED to 2025-12-31 using the slope implied
    by the two 2025 anchor points: 2025-01-01 and the last observed day of
    2025 in the panel (typically ~2025-07-30). This matches the upstream
    linear-interpolation scheme used to build the panel, so projecting along
    the same line to year-end is consistent with how the intermediate values
    were generated.

    Fallbacks:
      - If 2025-01-01 is missing, use the earliest 2025 observation as y0.
      - If the zone has only one 2025 observation (can't determine slope),
        fall back to the last observed value (no extrapolation).
      - If the zone has no 2025 observations at all, fall back to the last
        panel value.

    Returns:
        list of dicts: [{zone, dc_start, dc_end_2025, dc_growth, n_obs}, ...]
        meta: {n_zones, n_obs, date_start, date_end}
    """
    if not Path(panel_path).exists():
        return None, None

    df = pd.read_csv(panel_path, parse_dates=["date"])
    if df.empty:
        return None, None

    target_end = pd.Timestamp("2025-12-31")
    year_start = pd.Timestamp("2025-01-01")

    zones = []
    for zone, grp in df.groupby("zone"):
        grp = grp.sort_values("date")
        dc_start = grp["dc_local"].iloc[0]

        # Restrict to 2025 rows for the extrapolation
        g25 = grp[(grp["date"] >= year_start) & (grp["date"] <= target_end)]

        if g25.empty:
            # No 2025 data at all — fall back to last observed value
            dc_end = grp["dc_local"].iloc[-1]
        elif len(g25) == 1:
            # Only one 2025 obs — can't fit a slope, use it as-is
            dc_end = g25["dc_local"].iloc[0]
        else:
            # Anchor 1: first observed date ≥ 2025-01-01 (usually exactly Jan 1)
            d0 = g25["date"].iloc[0]
            y0 = g25["dc_local"].iloc[0]
            # Anchor 2: last observed 2025 date (e.g. 2025-07-30)
            d1 = g25["date"].iloc[-1]
            y1 = g25["dc_local"].iloc[-1]

            span_days = (d1 - d0).days
            if span_days <= 0:
                dc_end = y1
            else:
                slope_per_day = (y1 - y0) / span_days
                extrap_days = (target_end - d0).days
                dc_end = y0 + slope_per_day * extrap_days

        zones.append({
            "zone": zone,
            "dc_start": dc_start,
            "dc_end_2025": dc_end,
            "dc_growth": dc_end - dc_start,
            "n_obs": len(grp),
        })

    meta = {
        "n_zones": df["zone"].nunique(),
        "n_obs": len(df),
        "date_start": df["date"].min(),
        "date_end": df["date"].max(),
    }

    return zones, meta


# =============================================================================
# Discover all regions
# =============================================================================

def discover_regions(input_dir):
    regions = []

    # ISO regions
    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir() or sub.name in ("city",):
            continue
        iso_tag = sub.name
        json_path = sub / f"summary_{iso_tag}.json"
        summary = load_json(json_path)

        if summary is None:
            # Weak IV ISO: no JSON, check if first_stage exists
            if (sub / "coefficients_first_stage.xlsx").exists():
                regions.append({
                    "region": iso_tag, "type": "ISO", "dir": sub,
                    "summary": None,
                    "panel_path": sub / f"panel_with_bartik_{iso_tag}.csv",
                    "status": "weak_iv",
                })
            continue

        status = classify_region(summary)
        regions.append({
            "region": iso_tag, "type": "ISO", "dir": sub,
            "summary": summary,
            "panel_path": sub / f"panel_with_bartik_{iso_tag}.csv",
            "status": status,
        })

    # Non-ISO cities
    city_dir = input_dir / "city" / "cities"
    if city_dir.exists():
        summary = load_json(city_dir / "summary_cities.json")
        if summary is not None:
            status = classify_region(summary)
            regions.append({
                "region": "Non-ISO Cities", "type": "City", "dir": city_dir,
                "summary": summary,
                "panel_path": city_dir / "panel_with_bartik_cities.csv",
                "status": status,
            })

    return regions


# =============================================================================
# Formatting
# =============================================================================

STATUS_LABELS = {
    "iv_sig":   "IV significant",
    "iv_insig": "IV n.s. (check RF)",
    "iv_noisy": "IV noisy (SE inflated)",
    "weak_iv":  "Weak instrument",
    "ols_only": "OLS only",
}

def fmt_stars(p):
    if pd.isna(p) or p is None: return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def fmt_coef(c, p=None):
    if pd.isna(c): return ""
    s = fmt_stars(p) if p is not None else ""
    return f"{c:.4f}{s}"

def fmt_se(se):
    if pd.isna(se): return ""
    return f"({se:.4f})"

def fmt_ci(lo, hi):
    if pd.isna(lo) or pd.isna(hi): return ""
    return f"[{lo:.4f}, {hi:.4f}]"


# =============================================================================
# TABLE 1: Regression Summary
# =============================================================================

def build_table1(regions):
    rows = []
    for r in regions:
        s = r["summary"]
        status = r["status"]
        _, meta = compute_zone_dc_growth(r["panel_path"])

        row = {"Region": r["region"], "Status": STATUS_LABELS.get(status, status)}

        if meta:
            row["N"] = meta["n_obs"]
            row["Zones"] = meta["n_zones"]
            row["Period"] = (f"{meta['date_start'].strftime('%Y-%m')}"
                             f" to {meta['date_end'].strftime('%Y-%m')}")
        else:
            row["N"] = ""
            row["Zones"] = ""
            row["Period"] = ""

        if s is None:
            for c in ["FS_F","FS_pi1","FS_se","FS_p","FS_R2w",
                       "OLS_beta","OLS_se","OLS_p","OLS_ci_lo","OLS_ci_hi",
                       "IV_beta","IV_se","IV_p","IV_ci_lo","IV_ci_hi"]:
                row[c] = np.nan
            rows.append(row)
            continue

        # First stage
        fs = s.get("first_stage", {})
        row["FS_F"] = fs.get("F_stat", np.nan)
        row["FS_pi1"] = fs.get("pi_1", np.nan)
        row["FS_se"] = fs.get("se", np.nan)
        row["FS_p"] = fs.get("pval", np.nan)
        row["FS_R2w"] = fs.get("r_squared_within", np.nan)

        # OLS
        ols = s.get("ols", {})
        row["OLS_beta"] = ols.get("dc_local_coef", np.nan)
        row["OLS_se"] = ols.get("dc_local_se", np.nan)
        row["OLS_p"] = ols.get("dc_local_pval", np.nan)
        ci = get_ci(ols)
        row["OLS_ci_lo"], row["OLS_ci_hi"] = ci

        # IV — only if completed full pipeline and non-noisy
        iv = s.get("iv2sls", {})
        if iv and status in ("iv_sig", "iv_insig"):
            row["IV_beta"] = iv.get("dc_local_coef", np.nan)
            row["IV_se"] = iv.get("dc_local_se", np.nan)
            row["IV_p"] = iv.get("dc_local_pval", np.nan)
            ci = get_ci(iv)
            row["IV_ci_lo"], row["IV_ci_hi"] = ci
        else:
            row["IV_beta"] = row["IV_se"] = row["IV_p"] = np.nan
            row["IV_ci_lo"] = row["IV_ci_hi"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# TABLE 2: Per-GW Price Impact (OLS vs Causal Estimate)
# =============================================================================

def build_table2(regions):
    """
    Per-GW comparison: OLS vs best causal estimate.
    Causal = 2SLS (if significant) or Wald (if RF significant).
    Only regions with a credible causal estimate included.
    """
    rows = []
    for r in regions:
        s = r["summary"]
        if s is None:
            continue

        causal = compute_causal_estimate(r)
        if causal is None:
            continue

        ols = s.get("ols", {})
        ols_ci = get_ci(ols)
        fs = s.get("first_stage", {})

        source_label = "2SLS" if causal["source"] == "iv2sls" else "Wald (RF/FS)"

        row = {
            "Region": r["region"],
            "OLS_beta": ols.get("dc_local_coef", np.nan),
            "OLS_se": ols.get("dc_local_se", np.nan),
            "OLS_p": ols.get("dc_local_pval", np.nan),
            "OLS_ci_lo": ols_ci[0],
            "OLS_ci_hi": ols_ci[1],
            "Causal_source": source_label,
            "Causal_beta": causal["beta"],
            "Causal_se": causal["se"],
            "Causal_p": causal["p"],
            "Causal_ci_lo": causal["ci_lo"],
            "Causal_ci_hi": causal["ci_hi"],
            "FS_F": fs.get("F_stat", np.nan),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# TABLE 3: Zone-Level Total Price Increase by 2025
# =============================================================================

def build_table3(regions):
    """
    Zone-level price increase by end of 2025.
    Each load zone / city gets its own row.

    Only regions with a credible causal estimate (iv_sig or RF-based Wald).
    ΔP_zone = β_causal × ΔDC_zone
    CI propagated from β CI.
    """
    rows = []
    for r in regions:
        causal = compute_causal_estimate(r)
        if causal is None:
            continue

        zone_list, meta = compute_zone_dc_growth(r["panel_path"])
        if zone_list is None:
            continue

        beta = causal["beta"]
        ci_lo_beta = causal["ci_lo"]
        ci_hi_beta = causal["ci_hi"]
        source_label = "2SLS" if causal["source"] == "iv2sls" else "Wald (RF/FS)"

        for z in zone_list:
            dc_g = z["dc_growth"]

            # ΔP = β × ΔDC
            dp = beta * dc_g

            # CI: propagate β CI through ΔDC
            if not pd.isna(ci_lo_beta) and not pd.isna(ci_hi_beta):
                dp_a = ci_lo_beta * dc_g
                dp_b = ci_hi_beta * dc_g
                dp_ci_lo = min(dp_a, dp_b)
                dp_ci_hi = max(dp_a, dp_b)
            else:
                dp_ci_lo = dp_ci_hi = np.nan

            rows.append({
                "Region": r["region"],
                "Zone": z["zone"],
                "Estimate": source_label,
                "beta": beta,
                "beta_p": causal["p"],
                "DC_start_GW": round(z["dc_start"], 4),
                "DC_end2025_GW": round(z["dc_end_2025"], 4),
                "dDC_GW": round(dc_g, 4),
                "dP_dollar_MWh": round(dp, 4),
                "dP_CI_lo": round(dp_ci_lo, 4) if not pd.isna(dp_ci_lo) else np.nan,
                "dP_CI_hi": round(dp_ci_hi, 4) if not pd.isna(dp_ci_hi) else np.nan,
            })

    return pd.DataFrame(rows)


# =============================================================================
# TABLE 4: Zone-Level RELATIVE Price Increase by 2025 (% of base-year price)
# =============================================================================

def build_table4(table3_df, regions, iso_price_tables, city_price_table, base_year):
    """
    Relative price increase: (ΔP / P_base) × 100%, where P_base is the
    zone-level base price loaded from r1_panel_result/<ISO>/zone_price_diff_means.xlsx
    (or r1_panel_result/CITY/cities/city_price_means.xlsx for non-ISO cities).

    Built directly from Table 3 — same rows, same causal estimates, just
    divided by the base price. CI bounds are propagated proportionally.

    Zones with no matching base price are kept but with NaN in the relative
    columns, and a warning is printed so the user can inspect name mismatches.
    """
    if table3_df.empty:
        return pd.DataFrame()

    region_type = {r["region"]: r["type"] for r in regions}

    rows = []
    missing = []
    for _, tr in table3_df.iterrows():
        region = tr["Region"]
        zone = tr["Zone"]
        is_city = region_type.get(region) == "City"

        p_base = lookup_base_price(
            iso_price_tables, city_price_table, region, zone, is_city=is_city
        )

        dp = tr["dP_dollar_MWh"]
        dp_lo = tr["dP_CI_lo"]
        dp_hi = tr["dP_CI_hi"]

        if pd.isna(p_base) or p_base == 0:
            rel_dp = np.nan
            rel_lo = np.nan
            rel_hi = np.nan
            missing.append((region, zone))
        else:
            rel_dp = (dp / p_base) * 100.0
            rel_lo = (dp_lo / p_base) * 100.0 if not pd.isna(dp_lo) else np.nan
            rel_hi = (dp_hi / p_base) * 100.0 if not pd.isna(dp_hi) else np.nan

        rows.append({
            "Region": region,
            "Zone": zone,
            "Estimate": tr["Estimate"],
            "beta": tr["beta"],
            "dDC_GW": tr["dDC_GW"],
            "dP_dollar_MWh": tr["dP_dollar_MWh"],
            "P_base_dollar_MWh": round(p_base, 4) if not pd.isna(p_base) else np.nan,
            "dP_pct": round(rel_dp, 4) if not pd.isna(rel_dp) else np.nan,
            "dP_pct_CI_lo": round(rel_lo, 4) if not pd.isna(rel_lo) else np.nan,
            "dP_pct_CI_hi": round(rel_hi, 4) if not pd.isna(rel_hi) else np.nan,
        })

    if missing:
        print(f"  WARNING: no base-year price for {len(missing)} zone(s):")
        for region, zone in missing:
            print(f"    {region:<18s}  {zone}")

    return pd.DataFrame(rows)


# =============================================================================
# Pretty-print
# =============================================================================

def print_table1(df):
    if df.empty:
        print("\n  TABLE 1: No results.\n"); return

    print(f"\n{'='*130}")
    print(f"  TABLE 1: Regression Summary — Bartik IV (dc_local -> Wholesale Price)")
    print(f"  Dep var: LMP ($/MWh)  Endog: dc_local (GW)  Instrument: Bartik IV")
    print(f"  Controls: HDD2, CDD2, GasPrice, Zone/City FE, Year FE, MonthxDOW FE")
    print(f"  SE: Bartlett HAC (bw=7).  IV columns blank for noisy/weak/OLS-only regions.")
    print(f"{'='*130}")

    hdr = (f"  {'Region':<16s} {'Status':<22s} {'N':>7s} {'Z':>3s} "
           f"{'FS_F':>7s} "
           f"{'OLS':>12s} {'(SE)':>9s} "
           f"{'IV':>12s} {'(SE)':>9s} "
           f"{'IV 95% CI':>24s} "
           f"{'R2w':>6s}")
    print(hdr)
    print(f"  {'-'*126}")

    for _, r in df.iterrows():
        fs_f = f"{r['FS_F']:.1f}" if not pd.isna(r["FS_F"]) else ""
        r2w = f"{r['FS_R2w']:.4f}" if not pd.isna(r.get("FS_R2w")) else ""
        print(f"  {r['Region']:<16s} {r['Status']:<22s} "
              f"{str(r['N']):>7s} {str(r['Zones']):>3s} "
              f"{fs_f:>7s} "
              f"{fmt_coef(r['OLS_beta'], r['OLS_p']):>12s} {fmt_se(r['OLS_se']):>9s} "
              f"{fmt_coef(r['IV_beta'], r['IV_p']):>12s} {fmt_se(r['IV_se']):>9s} "
              f"{fmt_ci(r['IV_ci_lo'], r['IV_ci_hi']):>24s} "
              f"{r2w:>6s}")

    print(f"  {'-'*126}")
    print(f"  *** p<0.01  ** p<0.05  * p<0.10")
    print(f"  FS_F = First-stage F (>10=strong)  R2w = First-stage within-R2")
    print(f"{'='*130}\n")


def print_table2(df):
    if df.empty:
        print("\n  TABLE 2: No regions with credible causal estimate.\n"); return

    print(f"\n{'='*125}")
    print(f"  TABLE 2: Per-GW Price Impact — OLS vs Causal Estimate ($/MWh per GW)")
    print(f"  Causal = 2SLS (if IV significant) or Wald ratio RF/FS (if RF significant)")
    print(f"  Wald CI via delta method: se_wald = se_rf / |pi_FS|, CI = wald +/- 1.96 * se_wald")
    print(f"{'='*125}")

    hdr = (f"  {'Region':<16s} "
           f"{'OLS':>12s} {'OLS 95% CI':>24s} "
           f"{'Causal':>12s} {'Causal 95% CI':>24s} "
           f"{'Source':<14s} {'FS_F':>6s}")
    print(hdr)
    print(f"  {'-'*121}")

    for _, r in df.iterrows():
        fs_f = f"{r['FS_F']:.1f}" if not pd.isna(r.get('FS_F')) else ""
        print(f"  {r['Region']:<16s} "
              f"{fmt_coef(r['OLS_beta'], r['OLS_p']):>12s} "
              f"{fmt_ci(r['OLS_ci_lo'], r['OLS_ci_hi']):>24s} "
              f"{fmt_coef(r['Causal_beta'], r['Causal_p']):>12s} "
              f"{fmt_ci(r['Causal_ci_lo'], r['Causal_ci_hi']):>24s} "
              f"{r['Causal_source']:<14s} {fs_f:>6s}")

    print(f"  {'-'*121}")
    print(f"  *** p<0.01  ** p<0.05  * p<0.10")
    print(f"{'='*125}\n")


def print_table3(df):
    if df.empty:
        print("\n  TABLE 3: No zone-level impacts to report.\n"); return

    print(f"\n{'='*130}")
    print(f"  TABLE 3: Zone-Level Price Increase by 2025 Due to Data Centers")
    print(f"  dP = beta_causal x dDC_zone.  Each load zone / city shown separately.")
    print(f"  Causal beta = 2SLS (if IV sig.) or Wald RF/FS (if RF sig. but IV n.s.)")
    print(f"  95% CI propagated from beta CI through zone-specific dDC")
    print(f"{'='*130}")

    hdr = (f"  {'Region':<16s} {'Zone':<18s} {'Est.':<14s} "
           f"{'beta':>8s} "
           f"{'DC_start':>9s} {'DC_2025':>9s} {'dDC':>8s} "
           f"{'dP':>9s} {'dP 95% CI':>26s}")
    print(hdr)
    print(f"  {'-'*126}")

    prev_region = None
    for _, r in df.iterrows():
        # Separator between regions
        if r["Region"] != prev_region and prev_region is not None:
            print(f"  {'':.<126s}")
        prev_region = r["Region"]

        print(f"  {r['Region']:<16s} {r['Zone']:<18s} {r['Estimate']:<14s} "
              f"{r['beta']:>8.4f} "
              f"{r['DC_start_GW']:>9.4f} {r['DC_end2025_GW']:>9.4f} "
              f"{r['dDC_GW']:>8.4f} "
              f"{r['dP_dollar_MWh']:>9.4f} "
              f"{fmt_ci(r['dP_CI_lo'], r['dP_CI_hi']):>26s}")

    print(f"  {'-'*126}")

    # Region subtotals
    print(f"\n  Region subtotals:")
    print(f"    {'Region':<16s}  {'Zones':>5s}  "
          f"{'mean dDC':>10s}  {'total dDC':>10s}  "
          f"{'avg dP':>10s}  {'avg dP 95% CI':>26s}")
    print(f"    {'-'*90}")
    for region, grp in df.groupby("Region", sort=False):
        total_dc = grp["dDC_GW"].sum()
        beta = grp["beta"].iloc[0]
        avg_dc = grp["dDC_GW"].mean()
        avg_dp = beta * avg_dc
        # CI for avg zone dP: CI_β × mean(dDC)
        avg_dp_ci_lo = grp["dP_CI_lo"].mean() if grp["dP_CI_lo"].notna().all() else np.nan
        avg_dp_ci_hi = grp["dP_CI_hi"].mean() if grp["dP_CI_hi"].notna().all() else np.nan
        print(f"    {region:<16s}  {len(grp):>5d}  "
              f"{avg_dc:>10.4f}  {total_dc:>10.4f}  "
              f"{avg_dp:>10.4f}  {fmt_ci(avg_dp_ci_lo, avg_dp_ci_hi):>26s}")

    print(f"\n  dDC = DC capacity growth per zone (base year to 2025), in GW")
    print(f"  dP  = beta x dDC, zone-level avg price increase ($/MWh)")
    print(f"{'='*130}\n")


def print_table4(df, base_year):
    if df.empty:
        print("\n  TABLE 4: No zone-level relative impacts to report.\n"); return

    print(f"\n{'='*130}")
    print(f"  TABLE 4: Zone-Level RELATIVE Price Increase by 2025 (% of baseline price)")
    print(f"  dP_pct = (beta x dDC / P_base) x 100%")
    print(f"  P_base = zone-level price_diff_mean from r1_panel_result/")
    print(f"{'='*130}")

    hdr = (f"  {'Region':<16s} {'Zone':<18s} {'Est.':<14s} "
           f"{'dP $/MWh':>10s} {'P_base':>10s} "
           f"{'dP %':>9s} {'dP % 95% CI':>26s}")
    print(hdr)
    print(f"  {'-'*126}")

    prev_region = None
    for _, r in df.iterrows():
        if r["Region"] != prev_region and prev_region is not None:
            print(f"  {'':.<126s}")
        prev_region = r["Region"]

        p_base = r["P_base_dollar_MWh"]
        p_base_str = f"{p_base:>10.4f}" if not pd.isna(p_base) else f"{'—':>10s}"
        dp_pct_str = f"{r['dP_pct']:>9.2f}" if not pd.isna(r['dP_pct']) else f"{'—':>9s}"

        print(f"  {r['Region']:<16s} {r['Zone']:<18s} {r['Estimate']:<14s} "
              f"{r['dP_dollar_MWh']:>10.4f} {p_base_str} "
              f"{dp_pct_str} "
              f"{fmt_ci(r['dP_pct_CI_lo'], r['dP_pct_CI_hi']):>26s}")

    print(f"  {'-'*126}")

    # Region subtotals
    print(f"\n  Region subtotals (avg relative impact across zones):")
    print(f"    {'Region':<16s}  {'Zones':>5s}  "
          f"{'avg dP %':>10s}  {'avg dP % 95% CI':>26s}")
    print(f"    {'-'*64}")
    for region, grp in df.groupby("Region", sort=False):
        valid = grp.dropna(subset=["dP_pct"])
        if valid.empty:
            continue
        avg_pct = valid["dP_pct"].mean()
        avg_lo = valid["dP_pct_CI_lo"].mean() if valid["dP_pct_CI_lo"].notna().all() else np.nan
        avg_hi = valid["dP_pct_CI_hi"].mean() if valid["dP_pct_CI_hi"].notna().all() else np.nan
        print(f"    {region:<16s}  {len(valid):>5d}  "
              f"{avg_pct:>10.2f}  {fmt_ci(avg_lo, avg_hi):>26s}")

    print(f"\n  dP %% = price increase as a percentage of zone baseline price")
    print(f"{'='*130}\n")


# =============================================================================
# Save to Excel
# =============================================================================

def save_all(t1, t2, t3, t4, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = out_dir / "bartik_iv_summary.xlsx"

    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        if not t1.empty:
            t1.to_excel(w, sheet_name="Regression Summary", index=False)
        if not t2.empty:
            t2.to_excel(w, sheet_name="Per-GW Impact", index=False)
        if not t3.empty:
            t3.to_excel(w, sheet_name="Zone-Level Impact 2025", index=False)
        if not t4.empty:
            t4.to_excel(w, sheet_name="Zone-Level Relative 2025", index=False)

    print(f"  Saved: {xlsx}")

    t1.to_csv(out_dir / "table1_regression_summary.csv", index=False)
    if not t2.empty:
        t2.to_csv(out_dir / "table2_per_gw_impact.csv", index=False)
    if not t3.empty:
        t3.to_csv(out_dir / "table3_zone_impact_2025.csv", index=False)
    if not t4.empty:
        t4.to_csv(out_dir / "table4_zone_relative_2025.csv", index=False)
    print(f"  Saved CSVs to {out_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Summarize Bartik IV results — zone-level, with Wald fallback")
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--panel-root", default="./results/r3_bartik_iv",
                        help="Root directory containing <ISO>/zone_price_diff_means.xlsx "
                             "and CITY/cities/city_price_means.xlsx (for Table 4)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found."); sys.exit(1)

    print(f"{'='*70}")
    print(f"  Bartik IV Results Summary (v3)")
    print(f"  Input:  {input_dir.resolve()}")
    print(f"  Output: {out_dir.resolve()}")
    print(f"{'='*70}")

    # [1] Discover
    print("\n[1] Discovering regions...")
    regions = discover_regions(input_dir)
    if not regions:
        print("  No regions found."); sys.exit(1)

    for r in regions:
        label = STATUS_LABELS.get(r["status"], r["status"])
        print(f"    {r['region']:<20s}  {label}")

    # [2] Compute causal estimates
    print("\n[2] Computing causal estimates...")
    for r in regions:
        causal = compute_causal_estimate(r)
        if causal:
            src = "2SLS" if causal["source"] == "iv2sls" else "Wald (RF/FS)"
            print(f"    {r['region']:<20s}  {src:<16s}  "
                  f"beta={causal['beta']:.4f}  p={causal['p']:.4f}  "
                  f"CI={fmt_ci(causal['ci_lo'], causal['ci_hi'])}")
        else:
            print(f"    {r['region']:<20s}  — no credible causal estimate")

    # [3] Build tables
    print("\n[3] Building tables...")
    t1 = build_table1(regions)
    t2 = build_table2(regions)
    t3 = build_table3(regions)

    # Load price tables for relative growth (Table 4)
    print(f"  Loading baseline prices from {args.panel_root}...")
    iso_price_tables = load_iso_price_tables(args.panel_root)
    city_price_table = load_city_price_table(args.panel_root)
    n_iso_zones = sum(len(v) for v in iso_price_tables.values())
    print(f"    loaded {len(iso_price_tables)} ISO(s), {n_iso_zones} zone prices; "
          f"{len(city_price_table)} city prices")
    if not iso_price_tables and not city_price_table:
        print("  (no price tables found — skipping Table 4 relative growth)")
        t4 = pd.DataFrame()
    else:
        t4 = build_table4(t3, regions, iso_price_tables, city_price_table, BASE_YEAR)

    # [4] Print
    print_table1(t1)
    print_table2(t2)
    print_table3(t3)
    print_table4(t4, BASE_YEAR)

    # [5] Exclusion summary
    excluded = [r for r in regions
                if compute_causal_estimate(r) is None and r["summary"] is not None]
    no_json = [r for r in regions if r["summary"] is None]

    if excluded or no_json:
        print(f"  EXCLUDED from causal analysis (Tables 2 & 3):")
        for r in no_json:
            print(f"    {r['region']:<20s}  Weak instrument (no summary JSON)")
        for r in excluded:
            reason = STATUS_LABELS.get(r["status"], r["status"])
            extra = ""
            if r["status"] == "iv_insig":
                is_city = r["type"] == "City"
                rf = load_rf_robustness(r["dir"], r["region"], is_city)
                if rf:
                    rf_p = rf.get("rf_p", 1)
                    extra = f" (RF p={rf_p:.4f}, {'sig' if rf_p < 0.1 else 'n.s.'})"
                else:
                    extra = " (no RF robustness file)"
            print(f"    {r['region']:<20s}  {reason}{extra}")
        print()

    # [6] Save
    print("[6] Saving results...")
    save_all(t1, t2, t3, t4, out_dir)

    print(f"\n{'='*70}")
    print(f"  Done. Results: {out_dir.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
