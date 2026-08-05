#!/usr/bin/env python3
"""
Cross-Sectional Analysis: Why Do Data Centers Raise Prices in Some Regions?
===========================================================================
Links Bartik IV causal price effects to grid reliability and renewable capacity.

Step 1: Load zone-level ΔP from r03_summary (table3_zone_impact_2025.csv)
Step 2: Load NERC Summer Reliability (reserve margin, extreme margin)
Step 3: Load NERC Long-Term Reliability (EEU, LOLH — sheets: 2020,2022,2024,2025)
Step 4: Map ISOs/cities → NERC assessment areas
Step 5: Correlations at national (ISO-level) cross-section

Usage:
    python r02_cross_sectional.py
    python r02_cross_sectional.py --nerc-summer ./tables/nerc_summer_reliability.xlsx
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
BARTIK_DIR     = Path("./results/r3_bartik_iv")
SUMMARY_DIR    = Path("./results/r3_summary")
NERC_SUMMER    = Path("./LTRS/SRA.xlsx")
NERC_LONGTERM  = Path("./LTRS/LTRA.xlsx")
OUTPUT_DIR     = Path("./results/r3_cross_sectional")

# =============================================================================
# NERC Region Mapping (all 3 classification schemes)
# =============================================================================

ISO_TO_NERC = {
    "MISO":   ["MISO"],
    "PJM":    ["PJM"],
    "ERCOT":  ["ERCOT"],
    "SPP":    ["SPP"],
    "ISONE":  ["NPC-NE", "NPCC-NE", "NPCC New England"],
    "NYISO":  ["NPC-NY", "NPCC-NY", "NPCC New York"],
    "CAISO":  ["WECC-CAMX", "WECC-CA", "CAISO"],
}

CITY_STATE = {
    "Phoenix": "AZ", "Mesa": "AZ", "Goodyear": "AZ", "Chandler": "AZ",
    "Atlanta": "GA", "Lithia Springs": "GA",
    "Hillsboro": "OR", "Boardman": "OR",
    "Charlotte": "NC", "Las Vegas": "NV", "Sparks": "NV", "Reno": "NV",
    "Quincy": "WA", "Seattle": "WA", "Tukwila": "WA",
    "Miami": "FL", "Huntsville": "AL", "Moncks Corner": "SC",
}

STATE_TO_NERC = {
    "GA": ["SERC-SE"], "AL": ["SERC-SE"],
    "SC": ["SERC-E"], "NC": ["SERC-E", "SERC-C"],
    "FL": ["SERC-FP"],
    "AZ": ["SRSG", "WECC-SW"],
    "NV": ["SRSG", "WECC-SW", "NWPP-US"],
    "OR": ["NWPP-US", "WECC-NW"],
    "WA": ["NWPP-US", "WECC-NW"],
}

CITY_TO_NERC_OVERRIDE = {
    "Las Vegas":  ["SRSG", "WECC-SW"],
    "Sparks":     ["NWPP-US", "WECC-Basin"],
    "Reno":       ["NWPP-US", "WECC-Basin"],
}


def match_nerc_areas(candidates, available_areas):
    """Find ALL matches from candidates in available area list to handle renaming over time."""
    matched = set()
    norm = lambda s: str(s).upper().replace("-", "").replace("_", "").replace(" ", "")

    for c in candidates:
        # Exact match
        if c in available_areas:
            matched.add(c)
        # Normalized match
        for a in available_areas:
            if norm(c) == norm(a):
                matched.add(a)
            # Safe partial match (limit to length > 3 to avoid "CA" matching "SERC-CA")
            elif len(c) > 3 and (norm(c) in norm(a) or norm(a) in norm(c)):
                matched.add(a)

    return list(matched)


# =============================================================================
# Step 1: Load zone-level impacts from r03_summary
# =============================================================================

def load_zone_impacts(summary_dir, bartik_dir):
    """
    Load zone-level price impacts.
    Priority: table3_zone_impact_2025.csv from r03_summary.
    Fallback: reconstruct from Bartik IV JSONs + panel CSVs.
    """
    t3_path = summary_dir / "table3_zone_impact_2025.csv"
    if t3_path.exists():
        df = pd.read_csv(t3_path)
        if not df.empty:
            print(f"  Loaded {len(df)} zone-level impacts from {t3_path}")
            return df

    print(f"  WARNING: {t3_path} not found. Run r02_summary.py first.")
    return pd.DataFrame()


def load_all_region_results(bartik_dir):
    """
    Load ALL region results (including non-significant) for correlation analysis.
    Returns one row per ISO/city panel with OLS + IV + status.
    """
    results = []

    for sub in sorted(bartik_dir.iterdir()):
        if not sub.is_dir() or sub.name == "city":
            continue
        iso = sub.name
        json_path = sub / f"summary_{iso}.json"
        if not json_path.exists():
            results.append({"region": iso, "type": "ISO", "status": "weak_iv"})
            continue

        with open(json_path) as f:
            s = json.load(f)

        ols = s.get("ols", {})
        iv = s.get("iv2sls", {})
        fs = s.get("first_stage", {})

        row = {
            "region": iso, "type": "ISO",
            "ols_beta": ols.get("dc_local_coef", np.nan),
            "ols_se": ols.get("dc_local_se", np.nan),
            "ols_p": ols.get("dc_local_pval", np.nan),
            "iv_beta": iv.get("dc_local_coef", np.nan) if iv else np.nan,
            "iv_se": iv.get("dc_local_se", np.nan) if iv else np.nan,
            "iv_p": iv.get("dc_local_pval", np.nan) if iv else np.nan,
            "fs_F": fs.get("F_stat", np.nan),
            "fs_R2w": fs.get("r_squared_within", np.nan),
        }

        # Classify
        if "iv_skipped" in s:
            row["status"] = "weak_iv"
        elif not iv:
            row["status"] = "ols_only"
        else:
            ols_se_v = ols.get("dc_local_se", 1)
            iv_se_v = iv.get("dc_local_se", 0)
            if ols_se_v > 0 and iv_se_v / ols_se_v > 10:
                row["status"] = "iv_noisy"
            elif iv.get("dc_local_pval", 1) < 0.10:
                row["status"] = "iv_sig"
            else:
                # Check RF
                row["status"] = "iv_insig"
                rf_path = sub / f"rf_robustness_{iso}.xlsx"
                if not rf_path.exists():
                    rf_path = sub / f"rf_robustness_{iso}.csv"
                if rf_path.exists():
                    try:
                        rf = pd.read_excel(rf_path) if rf_path.suffix == ".xlsx" else pd.read_csv(rf_path)
                        if not rf.empty and rf.iloc[0].get("rf_p", 1) < 0.10:
                            row["status"] = "rf_sig"
                            row["wald_beta"] = rf.iloc[0].get("wald", np.nan)
                            row["rf_p"] = rf.iloc[0].get("rf_p", np.nan)
                    except Exception:
                        pass

        # Determine "best causal beta" for correlation
        if row.get("status") == "iv_sig":
            row["causal_beta"] = row["iv_beta"]
            row["causal_source"] = "2SLS"
        elif row.get("status") == "rf_sig":
            row["causal_beta"] = row.get("wald_beta", np.nan)
            row["causal_source"] = "Wald"
        else:
            row["causal_beta"] = np.nan
            row["causal_source"] = None

        results.append(row)

    # Cities
    city_dir = bartik_dir / "city" / "cities"
    if city_dir.exists():
        json_path = city_dir / "summary_cities.json"
        if json_path.exists():
            with open(json_path) as f:
                s = json.load(f)
            ols = s.get("ols", {})
            iv = s.get("iv2sls", {})
            fs = s.get("first_stage", {})
            row = {
                "region": "Non-ISO Cities", "type": "City",
                "ols_beta": ols.get("dc_local_coef", np.nan),
                "ols_p": ols.get("dc_local_pval", np.nan),
                "iv_beta": iv.get("dc_local_coef", np.nan) if iv else np.nan,
                "iv_p": iv.get("dc_local_pval", np.nan) if iv else np.nan,
                "fs_F": fs.get("F_stat", np.nan),
                "status": "city",
                "causal_beta": np.nan, "causal_source": None,
            }
            if iv and iv.get("dc_local_pval", 1) < 0.10:
                ols_se_v = ols.get("dc_local_se", 1)
                iv_se_v = iv.get("dc_local_se", 0)
                if not (ols_se_v > 0 and iv_se_v / ols_se_v > 10):
                    row["status"] = "iv_sig"
                    row["causal_beta"] = iv.get("dc_local_coef")
                    row["causal_source"] = "2SLS"
            else:
                # Check RF/Wald (same logic as ISO rows)
                for ext in (".xlsx", ".csv"):
                    rf_path = city_dir / f"rf_robustness_cities{ext}"
                    if rf_path.exists():
                        try:
                            rf = pd.read_excel(rf_path) if ext == ".xlsx" else pd.read_csv(rf_path)
                            if not rf.empty and rf.iloc[0].get("rf_p", 1) < 0.10:
                                row["status"] = "rf_sig"
                                row["causal_beta"] = rf.iloc[0].get("wald", np.nan)
                                row["causal_source"] = "Wald (RF/FS)"
                        except Exception:
                            pass
                        break
            results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# Step 2-3: Load NERC data
# =============================================================================

def load_nerc_summer(path):
    """Load NERC Summer Reliability Assessment. Auto-detect column names."""
    if not path.exists():
        print(f"  WARNING: {path} not found"); return pd.DataFrame()

    xls = pd.ExcelFile(path)
    dfs = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        # Try to infer year from sheet name
        for token in str(sheet).replace("_", " ").replace("-", " ").split():
            try:
                y = int(token)
                if 2018 <= y <= 2030:
                    df["_sheet_year"] = y
            except ValueError:
                pass
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)

    # Auto-detect columns by keyword matching
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        if c == "_sheet_year":
            continue
        elif any(k in cl for k in ["area", "region", "assessment"]):
            col_map[c] = "area"
        elif cl in ("year",):
            col_map[c] = "year"
        elif "reserve" in cl and "margin" in cl:
            col_map[c] = "reserve_margin"
        elif "extreme" in cl:
            col_map[c] = "extreme_margin"
        elif "anticipat" in cl and "margin" in cl:
            col_map[c] = "anticipated_margin"
        elif "prospective" in cl:
            col_map[c] = "prospective_margin"
        elif "reference" in cl or "ref" in cl and "margin" in cl:
            col_map[c] = "ref_margin_level"
        elif "resource" in cl:
            col_map[c] = "resources"
        elif "net_internal" in cl or "demand" in cl:
            if "50" in cl:
                col_map[c] = "demand_50"
            elif "90" in cl or "extreme" in cl:
                col_map[c] = "demand_90"
            else:
                col_map[c] = "demand"

    df = df.rename(columns=col_map)

    # If no area column, use first column
    if "area" not in df.columns:
        df = df.rename(columns={df.columns[0]: "area"})

    # If no year column, try sheet_year
    if "year" not in df.columns and "_sheet_year" in df.columns:
        df["year"] = df["_sheet_year"]

    # Coerce numerics
    for c in ["reserve_margin", "extreme_margin", "anticipated_margin",
              "prospective_margin", "ref_margin_level", "resources",
              "demand_50", "demand_90", "demand", "year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
    if "area" in df.columns:
        areas = sorted(df["area"].dropna().unique())
        print(f"  Assessment areas ({len(areas)}): {areas}")

    return df


def load_nerc_longterm(path):
    """
    Load NERC Long-Term Reliability Assessment.
    Each sheet = one report year (2020, 2022, 2024, 2025).
    Each sheet has a TWO-ROW header:
        Row 0:  ""          | EEU (ppm)     |               | LOLH          |
        Row 1:  area_name   | 2022          | 2024          | 2022          | 2024
    Data rows: WECC-CAMX   | 3721.39       | 8817.86       | 22.06         | 55.61

    Output: flat DataFrame with columns:
        area, report_year, EEU_2022, EEU_2024, LOLH_2022, LOLH_2024, ...
    """
    if not path.exists():
        print(f"  WARNING: {path} not found"); return pd.DataFrame()

    xls = pd.ExcelFile(path)
    dfs = []

    for sheet in xls.sheet_names:
        # Parse report year from sheet name
        report_year = None
        for token in str(sheet).replace("_", " ").replace("-", " ").split():
            try:
                y = int(token)
                if 2018 <= y <= 2030:
                    report_year = y
            except ValueError:
                pass

        # Read with multi-level header
        try:
            df = pd.read_excel(xls, sheet_name=sheet, header=[0, 1])
        except Exception:
            # Fallback: single header
            df = pd.read_excel(xls, sheet_name=sheet)
            if report_year:
                df["report_year"] = report_year
            dfs.append(df)
            continue

        # Flatten MultiIndex columns
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                top = str(c[0]).strip()
                sub = str(c[1]).strip()
                top_is_empty = ("Unnamed" in top or top == "")
                sub_is_empty = ("Unnamed" in sub or sub == "")

                if top_is_empty and sub_is_empty:
                    new_cols.append("area")
                elif top_is_empty:
                    # First column is area name
                    new_cols.append("area")
                elif sub_is_empty:
                    new_cols.append(top)
                else:
                    # Determine metric name
                    top_lower = top.lower()
                    if "eeu" in top_lower:
                        metric = "EEU"
                    elif "lolh" in top_lower:
                        metric = "LOLH"
                    else:
                        metric = top.replace(" ", "_")
                    # sub is the forecast year
                    new_cols.append(f"{metric}_{sub}")
            else:
                new_cols.append(str(c))

        df.columns = new_cols

        # If no 'area' column found, use first column
        if "area" not in df.columns:
            df = df.rename(columns={df.columns[0]: "area"})

        # Add report year
        if report_year:
            df["report_year"] = report_year

        # Coerce numeric columns
        for c in df.columns:
            if c not in ("area", "report_year"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Drop rows where area is NaN
        df = df.dropna(subset=["area"])

        dfs.append(df)
        print(f"    Sheet '{sheet}' (report_year={report_year}): "
              f"{len(df)} areas, cols={[c for c in df.columns if c not in ('area','report_year')]}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    print(f"  Loaded {len(df)} total rows from {len(xls.sheet_names)} sheets")
    if "area" in df.columns:
        areas = sorted(df["area"].dropna().unique())
        print(f"  Areas ({len(areas)}): {areas}")
    print(f"  Columns: {[c for c in df.columns if c not in ('area','report_year')]}")

    return df


# =============================================================================
# Step 5: Merge everything
# =============================================================================

def merge_nerc_to_region(region, rtype, nerc_df, col_prefix):
    """Match a region to NERC assessment areas (handling historical renames) and extract metrics."""
    result = {}
    if nerc_df.empty or "area" not in nerc_df.columns:
        return result

    available = nerc_df["area"].dropna().unique().tolist()

    if rtype == "ISO":
        candidates = ISO_TO_NERC.get(region, [region])
    else:
        return result  # Cities span multiple NERC areas

    matched_areas = match_nerc_areas(candidates, available)
    result[f"{col_prefix}_nerc_area"] = " | ".join(matched_areas) if matched_areas else None

    if not matched_areas:
        return result

    area_data = nerc_df[nerc_df["area"].isin(matched_areas)].copy()
    if area_data.empty:
        return result

    time_col = "year" if "year" in area_data.columns else "report_year"
    if time_col in area_data.columns:
        area_data = area_data.groupby(time_col).mean(numeric_only=True).reset_index()
        area_data = area_data.sort_values(time_col)

    numeric_cols = area_data.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("year", "report_year", "_sheet_year")]

    for c in numeric_cols:
        vals = area_data[c].dropna()
        if vals.empty:
            continue
        cname = str(c).strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        result[f"{col_prefix}_{cname}_earliest"] = vals.iloc[0]
        result[f"{col_prefix}_{cname}_latest"] = vals.iloc[-1]
        result[f"{col_prefix}_{cname}_change"] = vals.iloc[-1] - vals.iloc[0]

    return result


def build_merged_table(region_df, nerc_summer, nerc_lt):
    """Build the full cross-sectional table (one row per ISO)."""
    rows = []
    for _, r in region_df.iterrows():
        row = r.to_dict()

        # NERC Summer
        nerc_s = merge_nerc_to_region(r["region"], r["type"], nerc_summer, "summer")
        row.update(nerc_s)

        # NERC Long-Term
        nerc_l = merge_nerc_to_region(r["region"], r["type"], nerc_lt, "lt")
        keys_to_remove = [k for k in nerc_l.keys() if k.endswith("_change")]
        for k in keys_to_remove:
            nerc_l.pop(k)

        import re
        metrics = {}
        for k, v in nerc_l.items():
            if k.startswith("lt_") and k.endswith("_latest"):
                match = re.search(r"lt_([a-z]+)_(\d{4})_latest", k)
                if match:
                    metric = match.group(1)
                    year = int(match.group(2))
                    if metric not in metrics:
                        metrics[metric] = {}
                    metrics[metric][year] = v

        for metric, year_vals in metrics.items():
            valid_years = [y for y, val in year_vals.items() if not pd.isna(val)]
            if len(valid_years) >= 2:
                valid_years.sort()
                min_yr = valid_years[0]
                max_yr = valid_years[-1]
                nerc_l[f"lt_{metric}_forecast_change"] = year_vals[max_yr] - year_vals[min_yr]

        row.update(nerc_l)

        rows.append(row)

    result_df = pd.DataFrame(rows)
    # Compute (extreme margin - reference margin level) for summer
    for suffix in ["_latest", "_earliest"]:
        ext_col = f"summer_extreme_margin{suffix}"
        ref_col = f"summer_ref_margin_level{suffix}"
        new_col = f"summer_margin_above_ref{suffix}"
        if ext_col in result_df.columns and ref_col in result_df.columns:
            result_df[new_col] = result_df[ext_col] - result_df[ref_col]
    return result_df


def build_zone_nerc_table(zone_df, region_df, nerc_summer, nerc_lt, bartik_dir):
    """
    Build zone/city-level table for correlation analysis.

    Each row = one load zone or one city, with:
      - dP (price increase, 0 if region not significant)
      - NERC area + reliability metrics

    Mapping:
      ISO zones  → ISO → NERC area (via ISO_TO_NERC)
      Cities     → state → NERC area (via CITY_STATE + STATE_TO_NERC)
    """
    rows = []

    # --- Significant zones/cities from table3 ---
    sig_zones = set()
    if not zone_df.empty:
        for _, z in zone_df.iterrows():
            region = z.get("Region", "")
            zone = z.get("Zone", "")
            dp = z.get("dP_dollar_MWh", 0)
            ddc = z.get("dDC_GW", 0)
            beta = z.get("beta", np.nan)
            sig_zones.add((region, zone))

            rows.append({
                "region": region,
                "zone": zone,
                "dP": dp,
                "dDC_GW": ddc,
                "beta": beta,
                "significant": True,
            })

    # --- Non-significant regions: scan panel CSVs, set dP=0 ---
    for _, r in region_df.iterrows():
        region = r["region"]
        rtype = r["type"]
        has_causal = not pd.isna(r.get("causal_beta"))

        if has_causal:
            continue  # already in zone_df

        # Find panel CSV
        if rtype == "ISO":
            panel_path = bartik_dir / region / f"panel_with_bartik_{region}.csv"
        else:
            panel_path = bartik_dir / "city" / "cities" / "panel_with_bartik_cities.csv"

        if not panel_path.exists():
            continue

        try:
            panel = pd.read_csv(panel_path, usecols=["zone"], nrows=100000)
            zone_names = panel["zone"].unique()
        except Exception:
            continue

        for zname in zone_names:
            if (region, zname) not in sig_zones:
                rows.append({
                    "region": region,
                    "zone": zname,
                    "dP": 0.0,
                    "dDC_GW": 0.0,
                    "beta": 0.0,
                    "significant": False,
                })

    zdf = pd.DataFrame(rows)
    if zdf.empty:
        return zdf

    # --- Map each zone/city to NERC area ---
    def get_nerc_candidates(region, zone, rtype=None):
        """Determine NERC area candidates for a zone."""
        # Check if zone is a known city name
        if zone in CITY_TO_NERC_OVERRIDE:
            return CITY_TO_NERC_OVERRIDE[zone]
        if zone in CITY_STATE:
            state = CITY_STATE[zone]
            return STATE_TO_NERC.get(state, [])
        # ISO zone → use ISO mapping
        if region in ISO_TO_NERC:
            return ISO_TO_NERC[region]
        return [region]

    # Available NERC areas
    summer_areas = nerc_summer["area"].dropna().unique().tolist() if (not nerc_summer.empty and "area" in nerc_summer.columns) else []
    lt_areas = nerc_lt["area"].dropna().unique().tolist() if (not nerc_lt.empty and "area" in nerc_lt.columns) else []

    # Match and merge NERC data
    for idx, row in zdf.iterrows():
        candidates = get_nerc_candidates(row["region"], row["zone"])

        # Summer
        matched_s = match_nerc_areas(candidates, summer_areas)
        zdf.at[idx, "nerc_area_summer"] = " | ".join(matched_s) if matched_s else None
        if matched_s:
            nerc_data = merge_nerc_to_region_direct(matched_s, nerc_summer, "summer")
            for k, v in nerc_data.items():
                zdf.at[idx, k] = v

        # Long-Term
        matched_l = match_nerc_areas(candidates, lt_areas)
        zdf.at[idx, "nerc_area_lt"] = " | ".join(matched_l) if matched_l else None
        if matched_l:
            nerc_data = merge_nerc_to_region_direct(matched_l, nerc_lt, "lt")
            for k, v in nerc_data.items():
                zdf.at[idx, k] = v

    # Compute extreme - reference for summer
    for suffix in ["_latest", "_earliest"]:
        ext_col = f"summer_extreme_margin{suffix}"
        ref_col = f"summer_ref_margin_level{suffix}"
        new_col = f"summer_margin_above_ref{suffix}"
        if ext_col in zdf.columns and ref_col in zdf.columns:
            zdf[new_col] = zdf[ext_col] - zdf[ref_col]

    # Compute LTRA forecast change (max year - min year within latest report)
    import re
    lt_cols = [c for c in zdf.columns if c.startswith("lt_") and c.endswith("_latest")]
    metrics_years = {}
    for c in lt_cols:
        m = re.search(r"lt_([a-z]+)_(\d{4})_latest", c)
        if m:
            metric = m.group(1)
            year = int(m.group(2))
            if metric not in metrics_years:
                metrics_years[metric] = []
            metrics_years[metric].append((year, c))
    for metric, yc_list in metrics_years.items():
        yc_list.sort()
        if len(yc_list) >= 2:
            min_col = yc_list[0][1]
            max_col = yc_list[-1][1]
            new_col = f"lt_{metric}_forecast_change"
            zdf[new_col] = zdf[max_col] - zdf[min_col]

    print(f"  Zone-NERC table: {len(zdf)} rows "
          f"({zdf['significant'].sum()} significant, {(~zdf['significant']).sum()} non-sig)")

    return zdf


def merge_nerc_to_region_direct(matched_areas, nerc_df, col_prefix):
    """Extract NERC metrics for a list of area names. Returns dict."""
    result = {}
    if isinstance(matched_areas, str):
        matched_areas = [matched_areas]

    area_data = nerc_df[nerc_df["area"].isin(matched_areas)].copy()
    if area_data.empty:
        return result

    time_col = "year" if "year" in area_data.columns else "report_year"
    if time_col in area_data.columns:
        area_data = area_data.groupby(time_col).mean(numeric_only=True).reset_index()
        area_data = area_data.sort_values(time_col)

    numeric_cols = area_data.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols
                    if c not in ("year", "report_year", "_sheet_year")
                    and "unnamed" not in str(c).lower()]

    for c in numeric_cols:
        vals = area_data[c].dropna()
        if vals.empty:
            continue
        cname = str(c).strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        result[f"{col_prefix}_{cname}_earliest"] = vals.iloc[0]
        result[f"{col_prefix}_{cname}_latest"] = vals.iloc[-1]

    return result



# =============================================================================
# Step 6: Correlations
# =============================================================================

def run_correlations(zone_nerc_df):
    """
    Cross-sectional correlations at zone/city level.

    Y variable = dP (price increase per zone/city):
      - β × ΔDC for significant regions
      - 0 for non-significant regions

    X variables = NERC reliability metrics of the zone's NERC area.

    This gives more observations than ISO-level (N >> 7) and lets
    cities contribute genuine cross-sectional variation through
    different NERC areas.
    """
    df = zone_nerc_df.copy()
    if len(df) < 4:
        print(f"  Only {len(df)} zones — need ≥4.")
        return pd.DataFrame(), pd.DataFrame()

    # Auto-discover X candidates
    x_candidates = []

    for c in df.columns:
        if c.startswith("summer_") and c.endswith("_latest"):
            label = c.replace("summer_", "").replace("_latest", "").replace("_", " ").title()
            x_candidates.append((c, f"Summer: {label} (latest)"))
        if c.startswith("summer_") and c.endswith("_earliest"):
            label = c.replace("summer_", "").replace("_earliest", "").replace("_", " ").title()
            x_candidates.append((c, f"Summer: {label} (earliest)"))

    for c in df.columns:
        if c.startswith("lt_") and c.endswith("_latest"):
            label = c.replace("lt_", "").replace("_latest", "").replace("_", " ").title()
            x_candidates.append((c, f"Long-Term: {label} (latest rpt)"))
        if c.startswith("lt_") and c.endswith("_earliest"):
            label = c.replace("lt_", "").replace("_earliest", "").replace("_", " ").title()
            x_candidates.append((c, f"Long-Term: {label} (earliest rpt)"))
        if c.startswith("lt_") and c.endswith("_forecast_change"):
            metric = c.split("_")[1].upper()
            x_candidates.append((c, f"Long-Term: {metric} (Δ forecast)"))

    results = []
    for x_col, x_label in x_candidates:
        if x_col not in df.columns:
            continue
        valid = df[["dP", x_col]].dropna()
        if len(valid) < 4:
            continue
        if valid[x_col].std() == 0:
            continue

        r_val, p_val = sp_stats.pearsonr(valid["dP"], valid[x_col])
        rho, rho_p = sp_stats.spearmanr(valid["dP"], valid[x_col])

        if pd.isna(r_val):
            continue

        results.append({
            "Variable": x_label,
            "Column": x_col,
            "N": len(valid),
            "Pearson_r": r_val,
            "Pearson_p": p_val,
            "Spearman_rho": rho,
            "Spearman_p": rho_p,
        })

    # Build regression data sub-table (all tuples)
    show_cols = ["region", "zone", "significant", "dP", "dDC_GW", "beta",
                 "nerc_area_summer", "nerc_area_lt"]
    used_x = [col for col, _ in x_candidates if col in df.columns]
    show_cols.extend(used_x)
    avail = [c for c in show_cols if c in df.columns]
    regression_data = df[avail].copy()

    return pd.DataFrame(results), regression_data


# =============================================================================
# Step 6b: Region-level aggregation and correlation (ONE observation per ISO)
# =============================================================================

def build_region_level_table(zone_df, zone_pct_df, region_df, nerc_summer, nerc_lt):
    """
    Collapse zone-level dP (from table3) and dP_pct (from table4) to ONE observation
    per region. Aggregations: max, mean, median.

    Two types of rows:
      (1) ISO rows: dP and dP_pct from table3/table4 aggregated across the ISO's zones.
          Regions without any significant zone → dP = dP_pct = 0.
          NERC metrics merged via ISO_TO_NERC.
      (2) Cities-by-NERC rows: "Non-ISO Cities" rows in table3/table4 are mapped to
          NERC regions via CITY_STATE + STATE_TO_NERC (with CITY_TO_NERC_OVERRIDE).
          Each city is assigned to ONE NERC region (the first candidate), then
          cities within the same NERC region are aggregated (max/mean/median).

    Returns: DataFrame with columns
             region, type, status, n_sig_zones,
             dP_max / dP_mean / dP_median,
             dP_pct_max / dP_pct_mean / dP_pct_median,
             plus NERC summer_* and lt_* metrics.
    """
    CITY_ROW_LABEL = "Non-ISO Cities"

    def _city_nerc(city_name):
        """Return the primary NERC region for a city, or None."""
        if city_name in CITY_TO_NERC_OVERRIDE:
            cands = CITY_TO_NERC_OVERRIDE[city_name]
        else:
            state = CITY_STATE.get(city_name)
            if not state:
                return None
            cands = STATE_TO_NERC.get(state, [])
        return cands[0] if cands else None

    sig_by_region = {}    # ISO → {dP: [...], dP_pct: [...]}
    city_by_nerc = {}     # NERC region → {dP: [...], dP_pct: [...]}

    # --- dP from table3 ---
    if zone_df is not None and not zone_df.empty:
        region_col = "Region" if "Region" in zone_df.columns else zone_df.columns[0]
        zone_col = "Zone" if "Zone" in zone_df.columns else (
            "zone" if "zone" in zone_df.columns else zone_df.columns[1]
        )
        dp_col = None
        for c in zone_df.columns:
            if "dP_dollar" in c or c == "dP" or "delta_P" in c:
                dp_col = c
                break
        for _, z in zone_df.iterrows():
            reg = z.get(region_col)
            if pd.isna(reg):
                continue
            dp_val = z.get(dp_col, np.nan) if dp_col else np.nan
            if pd.isna(dp_val):
                continue

            if reg == CITY_ROW_LABEL:
                city = z.get(zone_col)
                nerc = _city_nerc(city)
                if not nerc:
                    continue
                city_by_nerc.setdefault(nerc, {"dP": [], "dP_pct": []})
                city_by_nerc[nerc]["dP"].append(float(dp_val))
            else:
                sig_by_region.setdefault(reg, {"dP": [], "dP_pct": []})
                sig_by_region[reg]["dP"].append(float(dp_val))

    # --- dP_pct from table4 ---
    if zone_pct_df is not None and not zone_pct_df.empty:
        region_col_p = "Region" if "Region" in zone_pct_df.columns else zone_pct_df.columns[0]
        zone_col_p = "Zone" if "Zone" in zone_pct_df.columns else (
            "zone" if "zone" in zone_pct_df.columns else zone_pct_df.columns[1]
        )
        dp_pct_col = None
        for c in zone_pct_df.columns:
            if c == "dP_pct" or ("dP_pct" in c and "CI" not in c):
                dp_pct_col = c
                break
        for _, z in zone_pct_df.iterrows():
            reg = z.get(region_col_p)
            if pd.isna(reg):
                continue
            dp_pct_val = z.get(dp_pct_col, np.nan) if dp_pct_col else np.nan
            if pd.isna(dp_pct_val):
                continue

            if reg == CITY_ROW_LABEL:
                city = z.get(zone_col_p)
                nerc = _city_nerc(city)
                if not nerc:
                    continue
                city_by_nerc.setdefault(nerc, {"dP": [], "dP_pct": []})
                city_by_nerc[nerc]["dP_pct"].append(float(dp_pct_val))
            else:
                sig_by_region.setdefault(reg, {"dP": [], "dP_pct": []})
                sig_by_region[reg]["dP_pct"].append(float(dp_pct_val))

    # --- Build one row per ISO region ---
    rows = []
    for _, r in region_df.iterrows():
        region = r["region"]
        rtype = r["type"]
        if rtype != "ISO":
            # Cities are handled separately as city-by-NERC rows below.
            continue

        vals = sig_by_region.get(region, {"dP": [], "dP_pct": []})
        dp_list = vals["dP"]
        dp_pct_list = vals["dP_pct"]

        # If no significant zones for this region → all aggregates = 0
        if len(dp_list) == 0:
            dp_max = dp_mean = dp_median = 0.0
        else:
            dp_max = float(np.max(dp_list))
            dp_mean = float(np.mean(dp_list))
            dp_median = float(np.median(dp_list))

        if len(dp_pct_list) == 0:
            dpp_max = dpp_mean = dpp_median = 0.0
        else:
            dpp_max = float(np.max(dp_pct_list))
            dpp_mean = float(np.mean(dp_pct_list))
            dpp_median = float(np.median(dp_pct_list))

        cb = r.get("causal_beta", np.nan)
        causal_beta = float(cb) if not pd.isna(cb) else 0.0

        row = {
            "region": region,
            "type": rtype,
            "status": r.get("status", ""),
            "n_sig_zones": len(dp_list),
            "causal_beta": causal_beta,
            "dP_max": dp_max,
            "dP_mean": dp_mean,
            "dP_median": dp_median,
            "dP_pct_max": dpp_max,
            "dP_pct_mean": dpp_mean,
            "dP_pct_median": dpp_median,
        }

        # --- Merge NERC summer metrics (ISO → NERC via ISO_TO_NERC) ---
        nerc_s = merge_nerc_to_region(region, rtype, nerc_summer, "summer")
        row.update(nerc_s)

        # --- Merge NERC long-term metrics ---
        nerc_l = merge_nerc_to_region(region, rtype, nerc_lt, "lt")
        # drop _change cols from lt (we recompute forecast_change below)
        nerc_l = {k: v for k, v in nerc_l.items() if not k.endswith("_change")}
        row.update(nerc_l)

        rows.append(row)

    # --- Append city-by-NERC aggregated rows ---
    city_row = region_df[region_df["region"] == "Non-ISO Cities"]
    if not city_row.empty:
        cb_city = city_row.iloc[0].get("causal_beta", np.nan)
        city_causal_beta = float(cb_city) if not pd.isna(cb_city) else 0.0
    else:
        city_causal_beta = 0.0

    summer_areas_all = (nerc_summer["area"].dropna().unique().tolist()
                        if nerc_summer is not None and not nerc_summer.empty
                           and "area" in nerc_summer.columns else [])
    lt_areas_all = (nerc_lt["area"].dropna().unique().tolist()
                    if nerc_lt is not None and not nerc_lt.empty
                       and "area" in nerc_lt.columns else [])

    for nerc, vals in sorted(city_by_nerc.items()):
        dp_list = vals["dP"]
        dp_pct_list = vals["dP_pct"]

        if len(dp_list) == 0:
            dp_max = dp_mean = dp_median = 0.0
        else:
            dp_max = float(np.max(dp_list))
            dp_mean = float(np.mean(dp_list))
            dp_median = float(np.median(dp_list))

        if len(dp_pct_list) == 0:
            dpp_max = dpp_mean = dpp_median = 0.0
        else:
            dpp_max = float(np.max(dp_pct_list))
            dpp_mean = float(np.mean(dp_pct_list))
            dpp_median = float(np.median(dp_pct_list))

        row = {
            "region": f"Cities:{nerc}",
            "type": "CityNERC",
            "status": "cities",
            "n_sig_zones": max(len(dp_list), len(dp_pct_list)),
            "causal_beta": city_causal_beta,
            "dP_max": dp_max,
            "dP_mean": dp_mean,
            "dP_median": dp_median,
            "dP_pct_max": dpp_max,
            "dP_pct_mean": dpp_mean,
            "dP_pct_median": dpp_median,
        }

        # Merge NERC summer metrics for this single NERC region
        if summer_areas_all:
            matched_s = match_nerc_areas([nerc], summer_areas_all)
            if matched_s:
                row["summer_nerc_area"] = " | ".join(matched_s)
                s_data = merge_nerc_to_region_direct(matched_s, nerc_summer, "summer")
                row.update(s_data)
                # Also compute _change = latest - earliest (parity with ISO rows)
                for k in list(s_data.keys()):
                    if k.endswith("_latest"):
                        base = k[:-len("_latest")]
                        earliest_k = base + "_earliest"
                        if earliest_k in s_data:
                            row[base + "_change"] = s_data[k] - s_data[earliest_k]

        # Merge NERC long-term metrics
        if lt_areas_all:
            matched_l = match_nerc_areas([nerc], lt_areas_all)
            if matched_l:
                row["lt_nerc_area"] = " | ".join(matched_l)
                nerc_l = merge_nerc_to_region_direct(matched_l, nerc_lt, "lt")
                nerc_l = {k: v for k, v in nerc_l.items() if not k.endswith("_change")}
                row.update(nerc_l)

        rows.append(row)

    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return rdf

    # extreme - reference margin
    for suffix in ["_latest", "_earliest"]:
        ext_col = f"summer_extreme_margin{suffix}"
        ref_col = f"summer_ref_margin_level{suffix}"
        new_col = f"summer_margin_above_ref{suffix}"
        if ext_col in rdf.columns and ref_col in rdf.columns:
            rdf[new_col] = rdf[ext_col] - rdf[ref_col]

    # LTRA forecast change (latest report: max forecast year - min forecast year)
    import re
    lt_cols = [c for c in rdf.columns if c.startswith("lt_") and c.endswith("_latest")]
    metrics_years = {}
    for c in lt_cols:
        m = re.search(r"lt_([a-z]+)_(\d{4})_latest", c)
        if m:
            metric = m.group(1)
            year = int(m.group(2))
            metrics_years.setdefault(metric, []).append((year, c))
    for metric, yc_list in metrics_years.items():
        yc_list.sort()
        if len(yc_list) >= 2:
            min_col = yc_list[0][1]
            max_col = yc_list[-1][1]
            rdf[f"lt_{metric}_forecast_change"] = rdf[max_col] - rdf[min_col]

    n_iso = int((rdf["type"] == "ISO").sum()) if "type" in rdf.columns else 0
    n_city_nerc = int((rdf["type"] == "CityNERC").sum()) if "type" in rdf.columns else 0
    print(f"  Region-level table: {len(rdf)} rows "
          f"({n_iso} ISO + {n_city_nerc} city-NERC; non-sig regions have dP=dP_pct=0)")
    return rdf


def run_region_correlations(region_level_df):
    """
    Correlation analysis at the REGION level (one obs per ISO).

    For each Y in {dP_max, dP_mean, dP_median, dP_pct_max, dP_pct_mean, dP_pct_median}
    and each X (NERC reliability metric), compute Pearson r and Spearman rho.

    Returns:
        corr_df: long-format DataFrame with columns
                 [Y, Variable, Column, N, Pearson_r, Pearson_p,
                  Spearman_rho, Spearman_p]
    """
    df = region_level_df.copy()
    if df.empty or len(df) < 4:
        print(f"  Only {len(df)} regions — need ≥4 for correlation.")
        return pd.DataFrame()

    y_vars = [
        ("dP_max",        "ΔP ($/MWh) — max"),
        ("dP_mean",       "ΔP ($/MWh) — mean"),
        ("dP_median",     "ΔP ($/MWh) — median"),
        ("dP_pct_max",    "ΔP (%) — max"),
        ("dP_pct_mean",   "ΔP (%) — mean"),
        ("dP_pct_median", "ΔP (%) — median"),
        ("causal_beta",   "β ($/MWh per GW DC)"),
    ]

    # Auto-discover X candidates (same logic as run_correlations)
    x_candidates = []
    for c in df.columns:
        if c.startswith("summer_") and c.endswith("_latest"):
            label = c.replace("summer_", "").replace("_latest", "").replace("_", " ").title()
            x_candidates.append((c, f"Summer: {label} (latest)"))
        elif c.startswith("summer_") and c.endswith("_earliest"):
            label = c.replace("summer_", "").replace("_earliest", "").replace("_", " ").title()
            x_candidates.append((c, f"Summer: {label} (earliest)"))
        elif c.startswith("lt_") and c.endswith("_latest"):
            label = c.replace("lt_", "").replace("_latest", "").replace("_", " ").title()
            x_candidates.append((c, f"Long-Term: {label} (latest rpt)"))
        elif c.startswith("lt_") and c.endswith("_earliest"):
            label = c.replace("lt_", "").replace("_earliest", "").replace("_", " ").title()
            x_candidates.append((c, f"Long-Term: {label} (earliest rpt)"))
        elif c.startswith("lt_") and c.endswith("_forecast_change"):
            metric = c.split("_")[1].upper()
            x_candidates.append((c, f"Long-Term: {metric} (Δ forecast)"))

    results = []
    for y_col, y_label in y_vars:
        if y_col not in df.columns:
            continue
        for x_col, x_label in x_candidates:
            if x_col not in df.columns:
                continue
            valid = df[[y_col, x_col]].dropna()
            if len(valid) < 4:
                continue
            if valid[x_col].std() == 0 or valid[y_col].std() == 0:
                continue

            r_val, p_val = sp_stats.pearsonr(valid[y_col], valid[x_col])
            rho, rho_p = sp_stats.spearmanr(valid[y_col], valid[x_col])
            if pd.isna(r_val):
                continue

            results.append({
                "Y": y_label,
                "Y_col": y_col,
                "Variable": x_label,
                "Column": x_col,
                "N": len(valid),
                "Pearson_r": r_val,
                "Pearson_p": p_val,
                "Spearman_rho": rho,
                "Spearman_p": rho_p,
            })

    return pd.DataFrame(results)


def print_region_correlation_table(corr_df):
    """Print region-level correlation table, grouped by Y variable."""
    if corr_df is None or corr_df.empty:
        print("\n  REGION-LEVEL CORRELATION: No results.\n")
        return

    print(f"\n{'='*120}")
    print(f"  REGION-LEVEL CORRELATION: One observation per ISO")
    print(f"  Y = dP or dP_pct aggregated across zones (max / mean / median)")
    print(f"  Non-significant regions: dP = dP_pct = 0")
    print(f"{'='*120}")

    for y_label, group in corr_df.groupby("Y", sort=False):
        print(f"\n  Y = {y_label}")
        print(f"  {'-'*116}")
        hdr = (f"  {'Variable':<42s} {'N':>3s} "
               f"{'Pearson r':>10s} {'p':>7s} {'':>3s} "
               f"{'Spearman ρ':>10s} {'p':>7s} {'':>3s} "
               f"{'Dir':>5s}")
        print(hdr)
        for _, r in group.iterrows():
            direction = ""
            if r["Pearson_p"] < 0.20 or r["Spearman_p"] < 0.20:
                direction = "+" if r["Pearson_r"] > 0 else "−"
            ps = fmt_stars(r["Pearson_p"])
            ss = fmt_stars(r["Spearman_p"])
            print(f"  {r['Variable']:<42s} {r['N']:>3.0f} "
                  f"{r['Pearson_r']:>10.4f} {r['Pearson_p']:>7.4f} {ps:<3s} "
                  f"{r['Spearman_rho']:>10.4f} {r['Spearman_p']:>7.4f} {ss:<3s} "
                  f"{direction:>5s}")

    print(f"\n  *** p<0.01  ** p<0.05  * p<0.10")
    print(f"  N = number of ISOs with non-missing X variable")
    print(f"{'='*120}\n")


# =============================================================================
# Output
# =============================================================================

def fmt_stars(p):
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def fmt_pct(v):
    if pd.isna(v): return ""
    return f"{v:.1%}"


def print_zone_impacts(zone_df):
    """Print zone-level price impacts (from r03_summary Table 3)."""
    if zone_df.empty:
        print("\n  No zone-level impacts available. Run r02_summary.py first.\n")
        return

    print(f"\n{'='*130}")
    print(f"  ZONE-LEVEL PRICE INCREASE BY 2025 (from Bartik IV causal estimates)")
    print(f"{'='*130}")

    # Determine column names (may vary)
    region_col = "Region" if "Region" in zone_df.columns else zone_df.columns[0]
    zone_col = "Zone" if "Zone" in zone_df.columns else "zone"
    dp_col = [c for c in zone_df.columns if "dP" in c or "delta_P" in c or "dollar" in c]
    dc_col = [c for c in zone_df.columns if "dDC" in c or "delta_DC" in c]

    if dp_col:
        dp_col = dp_col[0]
    else:
        print("  Cannot find dP column."); return

    dc_col = dc_col[0] if dc_col else None

    ci_lo = [c for c in zone_df.columns if "CI_lo" in c or "ci_lo" in c.lower()]
    ci_hi = [c for c in zone_df.columns if "CI_hi" in c or "ci_hi" in c.lower()]
    ci_lo = ci_lo[0] if ci_lo else None
    ci_hi = ci_hi[0] if ci_hi else None

    beta_col = "beta" if "beta" in zone_df.columns else None
    est_col = "Estimate" if "Estimate" in zone_df.columns else None

    prev_region = None
    for _, r in zone_df.iterrows():
        reg = r.get(region_col, "")
        if reg != prev_region:
            if prev_region is not None:
                print(f"  {'':.<120s}")
            prev_region = reg

        zone = r.get(zone_col, "")
        dp = r.get(dp_col, np.nan)
        dc = r.get(dc_col, np.nan) if dc_col else np.nan
        lo = r.get(ci_lo, np.nan) if ci_lo else np.nan
        hi = r.get(ci_hi, np.nan) if ci_hi else np.nan
        beta = r.get(beta_col, np.nan) if beta_col else np.nan
        est = r.get(est_col, "") if est_col else ""

        ci_str = f"[{lo:.4f}, {hi:.4f}]" if not (pd.isna(lo) or pd.isna(hi)) else ""

        print(f"  {reg:<16s} {zone:<18s} {est:<14s} "
              f"β={beta:>8.4f}  ΔDC={dc:>8.4f} GW  "
              f"ΔP={dp:>8.4f} $/MWh  {ci_str}")

    print(f"{'='*130}\n")


def print_merged_overview(df):
    """Print cross-sectional overview table."""
    print(f"\n{'='*140}")
    print(f"  CROSS-SECTIONAL OVERVIEW: Price Effect × Grid Characteristics")
    print(f"{'='*140}")

    # Decide which columns to show
    show_cols = ["region", "status", "ols_beta", "ols_p"]

    # Add NERC summer area
    if "summer_nerc_area" in df.columns:
        show_cols.append("summer_nerc_area")

    # Find reserve/extreme margin columns
    for c in df.columns:
        if "reserve_margin_latest" in c or "extreme_margin_latest" in c or "anticipated_margin_latest" in c:
            show_cols.append(c)

    # NERC long-term
    for c in df.columns:
        if c.startswith("lt_") and ("eeu" in c.lower() or "lolh" in c.lower()) and c.endswith("_latest"):
            show_cols.append(c)

    avail = [c for c in show_cols if c in df.columns]
    print(df[avail].to_string(index=False, na_rep="—", float_format="%.4f"))
    print(f"{'='*140}\n")


def print_correlation_table(corr_df):
    if corr_df.empty:
        print("\n  CORRELATION: No results (insufficient data or variables).\n")
        return

    print(f"\n{'='*110}")
    print(f"  CORRELATION: Zone/City ΔP vs Grid Characteristics")
    print(f"  ΔP = β×ΔDC for significant zones, 0 for non-significant")
    print(f"  Each load zone / city = one observation")
    print(f"{'='*110}")

    hdr = (f"  {'Variable':<40s} {'N':>3s} "
           f"{'Pearson r':>10s} {'p':>7s} {'':>3s} "
           f"{'Spearman ρ':>10s} {'p':>7s} {'':>3s} " 
           f"{'Direction':>12s}")
    print(hdr)
    print(f"  {'-'*106}")

    for _, r in corr_df.iterrows():
        direction = ""
        # If any p < 0.20, note the direction (optional)
        if r["Pearson_p"] < 0.20 or r["Spearman_p"] < 0.20:  
            direction = "+" if r["Pearson_r"] > 0 else "−"
            
        pearson_stars = fmt_stars(r["Pearson_p"])     # Pearson significance stars
        spearman_stars = fmt_stars(r["Spearman_p"])   # Spearman significance stars

        print(f"  {r['Variable']:<40s} {r['N']:>3.0f} "
              f"{r['Pearson_r']:>10.4f} {r['Pearson_p']:>7.4f} {pearson_stars:<3s} "
              f"{r['Spearman_rho']:>10.4f} {r['Spearman_p']:>7.4f} {spearman_stars:<3s} " # <-- Spearman stars
              f"{direction:>12s}")
        

    print(f"  {'-'*106}")
    print(f"  *** p<0.01  ** p<0.05  * p<0.10")
    print(f"  Note: Zones within same ISO share same NERC area → clustered observations.")
    print(f"  Cities provide independent cross-sectional variation.")
    print(f"{'='*110}\n")


def print_narrative(merged_df, corr_df):
    """Generate interpretive narrative."""
    print(f"\n{'='*90}")
    print(f"  INTERPRETATION: Why Some Regions Show Larger Price Effects")
    print(f"{'='*90}")

    isos = merged_df[merged_df["type"] == "ISO"].copy()
    causal = isos[isos["causal_beta"].notna()]
    no_causal = isos[isos["causal_beta"].isna()]

    if not causal.empty:
        print(f"\n  Regions with significant causal effect (2SLS or Wald):")
        for _, r in causal.sort_values("causal_beta", ascending=False).iterrows():
            parts = [f"β={r['causal_beta']:.4f} ({r.get('causal_source','')})"]
            for c in r.index:
                if "reserve_margin_latest" in str(c) and not pd.isna(r[c]):
                    parts.append(f"reserve margin={r[c]:.1%}")
                if "extreme_margin_latest" in str(c) and not pd.isna(r[c]):
                    parts.append(f"extreme margin={r[c]:.1%}")
            print(f"    {r['region']:<12s}  {', '.join(parts)}")

    if not no_causal.empty:
        print(f"\n  Regions without significant causal effect:")
        for _, r in no_causal.iterrows():
            parts = [f"OLS={r.get('ols_beta', np.nan):.4f}", f"status={r.get('status','')}"]
            print(f"    {r['region']:<12s}  {', '.join(parts)}")

    # Correlation highlights
    if not corr_df.empty:
        sig_any = corr_df[corr_df["Pearson_p"] < 0.20]  # relaxed for small N
        if not sig_any.empty:
            print(f"\n  Notable correlations (including suggestive, p<0.20):")
            for _, r in sig_any.sort_values("Pearson_p").iterrows():
                sign = "positively" if r["Pearson_r"] > 0 else "negatively"
                print(f"    {r['Variable']}: r={r['Pearson_r']:.3f} (p={r['Pearson_p']:.3f}), "
                      f"ρ={r['Spearman_rho']:.3f} (p={r['Spearman_p']:.3f}) — {sign} correlated")

    print(f"\n  Caveats:")
    print(f"  - N=7 ISOs: correlations are descriptive, not causal")
    print(f"  - NERC area boundaries ≠ ISO boundaries (approximate mapping)")
    print(f"  - Reserve margins reflect aggregate grid conditions, not DC-specific stress")
    print(f"{'='*90}\n")


def save_results(merged_df, corr_df, zone_df, regression_data,
                 region_level_df, region_corr_df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx = out_dir / "cross_sectional_analysis.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        merged_df.to_excel(w, sheet_name="Cross-Sectional", index=False)
        if corr_df is not None and not corr_df.empty:
            corr_df.to_excel(w, sheet_name="Correlations (zone)", index=False)
        if zone_df is not None and not zone_df.empty:
            zone_df.to_excel(w, sheet_name="Zone Impacts", index=False)
        if regression_data is not None and not regression_data.empty:
            regression_data.to_excel(w, sheet_name="Regression Data (zone)", index=False)
        if region_level_df is not None and not region_level_df.empty:
            region_level_df.to_excel(w, sheet_name="Region-Level Data", index=False)
        if region_corr_df is not None and not region_corr_df.empty:
            region_corr_df.to_excel(w, sheet_name="Correlations (region)", index=False)

    merged_df.to_csv(out_dir / "cross_sectional_merged.csv", index=False)
    if corr_df is not None and not corr_df.empty:
        corr_df.to_csv(out_dir / "correlations_zone.csv", index=False)
    if regression_data is not None and not regression_data.empty:
        regression_data.to_csv(out_dir / "regression_data_zone.csv", index=False)
    if region_level_df is not None and not region_level_df.empty:
        region_level_df.to_csv(out_dir / "region_level_data.csv", index=False)
    if region_corr_df is not None and not region_corr_df.empty:
        region_corr_df.to_csv(out_dir / "correlations_region.csv", index=False)

    print(f"  Saved: {xlsx}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-sectional analysis: price impact vs grid characteristics")
    parser.add_argument("--bartik-dir", default=str(BARTIK_DIR))
    parser.add_argument("--summary-dir", default=str(SUMMARY_DIR))
    parser.add_argument("--nerc-summer", default=str(NERC_SUMMER))
    parser.add_argument("--nerc-longterm", default=str(NERC_LONGTERM))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    bartik_dir = Path(args.bartik_dir)
    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.out_dir)

    print(f"{'='*70}")
    print(f"  Cross-Sectional Analysis v2")
    print(f"{'='*70}")

    # [1] Zone-level impacts
    print("\n[1] Loading zone-level price impacts...")
    zone_df = load_zone_impacts(summary_dir, bartik_dir)

    # [1b] Zone-level relative impacts (table4)
    print("\n[1b] Loading zone-level relative impacts (table4)...")
    t4_path = summary_dir / "table4_zone_relative_2025.csv"
    if t4_path.exists():
        zone_pct_df = pd.read_csv(t4_path)
        print(f"  Loaded {len(zone_pct_df)} rows from {t4_path}")
    else:
        print(f"  WARNING: {t4_path} not found — dP_pct aggregates will be 0")
        zone_pct_df = pd.DataFrame()

    # [2] All region results (for correlations)
    print("\n[2] Loading all region results...")
    region_df = load_all_region_results(bartik_dir)
    print(f"  Found {len(region_df)} regions:")
    for _, r in region_df.iterrows():
        causal_str = f"causal β={r.get('causal_beta', np.nan):.4f}" if not pd.isna(r.get("causal_beta")) else "no causal"
        print(f"    {r['region']:<20s}  OLS={r.get('ols_beta', np.nan):.4f}  "
              f"status={r.get('status',''):<12s}  ({causal_str})")

    # [3] NERC Summer
    print("\n[3] Loading NERC Summer Reliability Assessment...")
    nerc_summer = load_nerc_summer(Path(args.nerc_summer))

    # [4] NERC Long-Term
    print("\n[4] Loading NERC Long-Term Reliability Assessment...")
    nerc_lt = load_nerc_longterm(Path(args.nerc_longterm))

    # [5] Merge (ISO-level for overview)
    print("\n[5] Merging ISO-level datasets...")
    merged = build_merged_table(region_df, nerc_summer, nerc_lt)
    print(f"  Merged: {len(merged)} rows × {len(merged.columns)} columns")

    # [6] Build zone/city-level table for correlation
    print("\n[6] Building zone/city-level NERC table...")
    zone_nerc = build_zone_nerc_table(zone_df, region_df, nerc_summer, nerc_lt, bartik_dir)

    # [7] Correlations at zone/city level
    print("\n[7] Computing correlations (zone/city level)...")
    corr_df, regression_data = run_correlations(zone_nerc)

    # [7b] Region-level aggregation and correlation (one obs per ISO / city-NERC)
    print("\n[7b] Building region-level table (max/mean/median; ISO + city-NERC)...")
    region_level_df = build_region_level_table(zone_df, zone_pct_df, region_df, nerc_summer, nerc_lt)

    print("\n[7c] Computing region-level correlations...")
    region_corr_df = run_region_correlations(region_level_df)

    # [8] Print
    print_zone_impacts(zone_df)
    print_merged_overview(merged)
    print_correlation_table(corr_df)
    print_region_correlation_table(region_corr_df)

    # Also show the region-level data used
    if not region_level_df.empty:
        print(f"\n{'='*120}")
        print(f"  REGION-LEVEL DATA (one row per ISO, Y values used in region correlation)")
        print(f"{'='*120}")
        show_cols = ["region", "status", "n_sig_zones", "causal_beta",
                     "dP_max", "dP_mean", "dP_median",
                     "dP_pct_max", "dP_pct_mean", "dP_pct_median"]
        avail = [c for c in show_cols if c in region_level_df.columns]
        print(region_level_df[avail].to_string(index=False, na_rep="—", float_format="%.4f"))
        print(f"{'='*120}\n")

    # Print regression data sub-table
    if not regression_data.empty:
        print(f"\n{'='*140}")
        print(f"  REGRESSION DATA: Every (zone/city, ΔP, NERC metrics) tuple used in correlation")
        print(f"  ΔP = β×ΔDC for significant zones, 0 for non-significant")
        print(f"{'='*140}")
        # Show condensed version for terminal
        key_cols = ["region", "zone", "significant", "dP", "nerc_area_summer"]
        # Add a few representative NERC columns
        for c in regression_data.columns:
            if any(k in c for k in ["reserve_margin_latest", "extreme_margin_latest",
                                     "margin_above_ref_latest", "eeu", "lolh"]):
                if c.endswith("_latest"):
                    key_cols.append(c)
        avail = [c for c in key_cols if c in regression_data.columns]
        print(regression_data[avail].to_string(index=False, na_rep="—", float_format="%.4f"))
        print(f"  Total: {len(regression_data)} observations")
        print(f"  Full data with all X variables saved to Excel sheet 'Regression Data'")
        print(f"{'='*140}\n")

    print_narrative(merged, corr_df)

    # [9] Save
    print("[9] Saving results...")
    save_results(merged, corr_df, zone_df, regression_data,
                 region_level_df, region_corr_df, out_dir)

    print(f"\n{'='*70}")
    print(f"  Done. Results: {out_dir.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
