"""
r06 Stage 1: Build the clean state x year analysis panel.

Inputs (all read-only):
  employment/dc_facilities_by_state_year.csv
  employment/qwi_all_naics_annual.csv
  employment/pwc_multipliers.csv
  employment/pwc_state_data.csv
  employment/tax/state_subsidy_by_year_million_wide.xlsx
  econ_and_ai/state_accept/state_gdp.xlsx
  econ_and_ai/macroeconomic.xlsx

Outputs (new files only, all in employment/../results/r6_bartik/):
  panel_state_year.csv
  state_adjacency.json
  panel_data_dictionary.md   (written by a separate step)
  build_panel_log.txt        (stdout log)

Definitions of emp_5182_cleaned / emp_5182_cleaned_mult are copied EXACTLY from
09_ols_employment_dc.py so published numbers stay reproducible.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Release-set path rule: no absolute paths.  Resolved from this file's location.
EMP = Path(__file__).resolve().parent                     # employment/
REPO_ROOT = EMP.parent

def relpath(p) -> str:
    """Repo-relative path, so no absolute path reaches a shipped artefact."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)
ROOT = EMP.parent                                         # 
OUT = EMP / "../results/r6_bartik"                                         # generated results
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# This script imported NEITHER
# analysis_universe.py NOR any scope constant: it asserted the literal 51 in
# three places and never named the analysis universe at all, which is how a
# 51-unit panel could be handed downstream to consumers that disagreed about
# what to do with it.
# This script DELIBERATELY builds the full 51-unit frame -- the scope
# restriction belongs to apply_scope() in the estimators, because it also has
# to renormalise the exposure shares.  What is asserted here is therefore not
# "49 units" but the exact identity ALL_QWI_UNITS = ANALYSIS_UNITS +
# OUT_OF_SCOPE_UNITS, so that a unit which silently disappears from a source
# file or appears from nowhere aborts the build.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(EMP))
from analysis_universe import (ALL_QWI_UNITS, ANALYSIS_UNITS,  # noqa: E402
                               N_ANALYSIS_UNITS, OUT_OF_SCOPE_UNITS,
                               UNIVERSE_LABEL, UniverseError)

YEARS = list(range(2016, 2026))          # DC capacity 2016-2025
EMP_YEARS = list(range(2016, 2025))      # QWI employment 2016-2024

log_lines = []

def log(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    log_lines.append(s)

# ------------------------------------------------------------------
# 0. FIPS -> USPS abbreviation (standard, hand-checked)
# ------------------------------------------------------------------
FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}

ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

# ------------------------------------------------------------------
# Census regions / divisions (9 divisions, 4 regions)
# ------------------------------------------------------------------
DIVISION = {
    "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Middle Atlantic": ["NJ", "NY", "PA"],
    "East North Central": ["IL", "IN", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "Pacific": ["AK", "CA", "HI", "OR", "WA"],
}
DIV_TO_REGION = {
    "New England": "Northeast", "Middle Atlantic": "Northeast",
    "East North Central": "Midwest", "West North Central": "Midwest",
    "South Atlantic": "South", "East South Central": "South",
    "West South Central": "South",
    "Mountain": "West", "Pacific": "West",
}
STATE_DIVISION = {s: d for d, ss in DIVISION.items() for s in ss}
STATE_REGION = {s: DIV_TO_REGION[d] for s, d in STATE_DIVISION.items()}

# ------------------------------------------------------------------
# Contiguous (land/river bordering) states. Four-corners diagonal pairs
# (AZ-CO, NM-UT) ARE included. AK and HI have no neighbours.
# ------------------------------------------------------------------
ADJACENCY = {
    "AL": ["FL", "GA", "MS", "TN"],
    "AK": [],
    "AZ": ["CA", "CO", "NV", "NM", "UT"],
    "AR": ["LA", "MS", "MO", "OK", "TN", "TX"],
    "CA": ["AZ", "NV", "OR"],
    "CO": ["AZ", "KS", "NE", "NM", "OK", "UT", "WY"],
    "CT": ["MA", "NY", "RI"],
    "DE": ["MD", "NJ", "PA"],
    "DC": ["MD", "VA"],
    "FL": ["AL", "GA"],
    "GA": ["AL", "FL", "NC", "SC", "TN"],
    "HI": [],
    "ID": ["MT", "NV", "OR", "UT", "WA", "WY"],
    "IL": ["IN", "IA", "KY", "MO", "WI"],
    "IN": ["IL", "KY", "MI", "OH"],
    "IA": ["IL", "MN", "MO", "NE", "SD", "WI"],
    "KS": ["CO", "MO", "NE", "OK"],
    "KY": ["IL", "IN", "MO", "OH", "TN", "VA", "WV"],
    "LA": ["AR", "MS", "TX"],
    "ME": ["NH"],
    "MD": ["DE", "PA", "VA", "WV", "DC"],
    "MA": ["CT", "NH", "NY", "RI", "VT"],
    "MI": ["IN", "OH", "WI"],
    "MN": ["IA", "ND", "SD", "WI"],
    "MS": ["AL", "AR", "LA", "TN"],
    "MO": ["AR", "IL", "IA", "KS", "KY", "NE", "OK", "TN"],
    "MT": ["ID", "ND", "SD", "WY"],
    "NE": ["CO", "IA", "KS", "MO", "SD", "WY"],
    "NV": ["AZ", "CA", "ID", "OR", "UT"],
    "NH": ["ME", "MA", "VT"],
    "NJ": ["DE", "NY", "PA"],
    "NM": ["AZ", "CO", "OK", "TX", "UT"],
    "NY": ["CT", "MA", "NJ", "PA", "VT"],
    "NC": ["GA", "SC", "TN", "VA"],
    "ND": ["MN", "MT", "SD"],
    "OH": ["IN", "KY", "MI", "PA", "WV"],
    "OK": ["AR", "CO", "KS", "MO", "NM", "TX"],
    "OR": ["CA", "ID", "NV", "WA"],
    "PA": ["DE", "MD", "NJ", "NY", "OH", "WV"],
    "RI": ["CT", "MA"],
    "SC": ["GA", "NC"],
    "SD": ["IA", "MN", "MT", "NE", "ND", "WY"],
    "TN": ["AL", "AR", "GA", "KY", "MS", "MO", "NC", "VA"],
    "TX": ["AR", "LA", "NM", "OK"],
    "UT": ["AZ", "CO", "ID", "NM", "NV", "WY"],
    "VT": ["MA", "NH", "NY"],
    "VA": ["KY", "MD", "NC", "TN", "WV", "DC"],
    "WA": ["ID", "OR"],
    "WV": ["KY", "MD", "OH", "PA", "VA"],
    "WI": ["IA", "IL", "MI", "MN"],
    "WY": ["CO", "ID", "MT", "NE", "SD", "UT"],
}

# ---- validate adjacency: symmetry, no self loops, valid codes -------
valid = set(ABBR_TO_NAME)
assert set(ADJACENCY) == valid, set(ADJACENCY) ^ valid
asym = []
for a, nbrs in ADJACENCY.items():
    assert a not in nbrs, f"self-loop {a}"
    assert len(nbrs) == len(set(nbrs)), f"dup in {a}"
    for b in nbrs:
        assert b in valid, f"bad code {b} in {a}"
        if a not in ADJACENCY[b]:
            asym.append((a, b))
assert not asym, f"asymmetric adjacency: {asym}"
log(f"[adjacency] 51 units, symmetric, "
    f"{sum(len(v) for v in ADJACENCY.values()) // 2} unique borders. OK")

with open(OUT / "state_adjacency.json", "w") as f:
    json.dump({k: sorted(v) for k, v in sorted(ADJACENCY.items())}, f, indent=2)
log(f"[adjacency] wrote {relpath(OUT/'state_adjacency.json')}")

# ==================================================================
# 1. Data centre capacity: wide -> long
# ==================================================================
dc_raw = pd.read_csv(EMP / "dc_facilities_by_state_year.csv")
log(f"\n[dc] raw shape {dc_raw.shape}; rows incl. aggregate: "
    f"{sorted(set(dc_raw.state_abbr) - valid)}")
# Keep the source file's own US aggregate row before dropping it: it is an
# INDEPENDENTLY RECORDED total, and is used below to check the Bartik share
# denominator against something other than a re-sum of the same state rows
#
_us_rows = dc_raw[dc_raw["state_abbr"] == "US"]
if len(_us_rows) != 1:
    raise UniverseError(
        f"dc_facilities_by_state_year.csv carries {len(_us_rows)} 'US' aggregate "
        f"rows, expected exactly 1; the share-denominator cross-check has "
        f"nothing to compare against")
_US_ROW = _us_rows.iloc[0]
dc_raw = dc_raw[dc_raw["state_abbr"].isin(valid)].copy()
if set(dc_raw["state_abbr"]) != set(ALL_QWI_UNITS):
    raise UniverseError(
        "capacity file units != ANALYSIS_UNITS + OUT_OF_SCOPE_UNITS: "
        f"unexpected {sorted(set(dc_raw['state_abbr']) - set(ALL_QWI_UNITS))}, "
        f"absent {sorted(set(ALL_QWI_UNITS) - set(dc_raw['state_abbr']))}")

rows = []
for _, r in dc_raw.iterrows():
    for y in YEARS:
        rows.append({"state_abbr": r["state_abbr"], "year": y,
                     "dc_count": r[f"count_{y}"], "dc_mw": r[f"MW_{y}"]})
dc = pd.DataFrame(rows)
dc["dc_gw"] = dc["dc_mw"] / 1000.0

# monotonicity check (installed cumulative capacity should not fall)
bad_mono = []
for s, g in dc.sort_values("year").groupby("state_abbr"):
    d = np.diff(g["dc_mw"].values)
    if (d < -1e-9).any():
        bad_mono.append((s, g.loc[g.index[1:][d < -1e-9], "year"].tolist()))
log(f"[dc] non-monotonic MW series: {bad_mono if bad_mono else 'none'}")
log(f"[dc] total US MW 2019 = {dc.loc[dc.year==2019,'dc_mw'].sum():,.1f}; "
    f"2025 = {dc.loc[dc.year==2025,'dc_mw'].sum():,.1f}")

d19 = dc[dc.year == 2019].set_index("state_abbr")
log(f"[dc] 2019 states with ZERO capacity: {(d19.dc_mw == 0).sum()} -> "
    f"{sorted(d19.index[d19.dc_mw == 0])}")
log(f"[dc] 2019 MW min={d19.dc_mw.min():.3f} p25={d19.dc_mw.quantile(.25):.3f} "
    f"median={d19.dc_mw.median():.3f} p75={d19.dc_mw.quantile(.75):.3f} "
    f"max={d19.dc_mw.max():,.1f} mean={d19.dc_mw.mean():,.1f}")
log("[dc] 2019 top-10 (MW):")
for s, v in d19.dc_mw.sort_values(ascending=False).head(10).items():
    log(f"      {s} {ABBR_TO_NAME[s]:<15s} {v:10,.1f} MW  "
        f"({v/d19.dc_mw.sum()*100:5.2f}% of US)")

d16 = dc[dc.year == 2016].set_index("state_abbr")
log(f"[dc] 2016 states with ZERO capacity: {(d16.dc_mw == 0).sum()} -> "
    f"{sorted(d16.index[d16.dc_mw == 0])}")

# ------- Bartik base shares -------
share2019 = (d19.dc_gw / d19.dc_gw.sum()).rename("share2019")
share2016 = (d16.dc_gw / d16.dc_gw.sum()).rename("share2016")

# The assertion that used to stand here was
#     assert abs(share2019.sum() - 1) < 1e-12 and abs(share2016.sum() - 1) < 1e-12
# on the two lines directly above, where the shares are DEFINED as x / x.sum().
# The sum is one by construction; the assertion cannot fail and tests nothing.
# It is kept below only as a floating-point sanity note, and the load-bearing
# checks are the two that follow, neither of which is true by construction:
#   (i)  the share index is exactly ALL_QWI_UNITS -- 51 rows here, because
#        build_panel.py deliberately keeps AK and HI and lets apply_scope drop
#        them downstream.  A unit lost in the pivot changes the denominator and
#        every share silently, and the sum still comes to one.
#   (ii) the shares reproduce the capacity file's own US aggregate row, which is
#        an independently recorded total, not a re-sum of the state rows.
assert abs(share2019.sum() - 1) < 1e-12 and abs(share2016.sum() - 1) < 1e-12, \
    "floating-point: renormalised shares do not sum to one"

_missing19 = sorted(set(ALL_QWI_UNITS) - set(share2019.index))
_extra19 = sorted(set(share2019.index) - set(ALL_QWI_UNITS))
_missing16 = sorted(set(ALL_QWI_UNITS) - set(share2016.index))
_extra16 = sorted(set(share2016.index) - set(ALL_QWI_UNITS))
assert not (_missing19 or _extra19 or _missing16 or _extra16), (
    "the Bartik base-share denominator is not the full source universe: "
    f"2019 missing {_missing19} extra {_extra19}; "
    f"2016 missing {_missing16} extra {_extra16}. "
    "A unit dropped here rescales every share and the sum still equals one, "
    "so the sums-to-one assertion above would not have noticed.")

# (ii) two independently recorded totals: the US aggregate row that ships in
# dc_facilities_by_state_year.csv, and the sum of the state rows used as the
# share denominator.  These are separate records in the source file.
_us19 = float(_US_ROW["MW_2019"]) / 1000.0
_us16 = float(_US_ROW["MW_2016"]) / 1000.0
assert abs(d19.dc_gw.sum() - _us19) < 1e-9, (d19.dc_gw.sum(), _us19)
assert abs(d16.dc_gw.sum() - _us16) < 1e-9, (d16.dc_gw.sum(), _us16)
log(f"[dc] share denominators check out against the file's own US row: "
    f"2019 {d19.dc_gw.sum():.6f} GW vs US row {_us19:.6f} GW; "
    f"2016 {d16.dc_gw.sum():.6f} GW vs US row {_us16:.6f} GW")
log(f"[dc] base-share index is the full source universe "
    f"({len(share2019)} units == len(ALL_QWI_UNITS))")

# ==================================================================
# 2. QWI employment: long -> wide by NAICS
# ==================================================================
qwi = pd.read_csv(EMP / "qwi_all_naics_annual.csv",
                  dtype={"naics": str, "state": str})
qwi["state_abbr"] = qwi["state"].map(FIPS_TO_ABBR)
assert qwi["state_abbr"].notna().all(), "unmapped FIPS"
log(f"\n[qwi] raw shape {qwi.shape}; states {qwi.state_abbr.nunique()}; "
    f"years {qwi.year.min()}-{qwi.year.max()}; naics {sorted(qwi.naics.unique())}")

# missingness check
naics_list = ["23", "2362", "51", "517", "5182", "5415"]
full_idx = pd.MultiIndex.from_product(
    [sorted(valid), EMP_YEARS, naics_list],
    names=["state_abbr", "year", "naics"])
have = qwi.set_index(["state_abbr", "year", "naics"]).index
missing = full_idx.difference(have)
miss_df = missing.to_frame(index=False)
log(f"[qwi] missing (state,year,naics) cells: {len(missing)} of {len(full_idx)}")
for (s, y), g in miss_df.groupby(["state_abbr", "year"]):
    log(f"      MISSING {s} {y}: naics {sorted(g.naics)}")
part = qwi[qwi.quarters_available < 4]
log(f"[qwi] rows with quarters_available < 4: {len(part)}")
for (s, y), g in part.groupby(["state_abbr", "year"]):
    log(f"      PARTIAL {s} {y}: {g.quarters_available.iloc[0]} quarters "
        f"({len(g)} naics rows)")
log(f"[qwi] Emp_annual_avg NaN={qwi.Emp_annual_avg.isna().sum()} "
    f"zero={(qwi.Emp_annual_avg == 0).sum()}; "
    f"EarnBeg NaN={qwi.EarnBeg_annual_avg.isna().sum()} "
    f"zero={(qwi.EarnBeg_annual_avg == 0).sum()}; "
    f"Payroll zero={(qwi.Payroll_annual_total == 0).sum()} of {len(qwi)}")

emp_w = qwi.pivot_table(index=["state_abbr", "year"], columns="naics",
                        values="Emp_annual_avg", aggfunc="first")
emp_w.columns = [f"emp_{c}" for c in emp_w.columns]
earn_w = qwi.pivot_table(index=["state_abbr", "year"], columns="naics",
                         values="EarnBeg_annual_avg", aggfunc="first")
earn_w.columns = [f"earn_{c}" for c in earn_w.columns]
empend_w = qwi.pivot_table(index=["state_abbr", "year"], columns="naics",
                           values="EmpEnd_annual_avg", aggfunc="first")
empend_w.columns = [f"empend_{c}" for c in empend_w.columns]
q_min = (qwi.groupby(["state_abbr", "year"])["quarters_available"].min()
         .rename("quarters_available_min"))
qwi_wide = pd.concat([emp_w, earn_w, empend_w, q_min], axis=1).reset_index()
qwi_wide["partial_year"] = (qwi_wide["quarters_available_min"] < 4).astype(int)

# ==================================================================
# 3. Skeleton panel  (51 states x 2016-2025)
# ==================================================================
panel = dc.merge(qwi_wide, on=["state_abbr", "year"], how="left")
panel["state_name"] = panel["state_abbr"].map(ABBR_TO_NAME)
panel["state_fips"] = panel["state_abbr"].map(
    {v: k for k, v in FIPS_TO_ABBR.items()})
panel["census_division"] = panel["state_abbr"].map(STATE_DIVISION)
panel["census_region"] = panel["state_abbr"].map(STATE_REGION)
panel["n_neighbors"] = panel["state_abbr"].map(lambda s: len(ADJACENCY[s]))
panel["emp_available"] = ((panel.year <= 2024)
                          & panel["emp_5182"].notna()).astype(int)
panel["partial_year"] = panel["partial_year"].fillna(0).astype(int)

# ==================================================================
# 4. IMPLAN / PwC local share + multiplier  (definitions copied from
#    r05_ols_employment_dc.py, lines 62-111)
# ==================================================================
mult = pd.read_csv(EMP / "pwc_multipliers.csv")
pwc_sd = pd.read_csv(EMP / "pwc_state_data.csv")
log(f"\n[pwc] pwc_multipliers.csv shape {mult.shape}, cols {list(mult.columns)}")
log(f"[pwc] pwc_state_data.csv shape {pwc_sd.shape}, "
    f"{len(pwc_sd.columns)} cols (2022 & 2023 emp/li/gdp/tax detail)")

NAME_TO_ABBR = {v: k for k, v in ABBR_TO_NAME.items()}
NAME_TO_ABBR["DC"] = "DC"          # pwc + qwi spell DC as "DC"
for df in (mult, pwc_sd):
    df["state_abbr"] = df["state"].map(NAME_TO_ABBR)
    assert df["state_abbr"].notna().all(), df.loc[df.state_abbr.isna(), "state"]

NAT_MULT = (mult["emp_total_no_spillover_avg"].sum()
            / mult["emp_direct_avg"].sum())
mult["spillover_dc"] = mult["emp_spillover_avg"] / NAT_MULT
mult["local_share"] = (mult["emp_direct_avg"]
                       / (mult["emp_direct_avg"] + mult["spillover_dc"]))
mult = mult.rename(columns={"emp_multiplier_local": "local_multiplier",
                            "emp_spillover_rate": "spillover_rate",
                            "li_multiplier_local": "li_multiplier_local",
                            "gdp_multiplier_local": "gdp_multiplier_local"})
log(f"[pwc] national multiplier = {NAT_MULT:.4f} "
    f"(r05 reports ~5.43); DC industry share = {100/NAT_MULT:.2f}%")
log(f"[pwc] local_share min={mult.local_share.min():.3f} "
    f"median={mult.local_share.median():.3f} max={mult.local_share.max():.3f}")

panel = panel.merge(
    mult[["state_abbr", "local_share", "local_multiplier", "spillover_rate",
          "li_multiplier_local", "gdp_multiplier_local",
          "emp_direct_avg", "emp_spillover_avg"]]
    .rename(columns={"emp_direct_avg": "pwc_emp_direct_avg",
                     "emp_spillover_avg": "pwc_emp_spillover_avg"}),
    on="state_abbr", how="left")
assert panel["local_share"].notna().all()

# PwC 2022/2023 direct operational employment (a second, independent read on
# how many jobs a DC actually supports in-state) -- averaged over the 2 years.
pwc_sd["pwc_emp_direct_2223"] = pwc_sd[["emp_direct_2022",
                                        "emp_direct_2023"]].mean(axis=1)
pwc_sd["pwc_emp_total_local_2223"] = pwc_sd[
    ["emp_total_no_spillover_2022", "emp_total_no_spillover_2023"]].mean(axis=1)
panel = panel.merge(
    pwc_sd[["state_abbr", "pwc_emp_direct_2223", "pwc_emp_total_local_2223"]],
    on="state_abbr", how="left")

panel["emp_5182_cleaned"] = panel["emp_5182"] * panel["local_share"]
panel["emp_5182_cleaned_mult"] = (panel["emp_5182_cleaned"]
                                  * panel["local_multiplier"])

# r05 spec-2 "raw multiplied" (kept for backward comparability only; it
# double-counts, see r05 docstring). National direct total by year is taken
# over ALL 51 QWI states, exactly as r05 does.
direct_by_year = qwi[qwi.naics == "5182"].groupby("year")["Emp_annual_avg"].sum()
panel["emp_5182_raw_multiplied"] = (
    panel["emp_5182"] * panel["local_multiplier"]
    + (panel["year"].map(direct_by_year) - panel["emp_5182"])
    * panel["spillover_rate"])

# ---------------------------------------------------------------------------
# CORRECTED STATE TOTAL (2026-07-29).
# The spec-2 "raw x multiplier + spillover" column above double-counts the
# cross-state spillover, and its own comment has said so since it was written.
# Observed NAICS 5182 already contains the spillover-driven jobs that lambda_s
# isolates.  Multiplying the RAW series by m_s therefore applies the in-state
# multiplier to jobs that in-state activity did not create, and the national
# term then adds the same spillover a second time.
# The corrected total puts each component in exactly once: the in-state-driven
# part goes through the local multiplier, the spillover-driven part enters at
# its own level and is not multiplied, since it is not driven by in-state data
# center activity.
#     raw*lambda*m + raw*(1-lambda)  =  raw * (lambda*m + 1 - lambda)
# Note that no national aggregate appears.  The balanced-versus-unbalanced
# question that separates the published 13,102 from the settled 13,232 cannot
# arise for this quantity, because it never sums across states.
# ---------------------------------------------------------------------------
panel["emp_5182_state_total"] = panel["emp_5182"] * (
    panel["local_share"] * panel["local_multiplier"] + 1 - panel["local_share"])

# ==================================================================
# 5. GDP / macro / subsidy merges
# ==================================================================
gdp = pd.read_excel(ROOT / "econ_and_ai/state_accept/state_gdp.xlsx")
gdp["state_abbr"] = gdp["State"].map(NAME_TO_ABBR)
assert gdp["state_abbr"].notna().all(), gdp.loc[gdp.state_abbr.isna(), "State"]
log(f"\n[gdp] state_gdp.xlsx: {gdp.shape} -- CROSS-SECTION ONLY "
    f"(Nominal GDP per capita, 2024 USD). No year dimension.")
panel = panel.merge(
    gdp.rename(columns={"Nominal_GDP_per_capita_2024_USD": "gdp_pc_2024"})
       [["state_abbr", "gdp_pc_2024"]], on="state_abbr", how="left")
assert panel["gdp_pc_2024"].notna().all()

macro = pd.read_excel(ROOT / "econ_and_ai/macroeconomic.xlsx")
log(f"[macro] macroeconomic.xlsx: NATIONAL series only, no state dimension. "
    f"rows={macro.iloc[:,0].tolist()}; years {macro.columns[1]}-"
    f"{macro.columns[-1]}")
macro = macro.set_index(macro.columns[0])
mm = macro.T
mm.index.name = "year"
mm = mm.reset_index()
mm["year"] = mm["year"].astype(int)
mm = mm.rename(columns={
    "GDP Chain-type Price Index (2012=1.000)": "us_gdp_price_index",
    "Consumer Price Index (1982-84=1.00)": "us_cpi",
    "Real Disposable Personal Income (Nominal)": "us_rdpi_nominal"})
keep = [c for c in ["year", "us_gdp_price_index", "us_cpi", "us_rdpi_nominal"]
        if c in mm.columns]
panel = panel.merge(mm[keep], on="year", how="left")
log(f"[macro] merged national deflators; non-missing years in panel: "
    f"{sorted(panel.loc[panel.us_cpi.notna(),'year'].unique())}")

sub = pd.read_excel(EMP / "tax/state_subsidy_by_year_million_wide.xlsx",
                    sheet_name="Sheet1")
sub["state_abbr"] = sub["State"].map(NAME_TO_ABBR)
assert sub["state_abbr"].notna().all()
sub_long = sub.melt(id_vars="state_abbr",
                    value_vars=[c for c in sub.columns
                                if isinstance(c, (int, np.integer))],
                    var_name="year", value_name="subsidy_musd")
sub_long["year"] = sub_long["year"].astype(int)
log(f"[subsidy] state_subsidy_by_year_million_wide.xlsx: "
    f"{sub['state_abbr'].nunique()} states, years "
    f"{sub_long.year.min()}-{sub_long.year.max()}. "
    f"States absent from the file are NOT zeros - they are unobserved.")
panel = panel.merge(sub_long, on=["state_abbr", "year"], how="left")
panel["subsidy_observed"] = panel["subsidy_musd"].notna().astype(int)

# total state employment: not available in any supplied file
log("[total-emp] NO total (all-industry) state employment series exists in any "
    "supplied file. QWI extract contains only NAICS 23, 2362, 51, 517, 5182, "
    "5415 -- no '00'/all-industry row. Column emp_total_state is therefore NOT "
    "created. Downstream normalisation by total employment requires pulling "
    "BLS SAE/QCEW separately.")

# ==================================================================
# 6. Bartik base shares + derived variables
# ==================================================================
panel = panel.merge(share2019.reset_index(), on="state_abbr", how="left")
panel = panel.merge(share2016.reset_index(), on="state_abbr", how="left")

panel = panel.sort_values(["state_abbr", "year"]).reset_index(drop=True)
panel["dc_gw_lag1"] = panel.groupby("state_abbr")["dc_gw"].shift(1)
panel["d_dc_gw"] = panel["dc_gw"] - panel["dc_gw_lag1"]
panel["log_dc_gw_p1"] = np.log(panel["dc_gw"] + 1.0)
for c in ["emp_5182", "emp_517", "emp_5415", "emp_51", "emp_23", "emp_2362",
          "emp_5182_cleaned", "emp_5182_cleaned_mult"]:
    panel[f"log_{c}"] = np.log(panel[c].where(panel[c] > 0))

# national totals (for shift construction downstream)
nat_gw = dc.groupby("year")["dc_gw"].sum().rename("us_dc_gw")
panel = panel.merge(nat_gw.reset_index(), on="year", how="left")
panel["us_dc_gw_lag1"] = panel["year"].map(
    nat_gw.shift(1).reindex(YEARS)).values
panel["us_d_dc_gw"] = panel["us_dc_gw"] - panel["us_dc_gw_lag1"]

COL_ORDER = [
    "state_abbr", "state_name", "state_fips", "year",
    "census_region", "census_division", "n_neighbors",
    "dc_count", "dc_mw", "dc_gw", "dc_gw_lag1", "d_dc_gw", "log_dc_gw_p1",
    "us_dc_gw", "us_dc_gw_lag1", "us_d_dc_gw", "share2016", "share2019",
    "emp_5182", "emp_517", "emp_5415", "emp_51", "emp_23", "emp_2362",
    "earn_5182", "earn_517", "earn_5415", "earn_51", "earn_23", "earn_2362",
    "empend_5182", "empend_517", "empend_5415", "empend_51", "empend_23",
    "empend_2362",
    "log_emp_5182", "log_emp_517", "log_emp_5415", "log_emp_51",
    "log_emp_23", "log_emp_2362",
    "local_share", "local_multiplier", "spillover_rate",
    "li_multiplier_local", "gdp_multiplier_local",
    "pwc_emp_direct_avg", "pwc_emp_spillover_avg",
    "pwc_emp_direct_2223", "pwc_emp_total_local_2223",
    "emp_5182_cleaned", "emp_5182_cleaned_mult", "emp_5182_raw_multiplied",
    "emp_5182_state_total",
    "log_emp_5182_cleaned", "log_emp_5182_cleaned_mult",
    "gdp_pc_2024", "subsidy_musd", "subsidy_observed",
    "us_gdp_price_index", "us_cpi", "us_rdpi_nominal",
    "quarters_available_min", "partial_year", "emp_available",
]
missing_cols = [c for c in COL_ORDER if c not in panel.columns]
assert not missing_cols, missing_cols
extra = [c for c in panel.columns if c not in COL_ORDER]
panel = panel[COL_ORDER + extra]

# ==================================================================
# 7. Sanity checks
# ==================================================================
log("\n=== SANITY CHECKS ===")
log(f"panel shape {panel.shape}; states {panel.state_abbr.nunique()}; "
    f"years {panel.year.min()}-{panel.year.max()}")
assert len(panel) == len(ALL_QWI_UNITS) * len(YEARS)
assert not panel.duplicated(["state_abbr", "year"]).any()
# the shipped panel is asserted against the universe module, in
# both directions, before it is written.  Every downstream estimator calls
# apply_scope() on this file; if a unit were missing here the estimators'
# check_universe() would fire, but it would fire three stages downstream with a
# far less useful message.
_units = set(panel["state_abbr"])
if _units != set(ALL_QWI_UNITS):
    raise UniverseError(
        "panel units != ANALYSIS_UNITS + OUT_OF_SCOPE_UNITS: "
        f"unexpected {sorted(_units - set(ALL_QWI_UNITS))}, "
        f"absent {sorted(set(ALL_QWI_UNITS) - _units)}")
_missing_analysis = sorted(set(ANALYSIS_UNITS) - _units)
if _missing_analysis:
    raise UniverseError(
        f"analysis units absent from the built panel: {_missing_analysis}. "
        "If 'DC' is in that list the cause is a name map that sends "
        "'District of Columbia'->'DC' while the source spells the unit 'DC'.")
log(f"[universe] panel carries all {len(ALL_QWI_UNITS)} source units = the "
    f"{N_ANALYSIS_UNITS} analysis units ({UNIVERSE_LABEL}) "
    f"+ {list(OUT_OF_SCOPE_UNITS)}, which apply_scope() removes downstream")
log(f"rows with employment observed: {panel.emp_available.sum()} "
    f"(expected {51*9} - {66//6} missing = {51*9-11})")
log(f"partial_year rows: {panel.partial_year.sum()} -> "
    f"{panel.loc[panel.partial_year==1,['state_abbr','year']].values.tolist()}")
log(f"division counts: {panel.drop_duplicates('state_abbr').census_division.value_counts().to_dict()}")
log(f"region counts: {panel.drop_duplicates('state_abbr').census_region.value_counts().to_dict()}")
log(f"share2019 sum = {panel.drop_duplicates('state_abbr').share2019.sum():.10f}")
log(f"share2016 sum = {panel.drop_duplicates('state_abbr').share2016.sum():.10f}")
log("\nshare2019 top-10:")
s19 = panel.drop_duplicates("state_abbr").set_index("state_abbr")["share2019"]
for s, v in s19.sort_values(ascending=False).head(10).items():
    log(f"   {s} {ABBR_TO_NAME[s]:<15s} {v:.5f}")
log(f"share2019 == 0 for {int((s19==0).sum())} states: "
    f"{sorted(s19.index[s19==0])}")
s16 = panel.drop_duplicates("state_abbr").set_index("state_abbr")["share2016"]
log("share2016 top-5: " + ", ".join(
    f"{s}:{v:.4f}" for s, v in s16.sort_values(ascending=False).head(5).items()))
log(f"share2016 == 0 for {int((s16==0).sum())} states")

log("\nmissingness by column (panel rows = %d):" % len(panel))
for c in panel.columns:
    n = panel[c].isna().sum()
    if n:
        log(f"   {c:28s} missing {n:4d}  "
            f"({panel.loc[panel[c].isna(),'year'].min()}-"
            f"{panel.loc[panel[c].isna(),'year'].max()})")

# cross-check against r05 published construction
r05_check = panel[(panel.emp_available == 1) & (panel.state_abbr != "DC")]
log(f"\n[r05 cross-check] r05 panel = qwi 5182 x dc inner-join, DC dropped "
    f"(its STATE_ABBR_MAP has 'District of Columbia' but QWI writes 'DC'), "
    f"years 2016-2024 -> {len(r05_check)} rows expected in r05; "
    f"r06 keeps DC as well ({int((panel.emp_available==1).sum())} rows).")

panel.to_csv(OUT / "panel_state_year.csv", index=False)
log(f"\nWROTE {relpath(OUT/'panel_state_year.csv')}  ({panel.shape[0]} rows x "
    f"{panel.shape[1]} cols)")

with open(OUT / "build_panel_log.txt", "w") as f:
    f.write("\n".join(log_lines) + "\n")
print(f"WROTE {relpath(OUT/'build_panel_log.txt')}")
