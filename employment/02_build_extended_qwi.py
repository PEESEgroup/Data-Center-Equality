#!/usr/bin/env python3
"""
Assemble LEHD QWI R2026Q1 state-level quarterly extracts into the annual layout
used by employment/qwi_all_naics_annual.csv, and validate on the 2016-2024
overlap.
"""
import glob
import os

import numpy as np
import pandas as pd

# Release-set path rule: no absolute paths.  Resolved from this file's location.
EMP = os.path.dirname(os.path.abspath(__file__))          # employment/
SCRATCH = os.path.join(EMP, "_scratch")                   # written by step 01
OUT = os.path.join(EMP, "../results/r6_share_test")                 # generated results
EXISTING = os.path.join(EMP, "qwi_all_naics_annual.csv")  # shipped input

NAICS_DESC = {
    "5182": "Computing Infrastructure, Data Processing, Hosting",
    "51": "Information Sector",
    "517": "Telecommunications",
    "5415": "Computer Systems Design",
    "23": "Construction Sector",
    "2362": "Nonresidential Building Construction",
}

STATE_NAMES = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "DC", 12: "Florida",
    13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana",
    19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana", 23: "Maine",
    24: "Maryland", 25: "Massachusetts", 26: "Michigan", 27: "Minnesota",
    28: "Mississippi", 29: "Missouri", 30: "Montana", 31: "Nebraska",
    32: "Nevada", 33: "New Hampshire", 34: "New Jersey", 35: "New Mexico",
    36: "New York", 37: "North Carolina", 38: "North Dakota", 39: "Ohio",
    40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania", 44: "Rhode Island",
    45: "South Carolina", 46: "South Dakota", 47: "Tennessee", 48: "Texas",
    49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
}

def load_quarterly():
    frames = []
    for path in sorted(glob.glob(os.path.join(SCRATCH, "raw", "*.csv"))):
        d = pd.read_csv(path, dtype={"industry": str, "geography": str})
        frames.append(d)
    q = pd.concat(frames, ignore_index=True)
    q["state"] = q["geography"].astype(int)
    for c in ["Emp", "EmpEnd", "EarnBeg", "Payroll"]:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    return q

def annualise(q):
    ann = (q.groupby(["state", "industry", "year"])
             .agg(Emp_annual_avg=("Emp", "mean"),
                  EmpEnd_annual_avg=("EmpEnd", "mean"),
                  Payroll_annual_total=("Payroll", "sum"),
                  EarnBeg_annual_avg=("EarnBeg", "mean"),
                  quarters_available=("quarter", "count"))
             .reset_index())
    ann["state_name"] = ann["state"].map(STATE_NAMES)
    ann["naics"] = ann["industry"]
    ann["naics_desc"] = ann["naics"].map(NAICS_DESC)
    cols = ["state", "state_name", "naics", "naics_desc", "year",
            "Emp_annual_avg", "EmpEnd_annual_avg", "Payroll_annual_total",
            "EarnBeg_annual_avg", "quarters_available"]
    return ann[cols].sort_values(["state", "naics", "year"]).reset_index(drop=True)

def validate(ann):
    ex = pd.read_csv(EXISTING)
    ex["naics"] = ex["naics"].astype(str)
    mine = ann[(ann.year >= 2016) & (ann.year <= 2024)].copy()
    m = ex.merge(mine, on=["state", "naics", "year"], suffixes=("_ex", "_new"),
                 how="outer", indicator=True)
    report = {}
    report["n_existing"] = len(ex)
    report["n_mine_overlap"] = len(mine)
    report["only_existing"] = m[m._merge == "left_only"][["state", "naics", "year"]]
    report["only_mine"] = m[m._merge == "right_only"][["state", "naics", "year"]]
    b = m[m._merge == "both"].copy()
    rows = []
    for v in ["Emp_annual_avg", "EmpEnd_annual_avg", "EarnBeg_annual_avg",
              "quarters_available"]:
        a = pd.to_numeric(b[v + "_ex"], errors="coerce")
        c = pd.to_numeric(b[v + "_new"], errors="coerce")
        d = (c - a).abs()
        rel = d / a.abs().replace(0, np.nan)
        i = d.idxmax() if d.notna().any() else None
        rows.append({
            "variable": v, "n_compared": int(d.notna().sum()),
            "max_abs_diff": float(np.nanmax(d)) if d.notna().any() else np.nan,
            "max_rel_diff": float(np.nanmax(rel)) if rel.notna().any() else np.nan,
            "n_exact": int((d == 0).sum()),
            "worst_cell": (f"{b.loc[i,'state_name_ex']}/{b.loc[i,'naics']}/"
                           f"{int(b.loc[i,'year'])}" if i is not None else ""),
        })
    report["compare"] = pd.DataFrame(rows)
    report["n_both"] = len(b)
    return report

def main():
    os.makedirs(OUT, exist_ok=True)
    q = load_quarterly()
    print(f"quarterly rows: {len(q):,}  years {q.year.min()}-{q.year.max()}")
    ann = annualise(q)
    q.to_csv(os.path.join(SCRATCH, "qwi_quarterly_all_years.csv"), index=False)
    ann.to_csv(os.path.join(SCRATCH, "qwi_annual_all_years.csv"), index=False)

    rep = validate(ann)
    print("\n=== OVERLAP VALIDATION 2016-2024 ===")
    print(f"existing rows: {rep['n_existing']}, matched: {rep['n_both']}")
    print(f"only in existing: {len(rep['only_existing'])}, "
          f"only in mine: {len(rep['only_mine'])}")
    if len(rep["only_mine"]):
        print(rep["only_mine"].to_string())
    if len(rep["only_existing"]):
        print(rep["only_existing"].to_string())
    print(rep["compare"].to_string(index=False))

    ext = ann[(ann.year >= 2005) & (ann.year <= 2015)].copy()
    ext.to_csv(os.path.join(OUT, "qwi_extended_annual.csv"), index=False)
    print(f"\nWrote {len(ext)} rows -> {OUT}/qwi_extended_annual.csv")
    print(f"  years {ext.year.min()}-{ext.year.max()}, "
          f"{ext.state.nunique()} states, {ext.naics.nunique()} naics")

if __name__ == "__main__":
    main()
