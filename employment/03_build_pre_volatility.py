"""Build the pre-period employment-volatility variable used by the two new
robustness rows of Supplementary Table 22 (tab:emp_iv_robust).

Definition (one value per state):

    vol_pre_5182_s = SD over 2005-2015 of  d ln Emp_5182_{s,t}
                   = SD of  ln Emp_{s,t} - ln Emp_{s,t-1}

computed on consecutive year pairs only, from full-quarter annual averages, in
exact parallel with the `pre_volatility_sd` row of the balance test in
06_share_exogeneity.py (which uses pandas .std(), i.e. ddof = 1).

Input : r06_share_test/qwi_extended_annual.csv   (2005-2015, NAICS 5182)
Output: r06/pre_volatility_5182.csv              (state_abbr, vol_pre_5182, n_diff)

Reads only; writes one new file.  No pre-existing file is modified.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

# Release-set path rule: no absolute paths.  Resolved from this file's location.
EMP = os.path.dirname(os.path.abspath(__file__))          # employment
SHARE = os.path.join(EMP, "../results/r6_share_test")               # generated results
OUT_PATH = os.path.join(EMP, "../results/r6_bartik", "pre_volatility_5182.csv")

# The estimator helpers are reused from step 06 rather than reimplemented, so
# that the volatility variable is built by the same code as the balance test.
_spec = importlib.util.spec_from_file_location(
    "r06_share_exogeneity_state",
    os.path.join(EMP, "06_share_exogeneity.py"))
SX = importlib.util.module_from_spec(_spec)
sys.modules["r06_share_exogeneity_state"] = SX
_spec.loader.exec_module(SX)

YEAR_LO, YEAR_HI = 2005, 2015

def main():
    d = pd.read_csv(os.path.join(SHARE, "qwi_extended_annual.csv"))
    d["naics"] = d["naics"].astype(str)
    d = d[(d.naics == "5182")
          & (d.year >= YEAR_LO) & (d.year <= YEAR_HI)
          & (d.quarters_available == 4)
          & d.Emp_annual_avg.notna() & (d.Emp_annual_avg > 0)].copy()
    d["state_abbr"] = d["state"].map(SX.FIPS_TO_ABBR)
    assert d["state_abbr"].notna().all(), "unmapped FIPS code"

    piv = d.pivot_table(index="state_abbr", columns="year",
                        values="Emp_annual_avg").sort_index(axis=1)
    # reindex onto the full calendar so that a missing year breaks the
    # difference rather than joining two non-consecutive years
    piv = piv.reindex(columns=range(YEAR_LO, YEAR_HI + 1))
    g = np.log(piv).diff(axis=1)
    g = g.replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame({
        "vol_pre_5182": g.std(axis=1),          # ddof = 1
        "n_diff": g.notna().sum(axis=1).astype(int),
    }).reset_index()
    out = out.sort_values("state_abbr").reset_index(drop=True)

    assert out["vol_pre_5182"].notna().all(), "missing volatility for some state"
    assert (out["n_diff"] >= 2).all(), "fewer than 2 usable changes for some state"

    out.to_csv(OUT_PATH, index=False, float_format="%.10g")
    print(f"wrote {os.path.relpath(OUT_PATH, os.path.dirname(EMP))}  ({len(out)} states)")
    print(out.describe().to_string())
    short = out[out["n_diff"] < YEAR_HI - YEAR_LO]
    if len(short):
        print("\nstates with fewer than the full 10 annual changes:")
        print(short.to_string(index=False))
    print("\nhead:")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
