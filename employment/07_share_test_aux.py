"""Auxiliary recomputation for Supplementary Tables 21 and 22.

Re-runs the pre-trend and balance regressions of 06_share_exogeneity.py
in order to recover two quantities that the published result CSVs do not
record: the number of Census division fixed effects absorbed in each balance
regression, and the residual degrees of freedom of every regression.

Reads only.  Writes table_aux_pretrend.csv and table_aux_balance.csv next to
the existing result files.  Does not modify any pre-existing file.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

# Release-set path rule: no absolute paths.  Resolved from this file's location.
EMP = os.path.dirname(os.path.abspath(__file__))          # employment
HERE = os.path.join(EMP, "../results/r6_share_test")                # generated results
_spec = importlib.util.spec_from_file_location(
    "r06_share_exogeneity_state",
    os.path.join(EMP, "06_share_exogeneity.py"))
SX = importlib.util.module_from_spec(_spec)
sys.modules["r06_share_exogeneity_state"] = SX
_spec.loader.exec_module(SX)

def balance_aux(emp, shares, pre_end=2019):
    """Mirror of SX.run_balance, additionally recording n_div and df_resid."""
    sh = shares.set_index("state_abbr")
    rows = []
    for naics in SX.NAICS_ORDER:
        sub = emp[(emp.naics == naics) & emp.full_year & (emp.year <= pre_end)]
        piv = sub.pivot_table(index="state_abbr", columns="year",
                              values="Emp_annual_avg").sort_index(axis=1)
        if piv.empty:
            continue
        base_year = int(piv.columns.min())
        base_lvl = piv[base_year]
        pre_mean = piv.mean(axis=1)
        g = np.log(piv).diff(axis=1)
        outcomes = {
            f"base{base_year}_level": base_lvl,
            f"base{base_year}_log_level": np.log(base_lvl),
            "pre_mean_level": pre_mean,
            "pre_mean_log_level": np.log(pre_mean),
            "pre_growth_rate": g.mean(axis=1),
            "pre_volatility_sd": g.std(axis=1),
        }
        for name, yv in outcomes.items():
            yv = yv.replace([np.inf, -np.inf], np.nan).dropna()
            idx = [s for s in yv.index if s in sh.index]
            res, df = SX._ols(yv.loc[idx].values, sh.loc[idx, "share"].values,
                              sh.loc[idx, "division"].values)
            if res is None:
                continue
            rows.append({
                "outcome": f"emp_{naics}", "naics": naics, "variable": name,
                "beta": res.params["share"], "se": res.bse["share"],
                "pval": res.pvalues["share"], "n_obs": int(res.nobs),
                "n_div": int(df["div"].nunique()),
                "n_params": int(res.df_model) + 1,
                "df_resid": int(res.df_resid),
            })
    return pd.DataFrame(rows)

def pretrend_aux(emp, shares):
    """Residual degrees of freedom for the log pre-trend regressions."""
    sh = shares.set_index("state_abbr")
    rows = []
    for naics in SX.NAICS_ORDER:
        sub = emp[(emp.naics == naics) & emp.full_year]
        piv = sub.pivot_table(index="state_abbr", columns="year",
                              values="Emp_annual_avg")
        years = sorted(piv.columns)
        for i in range(len(years) - 1):
            y1, y2 = int(years[i]), int(years[i + 1])
            # PRETREND_START_YEAR must be applied here too, or this auxiliary
            # rebuild generates pairs the primary run no longer produces and the
            # cross-check against pretrend_results.csv loses rows.
            if (y2 - y1 != 1 or y2 > SX.STUDY_START_YEAR
                    or y1 < SX.PRETREND_START_YEAR):
                continue
            yv = (np.log(piv[y2]) - np.log(piv[y1])).replace(
                [np.inf, -np.inf], np.nan).dropna()
            idx = [s for s in yv.index if s in sh.index]
            res, df = SX._ols(yv.loc[idx].values, sh.loc[idx, "share"].values,
                              sh.loc[idx, "division"].values)
            if res is None:
                continue
            rows.append({
                "outcome": f"emp_{naics}", "year_pair": f"{y1}-{y2}",
                "beta": res.params["share"], "se": res.bse["share"],
                "pval": res.pvalues["share"], "n_obs": int(res.nobs),
                "n_div": int(df["div"].nunique()),
                "n_params": int(res.df_model) + 1,
                "df_resid": int(res.df_resid),
            })
    return pd.DataFrame(rows)

def main():
    emp = SX.load_employment()
    shares = SX.load_shares()
    bal = balance_aux(emp, shares)
    pre = pretrend_aux(emp, shares)
    bal.to_csv(os.path.join(HERE, "table_aux_balance.csv"), index=False)
    pre.to_csv(os.path.join(HERE, "table_aux_pretrend.csv"), index=False)

    ref_b = pd.read_csv(os.path.join(HERE, "balance_results.csv"))
    mb = bal.merge(ref_b, on=["outcome", "variable"], suffixes=("", "_ref"))
    assert len(mb) == len(bal), "balance merge lost rows"
    assert np.allclose(mb.beta, mb.beta_ref), "balance beta mismatch"
    assert np.allclose(mb.se, mb.se_ref), "balance se mismatch"
    assert (mb.n_obs == mb.n_obs_ref).all(), "balance n_obs mismatch"

    ref_p = pd.read_csv(os.path.join(HERE, "pretrend_results.csv"))
    ref_p = ref_p[(ref_p.form == "log") & (ref_p.is_pre_period)]
    mp = pre.merge(ref_p, on=["outcome", "year_pair"], suffixes=("", "_ref"))
    assert len(mp) == len(pre), "pretrend merge lost rows"
    assert np.allclose(mp.beta, mp.beta_ref), "pretrend beta mismatch"
    assert np.allclose(mp.se, mp.se_ref), "pretrend se mismatch"
    assert (mp.n_div == mp.n_div_ref).all(), "pretrend n_div mismatch"

    print("reproduction check: PASS")
    print("\nbalance n_obs / n_div / df_resid by variable:")
    print(bal.groupby("variable")[["n_obs", "n_div", "n_params",
                                   "df_resid"]].agg(["min", "max"]))
    print("\npretrend n_obs / n_div / df_resid by year pair:")
    print(pre.groupby("year_pair")[["n_obs", "n_div", "n_params",
                                    "df_resid"]].agg(["min", "max"]))

if __name__ == "__main__":
    main()
