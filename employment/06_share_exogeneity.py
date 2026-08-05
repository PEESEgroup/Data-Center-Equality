#!/usr/bin/env python3
"""
Bartik IV Share Exogeneity Test — STATE level (employment outcomes)
===================================================================
State-level analogue of r02_share_exogeneity_test.py (zone level, wholesale
prices). Same regression form, same HC1 standard errors, same output
conventions; ISO fixed effects are replaced by the 9 Census division fixed
effects, and the outcome is the change in state annual employment in each of
six data-center-relevant NAICS industries.

    Pre-trend:  Delta Emp_{s,tau} = a_tau + delta_{div(s)} + beta_tau * Share_s + e
    Balance:    Outcome_s        = a     + delta_{div(s)} + beta      * Share_s + e

Share_s = MW_2019[s] / MW_2019[analysis universe], 49 units.  The analysis
universe is the contiguous United States: the 48 contiguous states plus the
District of Columbia.  Alaska and Hawaii are out of scope by definition, not by
data availability, because both sit on isolated grids that none of the seven
ISOs analysed in this paper reaches, the ISO transmission-planning portfolios
used for the cost attribution do not cover them, and the cross-state spillover
and regional cost-sharing mechanisms studied here cannot operate on an isolated
grid.  The share is renormalised over the analysis universe, so the denominator
is the sum over the 49 units and not the 'US' national row of the capacity file
(that row still includes Alaska and Hawaii and is therefore never used here).
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

# Release-set path rule: no absolute paths.  Resolved from this file's location.
EMP = os.path.dirname(os.path.abspath(__file__))          # employment/
BASE = os.path.dirname(EMP)                               # 
OUT = os.path.join(EMP, "../results/r6_share_test")                 # generated results

DC_FILE = os.path.join(EMP, "dc_facilities_by_state_year.csv")
EXISTING_QWI = os.path.join(EMP, "qwi_all_naics_annual.csv")
EXTENDED_QWI = os.path.join(OUT, "qwi_extended_annual.csv")

SHARE_BASE_YEAR = 2019
STUDY_START_YEAR = 2019          # last PRE year; pairs ending <= 2019 are pre

# First pre-period year used by the pre-trend test.  When the estimation window opened
# in 2016 the panel left only three pre-period pairs, so the series was extended back
# to 2005 to buy power.  The window now opens at the 2019 share baseline, so 2015-2019
# supplies four pairs -- the same number the wholesale price analysis of Section 2 uses
# (2015-2016 through 2018-2019) -- and the deep extension is no longer needed.  The
# 2015 level itself still comes from the LEHD extension, because QWI starts in 2016.
PRETREND_START_YEAR = 2015
NAICS_ORDER = ["5182", "517", "23", "2362", "5415", "51"]

# ---------------------------------------------------------------------------
# Analysis scope: the contiguous United States (48 states + DC), 49 units.
# Alaska and Hawaii are excluded by scope, not by a data-availability rule.
# the two names below were local literals here as well -- a fifth
# independent declaration.  They are now imported from analysis_universe.py.
# ---------------------------------------------------------------------------
sys.path.insert(0, EMP)
from analysis_universe import (ANALYSIS_UNITS, ALL_QWI_UNITS,  # noqa: E402
                               N_ANALYSIS_UNITS, OUT_OF_SCOPE_UNITS,
                               check_universe)

MIN_DIVISION_SIZE = 3

DIVISION = {
    "CT": 1, "ME": 1, "MA": 1, "NH": 1, "RI": 1, "VT": 1,
    "NJ": 2, "NY": 2, "PA": 2,
    "IL": 3, "IN": 3, "MI": 3, "OH": 3, "WI": 3,
    "IA": 4, "KS": 4, "MN": 4, "MO": 4, "NE": 4, "ND": 4, "SD": 4,
    "DE": 5, "DC": 5, "FL": 5, "GA": 5, "MD": 5, "NC": 5, "SC": 5, "VA": 5, "WV": 5,
    "AL": 6, "KY": 6, "MS": 6, "TN": 6,
    "AR": 7, "LA": 7, "OK": 7, "TX": 7,
    "AZ": 8, "CO": 8, "ID": 8, "MT": 8, "NV": 8, "NM": 8, "UT": 8, "WY": 8,
    "AK": 9, "CA": 9, "HI": 9, "OR": 9, "WA": 9,
}
DIVISION_NAME = {
    1: "New England", 2: "Middle Atlantic", 3: "East North Central",
    4: "West North Central", 5: "South Atlantic", 6: "East South Central",
    7: "West South Central", 8: "Mountain", 9: "Pacific",
}

FIPS_TO_ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}

# ---------------------------------------------------------------- shares -----
def load_shares():
    dc = pd.read_csv(DC_FILE)
    nat = dc[dc.state_abbr == "US"]
    assert len(nat) == 1, "expected exactly one 'US' national row"
    us_row = float(nat["MW_2019"].iloc[0])

    st = dc[dc.state_abbr != "US"].copy()          # <- 'US' is NEVER a unit
    assert set(st["state_abbr"]) == set(ALL_QWI_UNITS), (
        "source capacity file units != ANALYSIS_UNITS + OUT_OF_SCOPE_UNITS "
        f"({len(st)} rows)")
    resid = abs(st["MW_2019"].sum() - us_row)
    assert resid < 1e-6, f"state sum != US row ({resid})"

    # Scope restriction, applied before the share is formed.
    dropped = st[st.state_abbr.isin(OUT_OF_SCOPE_UNITS)]
    st = st[~st.state_abbr.isin(OUT_OF_SCOPE_UNITS)].copy()
    check_universe(st, "state_abbr", where="share-exogeneity exposure shares")
    assert len(st) == N_ANALYSIS_UNITS, \
        f"expected {N_ANALYSIS_UNITS} in-scope units, got {len(st)}"

    # Renormalise over the analysis universe so the 49 shares sum to exactly one.
    denom = float(st["MW_2019"].sum())
    st["share"] = st["MW_2019"] / denom
    st["division"] = st["state_abbr"].map(DIVISION)
    assert st["division"].notna().all()

    print("=== ANALYSIS SCOPE: contiguous United States (48 states + DC) ===")
    for _, r in dropped.iterrows():
        print(f"  excluded by scope: {r.state_abbr} ({r.MW_2019:.2f} MW in 2019)")
    print(f"  2019 capacity denominator: {us_row:.3f} -> {denom:.3f} MW "
          f"({100*(us_row-denom)/us_row:.4f}% removed)")
    print(f"  units: 51 -> {len(st)}")

    # Leave-one-division-out is the primary design, so no division may become a
    # singleton.  Pacific falls from five units to three (CA, OR, WA).
    counts = st.groupby("division").size()
    print("  division sizes after the restriction:")
    for d, n in counts.sort_index().items():
        print(f"    {DIVISION_NAME[d]:<20s} {n}")
    assert len(counts) == 9, f"expected 9 divisions, got {len(counts)}"
    assert counts.min() >= MIN_DIVISION_SIZE, \
        f"division reduced below {MIN_DIVISION_SIZE} units: {counts.to_dict()}"

    return st[["state_abbr", "state_name", "MW_2019", "share", "division"]]

def validate_shares(sh):
    # Reference shares renormalised over the 49-unit analysis universe.
    known = {"TX": 0.138952, "VA": 0.133208, "CA": 0.089361, "IL": 0.057256,
             "NC": 0.055833, "GA": 0.053735, "WA": 0.049288, "IA": 0.046002}
    s = sh.set_index("state_abbr")["share"]
    lines = []
    ok = True
    assert not (set(OUT_OF_SCOPE_UNITS) & set(s.index)), "out-of-scope unit present"
    for k, v in known.items():
        got = float(s[k])
        good = abs(got - v) < 5e-5
        ok &= good
        lines.append(f"  {k}: {got:.6f} (expected {v:.6f}) {'OK' if good else 'MISMATCH'}")
    top = s.sort_values(ascending=False)
    for label, val, exp in [("top-3", top.head(3).sum(), 0.3615),
                            ("top-5", top.head(5).sum(), 0.4746),
                            ("sum-49", s.sum(), 1.000)]:
        good = abs(val - exp) < 1e-3
        ok &= good
        lines.append(f"  {label}: {val:.6f} (expected {exp:.4f}) {'OK' if good else 'MISMATCH'}")
    return ok, "\n".join(lines)

# ------------------------------------------------------------- employment ----
def load_employment():
    ex = pd.read_csv(EXISTING_QWI)
    ex["naics"] = ex["naics"].astype(str)
    ex["source"] = "existing(2016-2024)"
    new = pd.read_csv(EXTENDED_QWI)
    new["naics"] = new["naics"].astype(str)
    new["source"] = "extended(2005-2015)"
    d = pd.concat([new, ex], ignore_index=True)
    d["state_abbr"] = d["state"].map(FIPS_TO_ABBR)
    # Scope restriction: the analysis universe is the contiguous US (48 + DC).
    d = d[~d["state_abbr"].isin(OUT_OF_SCOPE_UNITS)]
    d["division"] = d["state_abbr"].map(DIVISION)
    d = d[d["Emp_annual_avg"].notna() & (d["Emp_annual_avg"] > 0)]
    # Only use years where all four quarters are present, so that the annual
    # mean is comparable across states and years.
    d["full_year"] = d["quarters_available"] == 4
    return d

# ------------------------------------------------------------ regressions ----
def _ols(y, share, division):
    df = pd.DataFrame({"y": y, "share": share, "div": division}).dropna()
    if len(df) < 8 or df["div"].nunique() < 2:
        return None, df
    dummies = pd.get_dummies(df["div"].astype(int), prefix="div",
                             drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                   df[["share"]], dummies], axis=1)
    return sm.OLS(df["y"], X).fit(cov_type="HC1"), df

def run_pretrend(emp, shares):
    rows = []
    sh = shares.set_index("state_abbr")
    for naics in NAICS_ORDER:
        sub = emp[(emp.naics == naics) & emp.full_year]
        piv = sub.pivot_table(index="state_abbr", columns="year",
                              values="Emp_annual_avg")
        years = sorted(piv.columns)
        for i in range(len(years) - 1):
            y1, y2 = int(years[i]), int(years[i + 1])
            if y2 - y1 != 1:
                continue
            if y2 > STUDY_START_YEAR or y1 < PRETREND_START_YEAR:
                continue                       # pre-period pairs only
            lev = piv[y2] - piv[y1]
            log = np.log(piv[y2]) - np.log(piv[y1])
            for form, yv in [("level", lev), ("log", log)]:
                yv = yv.replace([np.inf, -np.inf], np.nan).dropna()
                idx = [s for s in yv.index if s in sh.index]
                res, df = _ols(yv.loc[idx].values,
                               sh.loc[idx, "share"].values,
                               sh.loc[idx, "division"].values)
                if res is None:
                    continue
                ci = res.conf_int(alpha=0.05).loc["share"]
                rows.append({
                    "test": "pretrend", "outcome": f"emp_{naics}",
                    "naics": naics, "form": form,
                    "year_pair": f"{y1}-{y2}", "year_start": y1, "year_end": y2,
                    "beta": res.params["share"], "se": res.bse["share"],
                    "t_stat": res.tvalues["share"], "pval": res.pvalues["share"],
                    "ci_lower": ci.iloc[0], "ci_upper": ci.iloc[1],
                    "n_obs": int(res.nobs), "n_div": int(df["div"].nunique()),
                    "r_squared": res.rsquared,
                    "mean_dep": float(np.mean(df["y"])),
                    "is_pre_period": True,
                })
    return pd.DataFrame(rows)

def run_balance(emp, shares, pre_end=2019):
    sh = shares.set_index("state_abbr")
    rows = []
    for naics in NAICS_ORDER:
        sub = emp[(emp.naics == naics) & emp.full_year & (emp.year <= pre_end)]
        piv = sub.pivot_table(index="state_abbr", columns="year",
                              values="Emp_annual_avg").sort_index(axis=1)
        if piv.empty:
            continue
        base_year = int(piv.columns.min())
        base_lvl = piv[base_year]
        pre_mean = piv.mean(axis=1)
        g = np.log(piv).diff(axis=1)                    # annual log growth
        pre_growth = g.mean(axis=1)
        pre_vol = g.std(axis=1)
        n_years = piv.notna().sum(axis=1)

        outcomes = {
            f"base{base_year}_level": base_lvl,
            f"base{base_year}_log_level": np.log(base_lvl),
            "pre_mean_level": pre_mean,
            "pre_mean_log_level": np.log(pre_mean),
            "pre_growth_rate": pre_growth,
            "pre_volatility_sd": pre_vol,
            "pre_n_years": n_years.astype(float),
        }
        for name, yv in outcomes.items():
            if name == "pre_n_years":
                continue
            yv = yv.replace([np.inf, -np.inf], np.nan).dropna()
            idx = [s for s in yv.index if s in sh.index]
            res, df = _ols(yv.loc[idx].values, sh.loc[idx, "share"].values,
                           sh.loc[idx, "division"].values)
            if res is None:
                continue
            ci = res.conf_int(alpha=0.05).loc["share"]
            rows.append({
                "test": "balance", "outcome": f"emp_{naics}", "naics": naics,
                "variable": name, "base_year": base_year,
                "beta": res.params["share"], "se": res.bse["share"],
                "t_stat": res.tvalues["share"], "pval": res.pvalues["share"],
                "ci_lower": ci.iloc[0], "ci_upper": ci.iloc[1],
                "n_obs": int(res.nobs), "r_squared": res.rsquared,
                "mean_dep": float(np.mean(df["y"])),
            })
    return pd.DataFrame(rows)

def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

def plot_event_study(pre, form, out_path):
    """gamma_tau with 95% CI, one panel per outcome (mirrors the zone-level fig)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pre[pre.form == form]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, naics in zip(axes.ravel(), NAICS_ORDER):
        q = d[d.naics == naics].sort_values("year_start").reset_index(drop=True)
        x = np.arange(len(q))
        # Plain numpy throughout: a boolean pandas Series cannot index the
        # second axis of an array, and pandas raises rather than aligning.
        beta = q.beta.to_numpy(dtype=float)
        lo = q.ci_lower.to_numpy(dtype=float)
        hi = q.ci_upper.to_numpy(dtype=float)
        yerr = np.vstack([beta - lo, hi - beta])
        sig = (q.pval < 0.05).to_numpy(dtype=bool)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.errorbar(x[~sig], beta[~sig], yerr=yerr[:, ~sig], fmt="o",
                    color="steelblue", capsize=3, markersize=5, elinewidth=1.2)
        if sig.any():
            ax.errorbar(x[sig], beta[sig], yerr=yerr[:, sig], fmt="s",
                        color="firebrick", capsize=3, markersize=6, elinewidth=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(q.year_pair, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"emp_{naics}", fontsize=10)
        ax.set_ylabel(r"$\beta_\tau$", fontsize=9)
    fig.suptitle(f"State-level Bartik share pre-trend test ({form} form, "
                 f"Census division FE, HC1). Red = p<0.05", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs(OUT, exist_ok=True)
    shares = load_shares()
    ok, txt = validate_shares(shares)
    print("=== SHARE VALIDATION ===")
    print(txt)
    if not ok:
        raise SystemExit("Share validation FAILED — aborting.")
    shares.to_csv(os.path.join(OUT, "bartik_shares_2019.csv"), index=False)

    emp = load_employment()
    print(f"\nEmployment panel: {len(emp):,} state-naics-year rows, "
          f"{emp.year.min()}-{emp.year.max()}, {emp.state_abbr.nunique()} states")

    pre = run_pretrend(emp, shares)
    bal = run_balance(emp, shares)
    pre.to_csv(os.path.join(OUT, "pretrend_results.csv"), index=False)
    bal.to_csv(os.path.join(OUT, "balance_results.csv"), index=False)

    for form in ["level", "log"]:
        print(f"\n=== PRE-TREND ({form}) ===")
        p = pre[pre.form == form]
        for naics in NAICS_ORDER:
            q = p[p.naics == naics]
            if q.empty:
                continue
            nsig = (q.pval < 0.05).sum()
            print(f"  emp_{naics}: {nsig}/{len(q)} year-pairs p<0.05")
    print("\n=== BALANCE ===")
    print(f"  {(bal.pval < 0.05).sum()}/{len(bal)} p<0.05")

    for form in ["log", "level"]:
        plot_event_study(pre, form,
                         os.path.join(OUT, f"pretrend_event_study_{form}.png"))
    print(f"Wrote event-study plots to {OUT}")
    print(f"\nWrote {OUT}/pretrend_results.csv, balance_results.csv")

if __name__ == "__main__":
    main()
