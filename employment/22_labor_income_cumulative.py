#!/usr/bin/env python3
"""Cumulative incremental labour income, 2020-2024, against cumulative incentives.

The comparison is incremental labour income growth from 2020 to 2024 against
cumulative tax incentives over the same period, so the numerator must be a
five-year sum, not an end-year flow.  The end-year quantity,

    (emp_2024 - emp_2020) x lambda x earnings_2024 x li_multiplier,

is the labour income paid in a single year, 2024, by the jobs added over the
window; dividing it by a five-year sum of outlays compares quantities of
different dimensions.  This script computes the cumulative numerator, and
reproduces the end-year quantity first as a guard on the code path.

THE FORMULA, AND WHY EACH YEAR IS DIFFERENCED AGAINST 2020 RATHER THAN AGAINST
THE PREVIOUS YEAR.

    Delta LI = sum_{t=2021..2024} (emp_t - emp_2020) x lambda_s
                                   x EarnBeg_t x 12 x li_multiplier_s

Both QWI employment and PwC employment are annual STOCKS -- the count of jobs
existing in a year, not jobs added in it.  (Checked: the 2022-2023 growth rates
of the two series correlate at 0.905.)  So emp_t - emp_2020 is the number of
extra jobs in existence in year t, and multiplying by that year's annual
earnings gives the extra wages actually paid in year t.  Summing over t gives
the cumulative extra wages over the window, which is what a five-year outlay
should be compared with.

Differencing against the previous year instead would telescope: the sum of
(emp_t - emp_{t-1}) collapses to (emp_2024 - emp_2020), so each new job would be
counted only in its first year.  On these data that undercounts by a factor of
about three, giving 36,391 against the correct 123,232 million dollars.

x12 annualises: QWI EarnBeg is average MONTHLY earnings.  Cross-checked against
PwC's own labour income per direct job, which agrees to 1-2 percent for the
large states (California 256,358 against 258,446).

WHAT IS NOT CHANGED.  lambda_s and both multipliers stay at their 2022-2023
values, applied to every year.  PwC's direct employment moves by a median of
2.5 percent between those two years, so the approximation is small, but it is an
approximation and the Supplementary Information says so.
"""
import os
import sys

import numpy as np
import pandas as pd

EMP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EMP, "../results/r6_employment", "labor_income_cumulative.csv")
YEARS = (2021, 2022, 2023, 2024)
BASE = 2020
# End-year reference values, reproduced before anything new is reported.
ENDYEAR_REFERENCE = {"Virginia": 1780.223168, "California": 13667.45,
                     "Oklahoma": 38.63, "Texas": 3359.81161}
TOL = 0.01

def main():
    q = pd.read_csv(os.path.join(EMP, "qwi_all_naics_annual.csv"), dtype={"naics": str})
    mult = pd.read_csv(os.path.join(EMP, "pwc_multipliers.csv"))
    sub = pd.read_excel(os.path.join(EMP, "tax",
                                     "state_subsidy_by_year_million_wide.xlsx"))

    M = mult.emp_total_no_spillover_avg.sum() / mult.emp_direct_avg.sum()
    mult["spill_dc"] = mult.emp_spillover_avg / M
    mult["local_share"] = mult.emp_direct_avg / (mult.emp_direct_avg + mult.spill_dc)
    lam = dict(zip(mult.state, mult.local_share))
    limult = dict(zip(mult.state, mult.li_multiplier_local))

    sub["subsidy"] = sub[[2020, 2021, 2022, 2023, 2024]].sum(axis=1)
    subsidy = dict(zip(sub.State, sub.subsidy))

    e = q[q.naics == "5182"]
    rows = []
    for st in sub.State:
        g = e[e.state_name == st].set_index("year")
        if BASE not in g.index or 2024 not in g.index or st not in lam:
            continue
        b = g.loc[BASE, "Emp_annual_avg"]

        def li(year, emp_year):
            return ((g.loc[emp_year, "Emp_annual_avg"] - b) * lam[st]
                    * g.loc[year, "EarnBeg_annual_avg"] * 12 / 1e6 * limult[st])

        cum = sum(li(t, t) for t in YEARS if t in g.index)
        endyear = li(2024, 2024)
        rows.append(dict(state=st, subsidy_M=subsidy[st],
                         li_cumulative_M=cum, li_endyear_flow_M=endyear,
                         ratio_cumulative=cum / subsidy[st] if subsidy[st] else np.nan,
                         ratio_endyear=endyear / subsidy[st] if subsidy[st] else np.nan,
                         local_share=lam[st], li_multiplier=limult[st]))
    d = pd.DataFrame(rows)

    # RECONSTRUCTION GUARD.  The new quantity is only trustworthy if the same
    # code path reproduces the published one first.
    print("  reconstruction of the end-year quantity:")
    for st, want in ENDYEAR_REFERENCE.items():
        got = float(d[d.state == st].li_endyear_flow_M.iloc[0])
        rel = abs(got - want) / abs(want)
        print(f"    {st:12s} {got:12,.2f}   reference {want:12,.2f}   rel {rel:.5f}")
        if rel > TOL:
            raise SystemExit(
                f"{st}: cannot reproduce the published end-year value "
                f"({rel:.2%} off). The cumulative figures are not reported.")

    d.to_csv(OUT, index=False)
    print()
    print(d[["state", "subsidy_M", "li_endyear_flow_M", "li_cumulative_M",
             "ratio_endyear", "ratio_cumulative"]]
          .sort_values("subsidy_M", ascending=False).head(8).round(2).to_string(index=False))
    print()
    print(f"  states with subsidy data      : {len(d)}")
    print(f"  below break-even, end-year    : {(d.ratio_endyear < 1).sum()}"
          f"  ({', '.join(d[d.ratio_endyear < 1].state)})")
    print(f"  below break-even, cumulative  : {(d.ratio_cumulative < 1).sum()}"
          f"  ({', '.join(d[d.ratio_cumulative < 1].state)})")
    print(f"  aggregate ratio  end-year {d.li_endyear_flow_M.sum()/d.subsidy_M.sum():.2f}"
          f"   cumulative {d.li_cumulative_M.sum()/d.subsidy_M.sum():.2f}")
    print(f"\nwrote {os.path.relpath(OUT, os.path.dirname(EMP))}")

if __name__ == "__main__":
    main()
