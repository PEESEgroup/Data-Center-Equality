#!/usr/bin/env python3
"""Robustness of the Bartik IV across specifications, for EVERY sector.

WHY.  Supplementary Table tab:emp_iv_robust reports the twelve alternative
specifications for two outcomes only, NAICS 5182 raw and NAICS 517.  Every
other sector in tab:emp_iv_main -- construction, nonresidential building,
computer systems design, the Information parent -- appears with a primary
estimate and no robustness at all.  A readers who sees construction at 15,468
jobs per gigawatt in the main table and in Figure 5 will ask whether it
survives dropping Texas.  This answers that for all of them.

THE FIRST-STAGE SPLIT IS THE POINT.  Five of the twelve specifications have an
effective F below the 23.1 critical value, and they are exactly the ones that
drop influential units or move the share baseline -- unsurprising, since Texas
and Virginia carry most of the first-stage moment.  Pooling strong and weak
first stages into one range would let a weak-instrument estimate masquerade as
a robustness result, which matters here: the single specification in which
NAICS 517 turns significantly negative is drop_top3_share, whose effective F is
10.9.  The summary therefore reports the two groups separately and never quotes
a range that mixes them.
"""
import os
import sys

import numpy as np
import pandas as pd

EMP = os.path.dirname(os.path.abspath(__file__))
R06 = os.path.join(EMP, "../results/r6_bartik")
sys.path.insert(0, EMP)

F_CRIT = 23.1
# Dropped from the SI display table as a weak-instrument artefact (F_eff = 5.8; the
# control is nearly collinear with the instrument).  Excluded here too so the weak-
# group counts match what the SI actually shows.
EXCLUDE_TAGS = {"share_linear_trend"}
OUTCOMES = [
    ("emp_5182", "5182 (data processing, hosting), raw"),
    ("emp_5182_cleaned", "5182, cleaned direct"),
    ("emp_5182_cleaned_mult", "5182, cleaned $\\times$ local multiplier"),
    ("emp_5182_state_total", "5182, state total"),
    ("emp_517", "517 (telecommunications)"),
    ("emp_51", "51 (Information)"),
    ("emp_23", "23 (construction)"),
    ("emp_2362", "2362 (nonresidential building)"),
    ("emp_5415", "5415 (computer systems design)"),
]

def main():
    r = pd.read_csv(os.path.join(R06, "results_robustness__union.csv"))
    m = pd.read_csv(os.path.join(R06, "results_main__union.csv"))
    prim = m[(m.leave_out_variant == "union") & (m.specification == "level")
             & (m.estimator == "IV2SLS")].set_index("outcome")
    q = r[(r.estimator == "IV2SLS") & (r.specification == "level")
          & (~r.robustness_tag.isin(EXCLUDE_TAGS))]

    strong = sorted(q[q.fs_F_effective >= F_CRIT].robustness_tag.unique())
    weak = sorted(q[q.fs_F_effective < F_CRIT].robustness_tag.unique())
    print(f"strong first stage (F_eff >= {F_CRIT}): {len(strong)}  {strong}")
    print(f"weak   first stage (F_eff <  {F_CRIT}): {len(weak)}  {weak}")

    rows = []
    for oc, label in OUTCOMES:
        s = q[(q.outcome == oc) & (q.fs_F_effective >= F_CRIT)]
        w = q[(q.outcome == oc) & (q.fs_F_effective < F_CRIT)]
        if not len(s):
            raise SystemExit(f"no strong-first-stage rows for {oc}")
        p0 = float(prim.loc[oc, "coef"])
        rows.append(dict(
            outcome=oc, label=label, primary=p0,
            n_strong=len(s), lo=float(s.coef.min()), hi=float(s.coef.max()),
            sig05=int((s.p_value < 0.05).sum()),
            sig10=int((s.p_value < 0.10).sum()),
            flips=int((np.sign(s.coef) != np.sign(p0)).sum()),
            n_weak=len(w),
            weak_sig05=int((w.p_value < 0.05).sum()),
            F_min_strong=float(s.fs_F_effective.min()),
            F_max_strong=float(s.fs_F_effective.max()),
        ))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(EMP, "../results/r6_employment", "robustness_by_sector.csv"),
             index=False)
    pd.set_option("display.width", 220)
    print()
    print(d[["outcome", "primary", "lo", "hi", "sig05", "sig10", "flips",
             "weak_sig05"]].round(0).to_string(index=False))
    print("\nwrote robustness_by_sector.csv")

if __name__ == "__main__":
    main()
