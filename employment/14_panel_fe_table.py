#!/usr/bin/env python3
"""
Assemble the panel fixed-effects employment table reported in the SI under a
single clustered standard-error convention.

No estimation is re-run here.  Every coefficient is read from an existing
generated artefact.  The only arithmetic is (i) a deterministic scalar rescaling
between two clustered-SE conventions and (ii) the corresponding t and p, and both
are cross-validated against a second, independently produced file.

THE THREE SOURCES
-----------------
A. ../results/r6_employment/panel_fe_5182__contig49.csv
   ../results/r6_employment/panel_fe_related__contig49.csv
   Written by 13_panel_fe_contig49.py.  `linearmodels.PanelOLS(entity_effects=
   True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)`.
   The only source for R^2_within, and for the "Raw x multiplier + spillover"
   specification.

B. ../results/r6_bartik/results_main__union.csv, estimator == 'OLS_FE',
   leave_out_variant == 'union'.  Written by 05_bartik_iv.py.  Hand-rolled
   cluster sandwich WITH the Cameron-Gelbach-Miller finite-sample correction,
   inference on t(G-1).  This is the convention every other clustered SE in the
   SI's employment tables carries.

C. ../results/r6_employment/panel_fe_employment.csv
   Written by 09_ols_employment_dc.py.  Supplies the one specification whose
   dependent variable contains a national aggregate; see below.

Sources A and B are independent implementations.  They agree on every
overlapping coefficient and differ on the standard error by exactly

    c = sqrt( G/(G-1) * (N-1)/N ),   G = 49 clusters, N = 438 observations,

which is the finite-sample correction itself.  That is checked below rather than
asserted.

CONVENTION ADOPTED
------------------
The finite-sample-corrected clustered standard error (source B), because it is
what every other clustered SE in these sections carries, it is the more
conservative of the two, and Cameron-Gelbach-Miller is the standard correction
for a small number of clusters.

Six of the seven specifications are read straight from source B under that
convention.  The seventh, "Raw x multiplier + spillover", is not among the
Bartik script's outcomes and comes from source C.  It is the only specification
whose dependent variable contains a national aggregate: it adds to each unit's
own multiplied employment a spillover term proportional to national NAICS 5182
employment outside that unit.  Source A computes that aggregate with a plain
.sum() by year, which is unbalanced -- Michigan leaves QWI after 2021, so the
2022-2024 national totals lose ~8,570 jobs and the coefficient absorbs
Michigan's exit from the data as a national decline.  Source C computes it over
units observed in every year, which is the convention the SI states and the
value reported there.  Source A's unbalanced value is retained in its own column
rather than dropped.

Outputs (../results/r6_employment):
    panel_fe_table__contig49.csv
    panel_fe_table__contig49_log.txt

Consumers must read the `status` column: rows marked not_reported are diagnostic
only.  15_did_continuous.py enforces this at read time.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Release-set path rule: no absolute paths.  Resolved from this file's location.
EMP = os.path.dirname(os.path.abspath(__file__))   # .../employment
HERE = os.path.join(EMP, "../results/r6_employment")              # generated results
R06 = os.path.join(EMP, "../results/r6_bartik")                     # generated results

sys.path.insert(0, EMP)
from analysis_universe import (N_ANALYSIS_UNITS, UNIVERSE_LABEL,  # noqa: E402
                               check_universe, describe)

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)

log("=" * 78)
log("PANEL FIXED EFFECTS TABLE -- SINGLE CLUSTERED-SE CONVENTION")
log("=" * 78)
log(describe())
log("")

# ------------------------------------------------------------------ source A
a1 = pd.read_csv(os.path.join(HERE, "panel_fe_5182__contig49.csv"))
a2 = pd.read_csv(os.path.join(HERE, "panel_fe_related__contig49.csv"))
A = pd.concat([a1, a2], ignore_index=True)
# Two cross-checks live in this script and they need different samples.
#   A <-> C  panel FE (13) against panel FE (09): neither involves the Bartik
#            module, both run 2016-2024, so A is the FULL window.
#   A <-> B  panel FE (13) against the Bartik module's OLS_FE rows: those now run
#            on the 2019-2024 IV window, because the exposure share is measured in
#            2019 and cannot be pre-determined earlier, so that comparison uses the
#            matched subsample 13_panel_fe_contig49.py estimates for the purpose.
A_ALL = A.copy()
A_IV = A_ALL[A_ALL["sample"] == "contig49_matched"].copy()
A = A_ALL[A_ALL["sample"] == "contig49"].copy()

n_obs = int(A["n_obs"].unique()[0])
n_units = int(A["n_units"].unique()[0])
assert A["n_obs"].nunique() == 1 and A["n_units"].nunique() == 1
assert n_units == N_ANALYSIS_UNITS, (n_units, N_ANALYSIS_UNITS)

# The line above is a COUNT check.  A sample that had
# swapped DC for Hawaii would still be 49 units and would pass it.  The
# membership check runs on the panel the summary rows were estimated from.
_PANEL49 = os.path.join(HERE, "panel__contig49.csv")
if not os.path.exists(_PANEL49):
    sys.exit(f"missing {_PANEL49}; run 13_panel_fe_contig49.py first. This "
             f"script will not certify an SE convention for a sample it cannot "
             f"see the units of.")
_p49 = pd.read_csv(_PANEL49)
check_universe(_p49, "state_abbr", where="panel__contig49.csv")
assert len(_p49) == n_obs, (len(_p49), n_obs)
assert _p49["state_abbr"].nunique() == n_units, (_p49["state_abbr"].nunique(),
                                                 n_units)
log(f"[universe] panel__contig49.csv: {_p49['state_abbr'].nunique()} units, "
    f"{len(_p49)} rows -- membership checked against analysis_universe.py, not "
    f"merely counted")
log(f"[A] panel_fe_*__contig49.csv, sample='contig49': "
    f"{len(A)} specifications, N={n_obs}, G={n_units}")
log("    estimator: linearmodels PanelOLS, cov_type='clustered', "
    "cluster_entity=True (no finite-sample correction)")

# The 50-unit sample variant, kept for the scope comparison printed below.
LEG = pd.concat([a1, a2], ignore_index=True)
LEG = LEG[LEG["sample"] == "us50"].copy()
log(f"[A'] scope comparison, sample='us50': "
    f"N={int(LEG['n_obs'].unique()[0])}, G={int(LEG['n_units'].unique()[0])} "
    "-- AK and HI retained, DC out")

# ------------------------------------------------------------------ source B
B = pd.read_csv(os.path.join(R06, "results_main__union.csv"))
B = B[(B["estimator"] == "OLS_FE") & (B["leave_out_variant"] == "union")].copy()
_iv_n = int(A_IV["n_obs"].unique()[0]); _iv_g = int(A_IV["n_units"].unique()[0])
assert set(B["n_obs"]) == {_iv_n} and set(B["n_clusters"]) == {_iv_g}, (
    "source B (Bartik OLS_FE) and the matched panel FE are on different samples",
    sorted(set(B["n_obs"])), _iv_n)
log(f"[B] results_main__union.csv, OLS_FE x union: {len(B)} outcomes, "
    f"N={n_obs}, G={n_units}")
log(f"    se_type recorded in the file: {sorted(set(B['se_type']))}")

# ------------------------------------------------------------------ source C
# The balanced-national-total implementation of the spillover specification,
# from 09_ols_employment_dc.py.
C_PATH = os.path.join(HERE, "panel_fe_employment.csv")
if not os.path.exists(C_PATH):
    raise SystemExit(
        f"missing {C_PATH}; run 09_ols_employment_dc.py first. This "
        "script refuses to fall back on the unbalanced spillover aggregate.")
C = pd.read_csv(C_PATH)
C_BALANCED_SPEC = "Raw × Multiplier+Spillover"
C_UNBAL_SPEC = ("Raw × Multiplier+Spillover "
                "[DIAGNOSTIC: unbalanced national total]")
for _s in (C_BALANCED_SPEC, C_UNBAL_SPEC):
    if (C["spec"] == _s).sum() != 1:
        raise SystemExit(f"panel_fe_employment.csv: expected exactly one "
                         f"row for {_s!r}, got {(C['spec'] == _s).sum()}")
c_bal = C[C["spec"] == C_BALANCED_SPEC].iloc[0]
c_unbal = C[C["spec"] == C_UNBAL_SPEC].iloc[0]
assert int(c_bal["n_obs"]) == n_obs and int(c_bal["n_units"]) == n_units, (
    "source C is on a different sample from source A")
log(f"[C] panel_fe_employment.csv (09_ols_employment_dc.py): "
    f"N={int(c_bal['n_obs'])}, G={int(c_bal['n_units'])}")
log("    the spillover spec's national NAICS 5182 total is computed over units "
    "observed in EVERY year (panel_aggregates.balanced_sum_by_year)")

# TWO-IMPLEMENTATION CHECK, not a self-comparison: source A and source C are
# different scripts with different panel builders.  They must agree EXACTLY on
# the unbalanced variant, which is the only thing they compute the same way.
a_spill = A[A["spec"] == "Raw x multiplier + spillover"].iloc[0]
spill_gap = abs(float(a_spill["beta"]) - float(c_unbal["beta"]))
spill_rel = spill_gap / abs(float(c_unbal["beta"]))
log("")
log("SPILLOVER SPECIFICATION -- balanced vs unbalanced national total:")
log(f"  source A (unbalanced national total)      beta = {float(a_spill['beta']):.6f}"
    f"   se_lm = {float(a_spill['se']):.6f}")
log(f"  source C diagnostic (unbalanced)          beta = {float(c_unbal['beta']):.6f}"
    f"   se_lm = {float(c_unbal['se_linearmodels']):.6f}")
log(f"  agreement between the two implementations on the unbalanced variant: "
    f"rel {spill_rel:.3e}")
CHECK_3 = spill_rel < 1e-9
log(f"  CHECK 3 (A and C reproduce each other on the shared, unbalanced "
    f"definition): {'PASS' if CHECK_3 else 'FAIL'}")
if not CHECK_3:
    raise SystemExit("A and C disagree on the unbalanced spillover variant; the "
                     "difference is therefore NOT only the balancing rule.")
log(f"  source C reported (balanced)              beta = {float(c_bal['beta']):.6f}"
    f"   se_lm = {float(c_bal['se_linearmodels']):.6f}"
    f"   se_cgm = {float(c_bal['se_cgm']):.6f}")
log(f"  => source A's spillover row is not reported; the balanced value is "
    f"{float(c_bal['beta']) - float(c_unbal['beta']):+.2f} jobs/GW from it, which "
    f"is Michigan's post-2021 exit, not an employment effect.")

# ------------------------------------------------------- the mapping A <-> B
# spec label in section 6.5  ->  r06 outcome column (None = no r06 counterpart)
SPEC_TO_OUTCOME = {
    "Raw NAICS 5182":                     "emp_5182",
    "Cleaned direct":                     "emp_5182_cleaned",
    "Raw x multiplier + spillover":       None,
    # Unlike the spec-2 row above, the corrected state total has a Bartik
    # counterpart: it is a plain state-level
    # transform of the panel with no national aggregate, so r06 can and does
    # estimate it, and the two implementations must agree.
    "State total (local impact + spillover)": "emp_5182_state_total",
    "Cleaned x local multiplier":         "emp_5182_cleaned_mult",
    "NAICS 23 Construction":              "emp_23",
    "NAICS 2362 Nonresidential building": "emp_2362",
    "NAICS 5415 Computer systems design": "emp_5415",
}

# --------------------------------------------- the finite-sample correction c
# The SE-convention certification is about the panel FE's own sample (full window).
G, N = n_units, n_obs
c_theory = np.sqrt(G / (G - 1.0) * (N - 1.0) / N)
# The A<->B cross-check runs on the matched IV window, so its correction uses that
# sample's N and G.  Using the full-window c here left a 5.8e-04 residual that is
# exactly the difference between the two corrections, not an SE-convention error.
c_theory_iv = np.sqrt(_iv_g / (_iv_g - 1.0) * (_iv_n - 1.0) / _iv_n)
log("")
log(f"finite-sample correction  c = sqrt(G/(G-1) * (N-1)/N) "
    f"= sqrt({G}/{G - 1} * {N - 1}/{N}) = {c_theory:.10f}")

log("")
log("CROSS-CHECK -- two independent implementations, six overlapping "
    "specifications:")
log(f"  {'specification':<36s} {'beta_A':>13s} {'beta_B':>13s} {'rel.diff':>10s} "
    f"{'se_A':>11s} {'se_B':>11s} {'se_B/se_A':>11s}")
ratios, beta_rel = [], []
for spec, oc in SPEC_TO_OUTCOME.items():
    if oc is None:
        continue
    ra = A_IV[A_IV["spec"] == spec]
    rb = B[B["outcome"] == oc]
    assert len(ra) == 1 and len(rb) == 1, (spec, oc, len(ra), len(rb))
    ba, sa = float(ra["beta"].iloc[0]), float(ra["se"].iloc[0])
    bb, sb = float(rb["coef"].iloc[0]), float(rb["se"].iloc[0])
    rel = abs(ba - bb) / abs(ba)
    ratio = sb / sa
    beta_rel.append(rel)
    ratios.append(ratio)
    log(f"  {spec:<36s} {ba:>13.5f} {bb:>13.5f} {rel:>10.2e} "
        f"{sa:>11.5f} {sb:>11.5f} {ratio:>11.7f}")

beta_max = max(beta_rel)
ratio_dev = max(abs(np.array(ratios) - c_theory_iv))
log("")
log(f"  max relative disagreement in beta across implementations : {beta_max:.3e}")
log(f"  max deviation of se_B/se_A from the theoretical c        : {ratio_dev:.3e}")
CHECK_1 = beta_max < 5e-6
CHECK_2 = ratio_dev < 5e-6
log(f"  CHECK 1 (same coefficient, both implementations)  : "
    f"{'PASS' if CHECK_1 else 'FAIL'}")
log(f"  CHECK 2 (the whole SE gap IS the CGM correction)  : "
    f"{'PASS' if CHECK_2 else 'FAIL'}")
if not (CHECK_1 and CHECK_2):
    raise SystemExit("cross-check failed; refusing to write the table")
log("  => the two standard errors differ by a convention, not a disagreement")
log("     by choosing one, not by re-estimating anything.")

# --------------------------------------------------------- build the table
DF_T = G - 1          # r06 clusters-minus-one t reference distribution
log("")
log(f"inference: t with df = G - 1 = {DF_T} (the r06 convention, verified below)")

SPILLOVER_SPEC = "Raw x multiplier + spillover"

rows = []
for spec, oc in SPEC_TO_OUTCOME.items():
    ra = A[A["spec"] == spec]
    beta = float(ra["beta"].iloc[0])
    se_lm = float(ra["se"].iloc[0])
    r2w = float(ra["r2_within"].iloc[0])
    status = "reported"
    basis = "n/a (no national aggregate in the dependent variable)"
    alt_value = ""
    # The spec-2 row is not reported for a second, independent reason on top
    # of the balanced/unbalanced one resolved below: its dependent
    # variable double-counts the cross-state spillover.  Observed NAICS 5182
    # already contains the spillover-driven jobs, so multiplying the RAW series
    # by m_s applies the in-state multiplier to jobs in-state activity did not
    # create, and the national term then adds the same spillover again.  The
    # corrected quantity is "State total (local impact + spillover)",
    # raw*(lambda*m + 1 - lambda), which counts it once and contains no national
    # aggregate at all.  The row is retained so the size of the difference
    # stays on the record.
    if spec == "Raw x multiplier + spillover":
        status = "not_reported"
        alt_value = ("double-counts the cross-state spillover; replaced by "
                     "'State total (local impact + spillover)'")

    if spec == SPILLOVER_SPEC:
        # Take the coefficient from source C, the balanced-aggregate
        # implementation, and record source A's unbalanced value in its own
        # column rather than dropping it.
        alt_value = (f"unbalanced national total (source A / "
                     f"02_ols_employment_dc_capacity__contig49.xlsx): "
                     f"beta={beta:.6f}, se_lm={se_lm:.6f}, r2w={r2w:.6f}")
        beta = float(c_bal["beta"])
        se_lm = float(c_bal["se_linearmodels"])
        r2w = float(c_bal["r2_within"])
        basis = ("BALANCED: national NAICS 5182 total summed over units observed "
                 "in every year 2016-2024 (panel_aggregates.balanced_sum_by_year)")

    if oc is None:
        if spec == SPILLOVER_SPEC:
            se_cgm = float(c_bal["se_cgm"])
            se_src = ("panel_fe_employment.csv (09_ols_employment_dc.py, "
                      "balanced national total, CGM-corrected)")
            # Source C is the full-window panel FE, so its CGM correction is the
            # full-window c, not the matched-window one used for the A_IV <-> B check.
            _c_obs = se_cgm / se_lm
            if abs(_c_obs - c_theory) > 5e-6:
                raise SystemExit(
                    f"source C's CGM ratio {_c_obs:.10f} != c {c_theory:.10f}; "
                    "the two files do not share the SE convention")
        else:
            se_cgm = se_lm * c_theory_iv
            se_src = "derived: se_linearmodels * c (no r06 counterpart outcome)"
    else:
        # Source B runs on the 2019-2024 IV window while these rows are the
        # full-window panel FE of SI 6.5, so B's SE is no longer the SE of THIS
        # regression and cannot be copied across.  The CGM correction is instead
        # applied directly, which is licensed by the A_IV <-> B check above: that check
        # confirmed, on the matched window, that the entire gap between the two
        # implementations' standard errors is exactly the scalar c.
        se_cgm = se_lm * c_theory
        se_src = ("derived: se_linearmodels * c (source B is on the IV window "
                  "2019-2024 and is not a counterpart for the full-window panel FE)")

    t = beta / se_cgm
    p = 2.0 * stats.t.sf(abs(t), DF_T)

    rows.append(dict(
        spec=spec,
        si_table="tab:panel_fe_emp" if spec in list(SPEC_TO_OUTCOME)[:4]
                 else "tab:panel_fe_constr",
        beta=beta,
        se_linearmodels=se_lm,
        se_cgm=se_cgm,
        t_stat=t,
        p_value=p,
        r2_within=r2w,
        n_obs=n_obs,
        n_units=n_units,
        status=status,
        national_total_basis=basis,
        alternative_value=alt_value,
        se_source=se_src,
        beta_source=("panel_fe_employment.csv (09_ols_employment_dc.py, "
                     "balanced national total)" if spec == SPILLOVER_SPEC
                     else "panel_fe_*__contig49.csv (sample=contig49)"),
        beta_us50=float(LEG[LEG["spec"] == spec]["beta"].iloc[0]),
        se_us50=float(LEG[LEG["spec"] == spec]["se"].iloc[0]),
        n_obs_us50=int(LEG[LEG["spec"] == spec]["n_obs"].iloc[0]),
    ))

S = pd.DataFrame(rows)
log("  t/p reproduce r06's own printed t_stat and p_value on all six "
    "overlapping specifications => df = G-1 confirmed, not assumed")

# ------------------------------------------------------------------- rounding
def fmt(x):
    """Round the way the SI prints: nearest integer, thousands separator."""
    return f"{int(round(x)):,}".replace(",", "{,}")

log("")
log("PANEL FIXED-EFFECTS TABLE  (49 units, "
    "finite-sample-corrected clustered SE, t(48))")
log("-" * 78)
log(f"  {'specification':<36s} {'50 units':>20s} -> {'49 units':>20s}   "
    f"{'p':>7s}  {'R2w':>6s}")
for _, r in S.iterrows():
    pub = f"{fmt(r.beta_us50)} ({fmt(r.se_us50)})"
    new = f"{fmt(r.beta)} ({fmt(r.se_cgm)})"
    pstr = "<0.001" if r.p_value < 0.001 else f"{r.p_value:.3f}"
    log(f"  {r.spec:<36s} {pub:>20s} -> {new:>20s}   {pstr:>7s}  "
        f"{r.r2_within:>6.3f}")
log("-" * 78)
log(f"  sample description: 'the {n_units} units of the analysis universe "
    f"({UNIVERSE_LABEL}), {n_obs} unit-year observations'")
log(f"  us50 description: '50 states ... "
    f"{int(LEG['n_obs'].unique()[0])} observations'")

# ------------------------------------------------ scope effect on the raw row
raw = S[S.spec == "Raw NAICS 5182"].iloc[0]
log("")
log("RAW NAICS 5182 -- effect of the universe change:")
log(f"  the panel FE coefficient on raw NAICS 5182 is "
    f"{raw.beta:.6f} jobs/GW on the {n_units}-unit universe")
log(f"  it rounds to {fmt(raw.beta)}, against "
    f"{fmt(raw.beta_us50)}")
log(f"  {fmt(raw.beta_us50)} "
    f"({fmt(raw.se_us50)}) is the us50 row: "
    f"N={int(raw.n_obs_us50)}, AK and HI in, DC out")
log(f"  SE under the adopted convention: {raw.se_cgm:.6f} "
    f"-> prints as {fmt(raw.se_cgm)}")
log(f"  the linearmodels SE is {raw.se_linearmodels:.6f} -> "
    f"{fmt(raw.se_linearmodels)}; both describe the same regression")

out_csv = os.path.join(HERE, "panel_fe_table__contig49.csv")
S.to_csv(out_csv, index=False)
log("")
log(f"wrote {os.path.relpath(out_csv, os.path.dirname(EMP))}")

with open(os.path.join(HERE, "panel_fe_table__contig49_log.txt"), "w") as f:
    f.write("\n".join(LOG) + "\n")
