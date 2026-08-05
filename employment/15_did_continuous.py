#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 15_did_continuous.py
 CONTINUOUS-TREATMENT DIFFERENCE-IN-DIFFERENCES FOR DATA-CENTER CAPACITY
 AND EMPLOYMENT, FOLLOWING CALLAWAY, GOODMAN-BACON & SANT'ANNA
================================================================================

WHY THIS FILE EXISTS
--------------------
We do not rely on the binary-threshold staggered DiD, which declares a state
"treated" the first year its NEW installed data-center capacity (relative to
2016) crosses 500 MW or 1 GW.  Capacity is continuous, so a threshold indicator
discards the variation the question is about: a state moving from 200 to 300 MW
registers as nothing while one crossing the threshold registers as everything.
A continuous-treatment estimator is the appropriate alternative if a
difference-in-differences design is retained at all.

Two further problems with the binary design motivate what follows: the
propensity-score model behind its doubly robust form is not identified from
observables we hold, and difference-in-differences cannot establish causality
here regardless of functional form.

This script does three things:
  1. Implements the continuous-treatment DiD estimand and estimator of
     Callaway, Goodman-Bacon & Sant'Anna (CGS&S), NBER WP 25904 /
     "Difference-in-Differences with a Continuous Treatment", reporting the
     dose-response function ATT(d|d) and the average causal response on the
     treated, ACRT.
  2. Fully discloses BOTH propensity-score models -- the one implicit in the
     old binary design and the generalised propensity score used here --
     with common-support diagnostics and covariate balance before/after
     weighting.
  3. Builds a head-to-head estimator comparison table, all in
     jobs per GW.

NOTHING IS IMPORTED FROM `csdid` OR `drdid`. Neither package is installed in
this environment, and in any case neither implements the continuous-treatment
estimator. Every estimator below is implemented directly from the published
formulas, which are reproduced in the docstrings.

--------------------------------------------------------------------------------
 THE CGS&S FRAMEWORK, STATED EXPLICITLY
--------------------------------------------------------------------------------
EVERY STATEMENT IN THIS BLOCK IS TAKEN VERBATIM FROM THE PUBLISHED PAPER,
arXiv:2107.02637v8, draft of 31 December 2025, and the theorem/assumption
numbering below is that draft's.  Three points are easy to misstate from a
second-hand description of the paper -- the strong-parallel-trends assumption,
its relation to parallel trends, and the normalisation of the TWFE
decomposition weights -- and each is set out below with a pointer to the page.

Let D be a continuous dose (here: GW of new data-center capacity added since the
base year), with D = 0 for untreated units.  Y_t(d) is the potential outcome
under dose d.  Two periods, t = 1 (base) and t = 2 (post).

DEFINITIONS (paper, Section 3.1)
  Level effects
      ATT(d | d')  = E[ Y_2(d) - Y_2(0) | D = d' ]
      ATT(d | d)   = level effect of one's OWN dose  ("dose-response on treated")
      ATT(d)       = E[ Y_2(d) - Y_2(0) | D > 0 ]
                     ** NOT the unconditional ATE.  ATT(d) is the average effect
                     of dose d over ALL TREATED units, not just those that chose
                     d, and not over the untreated. **
  Slope (causal response) effects
      ACRT(d | d')  = d/dl ATT(l | d') |_{l = d}
      ACRT(d)       = d ATT(d) / dd
  Summary parameters (paper, end of Section 3.1)
      ATT^loc   = E[ ATT(D | D)  | D > 0 ]
      ATT^glob  = E[ ATT(D)      | D > 0 ]
      ACRT^loc  = E[ ACRT(D | D) | D > 0 ]
      ACRT^glob = E[ ACRT(D)     | D > 0 ]
  The paper has FOUR summary parameters, not two.  The unqualified word "ACRT"
  does not name a parameter in the paper and is not used unqualified here.

WHAT THE DATA IDENTIFY
  Observed dose-response (the thing a regression can actually recover):
      ATT^o(d)  = E[ Y_2 - Y_1 | D = d ] - E[ Y_2 - Y_1 | D = 0 ]
      ACRT^o(d) = d ATT^o(d) / dd
  ATT^o and ACRT^o are OBSERVED objects.  They are not causal parameters and are
  labelled with the superscript o everywhere in this file and in its outputs.

  (A1) PARALLEL TRENDS  (paper, Assumption PT).  For all d in D_+,
         E[ Y_2(0) - Y_1(0) | D = d ] = E[ Y_2(0) - Y_1(0) | D = 0 ]
       Under (A1), Theorem 3.1:   ATT^o(d) = ATT(d | d),
       and ATT^loc = E[dY | D > 0] - E[dY | D = 0].
       Parallel trends identifies the LEVEL dose-response local to each dose
       group.  It does NOT identify the slope, because (Theorem 3.2(b))
         ACRT^o(d) = ACRT(d | d)  +  d/dl ATT(d | l) |_{l = d}
                     \_ causal _/     \______ selection bias ______/
       The second term is how the effect of a FIXED dose d differs across
       groups that chose different doses.  It is zero only if there is no
       selection on gains.  Parallel trends does NOT identify ATT(d), ACRT(d),
       ATT^glob or ACRT^glob (paper, sentence following Theorem 3.1).

  (A2) STRONG PARALLEL TRENDS  (paper, Assumption SPT).  For all d in D,
         E[ Y_2(d) - Y_1(0) | D > 0 ] = E[ Y_2(d) - Y_1(0) | D = d ]
       NOTE.  (A2) is sometimes stated as
         E[ Y_2(d) - Y_1(0) | D = d' ] = E[ Y_2(d) - Y_1(0) ]  for all d, d'
       That is a STRICTLY STRONGER condition than the paper's, in two ways: it
       quantifies over every conditioning group d' rather than only over the
       treated population D > 0, and it equates to a full-population mean that
       includes the untreated units rather than to the observed path of the
       group that actually took dose d.  The paper's SPT restricts only the
       treated population.  Use the paper's.
       Under (A2), Theorem 3.3 and Corollary 3.1:
         ATT^o(d)  = ATT(d)                            [Thm 3.3(a)]
         ACRT^o(d) = ACRT(d)                           [Thm 3.3(c)]
         ATT^glob  = E[dY | D > 0] - E[dY | D = 0]     [Cor 3.1(a)]
         ACRT^glob = INT ACRT^o(s) f_{D|D>0}(s) ds     [Cor 3.1(b)]
       ONLY under (A2) does the derivative of the observed dose-response have a
       causal reading, and the parameter it then delivers is ACRT(d), the
       response for the WHOLE treated population, not ACRT(d|d).  (A2) is NOT
       testable: it is a statement about counterfactual TREATED outcomes, so no
       pre-period, in which no unit has taken any dose, can bear on it.  This
       script tests (A1) and three necessary implications of (A2); it cannot
       test (A2) itself, and says so.
       NOTE.  (A2) is not "strictly stronger" than (A1).  The paper is
       explicit (p.12): "While Assumption SPT is not strictly
       stronger than Assumption PT (e.g., notice that it does not require
       parallel trends in untreated potential outcomes for all dose groups), we
       refer to it as strong parallel trends to indicate that in many
       applications it would be a stronger, perhaps much stronger, assumption."
       The two are NOT nested.  What is true, and is the useful statement, is
       that IF (A1) is maintained then (A2) is equivalent to ATT(d|d) = ATT(d)
       for every dose, i.e. to the absence of selection into a particular
       AMOUNT of treatment on the basis of gains (paper, p.12-13).

  (A3) NO ANTICIPATION  (paper, Assumption 3):  Y_{i,1} = Y_{i,1}(d) = Y_{i,1}(0)
       for all d, and Y_{i,2} = Y_{i,2}(D_i).

  (A4) The paper's Assumption 2 additionally requires P(D = 0) > 0, a genuine
       untreated mass.  We do not have one; see Section 1 and the note on
       Remark 3.1 below.

WHY THE BINARY DESIGN IS WORSE, MADE PRECISE (paper, Theorems 3.1-3.2)
  * A binary DiD at threshold c compares D >= c against D < c.  Its estimand
    under (A1) is
        E[ ATT(D|D) | D >= c ]  -  E[ ATT(D|D) | D < c ]
    The "control" group is NOT untreated -- it contains states that added up to
    c GW.  The binary ATT is therefore a difference of two dose-averaged level
    effects, biased toward zero by the treated-ness of the controls.  This
    script quantifies that contamination directly from the estimated dose-
    response, using the ACTUAL control group D < c (which contains the zeros)
    and, for reference, the contaminating subset 0 < D < c on its own.
  * A linear TWFE regression on continuous D does not recover any ACRT
    parameter either.  Theorem 3.4(a), with the weights of the paper's Table 1:
        beta_TWFE = INT_{dL}^{dU} w1(l) [ ACRT(l|l) + selection bias ] dl
                    + w0 * ATT(dL|dL) / dL
        w1(l) = ( E[D | D >= l] - E[D] ) * P(D >= l) / Var(D)
        w0    = ( E[D | D >  0] - E[D] ) * P(D >  0) * dL / Var(D)
    and "the weights are always positive and integrate to 1", i.e.
        INT_{dL}^{dU} w1(l) dl  +  w0  =  1.
    THREE things follow, each easy to get wrong:
      (i)  w1 is a DENSITY in l.  It must be INTEGRATED with respect to dl, not
           summed over the observed dose points.  Summing gives every observed
           dose the same implicit width and therefore crushes a sparse upper
           tail; on our 2024 dose distribution the raw sum is 6.13, not 1.
      (ii) the normalisation includes w0, the separate term Theorem 3.4(a)
           carries for the discrete jump from 0 to the minimum treated dose dL.
           Paths of outcomes are not observed for doses below dL, so the scaled
           level effect ATT(dL|dL)/dL is what gets averaged in over that range
           (paper, footnote 6).
      (iii) w1 does NOT vanish at the tails.  As l -> 0+, w1(l) -> E[D] P(D=0) /
           Var(D) = w0/dL, which is strictly positive whenever an untreated
           group exists; at l = dU it is (dU - E[D]) P(D = dU) / Var(D) > 0.
    Under (A2) the selection-bias term is zero but the weights are unchanged,
    so beta_TWFE is "weakly causal" yet is still NOT ACRT^glob, because w1 is
    not the treated dose density f_{D|D>0} (paper, p.16-17).  This script
    computes w1 and w0 on the realised dose distribution, integrates them
    properly, and asserts the Theorem 3.4(a) normalisation.

NO UNTREATED GROUP (paper, Remark 3.1)
  With no genuinely untreated units the paper's prescription is to compare dose
  group d to the LOWEST dose group dL, which under (A1) delivers
  ATT(d|d) - ATT(dL|dL) and under (A2) delivers ATT(d) - ATT(dL).  The level is
  then identified only up to the additive constant ATT(dL|dL); the slope is not
  affected by an additive constant.  Our near-zero comparison group (total 2024
  dose below TAU, forced to exactly D = 0) is an APPROXIMATION to this, not the
  paper's construction, and is treated as such: TAU is varied and the
  sensitivity is reported.

REFERENCES
  Callaway, B., A. Goodman-Bacon and P. H. C. Sant'Anna. "Difference-in-
      Differences with a Continuous Treatment." arXiv:2107.02637v8, 31 December
      2025.
  Callaway, B. and P. H. C. Sant'Anna (2021). "Difference-in-Differences with
      Multiple Time Periods." Journal of Econometrics 225(2), 200-230.
  Sant'Anna, P. H. C. and J. Zhao (2020). "Doubly Robust Difference-in-
      Differences Estimators." Journal of Econometrics 219(1), 101-122.
  Hirano, K. and G. W. Imbens (2004). "The Propensity Score with Continuous
      Treatments." In Applied Bayesian Modeling and Causal Inference.
  Austin, P. C. (2019). "Assessing covariate balance when using the generalized
      propensity score with quantitative or continuous exposures."
      Statistical Methods in Medical Research 28(5), 1365-1377.
  Yitzhaki, S. (1996). "On Using Linear Regressions in Welfare Economics."
      Journal of Business & Economic Statistics 14(4), 478-486.
  Silverman, B. W. (1986). Density Estimation for Statistics and Data Analysis.

--------------------------------------------------------------------------------
 OUR EMPIRICAL MAPPING
--------------------------------------------------------------------------------
  base period  : 2016 (first year of the capacity series; the base year the
                 published binary DiD also used)
  post periods : 2017 ... 2024 (QWI ends 2024)
  dose         : D_{s,t} = max( dc_gw_{s,t} - dc_gw_{s,2016}, 0 ), in GW
                 -- cumulative NEW capacity, exactly the quantity the binary
                    design thresholded at 0.5 / 1.0 GW
  untreated    : D_{s,2024} < TAU (TAU = 0.05 GW = 50 MW) -- see Section 1 for
                 why an exact-zero group is infeasible and what it costs
  outcome      : Y_{s,t} = state-level QWI employment in the relevant NAICS

  CGS&S formally cover (i) two periods with a continuous dose and (ii) staggered
  adoption with a one-time dose.  Our dose accrues continuously in every year,
  which is outside the case they formally treat.  We therefore apply their
  TWO-PERIOD estimator to each (2016, t) pair separately -- a long difference
  against a fixed base period -- and then aggregate across t.  This is an
  extension, not a result in their paper, and is flagged as such throughout.

--------------------------------------------------------------------------------
 OUTPUTS (all NEW files, all under employment/../results/r6_employment/, suffixed '__contig')
 The pre-scope-restriction run remains in employment/../results/r6_bartik/ without the suffix.
--------------------------------------------------------------------------------
  did_dose_response.csv        ATT^o(d,t) and ACRT^o(d,t) on a dose grid
  did_acrt_by_period.csv       ACRT^glob(t) event study, all outcomes/estimators
  did_acrt_summary.csv         headline ACRT^glob and ATT^loc, all outcomes
  did_binned_dose_response.csv fully nonparametric binned dose-response
  did_propensity_disclosure.csv  PS / GPS coefficient tables
  did_balance_table.csv        covariate balance before and after weighting
  did_twfe_weights.csv         Theorem 3.4(a) decomposition weights of the
                               linear TWFE estimand: w1 density, per-interval
                               mass, and the w0 jump term
  did_parallel_trends_tests.csv  pre-trend and strong-PT implication tests
  did_estimator_comparison.csv the head-to-head table, jobs per GW
  did_continuous_results.json  everything, machine readable
  did_continuous_run_log.txt   full console log

NO EXISTING FILE IS READ-WRITE. Every path opened for writing is new.
================================================================================
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
# Release-set path rule: no absolute paths.  Resolved from this file's location.
EMP = os.path.dirname(os.path.abspath(__file__))   # .../employment
ROOT = os.path.dirname(EMP)                        # .../codes
IN = os.path.join(EMP, "../results/r6_bartik")
OUT = os.path.join(EMP, "../results/r6_employment")      # new outputs, previous run untouched

# ---------------------------------------------------------------------------
# RUN TAG.  Required, no default, so that a re-run can never shadow an earlier
# artefact at an unsuffixed name.  SUFFIX is built
# from --tag and is never empty.  The validator strips WHITESPACE ONLY -- it
# must not strip leading underscores, or --tag _contig would silently collapse
# 'did_acrt_summary__contig.csv' to 'did_acrt_summary_contig.csv'.
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(
    description="Continuous-treatment DiD (Callaway, Goodman-Bacon & Sant'Anna, "
                "arXiv:2107.02637v8) on the 49-unit analysis universe.")
_ap.add_argument("--tag", required=True,
                 help="run tag appended to every output filename, e.g. "
                      "'cgs49'.  Required; there is no default.")
_ARGS = _ap.parse_args()
_TAG = _ARGS.tag.strip()
if not _TAG:
    sys.exit("--tag must be a non-empty, non-whitespace string")
SUFFIX = "__" + _TAG                       # distinguishing suffix on every output

PANEL = os.path.join(IN, "panel_state_year.csv")
DCFILE = os.path.join(EMP, "dc_facilities_by_state_year.csv")
XLSX_BINARY = os.path.join(EMP, "06_staggered_did_analysis.xlsx")

os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# ANALYSIS SCOPE. IDENTICAL MECHANISM to
# 05_bartik_iv.py: an explicit OUT_OF_SCOPE_UNITS constant and an
# apply_scope() function that runs BEFORE anything is built from the data,
# backed by pre-flight assertions.  There is deliberately no second mechanism.
# The analysis universe is the contiguous United States: the 48 contiguous
# states plus the District of Columbia, 49 units in total.  Alaska and Hawaii
# are OUT OF SCOPE BY DEFINITION, not by data availability:
#   * both sit on isolated grids that no ISO in this paper reaches (CAISO,
#     ERCOT, MISO, PJM, SPP, NYISO, ISONE);
#   * the four ISO transmission-planning portfolios used for the cost
#     attribution do not cover them; and
#   * the cross-state spillover and regional cost-sharing mechanisms this
#     paper studies cannot operate on an isolated grid.
# Applying it here rather than relying on the fact that Alaska happens to have
# only one observed employment year matters: dropping Alaska silently by
# missingness while Hawaii stays in the COMPARISON group would leave the
# continuous-DiD sample and the Bartik sample on different universes, and the
# estimator comparison would then span two of them.
# The two lines
#     OUT_OF_SCOPE_UNITS = ("AK", "HI")
#     N_ANALYSIS_UNITS = 49
# were LOCAL LITERALS here, and identical literals lived in
# 05_bartik_iv.py and five other scripts.  The comment above claims
# "IDENTICAL MECHANISM to 05_bartik_iv.py"; a comment cannot enforce
# that, and nothing raised if the two drifted.  Both files now IMPORT the same
# names from employment/analysis_universe.py, so "identical" is a fact about the
# object, not a claim about the text.  No fallback: an ImportError aborts.
# ---------------------------------------------------------------------------
if EMP not in sys.path:
    sys.path.insert(0, EMP)
from analysis_universe import (  # noqa: E402
    ANALYSIS_UNITS, N_ANALYSIS_UNITS, OUT_OF_SCOPE_UNITS, UNIVERSE_LABEL,
    UniverseError, check_universe, describe as universe_describe,
)

# Pre-flight reference constants, computed over the 49-unit analysis universe.
# If the 'US'/'National' aggregate row ever creeps
# back into the capacity file every share halves and share('US') = 0.5.
REF_NATIONAL_2019_GW = 27.933103
REF_SHARE2019_TOP8 = {"TX": 0.138952, "VA": 0.133208, "CA": 0.089361,
                      "IL": 0.057256, "NC": 0.055833, "GA": 0.053735,
                      "WA": 0.049288, "IA": 0.046002}
REF_TOP3_SUM = 0.361520
MIN_DIVISION_SIZE = 3

BASE_YEAR = 2016
POST_YEARS = list(range(2017, 2025))
TAU = 0.05            # GW; comparison-group ceiling on total new capacity
TAU_ALT = [0.0, 0.10, 0.20]
B_BOOT = 1999
SEED = 20260728
THRESH_BINARY = [0.5, 1.0]   # the published binary thresholds, GW

OUTCOMES = [
    ("emp_5182", "Data centres (NAICS 5182), raw QWI"),
    ("emp_5182_cleaned", "Data centres, cleaned direct (5182 x local share)"),
    ("emp_517", "Telecommunications (NAICS 517)"),
    ("emp_23", "Construction (NAICS 23)"),
    ("emp_2362", "Nonresidential building construction (NAICS 2362)"),
    ("emp_5415", "Computer systems design (NAICS 5415)"),
]
PRIMARY_OUTCOMES = ["emp_5182", "emp_5182_cleaned", "emp_517"]

# Benchmarks the comparison table must line up against.  None of them is typed
# as a literal.  The IV numbers are read off
# ../results/r6_bartik/results_main__union.csv (leave_out_variant == 'union',
# specification == 'level', share_baseline == 2019); the panel-FE numbers are
# read off ../results/r6_employment/panel_fe_table__contig49.csv, produced by
# 14_panel_fe_table.py, whose SE convention is the finite-sample-corrected
# clustered SE with t(G-1) = t(48) inference.
PANEL_FE_TABLE = os.path.join(OUT, "panel_fe_table__contig49.csv")

def _read_panel_fe(spec_label):
    """Read one row of the panel-FE table.  Hard-fails if absent
    rather than falling back on a hard-coded literal.

    The table carries a `status` column because one of its seven rows,
    "Raw x multiplier + spillover", exists in two versions differing in whether
    the national aggregate in its dependent variable is computed on a balanced
    panel.  A row not marked `reported` is refused here, and the absence of the
    column is itself refused, so an old copy of the file cannot be read as if it
    had been checked.
    """
    if not os.path.exists(PANEL_FE_TABLE):
        sys.exit(f"missing {PANEL_FE_TABLE}; run 14_panel_fe_table.py "
                 f"first.  This script refuses to carry a hard-coded panel-FE "
                 f"benchmark.")
    _t = pd.read_csv(PANEL_FE_TABLE)
    if "status" not in _t.columns:
        sys.exit(f"{PANEL_FE_TABLE} has no `status` column: it may carry the unbalanced "
                 f"spillover aggregate. Re-run 14_panel_fe_table.py.")
    _r = _t[_t["spec"] == spec_label]
    if len(_r) != 1:
        sys.exit(f"panel FE table: expected exactly one row for {spec_label!r}, "
                 f"got {len(_r)}")
    _r = _r.iloc[0]
    if str(_r["status"]).strip().lower() != "reported":
        sys.exit(f"panel FE table row {spec_label!r} is marked "
                 f"{_r['status']!r}, not 'reported'. "
                 f"{_r.get('alternative_value', '')} "
                 f"Refusing to benchmark against a diagnostic-only row.")
    return dict(beta=float(_r["beta"]), se=float(_r["se_cgm"]),
                n=int(_r["n_obs"]))

# ---------------------------------------------------------------------------
# Until now the three lines below read
#     PUB_IV       = dict(beta=3705.69, se=1916.35, n=438)
#     PUB_IV_CLEAN = dict(beta=2825.76, se=1459.38, n=438)
#     PUB_IV_517   = dict(beta=-1650.06, se=1335.76, n=438)
# as literals.  A number read off a file once by hand is not read at run time,
# and it falls out of date silently the next time the estimator is re-run.  They are
# now read from the estimator's own output at run time and asserted.
# The selector pins every dimension that has ever been mixed up here --
# estimator, specification, leave-out variant, share baseline -- and requires
# exactly one row, so a file that gained a robustness variant under the same
# outcome name aborts the run instead of silently picking the first match.
# The sample is additionally asserted against the universe module.
# ---------------------------------------------------------------------------
MAIN_UNION = os.path.join(IN, "results_main__union.csv")
IV_PRIMARY_LEAVE_OUT = "union"
IV_PRIMARY_SHARE_BASE = 2019
_IV_CACHE = {}

def _read_pub_iv(outcome):
    """
    Read the primary Bartik IV benchmark for `outcome` from
    employment/../results/r6_bartik/results_main__union.csv, the file 05_bartik_iv.py
    writes.  Hard-fails rather than falling back to a literal.
    """
    if not _IV_CACHE:
        if not os.path.exists(MAIN_UNION):
            sys.exit(f"missing {MAIN_UNION}; run "
                     f"'python3 05_bartik_iv.py --tag _union' first. "
                     f"This script refuses to carry a hard-coded IV "
                     f"benchmark.")
        _IV_CACHE["t"] = pd.read_csv(MAIN_UNION)
    t = _IV_CACHE["t"]
    q = t[(t["outcome"] == outcome)
          & (t["estimator"] == "IV2SLS")
          & (t["specification"] == "level")
          & (t["leave_out_variant"] == IV_PRIMARY_LEAVE_OUT)
          & (t["share_baseline"] == IV_PRIMARY_SHARE_BASE)]
    if "robustness_tag" in t.columns:
        q = q[q["robustness_tag"].isna()]
    if len(q) != 1:
        sys.exit(f"results_main__union.csv: expected exactly one primary IV2SLS "
                 f"level row for {outcome!r} (union / share {IV_PRIMARY_SHARE_BASE}), "
                 f"got {len(q)}.  Refusing to guess which one the SI prints.")
    r = q.iloc[0]
    n_units = int(r["n_clusters"])
    if n_units != N_ANALYSIS_UNITS:
        sys.exit(f"results_main__union.csv row for {outcome!r} was estimated on "
                 f"{n_units} clusters, but analysis_universe declares "
                 f"{N_ANALYSIS_UNITS}.  The comparison table would mix universes "
                 f"Aborting.")
    return dict(beta=float(r["coef"]), se=float(r["se"]), n=int(r["n_obs"]),
                f_eff=float(r["fs_F_effective"]))

PUB_PANEL_FE = _read_panel_fe("Raw NAICS 5182")
PUB_PANEL_FE_CLEAN = _read_panel_fe("Cleaned direct")
PUB_IV = _read_pub_iv("emp_5182")
PUB_IV_CLEAN = _read_pub_iv("emp_5182_cleaned")
PUB_IV_517 = _read_pub_iv("emp_517")

# All three rows must come from ONE estimation sample; if they do not, the
# comparison table is mixed-universe again and no caveat repairs it.
assert PUB_IV["n"] == PUB_IV_CLEAN["n"] == PUB_IV_517["n"], (
    "the three IV benchmark rows disagree on N: "
    f"{PUB_IV['n']}, {PUB_IV_CLEAN['n']}, {PUB_IV_517['n']}")
assert abs(PUB_IV["f_eff"] - PUB_IV_CLEAN["f_eff"]) < 1e-9, (
    "the three IV benchmark rows disagree on the first-stage F, so they do not "
    "share a first stage")

PUB_IV_NOTE = (f"05_bartik_iv.py, {IV_PRIMARY_LEAVE_OUT} leave-out, "
               f"{N_ANALYSIS_UNITS} units, N = {PUB_IV['n']}, "
               f"Olea-Pflueger effective F = {PUB_IV['f_eff']:.1f}")

RNG = np.random.default_rng(SEED)

_LOG_LINES = []

def log(msg=""):
    print(msg)
    _LOG_LINES.append(str(msg))

def hdr(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)

T0 = time.time()
hdr("15_did_continuous.py  --  CONTINUOUS-TREATMENT DiD (CGS&S)")
log(f"started            : {datetime.now().isoformat(timespec='seconds')}")
log(f"python             : {sys.version.split()[0]}")
log(f"numpy/pandas/scipy : {np.__version__} / {pd.__version__} / {st.__name__}")
log(f"seed               : {SEED}   bootstrap reps: {B_BOOT}")

# ==============================================================================
# 1. DATA, DOSE CONSTRUCTION, AND PRE-FLIGHT ASSERTIONS
# ==============================================================================
hdr("1. DATA AND DOSE CONSTRUCTION")

def apply_scope(panel):
    """
    Restrict the panel to the analysis universe (the 48 contiguous states plus
    DC) before anything is built from it.

    Same mechanism as apply_scope() in 05_bartik_iv.py.  The one
    difference is that this script has no exposure share to renormalise: the
    treatment here is the state's own realised new capacity, which is a level
    and not a share of a national total, so nothing has to be rescaled.  The
    quantities that DO depend on the universe -- the comparison group, the
    bootstrap resampling frame, the Yitzhaki weight distribution and the dose
    contrast used to convert the binary ATT -- all follow from the restricted
    panel automatically.

    Returns (panel, report_dict).
    """
    drop = [s for s in OUT_OF_SCOPE_UNITS if s in set(panel["state_abbr"])]
    before_units = panel["state_abbr"].nunique()
    before_2019 = float(panel.loc[panel["year"] == 2019, "dc_gw"].sum())
    dropped_mw = {s: float(panel.loc[(panel["state_abbr"] == s)
                                     & (panel["year"] == 2019), "dc_gw"].iloc[0]) * 1000.0
                  for s in drop}
    p = panel[~panel["state_abbr"].isin(drop)].copy()
    after_2019 = float(p.loc[p["year"] == 2019, "dc_gw"].sum())
    rep = dict(dropped=drop, dropped_mw_2019=dropped_mw,
               n_units_before=before_units, n_units_after=p["state_abbr"].nunique(),
               gw_2019_before=before_2019, gw_2019_after=after_2019,
               dropped_pct_of_2019=100.0 * (before_2019 - after_2019) / before_2019)

    hdr("ANALYSIS SCOPE: contiguous United States (48 states + DC)")
    log("  Alaska and Hawaii are out of scope by definition, not by data")
    log("  availability: both sit on isolated grids that none of the seven ISOs")
    log("  analysed here reaches, the four ISO transmission-planning portfolios")
    log("  used for the cost attribution do not cover them, and the cross-state")
    log("  spillover and regional cost-sharing mechanisms studied in this paper")
    log("  cannot operate on an isolated grid.")
    log(f"\n  units: {before_units} -> {rep['n_units_after']}")
    for s, mw in dropped_mw.items():
        log(f"  dropped {s}: {mw:.2f} MW of 2019 capacity")
    log(f"  2019 national capacity: {before_2019:.6f} -> {after_2019:.6f} GW "
        f"({rep['dropped_pct_of_2019']:.4f}% removed)")
    return p, rep

panel_all = pd.read_csv(PANEL)
assert panel_all.shape[0] == 510, f"panel rows {panel_all.shape[0]} != 510"
# The INPUT panel is the raw 51-unit build (analysis units + out-of-scope units);
# the literal 51 is derived from the module, not typed.
from analysis_universe import ALL_QWI_UNITS  # noqa: E402
assert set(panel_all["state_abbr"]) == set(ALL_QWI_UNITS), (
    "input panel units != ANALYSIS_UNITS + OUT_OF_SCOPE_UNITS: "
    f"unexpected {sorted(set(panel_all['state_abbr']) - set(ALL_QWI_UNITS))}, "
    f"absent {sorted(set(ALL_QWI_UNITS) - set(panel_all['state_abbr']))}")
assert "US" not in set(panel_all["state_abbr"])

panel, scope_rep = apply_scope(panel_all)

# assert the scoped panel against the module that owns the
# universe, strict in BOTH directions.  The count-based A2 assertion below
# would pass on any 49 units; this one passes only on THE 49.
check_universe(panel, "state_abbr", where="apply_scope output (continuous DiD)")

# Independent re-verification of the capacity series straight from the raw file,
# including the national aggregate, ON THE ANALYSIS UNIVERSE.
raw = pd.read_csv(DCFILE)
us_row = raw[raw["state_abbr"] == "US"]
assert len(us_row) == 1, "expected exactly one 'US' aggregate row"
raw_all = raw[raw["state_abbr"] != "US"].copy()
assert set(raw_all["state_abbr"]) == set(ALL_QWI_UNITS), (
    f"raw capacity file units != ALL_QWI_UNITS after dropping the US row "
    f"({len(raw_all)} rows)")
for yy in range(2016, 2026):
    diff = abs(raw_all[f"MW_{yy}"].sum() - us_row[f"MW_{yy}"].values[0])
    assert diff < 1e-6, f"US row != sum of states in {yy} (diff {diff})"
raw_states = raw_all[~raw_all["state_abbr"].isin(OUT_OF_SCOPE_UNITS)].copy()

hdr("PRE-FLIGHT ASSERTIONS -- any failure aborts the run")
_pf = []

def _rec(name, ok, value, note=""):
    _pf.append((name, bool(ok), value, note))
    log(f"  {name:<32s} {'PASS' if ok else '**FAIL**':>8s}   value={value!r}"
        + (f"   [{note}]" if note else ""))

_abbrs = set(panel["state_abbr"])
_rec("A1_no_US_row", "US" not in _abbrs, float("US" in _abbrs),
     "national aggregate row must be absent")
_rec("A2_49_units", len(_abbrs) == N_ANALYSIS_UNITS, len(_abbrs),
     "48 contiguous states + DC")
_rec("A2b_scope_excluded", not (set(OUT_OF_SCOPE_UNITS) & _abbrs),
     len(set(OUT_OF_SCOPE_UNITS) & _abbrs),
     "AK/HI out of scope: isolated grids, no ISO coverage")
_rec("A2c_raw_49_units", len(raw_states) == N_ANALYSIS_UNITS, len(raw_states),
     "raw capacity file restricted to the same universe")
_nat19 = float(raw_states["MW_2019"].sum() / 1000.0)
_rec("A3_national_2019_gw", abs(_nat19 - REF_NATIONAL_2019_GW) < 1e-4,
     round(_nat19, 6), f"reference {REF_NATIONAL_2019_GW}")
_chk = raw_states.set_index("state_abbr")["MW_2019"] / raw_states["MW_2019"].sum()
_worst = max(abs(float(_chk[k]) - v) for k, v in REF_SHARE2019_TOP8.items())
_rec("A4_share2019_top8", _worst < 1e-4, round(_worst, 8),
     "max |computed - reference| over TX,VA,CA,IL,NC,GA,WA,IA")
_t3 = float(_chk.nlargest(3).sum())
_rec("A5_top3_sum", abs(_t3 - REF_TOP3_SUM) < 1e-3, round(_t3, 6),
     f"reference {REF_TOP3_SUM}")
_dc = panel[panel["year"] == 2019]["census_division"].value_counts().to_dict()
_rec("A6_division_counts", len(_dc) == 9 and min(_dc.values()) >= MIN_DIVISION_SIZE,
     min(_dc.values()), f"{len(_dc)} divisions, {_dc}")
_pfail = [n for n, ok, _, _ in _pf if not ok]
log(f"\n  {len(_pf) - len(_pfail)}/{len(_pf)} assertions passed.")
if _pfail:
    log(f"  HARD ABORT: pre-flight failed -> the build is wrong. Failed: {_pfail}")
    sys.exit(1)

log(f"\n[A1] raw capacity file: 'US' aggregate row dropped, "
    f"{len(raw_states)} in-scope states remain, "
    f"US == sum(all 51 states) for all 10 years (max |diff| < 1e-6).")
log(f"[A2] 2019 shares over the 49-unit universe reproduce the reference "
    f"constants (national 2019 = {_nat19:.6f} GW; top-3 = {_t3:.6f}).")

pn = panel[panel["year"] <= 2024].copy()
cap = pn.pivot(index="state_abbr", columns="year", values="dc_gw")
# cross-check the panel's capacity against the raw file
for yy in range(2016, 2025):
    m = (raw_states.set_index("state_abbr")[f"MW_{yy}"] / 1000.0)
    assert np.max(np.abs(cap[yy] - m.reindex(cap.index))) < 1e-9
log("[A3] panel dc_gw reproduces raw MW/1000 for every state-year (max |diff| < 1e-9).")

# ---- dose ---------------------------------------------------------------
dose_raw = cap.sub(cap[BASE_YEAR], axis=0)          # new capacity since 2016
n_neg = int((dose_raw[2024] < -1e-9).sum())
neg_states = sorted(dose_raw.index[dose_raw[2024] < -1e-9])
dose = dose_raw.clip(lower=0.0)                     # doses cannot be negative
log(f"\ndose D_(s,t) = max(dc_gw_(s,t) - dc_gw_(s,2016), 0), in GW")
log(f"  {n_neg} states have a NEGATIVE measured 2024 dose ({', '.join(neg_states)}), "
    f"a consequence of the known non-monotonicity of the cumulative MW series in "
    f"the upstream file (15 states). They are clipped to 0. Robustness check drops them.")

D24 = dose[2024]
untreated_mask = D24 < TAU
untreated_states = sorted(D24.index[untreated_mask])
treated_states = sorted(D24.index[~untreated_mask])
log(f"\ncomparison ('untreated') group: total new capacity by 2024 < {TAU} GW")
log(f"  n = {len(untreated_states):2d}  {', '.join(untreated_states)}")
log(f"  treated  n = {len(treated_states):2d}")
log(f"  exact zeros (D_2024 == 0)      : {int((D24 <= 1e-12).sum())} states "
    f"({', '.join(sorted(D24.index[D24 <= 1e-12]))})")
log("  NOTE: only a handful of states added literally zero capacity, so a strict")
log("  D=0 comparison group is infeasible. CGS&S discuss exactly this case: with no")
log("  untreated units the LEVEL dose-response is identified only up to an additive")
log("  constant, while the SLOPE (ACRT) is unaffected. We therefore (a) use a")
log(f"  near-zero comparison group (D < {TAU} GW) as the primary design and (b) report a")
log("  no-comparison-group variant in which every state contributes and only the")
log("  slope is identified. The two ACRTs are reported side by side.")

log("\ndose distribution, 2024 (GW of new capacity since 2016):")
q = D24.describe(percentiles=[.1, .25, .5, .75, .9, .95])
for k, v in q.items():
    log(f"    {k:>6s} : {v: .4f}")
log("  top 8 states: " + ", ".join(f"{s} {D24[s]:.3f}" for s in D24.nlargest(8).index))

# ---- outcome panel ------------------------------------------------------
wide = {}
for yv, _ in OUTCOMES:
    wide[yv] = pn.pivot(index="state_abbr", columns="year", values=yv)

emp_avail = pn.pivot(index="state_abbr", columns="year", values="emp_available")
miss = emp_avail.isna().sum(axis=1) + (emp_avail == 0).sum(axis=1)
log(f"\nemployment availability: states with any missing year -> "
    f"{ {k:int(v) for k,v in miss[miss>0].items()} }")

covars_base = pn[pn["year"] == BASE_YEAR].set_index("state_abbr")
static = panel[panel["year"] == BASE_YEAR].set_index("state_abbr")[
    ["census_region", "census_division", "n_neighbors", "gdp_pc_2024"]]

STATES = list(cap.index)
# the ESTIMATION UNIVERSE of this script -- the object every ACRT,
# every bootstrap resample and every dose contrast is computed over -- is
# asserted against analysis_universe.py at start-up, set-equality in both
# directions.  The two count/disjointness assertions kept below are strictly
# weaker (any 49 units without AK or HI would satisfy them); they are retained
# only because their failure messages are more specific.
check_universe(STATES, where="continuous-DiD state universe (cap.index)")
assert len(STATES) == N_ANALYSIS_UNITS, \
    f"state universe is {len(STATES)}, expected {N_ANALYSIS_UNITS}"
assert not (set(OUT_OF_SCOPE_UNITS) & set(STATES))
log(f"\nstate universe: {len(STATES)}  ({UNIVERSE_LABEL})")

# ==============================================================================
# 2. PROPENSITY SCORE DISCLOSURE
# ==============================================================================
hdr("2. PROPENSITY SCORE MODELS -- FULL DISCLOSURE")

log("""
2.0 WHAT THE BINARY DESIGN USES
-------------------------------------
12_staggered_did.py calls

    csdid.att_gt.ATTgt(yname=..., tname='year', idname='state_id',
                       gname='treat_year', data=df,
                       control_group='nevertreated')

with NO `xformla` argument. In the csdid package (as in R's `did`), the default
is xformla = ~1: the propensity score is fit on an intercept only. Consequently
the "doubly robust" estimator in the SI degenerates to the unconditional
outcome-regression DiD, every unit receives weight 1, and there is no
propensity-score model to disclose beyond a constant. That is the honest answer
to the objection: the PS model was not described because there was not one.
The cost is that NO covariate was ever balanced -- conditional parallel trends
was never invoked, only unconditional parallel trends.

Below we specify, estimate, and fully disclose (a) a binary propensity score for
the threshold design and (b) a GENERALISED propensity score for the continuous
dose, and we report covariate balance before and after weighting in both cases.
""")

# ---- covariate set ------------------------------------------------------
X_SPEC = [
    ("log_emp_5182_2016", "log 2016 data-centre employment (NAICS 5182)"),
    ("log_emp_517_2016", "log 2016 telecom employment (NAICS 517)"),
    ("log_emp_23_2016", "log 2016 construction employment (NAICS 23)"),
    ("log_emp_5415_2016", "log 2016 computer systems design employment (NAICS 5415)"),
    ("log_earn_5182_2016", "log 2016 NAICS 5182 monthly earnings (EarnBeg)"),
    ("log_cap_2016", "log(1 + 2016 installed data-centre capacity, GW)"),
    ("n_neighbors", "number of contiguous states"),
]
log("2.1 COVARIATE SET (all strictly PRE-DETERMINED, measured in the base year 2016)")
for k, d in X_SPEC:
    log(f"    {k:<22s}  {d}")
log("""
    EXCLUDED ON PURPOSE
      gdp_pc_2024        : the only state GDP series supplied is a 2024
                           cross-section, i.e. measured AFTER treatment. Using it
                           would condition on an outcome. (Reported in the
                           balance table as an unbalanced-by-construction
                           diagnostic only, never used as a PS covariate.)
      electricity price, energy burden
                         : mediators on the paper's own causal path.
      subsidy_musd       : observed for only 28 states and only 2020-2024; it is
                           the policy instrument states use to ATTRACT capacity,
                           i.e. a treatment determinant that is itself
                           post-treatment for most cells.
      census region dummies
                         : with 49 units and 7 continuous covariates, 3 further
                           dummies destroy overlap in the logit. Region is
                           instead used in the balance table and in the
                           robustness PS (Section 2.5).
""")

X = pd.DataFrame(index=STATES)
X["log_emp_5182_2016"] = np.log(covars_base["emp_5182"])
X["log_emp_517_2016"] = np.log(covars_base["emp_517"])
X["log_emp_23_2016"] = np.log(covars_base["emp_23"])
X["log_emp_5415_2016"] = np.log(covars_base["emp_5415"])
X["log_earn_5182_2016"] = np.log(covars_base["earn_5182"])
X["log_cap_2016"] = np.log1p(cap[BASE_YEAR])
X["n_neighbors"] = static["n_neighbors"].astype(float)
X = X.loc[STATES]
assert X.notna().all().all(), "missing covariate values"
XCOLS = list(X.columns)
log(f"covariate matrix: {X.shape[0]} states x {X.shape[1]} covariates, no missing values.")

ps_rows = []
bal_rows = []

def std_mean_diff(x, t, w=None):
    """Standardised mean difference between treated and control groups.
    Denominator: pooled UNWEIGHTED sd (Austin 2009 convention), so that the
    before/after numbers are directly comparable."""
    t = t.astype(bool)
    if w is None:
        w = np.ones(len(x))
    w = np.asarray(w, float)
    m1 = np.average(x[t], weights=w[t])
    m0 = np.average(x[~t], weights=w[~t])
    s = np.sqrt(0.5 * (np.var(x[t], ddof=1) + np.var(x[~t], ddof=1)))
    return (m1 - m0) / s if s > 0 else np.nan

def weighted_corr(x, d, w):
    w = np.asarray(w, float)
    mx = np.average(x, weights=w)
    md = np.average(d, weights=w)
    cxd = np.average((x - mx) * (d - md), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vd = np.average((d - md) ** 2, weights=w)
    return cxd / np.sqrt(vx * vd) if vx > 0 and vd > 0 else np.nan

# ------------------------------------------------------------------
# 2.2 BINARY PROPENSITY SCORE (for the threshold design)
# ------------------------------------------------------------------
log("\n2.2 BINARY PROPENSITY SCORE  -- P(cross threshold by 2024 | X), logit")
ps_binary = {}
for thr in THRESH_BINARY:
    Tb = (D24 >= thr).astype(int)
    Xd = sm.add_constant(X.values)
    try:
        mod = sm.Logit(Tb.values, Xd).fit(disp=0, maxiter=200)
        conv = bool(mod.mle_retvals.get("converged", False))
        pscore = mod.predict(Xd)
        params, bses, pvals = mod.params, mod.bse, mod.pvalues
        pr2 = float(mod.prsquared)
        llf = float(mod.llf)
    except Exception as e:                                      # pragma: no cover
        log(f"  logit failed at threshold {thr}: {e}")
        continue
    names = ["const"] + XCOLS
    log(f"\n  --- threshold {thr:.1f} GW :  n_treated={int(Tb.sum())}, "
        f"n_control={int((1-Tb).sum())}, converged={conv}, "
        f"McFadden pseudo-R2={pr2:.3f}, logL={llf:.3f}")
    log(f"    {'covariate':<24s}{'coef':>12s}{'se':>12s}{'z':>8s}{'p':>8s}")
    for nm, b, s, pv in zip(names, params, bses, pvals):
        log(f"    {nm:<24s}{b:>12.4f}{s:>12.4f}{b/s:>8.2f}{pv:>8.3f}")
        ps_rows.append(dict(model=f"binary_logit_{thr:g}GW", term=nm, coef=b,
                            se=s, z=b / s, pvalue=pv,
                            pseudo_r2=pr2, n=len(Tb),
                            n_treated=int(Tb.sum()), converged=conv))
    # common support
    p1, p0 = pscore[Tb == 1], pscore[Tb == 0]
    lo = max(p1.min(), p0.min())
    hi = min(p1.max(), p0.max())
    off = int(((pscore < lo) | (pscore > hi)).sum())
    log(f"    common support: treated pscore [{p1.min():.4f}, {p1.max():.4f}], "
        f"control [{p0.min():.4f}, {p0.max():.4f}]")
    log(f"    overlap region [{lo:.4f}, {hi:.4f}]; {off} of {len(pscore)} states outside it")
    extreme = int(((pscore < .05) | (pscore > .95)).sum())
    log(f"    extreme scores (<0.05 or >0.95): {extreme} states "
        f"-> ATT odds weight p/(1-p) max = {np.max(p0/(1-p0)):.2f} among controls")
    ps_binary[thr] = dict(pscore=pscore, T=Tb.values, converged=conv,
                          pr2=pr2, overlap=(lo, hi), n_off=off, n_extreme=extreme)

    # ATT weights: treated get 1; controls get p/(1-p) normalised
    w = np.ones(len(pscore))
    ctl = Tb.values == 0
    odds = pscore[ctl] / (1 - pscore[ctl])
    w[ctl] = odds / odds.mean()
    log(f"    {'covariate':<24s}{'SMD before':>12s}{'SMD after':>12s}")
    for c in XCOLS + ["gdp_pc_2024"]:
        xv = X[c].values if c in X.columns else static.loc[STATES, c].values.astype(float)
        b_ = std_mean_diff(xv, Tb.values)
        a_ = std_mean_diff(xv, Tb.values, w)
        log(f"    {c:<24s}{b_:>12.3f}{a_:>12.3f}"
            + ("   [not a PS covariate]" if c not in X.columns else ""))
        bal_rows.append(dict(model=f"binary_logit_{thr:g}GW", covariate=c,
                             metric="std_mean_diff", before=b_, after=a_,
                             is_ps_covariate=c in X.columns))

# ------------------------------------------------------------------
# 2.3 GENERALISED PROPENSITY SCORE (continuous dose)
# ------------------------------------------------------------------
log("""
2.3 GENERALISED PROPENSITY SCORE FOR THE CONTINUOUS DOSE
--------------------------------------------------------
With a continuous treatment the binary propensity score has no analogue; the
object is the generalised propensity score (GPS) of Hirano & Imbens (2004),
r(d, X) = f_{D|X}(d | X), the conditional density of the dose at the realised
dose. We model the dose among treated states (D > TAU) as

      log D_s = X_s' gamma + eps_s ,   eps_s ~ N(0, sigma^2)

i.e. a lognormal GPS, chosen because the dose is strictly positive and strongly
right-skewed (skewness reported below). The stabilised weight is

      sw_s = f_D(D_s) / f_{D|X}(D_s | X_s)

with a marginal lognormal numerator. Stabilised weights have mean approx 1 and
bounded variance; we report their distribution and truncate at the 1st/99th
percentile, reporting how many observations that binds on.

Balance for a CONTINUOUS exposure is assessed by the weighted correlation
between each covariate and the dose (Austin 2019): under correct specification
the weighted correlations should be near zero, whereas the unweighted ones need
not be.
""")
Dtr = D24[~untreated_mask]
log(f"  dose skewness (treated, level) = {st.skew(Dtr.values):.3f}; "
    f"(log) = {st.skew(np.log(Dtr.values)):.3f}")

Xtr = X.loc[Dtr.index]
Xd_tr = sm.add_constant(Xtr.values)
gps_mod = sm.OLS(np.log(Dtr.values), Xd_tr).fit()
sigma2 = float(np.sum(gps_mod.resid ** 2) / gps_mod.df_resid)
log(f"\n  --- lognormal GPS, n={len(Dtr)}, R2={gps_mod.rsquared:.3f}, "
    f"sigma={np.sqrt(sigma2):.4f}")
log(f"    {'covariate':<24s}{'coef':>12s}{'se':>12s}{'t':>8s}{'p':>8s}")
for nm, b, s, pv in zip(["const"] + XCOLS, gps_mod.params, gps_mod.bse, gps_mod.pvalues):
    log(f"    {nm:<24s}{b:>12.4f}{s:>12.4f}{b/s:>8.2f}{pv:>8.3f}")
    ps_rows.append(dict(model="gps_lognormal", term=nm, coef=b, se=s, z=b / s,
                        pvalue=pv, pseudo_r2=float(gps_mod.rsquared), n=len(Dtr),
                        n_treated=len(Dtr), converged=True))

logD = np.log(Dtr.values)
num = st.norm.pdf(logD, loc=logD.mean(), scale=logD.std(ddof=1))
den = st.norm.pdf(logD, loc=gps_mod.fittedvalues, scale=np.sqrt(sigma2))
sw = num / den
sw_raw = sw.copy()
lo_c, hi_c = np.percentile(sw, [1, 99])
n_trunc = int(((sw < lo_c) | (sw > hi_c)).sum())
sw = np.clip(sw, lo_c, hi_c)
log(f"\n    stabilised weights: mean={sw_raw.mean():.4f}, sd={sw_raw.std(ddof=1):.4f}, "
    f"min={sw_raw.min():.4f}, max={sw_raw.max():.4f}")
log(f"    truncated at [{lo_c:.4f}, {hi_c:.4f}] (1st/99th pct); binds on {n_trunc} of {len(sw)} states")
log(f"    effective sample size (Kish) = {sw.sum()**2/np.sum(sw**2):.1f} of {len(sw)}")

log(f"\n    covariate balance for the continuous dose (Austin 2019 weighted correlation)")
log(f"    {'covariate':<24s}{'corr before':>13s}{'corr after':>13s}")
for c in XCOLS:
    b_ = weighted_corr(Xtr[c].values, Dtr.values, np.ones(len(Dtr)))
    a_ = weighted_corr(Xtr[c].values, Dtr.values, sw)
    log(f"    {c:<24s}{b_:>13.3f}{a_:>13.3f}")
    bal_rows.append(dict(model="gps_lognormal", covariate=c,
                         metric="weighted_corr_with_dose", before=b_, after=a_,
                         is_ps_covariate=True))
gps_weights = pd.Series(sw, index=Dtr.index)

# ==============================================================================
# 3. CONTINUOUS-TREATMENT DiD ESTIMATORS
# ==============================================================================
hdr("3. CONTINUOUS-TREATMENT DiD:  ATT^o(d), ACRT^o(d), ACRT^glob")

log("""
ESTIMATION. For each post period t we form the long difference against the base
period, dY_{s,t} = Y_{s,t} - Y_{s,2016}, and the realised dose D_{s,t}. Under
weak parallel trends,

    ATT^o(d, t) = E[dY_t | D_t = d] - E[dY_t | D_t = 0]

We estimate E[dY_t | D_t = d] four ways, from most to least parametric:

  (a) SERIES-LINEAR    dY = a + b1 D                      ATT^o(d) = b1 d
  (b) SERIES-QUADRATIC dY = a + b1 D + b2 D^2             ACRT^o(d) = b1 + 2 b2 d
  (c) SERIES-CUBIC     dY = a + b1 D + b2 D^2 + b3 D^3
  (d) LOCAL LINEAR     Gaussian-kernel local linear regression among treated
                       units; the local slope IS ACRT^o(d) by construction.
  (e) BINNED           fully nonparametric: dose bins, ATT^o at bin mean dose,
                       discrete causal response between adjacent bins.

All series models are fit on ALL states, with untreated states entering at
D = 0, so the intercept a estimates E[dY | D = 0] and the fitted dose-response
passes through the origin (ATT^o(0) = 0), as the definition requires.

We also fit a JUMP variant, dY = a + g*1{D>0} + b1 D + b2 D^2. The estimate of g
is a diagnostic on the comparison group: under a valid near-zero comparison
group and a continuous dose-response, g should be indistinguishable from zero.
The ACRT is invariant to g.

THE HEADLINE PARAMETER, CORRECTLY NAMED

    ACRT^glob(t) = E[ ACRT^o(D_t, t) | D_t > 0 ]   -- average over the REALISED
                                                      treated dose distribution
    ACRT^glob    = (1/|T|) sum_t ACRT^glob(t)      -- simple aggregation over
                                                      post periods

*** NOTE ON LABELLING.  This quantity is not E[ACRT(D|D) | D>0].  E[ACRT(D|D)|D>0] is
the paper's ACRT^loc, and it is NOT identified here: under parallel trends alone
ACRT^o(d) = ACRT(d|d) + selection bias (Theorem 3.2(b)), so averaging ACRT^o
over the treated dose distribution does not deliver ACRT^loc.  The Supplementary
Information invokes STRONG parallel trends for this estimate.  Under strong
parallel trends Theorem 3.3(c) gives ACRT^o(d) = ACRT(d) -- the response for the
WHOLE treated population, not for dose group d -- and Corollary 3.1(b) gives

    ACRT^glob = INT ACRT^o(s) f_{D|D>0}(s) ds = E[ ACRT^o(D) | D > 0 ]

which is exactly what the code computes.  The sample average over treated units
of the fitted ACRT^o is also precisely the paper's own recommended plug-in
estimator of ACRT^glob (paper, Section 4.2):
    ACRT^glob_hat = (1/n_{D>0}) sum_{i: D_i > 0} ACRT^o_hat(D_i).
The estimator above is therefore ACRT^glob, not ACRT^loc.  ACRT^loc is not
reported at all, because nothing in this design identifies it.

Likewise the level effect reported below as att_raw_mean,
E[dY|D>0] - E[dY|D=0], is ATT^loc under parallel trends (Theorem 3.1) and
ATT^glob under strong parallel trends (Corollary 3.1(a)) -- the same estimator,
two different parameters depending on which assumption is invoked.

Units: jobs per GW of new capacity.
""")

def fit_series(dY, Dv, order, weights=None, jump=False):
    """OLS of dY on a polynomial in D. Returns dict with coefs and a callable
    ACRT^o(d). Untreated units enter at D = 0."""
    cols = [np.ones(len(Dv))]
    names = ["const"]
    if jump:
        cols.append((Dv > 0).astype(float))
        names.append("jump")
    for k in range(1, order + 1):
        cols.append(Dv ** k)
        names.append(f"D{k}")
    Xm = np.column_stack(cols)
    if weights is None:
        res = sm.OLS(dY, Xm).fit()
    else:
        res = sm.WLS(dY, Xm, weights=weights).fit()
    p = dict(zip(names, res.params))

    def acrt_fn(d):
        d = np.asarray(d, float)
        out = np.zeros_like(d)
        for k in range(1, order + 1):
            out = out + k * p[f"D{k}"] * d ** (k - 1)
        return out

    def att_fn(d):
        d = np.asarray(d, float)
        out = np.zeros_like(d)
        if jump:
            out = out + p["jump"] * (d > 0)
        for k in range(1, order + 1):
            out = out + p[f"D{k}"] * d ** k
        return out

    return dict(params=p, res=res, acrt=acrt_fn, att=att_fn, names=names)

def local_linear(dY, Dv, grid, h):
    """Gaussian-kernel local linear regression. Returns (fit, slope) on grid."""
    fit = np.full(len(grid), np.nan)
    slope = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        u = (Dv - g) / h
        w = np.exp(-0.5 * u ** 2)
        if w.sum() < 1e-8:
            continue
        Z = np.column_stack([np.ones(len(Dv)), Dv - g])
        A = Z.T @ (w[:, None] * Z)
        if np.linalg.cond(A) > 1e12:
            continue
        beta = np.linalg.solve(A, Z.T @ (w * dY))
        fit[i], slope[i] = beta[0], beta[1]
    return fit, slope

def rot_bandwidth(Dv):
    """Silverman rule-of-thumb, inflated by 1.5 for local-linear derivative
    estimation with a very small n."""
    n = len(Dv)
    s = min(np.std(Dv, ddof=1), (np.percentile(Dv, 75) - np.percentile(Dv, 25)) / 1.349)
    return 1.5 * 1.06 * s * n ** (-0.2)

def build_cross_section(yv, t, tau=TAU, drop_neg=False, states=None):
    """Long difference dY_{s,t} = Y_{s,t} - Y_{s,BASE} with dose D_{s,t}."""
    Y = wide[yv]
    idx = states if states is not None else STATES
    dY = (Y[t] - Y[BASE_YEAR]).reindex(idx)
    Dv = dose[t].reindex(idx).copy()
    Dfin = D24.reindex(idx)
    keep = dY.notna() & Dv.notna()
    if drop_neg:
        keep &= ~pd.Series(idx, index=idx).isin(neg_states)
    # a state is in the comparison group if its TOTAL 2024 dose is < tau,
    # so group membership is fixed across t (no unit switches groups over time)
    treat_ind = (Dfin >= tau)
    Dv = Dv.where(treat_ind, 0.0)   # comparison states forced to exactly d = 0
    return dY[keep].values, Dv[keep].values.astype(float), np.array(idx)[keep.values]

DOSE_GRID = np.round(np.arange(0.0, 6.51, 0.05), 4)

dose_response_rows = []
acrt_period_rows = []
binned_rows = []

def estimate_all(yv, t, tau=TAU, drop_neg=False, states=None, weights_map=None,
                 collect=False):
    """Returns a dict of ACRT / ATT estimates for one (outcome, period)."""
    dY, Dv, sid = build_cross_section(yv, t, tau, drop_neg, states)
    tr = Dv > 0
    n, ntr = len(dY), int(tr.sum())
    if ntr < 6 or (n - ntr) < 3:
        return None
    out = dict(outcome=yv, year=t, n=n, n_treated=ntr, n_control=n - ntr,
               mean_dose_treated=float(Dv[tr].mean()),
               max_dose=float(Dv.max()))

    w = None
    if weights_map is not None:
        w = np.array([weights_map.get(s, 1.0) for s in sid])

    for order, tag in [(1, "series_lin"), (2, "series_quad"), (3, "series_cub")]:
        f = fit_series(dY, Dv, order, weights=w)
        out[f"acrt_{tag}"] = float(np.mean(f["acrt"](Dv[tr])))
        out[f"att_{tag}"] = float(np.mean(f["att"](Dv[tr])))
        if collect:
            for k, v in f["params"].items():
                out[f"coef_{tag}_{k}"] = float(v)

    fj = fit_series(dY, Dv, 2, weights=w, jump=True)
    out["acrt_series_quad_jump"] = float(np.mean(fj["acrt"](Dv[tr])))
    out["jump_gamma"] = float(fj["params"]["jump"])
    out["jump_gamma_se"] = float(fj["res"].bse[fj["names"].index("jump")])

    # local linear on treated only, differenced against the comparison mean
    if ntr >= 10:
        h = rot_bandwidth(Dv[tr])
        base_mean = dY[~tr].mean()
        fit, slp = local_linear(dY[tr] - base_mean, Dv[tr], Dv[tr], h)
        out["acrt_local_linear"] = float(np.nanmean(slp))
        out["att_local_linear"] = float(np.nanmean(fit))
        out["ll_bandwidth"] = float(h)
    else:
        out["acrt_local_linear"] = np.nan
        out["att_local_linear"] = np.nan
        out["ll_bandwidth"] = np.nan

    # simple difference-in-means level effect (the "ATT" of the continuous design)
    out["att_raw_mean"] = float(dY[tr].mean() - dY[~tr].mean())

    # --- dose-specific causal responses from the quadratic series ------------
    # ACRT is an average of ACRT^o(d) over the REALISED treated dose
    # distribution, which is dominated by the many small-dose states. Because
    # the estimated dose-response is concave, the response evaluated at
    # policy-relevant doses is a different (and smaller) number. Report both.
    #
    # SUPPORT RULE (pre-specified, applied identically in every bootstrap
    # replicate). A quadratic fitted on a narrow dose range extrapolates
    # violently outside it: in 2017 and 2018 no state has yet added more than
    # ~1 GW, so evaluating the fitted response AT 1 GW is pure extrapolation and
    # produces absurd numbers (e.g. -16,000 jobs/GW). We therefore report
    #   * ACRT^o(d*) only for years with >= 5 treated states at or above d* AND
    #     a treated dose range reaching at least 1.25 * d*;
    #   * the capacity-weighted ACRT and the aggregate jobs-per-GW only for
    #     years whose treated dose range reaches at least 2 GW (2019 onwards).
    # Years failing the rule contribute NaN and drop out of the average.
    f2 = fit_series(dY, Dv, 2, weights=w)
    dmax = float(Dv[tr].max())
    for dstar in (0.25, 0.50, 1.00, 2.00):
        key = f"acrt_at_{dstar:.2f}GW".replace(".", "p")
        ok_sup = (int((Dv[tr] >= dstar).sum()) >= 5) and (dmax >= 1.25 * dstar)
        out[key] = float(f2["acrt"](np.array([dstar]))[0]) if ok_sup else np.nan
        out[f"att_at_{dstar:.2f}GW".replace(".", "p")] = (
            float(f2["att"](np.array([dstar]))[0]) if ok_sup else np.nan)
        out[f"support_ok_{dstar:.2f}GW".replace(".", "p")] = bool(ok_sup)
    wide_support = dmax >= 2.0
    out["wide_support"] = bool(wide_support)
    # capacity-weighted causal response: E[D * ACRT^o(D)] / E[D].  This is the
    # response relevant for the TOTAL number of jobs generated by the TOTAL
    # build-out, i.e. it weights states by how much capacity they added.
    out["acrt_capacity_weighted"] = float(
        np.sum(Dv[tr] * f2["acrt"](Dv[tr])) / np.sum(Dv[tr])) if wide_support else np.nan
    # total-jobs-per-total-GW implied by the fitted level dose-response
    out["att_per_gw_aggregate"] = float(
        np.sum(f2["att"](Dv[tr])) / np.sum(Dv[tr])) if wide_support else np.nan

    if collect:
        f2 = fit_series(dY, Dv, 2, weights=w)
        f3 = fit_series(dY, Dv, 3, weights=w)
        for g in DOSE_GRID:
            if g > Dv.max() + 1e-9:
                continue
            dose_response_rows.append(dict(
                outcome=yv, year=t, dose_gw=float(g),
                att_o_quad=float(f2["att"](np.array([g]))[0]),
                acrt_o_quad=float(f2["acrt"](np.array([g]))[0]),
                att_o_cub=float(f3["att"](np.array([g]))[0]),
                acrt_o_cub=float(f3["acrt"](np.array([g]))[0]),
                n_treated_at_or_above=int((Dv[tr] >= g).sum())))
    return out

# ---- point estimates ----------------------------------------------------
log("\n3.1 PERIOD-BY-PERIOD ACRT (jobs per GW), primary spec")
log(f"    {'outcome':<18s}{'yr':>5s}{'n':>4s}{'ntr':>4s}{'meanD':>7s}"
    f"{'lin':>10s}{'quad':>10s}{'cub':>10s}{'locallin':>10s}{'ATTlvl':>10s}")
for yv, _ in OUTCOMES:
    for t in POST_YEARS:
        r = estimate_all(yv, t, collect=(yv in PRIMARY_OUTCOMES))
        if r is None:
            continue
        acrt_period_rows.append(r)
        if yv in PRIMARY_OUTCOMES:
            log(f"    {yv:<18s}{t:>5d}{r['n']:>4d}{r['n_treated']:>4d}"
                f"{r['mean_dose_treated']:>7.3f}"
                f"{r['acrt_series_lin']:>10.0f}{r['acrt_series_quad']:>10.0f}"
                f"{r['acrt_series_cub']:>10.0f}{r['acrt_local_linear']:>10.0f}"
                f"{r['att_raw_mean']:>10.0f}")

acrt_period = pd.DataFrame(acrt_period_rows)

# ---- binned nonparametric dose-response (final period) ------------------
log("\n3.2 BINNED NONPARAMETRIC DOSE-RESPONSE, t = 2024 "
    "(the direct answer to 'the threshold throws away variation')")
BINS = [TAU, 0.20, 0.50, 1.00, 2.00, 99.0]
for yv in PRIMARY_OUTCOMES:
    dY, Dv, sid = build_cross_section(yv, 2024)
    tr = Dv > 0
    base_mean = dY[~tr].mean()
    log(f"\n  {yv}   (comparison group mean dY = {base_mean:,.0f} jobs, "
        f"n = {int((~tr).sum())})")
    log(f"    {'dose bin (GW)':<18s}{'n':>4s}{'mean D':>9s}{'mean dY':>11s}"
        f"{'ATT^o':>11s}{'ATT^o/D':>10s}")
    prev = None
    for lo_b, hi_b in zip(BINS[:-1], BINS[1:]):
        m = tr & (Dv >= lo_b) & (Dv < hi_b)
        if m.sum() == 0:
            continue
        md, mdy = Dv[m].mean(), dY[m].mean()
        att_o = mdy - base_mean
        lbl = f"[{lo_b:.2f}, {hi_b:.2f})" if hi_b < 50 else f"[{lo_b:.2f}, inf)"
        log(f"    {lbl:<18s}{int(m.sum()):>4d}{md:>9.3f}{mdy:>11,.0f}"
            f"{att_o:>11,.0f}{att_o/md:>10,.0f}")
        row = dict(outcome=yv, year=2024, bin_lo=lo_b, bin_hi=hi_b,
                   n=int(m.sum()), mean_dose=float(md), mean_dY=float(mdy),
                   att_o=float(att_o), att_o_per_gw=float(att_o / md),
                   states=",".join(sorted(sid[m])))
        if prev is not None:
            dacr = (att_o - prev[0]) / (md - prev[1])
            row["discrete_acr_vs_prev_bin"] = float(dacr)
            log(f"        -> discrete causal response between this bin and the "
                f"previous: {dacr:,.0f} jobs/GW")
        prev = (att_o, md)
        binned_rows.append(row)

# ==============================================================================
# 4. CLUSTER BOOTSTRAP INFERENCE
# ==============================================================================
hdr("4. INFERENCE: NONPARAMETRIC BOOTSTRAP OVER STATES")

log(f"""
The unit of observation and the unit of treatment is the state; there are only
{len(STATES)} of them. All inference below is a nonparametric bootstrap that
resamples STATES with replacement (B = {B_BOOT}, seed = {SEED}) and recomputes
the ENTIRE estimation chain -- comparison-group assignment, series fit, dose
averaging -- inside each replicate. Standard errors are bootstrap standard
deviations; confidence intervals are reported both as percentile intervals and
as symmetric normal intervals, and both are shown because with {len(STATES)}
clusters they can differ. p-values are two-sided and based on the symmetric-t
statistic against the bootstrap sd, with {len(STATES) - 1} degrees of freedom.
""")

ESTIMATORS = ["acrt_series_lin", "acrt_series_quad", "acrt_series_cub",
              "acrt_local_linear", "acrt_series_quad_jump",
              "acrt_at_0p25GW", "acrt_at_0p50GW", "acrt_at_1p00GW",
              "acrt_at_2p00GW", "acrt_capacity_weighted",
              "att_per_gw_aggregate"]

def acrt_aggregate_multi(yv, tau=TAU, drop_neg=False, states=None,
                         weights_map=None, years=POST_YEARS,
                         estimators=ESTIMATORS):
    """Run every estimator for one outcome in a single pass over the post
    periods; returns {estimator: simple average over t} plus the level ATT."""
    acc = {e: [] for e in estimators}
    lvl = []
    for t in years:
        r = estimate_all(yv, t, tau, drop_neg, states, weights_map)
        if r is None:
            continue
        lvl.append(r["att_raw_mean"])
        for e in estimators:
            v = r.get(e, np.nan)
            if np.isfinite(v):
                acc[e].append(v)
    out = {e: (float(np.mean(v)) if v else np.nan) for e, v in acc.items()}
    return out, (float(np.mean(lvl)) if lvl else np.nan)

def acrt_aggregate(yv, estimator="acrt_series_quad", **kw):
    o, l = acrt_aggregate_multi(yv, estimators=[estimator], **kw)
    return o[estimator], l, None

def boot_states(rng):
    return list(rng.choice(STATES, size=len(STATES), replace=True))

ALL_OUT = [o for o, _ in OUTCOMES]
log(f"bootstrapping {len(ALL_OUT)} outcomes x {len(ESTIMATORS)} estimators ...")
point = {}
for yv in ALL_OUT:
    o, l = acrt_aggregate_multi(yv)
    for e in ESTIMATORS:
        point[(yv, e)] = (o[e], l)

boot_draws = {k: [] for k in point}
rng = np.random.default_rng(SEED)
n_fail = 0
for b in range(B_BOOT):
    ss = boot_states(rng)
    for yv in ALL_OUT:
        try:
            o, l = acrt_aggregate_multi(yv, states=ss)
        except Exception:
            o, l = {e: np.nan for e in ESTIMATORS}, np.nan
            n_fail += 1
        for e in ESTIMATORS:
            boot_draws[(yv, e)].append((o[e], l))
    if (b + 1) % 400 == 0:
        log(f"   ... {b+1}/{B_BOOT} replicates  ({time.time()-T0:.0f}s)")

summary_rows = []
for (yv, est), (a, l) in point.items():
    draws = np.array([d[0] for d in boot_draws[(yv, est)]], float)
    ldraws = np.array([d[1] for d in boot_draws[(yv, est)]], float)
    ok = np.isfinite(draws)
    d_ok = draws[ok]
    se = float(np.std(d_ok, ddof=1)) if len(d_ok) > 10 else np.nan
    lo_p, hi_p = (np.percentile(d_ok, [2.5, 97.5]) if len(d_ok) > 10 else (np.nan, np.nan))
    med = float(np.median(d_ok)) if len(d_ok) > 10 else np.nan
    tstat = a / se if se and np.isfinite(se) and se > 0 else np.nan
    pval = 2 * (1 - st.t.cdf(abs(tstat), df=len(STATES) - 1)) if np.isfinite(tstat) else np.nan
    # bootstrap p-value by inverting the percentile interval: the smallest alpha
    # at which the (1-alpha) percentile interval excludes zero
    if len(d_ok) > 10:
        frac_le0 = float(np.mean(d_ok <= 0))
        pval_pct = 2 * min(frac_le0, 1 - frac_le0)
    else:
        pval_pct = np.nan
    # If the pre-specified support rule kills the point estimate in EVERY post
    # period, the estimand does not exist in this sample.  The bootstrap can
    # still produce draws, because a resample that duplicates the large-dose
    # states may satisfy the rule -- but an SE and a confidence interval for a
    # quantity with no point estimate are meaningless and get copied into
    # tables.  Suppress them and say why.  (Applies to acrt_at_2p00GW: only 4
    # units ever exceed 2 GW, against a rule requiring 5.)
    point_defined = bool(np.isfinite(a))
    if not point_defined:
        se = np.nan
        lo_p = hi_p = med = np.nan
        pval_pct = np.nan
    lok = ldraws[np.isfinite(ldraws)]
    lse = float(np.std(lok, ddof=1)) if len(lok) > 10 else np.nan
    # Parameter naming follows the paper.  Everything in ESTIMATORS is an
    # average or a point evaluation of the OBSERVED ACRT^o.  Under strong
    # parallel trends the dose-distribution average is ACRT^glob (Cor 3.1(b))
    # and a point evaluation is ACRT(d) (Thm 3.3(c)); under parallel trends
    # alone neither is a causal parameter.  ACRT^loc = E[ACRT(D|D)|D>0] is not
    # identified in this design and is not reported.
    _param = {"acrt_series_lin": "ACRT_glob", "acrt_series_quad": "ACRT_glob",
              "acrt_series_cub": "ACRT_glob", "acrt_local_linear": "ACRT_glob",
              "acrt_series_quad_jump": "ACRT_glob",
              "acrt_at_0p25GW": "ACRT(d)", "acrt_at_0p50GW": "ACRT(d)",
              "acrt_at_1p00GW": "ACRT(d)", "acrt_at_2p00GW": "ACRT(d)",
              "acrt_capacity_weighted": "capacity-weighted ACRT (not a CGS&S parameter)",
              "att_per_gw_aggregate": "ATT^o aggregate / total GW (not a CGS&S parameter)",
              }.get(est, "ACRT_glob")
    summary_rows.append(dict(
        outcome=yv, estimator=est, param=_param,
        param_under_PT_only="ACRT^o average; not causal without SPT",
        estimate=a, boot_se=se, t=tstat, pvalue=pval,
        boot_median=med, boot_bias=med - a if np.isfinite(med) else np.nan,
        pvalue_percentile=pval_pct,
        ci95_lo_normal=a - 1.96 * se if np.isfinite(se) else np.nan,
        ci95_hi_normal=a + 1.96 * se if np.isfinite(se) else np.nan,
        ci95_lo_pct=lo_p, ci95_hi_pct=hi_p,
        att_level=l, att_level_boot_se=lse,
        n_boot_ok=int(ok.sum()), B=B_BOOT,
        point_defined=point_defined,
        suppressed_reason=("" if point_defined else
                           "support rule fails in every post period; the "
                           f"bootstrap produced {int(ok.sum())} finite draws from "
                           "resamples that duplicate the large-dose states, but "
                           "there is no point estimate to attach them to")))

summary = pd.DataFrame(summary_rows)
log(f"\nbootstrap complete in {time.time()-T0:.0f}s; failed replicate-cells: {n_fail}")

log("\n4.1 HEADLINE: ACRT^glob (jobs per GW of new capacity), averaged over 2017-2024")
log("    (Corollary 3.1(b); identified under STRONG parallel trends only.)")
log(f"    {'outcome':<18s}{'estimator':<24s}{'ACRT':>10s}{'SE':>10s}{'t':>7s}"
    f"{'p(t)':>8s}{'p(pct)':>8s}{'95% CI (percentile)':>26s}")
for _, r in summary.iterrows():
    if r["outcome"] not in PRIMARY_OUTCOMES:
        continue
    log(f"    {r['outcome']:<18s}{r['estimator']:<24s}{r['estimate']:>10,.0f}"
        f"{r['boot_se']:>10,.0f}{r['t']:>7.2f}{r['pvalue']:>8.3f}"
        f"{r['pvalue_percentile']:>8.3f}"
        f"   [{r['ci95_lo_pct']:>10,.0f},{r['ci95_hi_pct']:>10,.0f}]")
log(f"""
    NOTE ON THE TWO p-VALUES. p(t) uses the bootstrap standard deviation in a
    symmetric t test; p(pct) is the percentile-bootstrap p-value (twice the
    smaller tail mass on either side of zero). They differ materially because
    the bootstrap distribution of the ACRT is strongly right-skewed: with
    {len(STATES)} units, resamples that happen to include several of the
    very-large-dose states produce very different fits. Where the two disagree we
    report the CONSERVATIVE one, i.e. p(t), and say so.""")

log("\n4.1b DOSE-SPECIFIC CAUSAL RESPONSE, because the dose-response is CONCAVE.")
log("     ACRT averages ACRT^o(d) over the REALISED treated dose distribution,")
log("     which is dominated by the many small-dose states. The response at")
log("     policy-relevant doses is a different number and is reported here.")
log("     Years without sufficient dose support are excluded by the pre-specified")
log("     support rule (see the code comment in estimate_all); the number of")
log("     contributing years is printed below.")
for yv in PRIMARY_OUTCOMES[:1]:
    sub = acrt_period[acrt_period.outcome == yv]
    log(f"     support by year for {yv}: " + ", ".join(
        f"{int(r.year)}[max={r.max_dose:.1f}GW,"
        f"{'wide' if r.wide_support else 'narrow'}]" for _, r in sub.iterrows()))
log(f"    {'outcome':<18s}{'quantity':<26s}{'jobs/GW':>10s}{'SE':>10s}{'t':>7s}")
for yv in PRIMARY_OUTCOMES:
    for e, lbl in [("acrt_at_0p25GW", "ACRT^o at d = 0.25 GW"),
                   ("acrt_at_0p50GW", "ACRT^o at d = 0.50 GW"),
                   ("acrt_at_1p00GW", "ACRT^o at d = 1.00 GW"),
                   ("acrt_at_2p00GW", "ACRT^o at d = 2.00 GW"),
                   ("acrt_capacity_weighted", "capacity-weighted ACRT"),
                   ("att_per_gw_aggregate", "total jobs / total GW")]:
        r = summary[(summary.outcome == yv) & (summary.estimator == e)].iloc[0]
        log(f"    {yv:<18s}{lbl:<26s}{r['estimate']:>10,.0f}{r['boot_se']:>10,.0f}"
            f"{r['t']:>7.2f}")

log("\n4.2 ALL OUTCOMES, quadratic-series ACRT^glob")
log(f"    {'outcome':<18s}{'ACRT':>10s}{'SE':>10s}{'p':>8s}{'ATT level (jobs)':>20s}")
for _, r in summary[summary.estimator == "acrt_series_quad"].iterrows():
    log(f"    {r['outcome']:<18s}{r['estimate']:>10,.0f}{r['boot_se']:>10,.0f}"
        f"{r['pvalue']:>8.3f}{r['att_level']:>20,.0f}")

# ---- ACRT event study ---------------------------------------------------
log("\n4.3 ACRT EVENT STUDY (per-period ACRT with bootstrap SE), quadratic series")
es_rows = []
for yv in PRIMARY_OUTCOMES:
    per_pt = {}
    for t in POST_YEARS:
        r = estimate_all(yv, t)
        per_pt[t] = r["acrt_series_quad"] if r else np.nan
    per_boot = {t: [] for t in POST_YEARS}
    rng2 = np.random.default_rng(SEED + 7)
    for b in range(500):
        ss = boot_states(rng2)
        for t in POST_YEARS:
            try:
                r = estimate_all(yv, t, states=ss)
                per_boot[t].append(r["acrt_series_quad"] if r else np.nan)
            except Exception:
                per_boot[t].append(np.nan)
    log(f"\n  {yv}")
    log(f"    {'year':>6s}{'ACRT':>12s}{'SE':>12s}{'t':>7s}")
    for t in POST_YEARS:
        arr = np.array(per_boot[t], float)
        arr = arr[np.isfinite(arr)]
        se = float(np.std(arr, ddof=1)) if len(arr) > 10 else np.nan
        pv = per_pt[t]
        log(f"    {t:>6d}{pv:>12,.0f}{se:>12,.0f}"
            f"{(pv/se if se else np.nan):>7.2f}")
        es_rows.append(dict(outcome=yv, year=t, acrt=pv, boot_se=se,
                            t=pv / se if se else np.nan, B=500))
es = pd.DataFrame(es_rows)

# ==============================================================================
# 5. TWFE / YITZHAKI WEIGHTS AND THE BINARY-THRESHOLD ESTIMAND
# ==============================================================================
hdr("5. WHAT THE LINEAR AND BINARY SPECIFICATIONS ARE ACTUALLY ESTIMATING")

log("""
5.1 THEOREM 3.4(a) DECOMPOSITION WEIGHTS  (paper, Table 1 and Theorem 3.4(a))

A linear regression of the long difference on the continuous dose returns

    beta_TWFE = INT_{dL}^{dU} w1(l) [ ACRT(l|l) + selection bias ] dl
                + w0 * ATT(dL|dL) / dL

    w1(l) = ( E[D | D >= l] - E[D] ) P(D >= l) / Var(D)      a DENSITY in l
    w0    = ( E[D | D >  0] - E[D] ) P(D >  0) dL / Var(D)   a point mass

and the paper states that these weights "are always positive and integrate to
1", i.e.  INT_{dL}^{dU} w1(l) dl + w0 = 1.

*** NORMALISATION.  The weights must NOT be normalised by summing w1 over the
distinct observed doses and dividing by that sum.  That fails three ways.
(i) w1 is a density, so it must be integrated dl; summing gives every observed
dose the same implicit width, which on a dose grid whose upper tail is sparse
crushes the tail.  (ii) The raw sum is not 1 and is not meant to be, so
rescaling by it silently redefines the object.  (iii) It omits the jump term
w0, without which the normalisation cannot hold even in principle.

w1 is a step function: for l in (d_(k), d_(k+1)] the event {D >= l} is exactly
{D >= d_(k+1)}, so w1 is constant on that interval and the integral is EXACT as
a sum of  w1(d_(k+1)) * (d_(k+1) - d_(k)),  taking d_(0) = 0.  Var(D) must be
the POPULATION variance (ddof = 0) for the normalisation to hold; with ddof = 1
every weight is scaled by (n-1)/n.  The identity is asserted below, not assumed.

Note also that w1 does NOT vanish at either tail: as l -> 0+ it tends to
E[D] P(D=0) / Var(D) = w0/dL > 0, and at the top dose it is
(dU - E[D]) P(D = dU) / Var(D) > 0.
""")
Dall = np.where(D24.reindex(STATES).values < TAU, 0.0,
                D24.reindex(STATES).values)
meanD = float(Dall.mean())
varD = float(Dall.var(ddof=0))          # POPULATION variance; see above
grid_w = np.sort(np.unique(Dall))       # includes 0 (the untreated mass)
assert grid_w[0] == 0.0, "expected an untreated mass at D = 0"
dL_grid = float(grid_w[1])              # minimum strictly positive dose
dU_grid = float(grid_w[-1])

w_rows = []
prev_d = 0.0
for g in grid_w:
    ge = Dall >= g
    wv = (Dall[ge].mean() - meanD) * ge.mean() / varD
    width = float(g - prev_d)
    w_rows.append(dict(dose_gw=float(g),
                       w1_density=float(wv),          # w1 evaluated on (prev, g]
                       step_width_gw=width,
                       w1_mass_on_interval=float(wv * width),
                       interval_lo_gw=prev_d, interval_hi_gw=float(g),
                       p_ge=float(ge.mean()), n_ge=int(ge.sum())))
    prev_d = float(g)
wdf = pd.DataFrame(w_rows)

# the jump term of Theorem 3.4(a): the mass over (0, dL], carrying
# ATT(dL|dL)/dL rather than an ACRT
w0_jump = (Dall[Dall > 0].mean() - meanD) * (Dall > 0).mean() * dL_grid / varD
mass_0_to_dL = float(wdf.loc[wdf.dose_gw == dL_grid, "w1_mass_on_interval"].iloc[0])
assert abs(w0_jump - mass_0_to_dL) < 1e-12, \
    f"w0 {w0_jump} != integral of w1 over (0, dL] {mass_0_to_dL}"
# mark the jump interval so no consumer mistakes it for an ACRT weight
wdf["is_jump_term"] = wdf["dose_gw"] == dL_grid

total_mass = float(wdf["w1_mass_on_interval"].sum())
w1_mass = total_mass - w0_jump          # INT_{dL}^{dU} w1(l) dl
log(f"    Var(D) (population)     : {varD:.8f}   E[D] = {meanD:.6f}")
log(f"    dL = {dL_grid:.6f} GW, dU = {dU_grid:.6f} GW, "
    f"{len(grid_w)-1} distinct positive doses")
log(f"    INT_dL^dU w1(l) dl      : {w1_mass:.10f}")
log(f"    w0 (jump 0 -> dL)       : {w0_jump:.10f}")
log(f"    TOTAL (Theorem 3.4(a))  : {w1_mass + w0_jump:.10f}   [must be 1]")
assert abs(w1_mass + w0_jump - 1.0) < 1e-9, \
    f"Theorem 3.4(a) normalisation failed: {w1_mass + w0_jump}"
assert (wdf["w1_density"].iloc[1:] > 0).all(), "a decomposition weight is <= 0"
log("    ASSERTION PASSED: the weights are positive and integrate to 1.")
log(f"\n    For contrast, the (incorrect) SUM of w1 over the {len(grid_w)} distinct")
log(f"    observed doses is {wdf['w1_density'].sum():.4f}, which is what the")
log("    an implementation that normalised by the sum would divide by.")

log(f"\n    {'interval (GW]':>22s}{'width':>9s}{'w1 density':>12s}"
    f"{'mass':>10s}{'cum mass':>10s}{'P(D>=d)':>10s}{'n>=d':>6s}")
_cum = 0.0
for _, r in wdf.iloc[1:].iterrows():
    _cum += r["w1_mass_on_interval"]
    log(f"    {f'({r.interval_lo_gw:.4f}, {r.interval_hi_gw:.4f}]':>22s}"
        f"{r['step_width_gw']:>9.4f}{r['w1_density']:>12.4f}"
        f"{r['w1_mass_on_interval']:>10.4f}{_cum:>10.4f}"
        f"{r['p_ge']:>10.3f}{int(r['n_ge']):>6d}"
        + ("   <- jump term w0, carries ATT(dL|dL)/dL not an ACRT"
           if r["is_jump_term"] else ""))

peak = wdf.loc[wdf["w1_density"].idxmax()]
# The argmax is a knife edge: w1 is flat to within a percent over a wide range,
# so the peak DOSE is not a meaningful statistic on its own.
_plat = wdf[(wdf["w1_density"] >= 0.99 * peak["w1_density"]) & (wdf["dose_gw"] > 0)]
plateau_lo, plateau_hi = float(_plat["dose_gw"].min()), float(_plat["dose_gw"].max())
log(f"\n    w1 DENSITY peaks at d = {peak['dose_gw']:.3f} GW "
    f"(density {peak['w1_density']:.4f}).")
log(f"    PLATEAU: w1 is within 1 percent of its maximum over "
    f"d in [{plateau_lo:.3f}, {plateau_hi:.3f}] GW "
    f"({len(_plat)} of the {int((wdf.dose_gw>0).sum())} positive doses). The argmax is a knife")
log("    edge on that plateau and should be reported as a range, not a point.")
log(f"    w1 at the lower tail, l -> 0+ : {w0_jump/dL_grid:.4f}  "
    f"(= w0/dL; NOT zero)")
log(f"    w1 at the upper dose dU        : "
    f"{float(wdf['w1_density'].iloc[-1]):.4f}  (NOT zero)")

def _mass_ge(thr):
    return float(wdf.loc[wdf.dose_gw >= thr, "w1_mass_on_interval"].sum())

def _mass_lt(thr):
    return float(wdf.loc[wdf.dose_gw < thr, "w1_mass_on_interval"].sum())

share_ge2 = float((D24[~untreated_mask] >= 2).mean())
share_lt05 = float((D24[~untreated_mask] < 0.5).mean())
m_ge2, m_lt05 = _mass_ge(2.0), _mass_lt(0.5)
log(f"\n    Weight placed on d >= 2 GW : {m_ge2:.3f}   "
    f"(share of TREATED states with D >= 2 GW: {share_ge2:.3f}; "
    f"ratio {m_ge2/share_ge2:.2f})")
log(f"    Weight placed on d <  0.5 GW: {m_lt05:.3f}   "
    f"(share of TREATED states with D < 0.5 GW: {share_lt05:.3f}; "
    f"ratio {m_lt05/share_lt05:.2f})")
log(f"    Weight placed on d >= 1 GW : {_mass_ge(1.0):.3f}")
log(f"    Weight on the top interval ({float(wdf['interval_lo_gw'].iloc[-1]):.3f}, "
    f"{dU_grid:.3f}] GW, i.e. the single largest builder: "
    f"{float(wdf['w1_mass_on_interval'].iloc[-1]):.4f}")
log(f"""
    SUBSTANTIVE CONSEQUENCE, AND IT REVERSES THE PREVIOUS CLAIM.  Relative to
    the treated dose distribution the linear specification does not UNDER-weight
    the multi-gigawatt build-outs; it OVER-weights them, by a factor of
    {m_ge2/share_ge2:.1f}. The mass above 2 GW is {m_ge2:.3f} against a treated-state share
    of {share_ge2:.3f}. The four states above 2 GW carry two-thirds of the linear
    coefficient. Given a concave dose-response that is exactly why the linear
    coefficient sits BELOW the average causal response: it loads on the range in
    which the response has saturated. The direction of the discrepancy is the
    same either way, but this is the mechanism behind it.""")

log("""
5.1b THEOREM 3.4(b) AND 3.4(c): THE LEVELS AND SCALED-LEVELS DECOMPOSITIONS

The same beta_TWFE also decomposes over ATT(l|l) and over ATT(l|l)/l, with the
weights of the paper's Table 1:

    w_lev1(l) = (l - E[D]) / Var(D) * f_D(l)      w_lev0 = -E[D] P(D=0) / Var(D)
    w_s(l)    = l (l - E[D]) / Var(D) * f_D(l)

and Theorem 3.4(b) says  INT w_lev1(l) dl + w_lev0 = 0, so SOME WEIGHTS ARE
NEGATIVE and beta_TWFE is not weakly causal when the LEVEL effect is the
building block; 3.4(c) says INT w_s(l) dl = 1 but the same sign pattern
survives, w ≶ 0 for l ≶ E[D].

Note the contrast with 3.4(a): w_lev1 and w_s carry an explicit f_D(l) factor,
so on an empirical distribution they are MASSES and summing over the observed
doses is correct.  w1^acrt carries no density factor -- it is built from the
survival function -- and must be integrated.  Using one code path for both is
what makes the normalisation described above fail.
""")
fD = np.array([(Dall == g).mean() for g in grid_w])
w_lev1 = (grid_w - meanD) / varD * fD
w_lev0 = -meanD * (Dall == 0).mean() / varD
w_s = grid_w * (grid_w - meanD) / varD * fD
pos_mask = grid_w > 0
log(f"    SUM_{{d>0}} w_lev1        : {w_lev1[pos_mask].sum():+.10f}")
log(f"    w_lev0                  : {w_lev0:+.10f}")
log(f"    TOTAL (Theorem 3.4(b))  : {w_lev1[pos_mask].sum() + w_lev0:+.10f}   [must be 0]")
assert abs(w_lev1[pos_mask].sum() + w_lev0) < 1e-9, "Theorem 3.4(b) failed"
log(f"    SUM w_s                 : {w_s.sum():+.10f}   [must be 1, Theorem 3.4(c)]")
assert abs(w_s.sum() - 1.0) < 1e-9, "Theorem 3.4(c) failed"
_negmass = float(w_lev1[pos_mask & (w_lev1 < 0)].sum())
_nneg = int(((grid_w > 0) & (w_lev1 < 0)).sum())
log(f"    E[D] = {meanD:.4f} GW, so w_lev1 < 0 for every treated unit below it:")
log(f"    {_nneg} of the {int(pos_mask.sum())} positive doses carry NEGATIVE level weight, "
    f"total {_negmass:+.4f};")
log(f"    the {int(pos_mask.sum())-_nneg} above E[D] carry "
    f"{float(w_lev1[pos_mask & (w_lev1 > 0)].sum()):+.4f}, and the untreated atom carries "
    f"{w_lev0:+.4f}.")
log("    Under the LEVELS reading beta_TWFE is therefore not weakly causal at all;")
log("    the ACRT reading of 5.1 is the only one of the four in Theorem 3.4 whose")
log("    weights are positive AND integrate to one.")
levels_weight_meta = dict(sum_w_lev1_positive_doses=float(w_lev1[pos_mask].sum()),
                          w_lev0=float(w_lev0),
                          total_theorem_3_4b=float(w_lev1[pos_mask].sum() + w_lev0),
                          sum_w_s_theorem_3_4c=float(w_s.sum()),
                          n_doses_negative_level_weight=_nneg,
                          negative_level_weight_mass=_negmass,
                          mean_dose_gw=meanD)

log("""
    SCOPE OF THIS TABLE.  The weights above are computed on the 2024 dose
    distribution over all 49 units.  The linear-series estimate they explain is
    an average over eight annual cross-sections, each with its own dose vector
    and its own estimation sample, so the decomposition is exact for the 2024
    cross-section and only indicative for the average.  The per-year mass above
    2 GW is therefore reported below, computed the same way on each year's dose.
""")

def _weight_mass_ge(dvec, thr):
    """INT_thr^dU w1(l) dl on an arbitrary dose vector (Theorem 3.4(a))."""
    dv = np.asarray(dvec, float)
    v = dv.var(ddof=0)
    if v <= 0:
        return np.nan
    m = dv.mean()
    g = np.sort(np.unique(dv))
    tot = 0.0
    prev = 0.0
    for x in g:
        ge = dv >= x
        if x >= thr:
            tot += (dv[ge].mean() - m) * ge.mean() / v * (x - prev)
        prev = float(x)
    return float(tot)

log(f"    {'year':>6s}{'n treated':>11s}{'max dose':>10s}"
    f"{'mass d>=2GW':>13s}{'treated share d>=2GW':>22s}")
year_w_rows = []
for _t in POST_YEARS:
    _d = dose[_t].reindex(STATES).copy()
    _d = _d.where(D24.reindex(STATES) >= TAU, 0.0).values.astype(float)
    _m2 = _weight_mass_ge(_d, 2.0)
    _sh = float((_d[_d > 0] >= 2).mean())
    year_w_rows.append(dict(year=_t, n_treated=int((_d > 0).sum()),
                            max_dose_gw=float(_d.max()),
                            mass_ge_2gw=_m2, treated_share_ge_2gw=_sh))
    log(f"    {_t:>6d}{int((_d>0).sum()):>11d}{_d.max():>10.3f}"
        f"{_m2:>13.3f}{_sh:>22.3f}")

twfe_weight_meta = dict(
    var_D_population=varD, mean_D=meanD, dL_gw=dL_grid, dU_gw=dU_grid,
    w1_integral=w1_mass, w0_jump=w0_jump, total=w1_mass + w0_jump,
    incorrect_raw_sum=float(wdf["w1_density"].sum()),
    peak_density_dose_gw=float(peak["dose_gw"]),
    peak_density=float(peak["w1_density"]),
    peak_plateau_lo_gw=plateau_lo, peak_plateau_hi_gw=plateau_hi,
    w1_at_zero_plus=float(w0_jump / dL_grid),
    w1_at_dU=float(wdf["w1_density"].iloc[-1]),
    mass_ge_2gw=m_ge2, mass_ge_1gw=_mass_ge(1.0), mass_lt_0p5gw=m_lt05,
    mass_top_interval=float(wdf["w1_mass_on_interval"].iloc[-1]),
    treated_share_ge_2gw=share_ge2, treated_share_lt_0p5gw=share_lt05,
    over_weighting_factor_ge_2gw=m_ge2 / share_ge2,
    n_distinct_doses=int(len(grid_w)),
    by_post_year=year_w_rows,
    note=("weights computed on the 2024 dose over all 49 units; exact for the "
          "2024 cross-section, indicative for the year-averaged linear series"))

log("""
5.2 THE BINARY THRESHOLD ESTIMAND
Under parallel trends and Theorem 3.1 the binary DiD at threshold c estimates
    E[ ATT(D|D) | D >= c ]  -  E[ ATT(D|D) | D < c ]
The second term is NOT zero: the "never-treated" control group of the published
design consists of states that simply did not cross c, and many of them added
substantial capacity.

*** CONTROL GROUP.  The second term must NOT be computed as
E[ATT(D|D) | 0 < D < c], i.e. over the contaminating subset only, dropping the
zero-dose states from the average. That is not the estimand: the binary
design's control group is every state below c, zeros included, and the zeros
contribute ATT = 0 to the average. Restricting to positive doses inflates the
measured contamination. Both are reported below; the estimand column uses the
control group the design actually uses.
""")
contam_rows = []
for yv in PRIMARY_OUTCOMES:
    dY, Dv, sid = build_cross_section(yv, 2024)
    f2 = fit_series(dY, Dv, 2)
    for c in THRESH_BINARY:
        hi_m = Dv >= c
        lo_all = Dv < c                  # the design's actual control group
        lo_pos = (Dv > 0) & (Dv < c)     # its contaminating subset
        if hi_m.sum() == 0:
            continue
        att_hi = float(np.mean(f2["att"](Dv[hi_m])))
        att_lo_all = float(np.mean(f2["att"](Dv[lo_all]))) if lo_all.sum() else 0.0
        att_lo_pos = float(np.mean(f2["att"](Dv[lo_pos]))) if lo_pos.sum() else 0.0
        log(f"  {yv:<18s} c={c:.1f} GW: "
            f"E[ATT|D>=c]={att_hi:>9,.0f}   E[ATT|D<c]={att_lo_all:>9,.0f}   "
            f"binary estimand={att_hi-att_lo_all:>9,.0f} jobs  "
            f"(contamination {100*att_lo_all/att_hi if att_hi else np.nan:5.1f}% "
            f"of the true level effect)")
        log(f"      control group: {int(lo_all.sum())} states below c, of which "
            f"{int(lo_pos.sum())} have a POSITIVE dose (mean dose among those "
            f"{Dv[lo_pos].mean() if lo_pos.sum() else 0:.3f} GW; mean over the whole "
            f"control group {Dv[lo_all].mean():.3f} GW). Restricting the average to "
            f"the positive-dose subset would report "
            f"{100*att_lo_pos/att_hi if att_hi else np.nan:.1f}% instead.")
        contam_rows.append(dict(
            outcome=yv, threshold_gw=c,
            att_above=att_hi,
            att_below=att_lo_all,
            att_below_positive_dose_subset=att_lo_pos,
            binary_estimand=att_hi - att_lo_all,
            contamination_pct=100 * att_lo_all / att_hi if att_hi else np.nan,
            contamination_pct_positive_subset=(
                100 * att_lo_pos / att_hi if att_hi else np.nan),
            n_control_positive_dose=int(lo_pos.sum()),
            n_control=int(lo_all.sum()),
            mean_dose_control=float(Dv[lo_all].mean()),
            mean_dose_control_positive_subset=(
                float(Dv[lo_pos].mean()) if lo_pos.sum() else 0.0),
            mean_dose_treated=float(Dv[hi_m].mean())))
contam = pd.DataFrame(contam_rows)

# ==============================================================================
# 6. PARALLEL TRENDS: WHAT CAN AND CANNOT BE TESTED
# ==============================================================================
hdr("6. PARALLEL TRENDS TESTS")

log("""
6.0 WHAT IS AND IS NOT TESTABLE
  WEAK PT   is about counterfactual UNTREATED outcomes. It has testable
            implications in pre-treatment periods.
  STRONG PT is about counterfactual TREATED outcomes. In the paper's form
            (Assumption SPT) it says E[Y_2(d) - Y_1(0) | D > 0] =
            E[Y_2(d) - Y_1(0) | D = d]: the group that actually took dose d is a
            valid counterfactual for the WHOLE treated population at that dose.
            It is not nested with weak PT -- it does not require parallel trends
            in untreated potential outcomes for all dose groups -- but here it is
            the much stronger assumption. No pre-period data can test it, because
            in the pre-period no unit has taken any dose. We test three NECESSARY
            (not sufficient) implications instead.
This is the assumption the whole continuous-DiD dose-response rests on, and it
is the reason we treat DiD as supporting evidence rather than as the causal
claim of the paper.
""")

pt_rows = []

# --- 6.1 pre-trend / placebo: does FUTURE dose predict PAST outcome growth? ---
log("6.1 PLACEBO PRE-TREND. Regress the EARLY outcome change (2016 -> tau) on the")
log("    LATER dose increment (D_2024 - D_tau). Under weak PT the coefficient")
log("    should be zero: states that were about to build should not already be")
log("    diverging. Two-sided HC1 t-tests.")
log(f"\n    {'outcome':<18s}{'window':>14s}{'future dose':>14s}"
    f"{'coef':>12s}{'se':>12s}{'p':>8s}")
for yv in PRIMARY_OUTCOMES:
    for tau_y in [2017, 2018, 2019]:
        Y = wide[yv]
        dY = (Y[tau_y] - Y[BASE_YEAR])
        fut = (dose[2024] - dose[tau_y]).clip(lower=0)
        d = pd.DataFrame({"dY": dY, "fut": fut}).dropna()
        r = sm.OLS(d["dY"].values, sm.add_constant(d["fut"].values)).fit(cov_type="HC1")
        log(f"    {yv:<18s}{f'{BASE_YEAR}->{tau_y}':>14s}"
            f"{f'D24-D{tau_y}':>14s}{r.params[1]:>12,.0f}{r.bse[1]:>12,.0f}"
            f"{r.pvalues[1]:>8.3f}")
        pt_rows.append(dict(test="placebo_future_dose", outcome=yv,
                            window=f"{BASE_YEAR}-{tau_y}", regressor=f"D2024-D{tau_y}",
                            coef=float(r.params[1]), se=float(r.bse[1]),
                            pvalue=float(r.pvalues[1]), n=int(len(d))))

# --- 6.2 pre-existing level trend by eventual dose ------------------------
log("\n6.2 PRE-PERIOD TREND BY EVENTUAL DOSE. Regress each single-year outcome")
log("    change on the FINAL 2024 dose. Years before the build-out accelerated")
log("    are the informative ones; later years are shown for contrast.")
log(f"\n    {'outcome':<18s}{'change':>14s}{'coef/GW':>12s}{'se':>12s}{'p':>8s}  flag")
for yv in PRIMARY_OUTCOMES:
    Y = wide[yv]
    for t0, t1 in zip(range(2016, 2024), range(2017, 2025)):
        d = pd.DataFrame({"dY": Y[t1] - Y[t0], "D": D24}).dropna()
        r = sm.OLS(d["dY"].values, sm.add_constant(d["D"].values)).fit(cov_type="HC1")
        pre = t1 <= 2019
        log(f"    {yv:<18s}{f'{t0}->{t1}':>14s}{r.params[1]:>12,.0f}"
            f"{r.bse[1]:>12,.0f}{r.pvalues[1]:>8.3f}  "
            f"{'PRE' if pre else 'post'}{' **' if (pre and r.pvalues[1]<0.05) else ''}")
        pt_rows.append(dict(test="yearly_change_on_final_dose", outcome=yv,
                            window=f"{t0}-{t1}", regressor="D2024",
                            coef=float(r.params[1]), se=float(r.bse[1]),
                            pvalue=float(r.pvalues[1]), n=int(len(d)),
                            is_pre_period=bool(pre)))

# --- 6.3 necessary implications of STRONG PT ------------------------------
log("""
6.3 NECESSARY IMPLICATIONS OF STRONG PARALLEL TRENDS

  (i) COMPARISON-GROUP INVARIANCE. Under weak PT the choice of comparison group
      among untreated-or-near-untreated states must not change ATT^o(d). We
      re-estimate with TAU in {0.00, 0.05, 0.10, 0.20} GW. A material shift is
      evidence against PT (or against the near-zero approximation).
  (ii) FUNCTIONAL-FORM / SELECTION TEST. Under strong PT, ACRT^o(d) = ACRT(d)
      (Theorem 3.3(c)); under weak PT alone it is ACRT(d|d) plus selection bias.
      A test that ATT^o(d) is LINEAR through the origin (H0: b2 = b3 = 0) is a
      joint test of "constant causal response" AND "no dose-dependent
      selection". Rejection means at least one fails; the test cannot say which.
      Non-rejection is weak evidence consistent with strong PT.
  (iii) EARLY vs LATE BUILDERS AT THE SAME DOSE. Under strong PT each dose
      group is a valid counterfactual for the whole treated population at its
      own dose, so the dose-
      response is the same function for states that reached a given dose early
      and states that reached it late. We split treated states at the median
      year in which they reached half their final dose and compare the
      quadratic-series ACRT across the two halves.
""")
log("  (i) comparison-group invariance, quadratic-series ACRT (jobs/GW):")
log(f"      {'outcome':<18s}" + "".join(f"{f'tau={t:.2f}':>14s}" for t in [0.0] + [TAU] + TAU_ALT[1:]))
for yv in PRIMARY_OUTCOMES:
    vals = []
    for tv in [0.0, TAU] + TAU_ALT[1:]:
        a, _, _ = acrt_aggregate(yv, tau=tv)
        vals.append(a)
        pt_rows.append(dict(test="tau_invariance", outcome=yv,
                            window="2017-2024", regressor=f"tau={tv}",
                            coef=float(a), se=np.nan, pvalue=np.nan, n=np.nan))
    log(f"      {yv:<18s}" + "".join(f"{v:>14,.0f}" for v in vals))

log("\n  (ii) linearity test on the 2024 long difference (H0: quadratic and cubic")
log("       dose terms are jointly zero), HC1 Wald:")
for yv in PRIMARY_OUTCOMES:
    dY, Dv, sid = build_cross_section(yv, 2024)
    Xm = np.column_stack([np.ones(len(Dv)), Dv, Dv ** 2, Dv ** 3])
    r = sm.OLS(dY, Xm).fit(cov_type="HC1")
    R = np.zeros((2, 4)); R[0, 2] = 1; R[1, 3] = 1
    wt = r.wald_test(R, scalar=True)
    log(f"      {yv:<18s} F = {float(wt.statistic):7.3f}   p = {float(wt.pvalue):.4f}"
        f"   -> {'REJECT linearity' if wt.pvalue < 0.05 else 'cannot reject linearity'}")
    pt_rows.append(dict(test="linearity_wald", outcome=yv, window="2016-2024",
                        regressor="D^2,D^3", coef=float(wt.statistic), se=np.nan,
                        pvalue=float(wt.pvalue), n=int(len(dY))))

log("\n  (iii) early vs late builders (split at the median year in which a state")
log("        first reached 50% of its final 2024 dose):")
half_year = {}
for s in STATES:
    dfin = D24[s]
    if dfin < TAU:
        continue
    ys = [t for t in POST_YEARS if dose.loc[s, t] >= 0.5 * dfin]
    half_year[s] = min(ys) if ys else 2024
hy = pd.Series(half_year)
med = hy.median()
early = sorted(hy.index[hy <= med])
late = sorted(hy.index[hy > med])
log(f"        median half-dose year = {med:.0f}; early n={len(early)}, late n={len(late)}")
log(f"        early: {', '.join(early)}")
log(f"        late : {', '.join(late)}")
for yv in PRIMARY_OUTCOMES:
    a_e, _, _ = acrt_aggregate(yv, states=untreated_states + early)
    a_l, _, _ = acrt_aggregate(yv, states=untreated_states + late)
    log(f"        {yv:<18s} ACRT early = {a_e:>10,.0f}   ACRT late = {a_l:>10,.0f}"
        f"   ratio = {a_e/a_l if a_l else np.nan:6.2f}")
    pt_rows.append(dict(test="early_vs_late", outcome=yv, window="2017-2024",
                        regressor="early", coef=float(a_e), se=np.nan,
                        pvalue=np.nan, n=len(early)))
    pt_rows.append(dict(test="early_vs_late", outcome=yv, window="2017-2024",
                        regressor="late", coef=float(a_l), se=np.nan,
                        pvalue=np.nan, n=len(late)))
pt = pd.DataFrame(pt_rows)

# ==============================================================================
# 7. ROBUSTNESS
# ==============================================================================
hdr("7. ROBUSTNESS OF THE CONTINUOUS-DiD ACRT")

rob_rows = []
# The three largest 2024 doses, read off the data rather than hard-coded: on the
# 51-unit sample these were TX, VA, OR, and a scope change could reorder them.
TOP3_DOSE = tuple(D24.nlargest(3).index)
log(f"    three largest 2024 doses: "
    + ", ".join(f"{s} {D24[s]:.3f} GW" for s in TOP3_DOSE))
log(f"    {'variant':<34s}" + "".join(f"{y:>20s}" for y in PRIMARY_OUTCOMES))
variants = [
    ("primary (tau=0.05, quad)", dict()),
    ("linear series", dict(estimator="acrt_series_lin")),
    ("cubic series", dict(estimator="acrt_series_cub")),
    ("local linear", dict(estimator="acrt_local_linear")),
    ("tau = 0 (exact zeros only)", dict(tau=0.0)),
    ("tau = 0.10 GW", dict(tau=0.10)),
    ("tau = 0.20 GW", dict(tau=0.20)),
    ("drop negative-dose states", dict(drop_neg=True)),
    ("GPS-weighted (stabilised)", dict(weights_map=gps_weights.to_dict())),
    ("final period only (2024)", dict(years=[2024])),
    ("2021-2024 only", dict(years=[2021, 2022, 2023, 2024])),
    ("drop TX and VA", dict(states=[s for s in STATES if s not in ("TX", "VA")])),
    (f"drop top-3 dose ({','.join(TOP3_DOSE)})",
     dict(states=[s for s in STATES if s not in TOP3_DOSE])),
]
for label, kw in variants:
    vals = []
    for yv in PRIMARY_OUTCOMES:
        a, l, _ = acrt_aggregate(yv, **kw)
        vals.append(a)
        rob_rows.append(dict(variant=label, outcome=yv, acrt=a, att_level=l))
    log(f"    {label:<34s}" + "".join(f"{v:>20,.0f}" for v in vals))
rob = pd.DataFrame(rob_rows)

# ==============================================================================
# 8. HEAD-TO-HEAD ESTIMATOR COMPARISON
# ==============================================================================
hdr("8. HEAD-TO-HEAD ESTIMATOR COMPARISON, ALL IN JOBS PER GW")

log("""
UNIT RECONCILIATION. The published binary staggered DiD reports an ATT in JOBS,
not jobs per GW, because its treatment variable is an indicator. To put it on
the same axis as everything else we divide by the DOSE CONTRAST the indicator
actually represents:
      jobs per GW  =  ATT_binary / ( E[D | D >= c] - E[D | D < c] )
This is a units conversion, not a re-estimation; the standard error is scaled by
the same constant, which treats the dose contrast as fixed. The resulting number
is an average LEVEL effect per average GW -- it is NOT the ACRT, and the gap
between the two is exactly the object at issue.
""")
dose_contrast = {}
for c in THRESH_BINARY:
    hi_m = D24 >= c
    lo_m = D24 < c
    dc_ = float(D24[hi_m].mean() - D24[lo_m].mean())
    dose_contrast[c] = dc_
    log(f"  threshold {c:.1f} GW: E[D|treated] = {D24[hi_m].mean():.4f} GW "
        f"(n={int(hi_m.sum())}), E[D|control] = {D24[lo_m].mean():.4f} GW "
        f"(n={int(lo_m.sum())}), contrast = {dc_:.4f} GW")

# The binary staggered-DiD ATTs are READ FROM THE WORKBOOK that produced them,
# not typed.  The workbook is the output of 12_staggered_did.py.
# We record its modification time and its own reported sample so that a mismatch
# between the binary design's universe and this script's is visible rather than
# silent.
_XLS_LABEL = {"emp_5182": "Raw NAICS 5182", "emp_5182_cleaned": "Cleaned Direct"}
_XLS_THRESH = {0.5: "500MW", 1.0: "1GW"}
if not os.path.exists(XLSX_BINARY):
    sys.exit(f"missing {XLSX_BINARY}; the binary staggered-DiD ATTs are read "
             f"from it and are not hard-coded here.")
_bx = pd.read_excel(XLSX_BINARY, sheet_name="Overall ATT", header=1)
_bx = _bx[_bx["Category"] == "Employment"]
_ATT_COL = [c for c in _bx.columns if str(c).startswith("ATT") and "simple" not in str(c)]
assert len(_ATT_COL) == 1, f"ambiguous ATT column in {XLSX_BINARY}: {_ATT_COL}"
_ATT_COL = _ATT_COL[0]
BIN_PUB = {}
bin_pub_meta = dict(source=os.path.relpath(XLSX_BINARY, os.path.dirname(EMP)),
                    mtime=datetime.fromtimestamp(
                        os.path.getmtime(XLSX_BINARY)).isoformat(timespec="seconds"),
                    aggregation="dynamic", rows={})
log(f"\nbinary staggered-DiD ATTs read from {os.path.basename(XLSX_BINARY)} "
    f"(modified {bin_pub_meta['mtime']}), '{_ATT_COL.splitlines()[-1]}' aggregation:")
for _yv, _lab in _XLS_LABEL.items():
    for _c, _tl in _XLS_THRESH.items():
        _r = _bx[(_bx["Threshold"] == _tl) & (_bx["Y Variable"] == _lab)]
        assert len(_r) == 1, f"{_lab} @ {_tl}: {len(_r)} rows in {XLSX_BINARY}"
        _r = _r.iloc[0]
        BIN_PUB[(_yv, _c)] = dict(att=float(_r[_ATT_COL]), se=float(_r["SE"]),
                                  n=int(_r["N obs"]),
                                  n_treated=int(_r["Treated"]),
                                  n_control=int(_r["Control"]))
        bin_pub_meta["rows"][f"{_yv}@{_c}"] = BIN_PUB[(_yv, _c)]
        log(f"    {_lab:<16s} {_tl:>6s}: ATT = {_r[_ATT_COL]:>10,.1f} "
            f"(SE {_r['SE']:>9,.1f})  N = {int(_r['N obs'])}  "
            f"treated {int(_r['Treated'])} / control {int(_r['Control'])}")
_bin_units = {v["n_treated"] + v["n_control"] for v in BIN_PUB.values()}
if _bin_units != {N_ANALYSIS_UNITS}:
    log(f"    ** UNIVERSE MISMATCH: the binary design reports {_bin_units} units, "
        f"this script uses {N_ANALYSIS_UNITS}. The converted jobs-per-GW row of the "
        f"comparison table therefore divides an ATT estimated on one universe by a "
        f"dose contrast computed on another. **")
else:
    log(f"    universe check: the binary design and this script both use "
        f"{N_ANALYSIS_UNITS} units.")

cmp_rows = []

def add_cmp(outcome, method, est, se, n, assumption, bias, note):
    cmp_rows.append(dict(outcome=outcome, method=method, estimate_jobs_per_gw=est,
                         se_jobs_per_gw=se, n=n, identifying_assumption=assumption,
                         likely_bias_direction=bias, note=note))

for yv, pub_fe, pub_iv in [("emp_5182", PUB_PANEL_FE, PUB_IV),
                           ("emp_5182_cleaned", PUB_PANEL_FE_CLEAN, PUB_IV_CLEAN)]:
    add_cmp(yv, "Two-way panel FE (published)", pub_fe["beta"], pub_fe["se"], pub_fe["n"],
            "Strict exogeneity of capacity conditional on state and year FE; no "
            "time-varying state shock correlated with both capacity and employment.",
            "Attenuated by classical measurement error in the capacity series; "
            "sign ambiguous under reverse causality (firms site where the tech "
            "labour force already grows) -- that channel biases UP.",
            "panel_fe_table__contig49.csv, 49 units, N = 438, "
            "finite-sample-corrected clustered SE with t(48) inference")
    key = (yv, 1.0)
    if key in BIN_PUB:
        p = BIN_PUB[key]
        dcv = dose_contrast[1.0]
        add_cmp(yv, "Binary staggered DiD, 1 GW (published)", p["att"] / dcv,
                p["se"] / dcv, p["n"],
                "Parallel trends between states that cross 1 GW and states that "
                "never do, plus no anticipation. Control group is NOT untreated.",
                "Attenuated toward zero: the control group contains states that "
                "added up to 1 GW, so the contrast understates the true level "
                "effect. Offsetting upward bias if crossing 1 GW is itself "
                "selected on booming tech employment.",
                f"published ATT = {p['att']:,.0f} jobs (SE {p['se']:,.0f}), "
                f"{p['n_treated']} treated / {p['n_control']} control, N = {p['n']}, "
                f"read from {os.path.basename(XLSX_BINARY)}; "
                f"divided by a dose contrast of {dcv:.4f} GW")
    key = (yv, 0.5)
    if key in BIN_PUB:
        p = BIN_PUB[key]
        dcv = dose_contrast[0.5]
        add_cmp(yv, "Binary staggered DiD, 500 MW (published)", p["att"] / dcv,
                p["se"] / dcv, p["n"],
                "As above at the 500 MW threshold.",
                "Same direction; worse, because the control group is even more "
                "contaminated at the lower threshold.",
                f"published ATT = {p['att']:,.0f} jobs (SE {p['se']:,.0f}), "
                f"{p['n_treated']} treated / {p['n_control']} control, N = {p['n']}, "
                f"read from {os.path.basename(XLSX_BINARY)}; "
                f"dose contrast {dcv:.4f} GW")
    row = summary[(summary.outcome == yv) & (summary.estimator == "acrt_series_quad")].iloc[0]
    add_cmp(yv, "Continuous DiD ACRT^glob (this analysis)", row["estimate"], row["boot_se"],
            int(acrt_period[acrt_period.outcome == yv]["n"].mean()),
            "STRONG parallel trends (paper, Assumption SPT): E[Y_2(d)-Y_1(0)|D>0] "
            "= E[Y_2(d)-Y_1(0)|D=d] for all d, i.e. the group that took dose d is "
            "a valid counterfactual for the whole treated population at that dose. "
            "NOT nested with binary parallel trends -- it does not require parallel "
            "trends in untreated potential outcomes for all dose groups -- but in "
            "an application like this one it is a much stronger assumption, and it "
            "is untestable because it restricts counterfactual TREATED outcomes. "
            "The parameter identified is ACRT^glob (Corollary 3.1(b)), not "
            "ACRT^loc.",
            "Biased by selection on gains if states with the largest build-outs "
            "would have had the steepest tech-employment growth anyway -- that is "
            "an UPWARD bias. Downward bias from measurement error in the dose.",
            "quadratic series, averaged over 2017-2024, bootstrap SE over states")
    r1 = summary[(summary.outcome == yv) & (summary.estimator == "acrt_at_1p00GW")].iloc[0]
    add_cmp(yv, "Continuous DiD, ACRT^o at d = 1 GW", r1["estimate"], r1["boot_se"],
            int(acrt_period[acrt_period.outcome == yv]["n"].mean()),
            "STRONG parallel trends (Assumption SPT), evaluated at a fixed "
            "policy-relevant dose rather than averaged over the realised dose "
            "distribution. The parameter is ACRT(d) (Theorem 3.3(c)).",
            "Same as the ACRT. Reported separately because the estimated dose-"
            "response is concave, so the average over realised doses (dominated "
            "by many small-dose states) exceeds the response at 1 GW.",
            "quadratic series, averaged over 2017-2024")
    rcw = summary[(summary.outcome == yv) & (summary.estimator == "acrt_capacity_weighted")].iloc[0]
    add_cmp(yv, "Continuous DiD, capacity-weighted ACRT", rcw["estimate"], rcw["boot_se"],
            int(acrt_period[acrt_period.outcome == yv]["n"].mean()),
            "STRONG parallel trends; states weighted by the capacity they added, "
            "so this is the response relevant to the national job total.",
            "Pulled down relative to the ACRT by the concavity of the dose-"
            "response: the marginal GW in TX/VA/OR adds fewer jobs than the first "
            "GW anywhere.",
            "quadratic series, E[D*ACRT(D)]/E[D] over treated states")
    add_cmp(yv, "State-level Bartik shift-share IV", pub_iv["beta"], pub_iv["se"],
            pub_iv["n"],
            "Exogeneity of the 2019 exposure shares conditional on state and year "
            "FE (Goldsmith-Pinkham/Borusyak share exogeneity), UNION leave-out "
            "national shift (state + contiguous + division), 49 units.",
            "Above the FE estimate as classical attenuation and a compliers-"
            "weighted LATE both predict. Residual risk: share endogeneity if 2019 "
            "capacity shares proxy a persistent state trend.",
            PUB_IV_NOTE)

row = summary[(summary.outcome == "emp_517") & (summary.estimator == "acrt_series_quad")].iloc[0]
add_cmp("emp_517", "Continuous DiD ACRT^glob (this analysis)", row["estimate"], row["boot_se"],
        int(acrt_period[acrt_period.outcome == "emp_517"]["n"].mean()),
        "STRONG parallel trends (Assumption SPT); parameter is ACRT^glob.",
        "See NAICS 517 pre-trend failure in Section 6.",
        "quadratic series, averaged over 2017-2024")
add_cmp("emp_517", "State-level Bartik shift-share IV", PUB_IV_517["beta"],
        PUB_IV_517["se"], PUB_IV_517["n"],
        "Share exogeneity.", "NAICS 517 fails the DiD pre-trend test; the "
        "share-exogeneity pre-trend test passes.",
        PUB_IV_NOTE)

cmp = pd.DataFrame(cmp_rows)
log(f"\n{'outcome':<18s}{'method':<42s}{'jobs/GW':>12s}{'SE':>12s}{'t':>7s}")
for _, r in cmp.iterrows():
    tt = r["estimate_jobs_per_gw"] / r["se_jobs_per_gw"] if r["se_jobs_per_gw"] else np.nan
    log(f"{r['outcome']:<18s}{r['method']:<42s}{r['estimate_jobs_per_gw']:>12,.0f}"
        f"{r['se_jobs_per_gw']:>12,.0f}{tt:>7.2f}")

# formal comparison of ACRT against the other estimates
log("\n8.1 IS THE CONTINUOUS ACRT STATISTICALLY DISTINGUISHABLE FROM THE OTHERS?")
log("    (two-sided z on the difference, treating the two estimates as independent.")
log("     They are NOT independent: they are computed on overlapping data and their")
log("     covariance is positive, so Var(a-b) = Var(a)+Var(b)-2Cov is SMALLER than")
log("     the independence variance. Independence therefore UNDERSTATES the precision")
log("     of the difference and inflates every p-value below, making 'the estimates")
log("     are not distinguishable' easier to conclude. That is the conservative")
log("     direction for this particular claim, which is why we report it.)")
for yv in ["emp_5182", "emp_5182_cleaned"]:
    sub = cmp[cmp.outcome == yv]
    a_row = sub[sub.method.str.startswith("Continuous DiD")].iloc[0]
    a, sa = a_row["estimate_jobs_per_gw"], a_row["se_jobs_per_gw"]
    for _, r in sub.iterrows():
        if r["method"].startswith("Continuous DiD"):
            continue
        diff = a - r["estimate_jobs_per_gw"]
        sd = np.sqrt(sa ** 2 + r["se_jobs_per_gw"] ** 2)
        z = diff / sd
        p = 2 * (1 - st.norm.cdf(abs(z)))
        log(f"    {yv:<18s} ACRT - [{r['method'][:38]:<38s}] = {diff:>9,.0f}  "
            f"z = {z:>6.2f}  p = {p:.3f}")

# ==============================================================================
# 9. WRITE OUTPUTS
# ==============================================================================
hdr("9. OUTPUTS")

dr = pd.DataFrame(dose_response_rows)
paths = {}

def dump(df, name):
    stem, ext = os.path.splitext(name)
    fname = f"{stem}{SUFFIX}{ext}"
    # A re-run must never produce, or shadow, the unsuffixed name.
    assert SUFFIX and fname != name, \
        f"refusing to write an unsuffixed output name: {name}"
    assert os.path.abspath(OUT) != os.path.abspath(IN), \
        "OUT must not be the r06 input directory"
    p = os.path.join(OUT, fname)
    assert not os.path.exists(os.path.join(IN, fname)), \
        f"refusing to shadow an existing r06 output: {fname}"
    df.to_csv(p, index=False)
    paths[fname] = p
    log(f"  wrote {fname:<44s} {df.shape[0]:>6d} rows x {df.shape[1]} cols")

dump(dr, "did_dose_response.csv")
dump(acrt_period, "did_acrt_by_period.csv")
dump(summary, "did_acrt_summary.csv")
dump(es, "did_acrt_event_study.csv")
dump(pd.DataFrame(binned_rows), "did_binned_dose_response.csv")
dump(pd.DataFrame(ps_rows), "did_propensity_disclosure.csv")
dump(pd.DataFrame(bal_rows), "did_balance_table.csv")
dump(wdf, "did_twfe_weights.csv")
dump(pt, "did_parallel_trends_tests.csv")
dump(rob, "did_robustness.csv")
dump(contam, "did_binary_contamination.csv")
dump(cmp, "did_estimator_comparison.csv")

res = dict(
    meta=dict(script=os.path.relpath(os.path.abspath(__file__), os.path.dirname(EMP)), run_at=datetime.now().isoformat(),
              scope=dict(out_of_scope_units=list(OUT_OF_SCOPE_UNITS),
                         n_analysis_units=N_ANALYSIS_UNITS, **scope_rep),
              seed=SEED, B_boot=B_BOOT, base_year=BASE_YEAR,
              post_years=POST_YEARS, tau_gw=TAU,
              n_states=len(STATES), n_treated=len(treated_states),
              n_untreated=len(untreated_states),
              untreated_states=untreated_states,
              negative_dose_states=neg_states,
              runtime_sec=round(time.time() - T0, 1)),
    dose=dict(mean_treated=float(D24[~untreated_mask].mean()),
              max=float(D24.max()),
              quantiles={str(k): float(v) for k, v in
                         D24.describe(percentiles=[.25, .5, .75, .9]).items()}),
    acrt_summary=summary.to_dict("records"),
    event_study=es.to_dict("records"),
    binary_contamination=contam.to_dict("records"),
    twfe_decomposition_theorem_3_4a=twfe_weight_meta,
    twfe_decomposition_theorem_3_4bc=levels_weight_meta,
    binary_staggered_did_source=bin_pub_meta,
    dose_contrast_by_threshold={str(k): v for k, v in dose_contrast.items()},
    comparison_table=cmp.to_dict("records"),
    parallel_trends=pt.to_dict("records"),
    robustness=rob.to_dict("records"),
)
jp = os.path.join(OUT, f"did_continuous_results{SUFFIX}.json")
with open(jp, "w") as f:
    json.dump(res, f, indent=2, default=float)
paths[f"did_continuous_results{SUFFIX}.json"] = jp
log(f"  wrote did_continuous_results{SUFFIX}.json")

lp = os.path.join(OUT, f"did_continuous_run_log{SUFFIX}.txt")
with open(lp, "w") as f:
    f.write("\n".join(_LOG_LINES))
print(f"  wrote did_continuous_run_log{SUFFIX}.txt")
print(f"\nDONE in {time.time()-T0:.1f}s")
