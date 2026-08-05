#!/usr/bin/env python3
"""
State-level Bartik shift-share IV: data-center capacity -> employment.

State-level analogue of the zone-level instrument used for the wholesale price
analysis:

    Z_{s,t} = Share_s x (G_{t}^{-g(s)} - G_{base}^{-g(s)} )

    Share_s        = Capacity_{s,2019} / Capacity_{national,2019}   (time-invariant)
    G_{t}^{-g(s)}  = Capacity_{national,t} - sum_{j in g(s)} Capacity_{j,t}
    g(s)           = the leave-out group of state s.

Correspondence with the price design: pricing zone -> state; ISO -> Census
division (primary leave-out group); daily panel 2020-2025 -> annual panel
2016-2024; zone and year fixed effects -> state and year fixed effects; HAC
Bartlett -> clustering by state plus a wild cluster bootstrap; $/MWh per GW ->
jobs per GW.

    First stage    Capacity_{s,t} = b_1st  * Z_{s,t}         + StateFE + YearFE + nu
    Second stage   Y_{s,t}        = b_2SLS * Capacity_hat    + StateFE + YearFE + e
    Reduced form   Y_{s,t}        = b_RF   * Z_{s,t}         + StateFE + YearFE + e
    OLS comparison Y_{s,t}        = b_OLS  * Capacity_{s,t}  + StateFE + YearFE + e

Everything is estimated in the Frisch-Waugh-Lovell residualised space: y, x and z
are each projected off W = [const, state FE, year FE, extra FE / controls], after
which every estimator collapses to a scalar ratio and every cluster-robust
variance to a scalar sandwich.  This is algebraically identical to the full dummy
regression and makes the Anderson-Rubin grid, the wild cluster bootstrap and the
Rotemberg decomposition cheap enough to run exhaustively.

Estimation is implemented directly rather than delegated to linearmodels, which
applies no finite-sample correction to clustered IV covariances; the correction
used here is G/(G-1) * (N-1)/(N-K).  Coefficients agree with linearmodels and
statsmodels to ~1e-9 and the standard errors differ from linearmodels by exactly
that factor.

Outputs land in --out-dir.  --tag is required and is appended to every name:

    results_main_<tag>.csv           27 columns
    results_robustness_<tag>.csv     30 columns (superset of the above)
    results_diagnostics_<tag>.csv    13 columns, long format
    results_<tag>.json               full nested summary
    bartik_instrument_panel_<tag>.csv
    run_log_<tag>.txt

The published run is `--tag _union`.  No existing file is ever modified.

Usage:
    python3 05_bartik_iv.py --tag _union
    python3 05_bartik_iv.py --tag _union --outcomes headline
    python3 05_bartik_iv.py --tag contig --outcomes all --leave-out contig
    python3 05_bartik_iv.py --tag base2016 --share-baseline 2016
    python3 05_bartik_iv.py --list-outcomes
"""

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

# Release-set path rule: no absolute paths.  Resolved from this file's location.
EMP_DIR = Path(__file__).resolve().parent                 # employment/
REPO_ROOT = EMP_DIR.parent

def relpath(p) -> str:
    """Repo-relative path, so no absolute path reaches a shipped artefact."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)
R06_DIR = EMP_DIR / "../results/r6_bartik"                                 # generated results

DATA_PATHS = {
    "panel": R06_DIR / "panel_state_year.csv",
    "adjacency": R06_DIR / "state_adjacency.json",
    # One value per state: SD of the annual change in log NAICS 5182 employment
    # over 2005-2015.  Built by r06_share_test/build_pre_volatility.py.  Used
    # only by the vol_linear_trend and vol_year_fe robustness tags.
    "pre_volatility": R06_DIR / "pre_volatility_5182.csv",
}

OUTPUT_DIR = R06_DIR

SEED = 20260728                 # one global seed, recorded in results.json
N_BOOT = 999                    # WCR bootstrap and pairs cluster bootstrap replications
N_PERM = 1000                   # share-permutation placebo draws

YEAR_MIN, YEAR_MAX = 2016, 2024  # QWI stops at 2024; 2025 capacity rows carry no outcome

# The estimation window opens at the exposure-share baseline, not at the start of the
# panel.  The share is measured in 2019, so on a sample beginning in 2016 the share
# would not be pre-determined for 2016-2018: it embeds capacity built during exactly
# those years, which is the reverse-causality channel the shift-share design exists to
# break.  The wholesale price analysis of Section 2 already does this correctly, with a
# 2019 baseline and a 2020-2025 estimation window; this aligns the employment design
# with it.  YEAR_MIN stays at 2016 because the pre-trend test must run OUTSIDE the
# estimation window, which is its whole purpose.
IV_YEAR_MIN = 2019
MIN_OBS_YEARS = 3                # drop states with < 3 observed years

# ---------------------------------------------------------------------------
# ANALYSIS SCOPE. #
# The analysis universe is the contiguous United States: the 48 contiguous
# states plus the District of Columbia, 49 units in total.  Alaska and Hawaii
# are OUT OF SCOPE BY DEFINITION, not by data availability:
#   * both sit on isolated grids that no ISO in this paper reaches (CAISO,
#     ERCOT, MISO, PJM, SPP, NYISO, ISONE);
#   * the four ISO transmission-planning portfolios used for the cost
#     attribution do not cover them; and
#   * the cross-state spillover and regional cost-sharing mechanisms this
#     paper studies cannot operate on an isolated grid.
# This is a scope definition and it is applied BEFORE anything is estimated,
# so every part of the employment analysis -- pre-trend, balance, first stage,
# reduced form, IV and every robustness variant -- runs on the same 49 units.
# MIN_OBS_YEARS remains in force as an independent safeguard against a unit
# that is a singleton under state fixed effects; it is not the mechanism by
# which Alaska and Hawaii leave the sample.
# The two lines below read
#     OUT_OF_SCOPE_UNITS = ("AK", "HI")
#     N_ANALYSIS_UNITS = 49
# as LOCAL LITERALS, one declaration per script, none of which could raise if
# another drifted.  They are now IMPORTED from analysis_universe.py, the only
# place in this codebase where the universe is written down.  There is
# deliberately NO fallback: if the import fails the run aborts, because a
# fallback literal would silently outlive the run that produced it.
# ---------------------------------------------------------------------------
_EMP_DIR = R06_DIR.parent
if str(_EMP_DIR) not in sys.path:
    sys.path.insert(0, str(_EMP_DIR))
from analysis_universe import (# noqa: E402
    ANALYSIS_UNITS, N_ANALYSIS_UNITS, OUT_OF_SCOPE_UNITS, UNIVERSE_LABEL,
    UniverseError, check_universe, describe as universe_describe,
)

SHARE_BASE_PRIMARY = 2019
PRIMARY_LEAVE_OUT = "union"
LEAVE_OUT_VARIANTS = ["state", "division", "contig", "union"]

# Olea-Pflueger (2013) critical value, K = 1 excluded instrument, 10% worst-case bias.
# NOT the Staiger-Stock 10.
OP_F_CRIT = 23.1
WEAK_F_FOR_BOOTSTRAP_DROP = 5.0  # drop pairs-bootstrap reps below this

# ---------------------------------------------------------------------------
# BENCHMARKS ARE RESOLVED AT RUN TIME, NEVER DECLARED.
# These two are resolved at run time, never declared:
#     PANEL_FE_BENCHMARK = 1811.0     LOCAL_SHARE_MEDIAN = 0.637
# Neither could ever raise, because a literal cannot.  1811.0 was
# the two-way FE estimate on the 51-unit-complement sample (AK and HI in, DC out
# via the r05 STATE_ABBR_MAP bug); this script's OWN output has printed 1804.32
# for the same regression since the scope restriction was applied, twelve lines
# below the constant that contradicted it.  0.637 was the median local share over
# all 51 PwC rows; over the 49 units of the analysis universe it is 0.644458.
# They are now computed inside this run, from this run's own sample:
#   PANEL_FE_BENCHMARK  = the OLS_FE (state + year FE, clustered) coefficient of
#                         dc_gw on emp_5182 for the primary leave-out variant --
#                         i.e. the identical regression reported in the OLS FE
#                         column of SI tab:emp_iv_main and in SI tab:panel_fe_emp,
#                         computed here rather than copied across a script boundary.
#   LOCAL_SHARE_MEDIAN  = median of local_share over the units of the estimation
#                         sample, one value per unit.
# Both are asserted against the estimated cells later in the run, so a divergence
# is a hard error rather than a silently outdated number.  See resolve_benchmarks().
# ---------------------------------------------------------------------------
PANEL_FE_BENCHMARK = None
LOCAL_SHARE_MEDIAN = None
BENCHMARK_PROVENANCE = {"panel_fe_benchmark": "NOT_RESOLVED",
                        "local_share_median": "NOT_RESOLVED"}

def _benchmark():
    """The run-time panel FE benchmark; raises rather than defaulting to a literal."""
    if PANEL_FE_BENCHMARK is None:
        raise RuntimeError("PANEL_FE_BENCHMARK was read before resolve_benchmarks() ran. It is "
            "deliberately not a literal; nothing may fall back to a stored value.")
    return PANEL_FE_BENCHMARK

def _local_share_median():
    """The run-time median local share; raises rather than defaulting to a literal."""
    if LOCAL_SHARE_MEDIAN is None:
        raise RuntimeError("LOCAL_SHARE_MEDIAN was read before resolve_benchmarks() ran. It is "
            "deliberately not a literal; nothing may fall back to a stored value.")
    return LOCAL_SHARE_MEDIAN

MDE_MULT = 2.80                  # 80% power, alpha = 0.05, two-sided
# NOTE: 2.80 = z_.975 + z_.80, the
# normal-approximation constant, while every test and CI in this script uses
# t(G-1) = t(48).  The matching t constant is t_.975,48 + t_.80,48 = 2.8598, which
# would raise every MDE by 2.1 percent (raw 5182: 5,366 -> 5,480).  The SI caption
# of tab:emp_iv_inference now discloses the multiplier, the alternative and the
# 2.1 percent, so changing it here would contradict the SI text.  Whoever adopts
# the t constant must change the caption in the same commit.

# ---------------------------------------------------------------------------
# Pre-flight reference constants.  Any of these failing aborts the run:
# if the 'US'/'National' aggregate row ever creeps back into the capacity file,
# every share halves and share('US') = 0.5.  Fail loudly.
# ---------------------------------------------------------------------------
# All reference values below are computed over the 49-unit analysis universe,
# with the exposure share renormalised so that the 49 shares sum to one.
REF_NATIONAL_2019_GW = 27.933103
REF_SHARE2019_TOP8 = {"TX": 0.138952, "VA": 0.133208, "CA": 0.089361,
                      "IL": 0.057256, "NC": 0.055833, "GA": 0.053735,
                      "WA": 0.049288, "IA": 0.046002}
REF_TOP3_SUM = 0.3615
REF_TOP5_SUM = 0.4746
# Pacific falls from five units to three (CA, OR, WA) once Alaska and Hawaii are
# out of scope.  Every other division is unchanged and every division retains at
# least three units, so leave-one-division-out cannot degenerate to a singleton.
REF_DIVISION_COUNTS = {
    "South Atlantic": 9, "Mountain": 8, "West North Central": 7, "New England": 6,
    "East North Central": 5, "East South Central": 4, "West South Central": 4,
    "Pacific": 3, "Middle Atlantic": 3,
}
MIN_DIVISION_SIZE = 3
# Known first-stage state-influence values under the union leave-out, which is
# what this design computes and what the SI reports.
REF_INFL_FS = {"TX": 0.4247, "VA": 0.3020}

# ---------------------------------------------------------------------------
# Outcomes.  `outcome_units` is a required output column so no
# downstream stage has to guess: jobs/GW and $/month/GW are never comparable.
# ---------------------------------------------------------------------------
LEVEL_OUTCOMES = [
    "emp_5182", "emp_5182_cleaned", "emp_5182_cleaned_mult",
    "emp_5182_state_total", "emp_517",
    "emp_23", "emp_2362", "emp_5415", "emp_51",
    "earn_5182", "earn_517", "earn_23", "earn_2362", "earn_5415", "earn_51",
]
LOG_OUTCOMES = [
    "log_emp_5182", "log_emp_517", "log_emp_5415", "log_emp_51",
    "log_emp_23", "log_emp_2362", "log_emp_5182_cleaned", "log_emp_5182_cleaned_mult",
]
HEADLINE_OUTCOMES = ["emp_5182", "emp_517"]

OUTCOME_SETS = {
    "headline": HEADLINE_OUTCOMES,
    "employment": [c for c in LEVEL_OUTCOMES if c.startswith("emp_")],
    "earnings": [c for c in LEVEL_OUTCOMES if c.startswith("earn_")],
    "level": LEVEL_OUTCOMES,
    "all": LEVEL_OUTCOMES,
    "log": LOG_OUTCOMES,
}

def outcome_units(col):
    if col.startswith("log_"):
        return "log_points_per_GW"
    if col.startswith("earn_"):
        return "usd_per_month_per_GW"
    return "jobs_per_GW"

# Anderson-Rubin grid: +/-20,000 in steps of 25 for employment.
AR_GRID = {
    "jobs_per_GW": (20000.0, 25.0),
    "usd_per_month_per_GW": (20000.0, 25.0),
    "log_points_per_GW": (20.0, 0.025),
}

# Robustness tags
ROBUSTNESS_TAGS = [
    "leaveout_state", "leaveout_contig", "leaveout_division", "share2016",
    "drop_VA", "drop_TX", "drop_top3_share", "drop_partial_years",
    "fe_region_year", "fe_division_year", "share_linear_trend", "log",
    # the balance test finds pre-period employment growth
    # significantly less volatile in high-exposure states, so allow high- and
    # low-volatility states their own trend, and then their own year effects.
    "vol_linear_trend", "vol_year_fe",
]

# Balance-test characteristics.  gdp_pc_2024 is deliberately
# EXCLUDED: it is a 2024 measurement, hence post-treatment.
BALANCE_VARS = [
    "emp_5182_2016", "emp_517_2016", "emp_23_2016", "emp_5415_2016", "emp_51_2016",
    "emp_5182_mean_pre", "emp_5182_sd_pre", "emp_517_mean_pre",
    "earn_5182_2016", "earn_517_2016",
    "emp_5182_growth_1619", "emp_517_growth_1619", "n_neighbors",
]

# Closed enumeration of `diagnostic`.  Every row written is checked
# against this set before the file is saved.
DIAGNOSTIC_NAMES = {
    "preflight_assertion", "rotemberg_year_weight", "rotemberg_year_beta",
    "rotemberg_validation", "state_influence_fs", "state_influence_rf",
    "influence_concentration", "rf_fs_influence_divergence", "pretrend_gamma",
    "balance_gamma", "placebo_permutation", "placebo_lead",
    "first_stage_F", "first_stage_F_effective", "anderson_rubin_ci",
    "wcr_bootstrap", "wald_se_delta_full", "wald_se_delta_simplified",
    "wald_cov_rf_fs", "mde", "sample_composition",
    # The pairwise instrument correlations SI tab:leaveout_compare prints.
    "instrument_correlation",
}

MAIN_COLUMNS = [
    "outcome", "outcome_units", "specification", "leave_out_variant", "share_baseline",
    "estimator", "coef", "se", "se_type", "t_stat", "p_value", "ci_low", "ci_high",
    "ci_type", "n_obs", "n_clusters", "n_states", "n_years",
    "fs_coef", "fs_se", "fs_F", "fs_F_effective", "fs_F_crit", "first_stage_ok",
    "r2_within", "mde_80", "notes",
]
ROBUST_COLUMNS = MAIN_COLUMNS + ["robustness_tag", "dropped_units", "extra_fe"]
DIAG_COLUMNS = ["diagnostic", "outcome", "leave_out_variant", "share_baseline", "subject",
                "statistic", "value", "se", "p_value", "n_obs", "threshold",
                "pass_fail", "notes"]

# =============================================================================
# Logging (mirrors r02: everything printed also lands in the run log)
# =============================================================================

class Tee:
    """Duplicate stdout into run_log.txt so the log IS the printed output."""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def banner(msg, char="=", width=78):
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}")

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# =============================================================================
# Pre-flight assertions (A1-A12).  Any failure = hard abort.
# =============================================================================

def apply_scope(panel, adjacency):
    """
    Restrict the panel and the adjacency map to the analysis universe (the 48
    contiguous states plus DC) and renormalise every exposure share over it.

    This runs before the instrument is built and before anything is estimated,
    so the national shift G_t, the exposure share Share_s, and the estimation
    sample all refer to the same 49 units.  Renormalising matters: if the 49
    shares were left as fractions of a 51-unit total they would sum to
    0.99942 rather than to one, and Share_s would no longer be a share of the
    universe over which the shift is defined.

    Returns (panel, adjacency, report_dict).
    """
    drop = [s for s in OUT_OF_SCOPE_UNITS if s in set(panel["state_abbr"])]
    before_units = panel["state_abbr"].nunique()
    before_2019 = float(panel.loc[panel["year"] == 2019, "dc_gw"].sum())
    dropped_mw = {s: float(panel.loc[(panel["state_abbr"] == s)
                                     & (panel["year"] == 2019), "dc_gw"].iloc[0]) * 1000.0
                  for s in drop}

    p = panel[~panel["state_abbr"].isin(drop)].copy()

    # National capacity is the sum over the analysis universe, by year.
    nat = p.groupby("year")["dc_gw"].sum()
    p["us_dc_gw"] = p["year"].map(nat)
    p["us_dc_gw_lag1"] = p["year"].map(nat.shift(1))
    p["us_d_dc_gw"] = p["year"].map(nat.diff())

    # Renormalise the exposure shares over the analysis universe.  Shares are
    # time-invariant, so each is rebuilt from its own base year and broadcast.
    for base_year in (2016, 2019):
        base = p.loc[p["year"] == base_year].set_index("state_abbr")["dc_gw"]
        p[f"share{base_year}"] = p["state_abbr"].map(base / base.sum())

    adj = {k: [n for n in v if n not in drop]
           for k, v in adjacency.items() if k not in drop}

    after_2019 = float(p.loc[p["year"] == 2019, "dc_gw"].sum())
    rep = dict(dropped=drop, dropped_mw_2019=dropped_mw,
               n_units_before=before_units, n_units_after=p["state_abbr"].nunique(),
               gw_2019_before=before_2019, gw_2019_after=after_2019,
               dropped_pct_of_2019=100.0 * (before_2019 - after_2019) / before_2019)

    banner("ANALYSIS SCOPE: contiguous United States (48 states + DC)")
    print("  Alaska and Hawaii are out of scope by definition, not by data")
    print("  availability: both sit on isolated grids that none of the seven ISOs")
    print("  analysed here reaches, the four ISO transmission-planning portfolios")
    print("  used for the cost attribution do not cover them, and the cross-state")
    print("  spillover and regional cost-sharing mechanisms studied in this paper")
    print("  cannot operate on an isolated grid.")
    print(f"\n  units: {before_units} -> {rep['n_units_after']}")
    for s, mw in dropped_mw.items():
        print(f"  dropped {s}: {mw:.2f} MW of 2019 capacity")
    print(f"  2019 national capacity: {before_2019:.6f} -> {after_2019:.6f} GW "
          f"({rep['dropped_pct_of_2019']:.4f}% removed)")
    print("  exposure shares renormalised over the analysis universe "
          f"(sum = {float(p.loc[p['year'] == 2019, 'share2019'].sum()):.6f})")
    return p, adj, rep

def preflight(panel, adjacency, diag):
    """Run A1-A12.  Returns (n_passed, failures).  Writes one diagnostic row each."""
    banner("PRE-FLIGHT ASSERTIONS  --  any failure aborts the run")
    checks = []

    def rec(name, ok, value, threshold=None, note=""):
        checks.append((name, bool(ok), value, note))
        diag.append(dict(diagnostic="preflight_assertion", outcome="", leave_out_variant="",
                         share_baseline="", subject=name, statistic="value",
                         value=value, se=np.nan, p_value=np.nan, n_obs=len(panel),
                         threshold=threshold, pass_fail="PASS" if ok else "FAIL", notes=note))

    abbrs = panel["state_abbr"].unique()
    rec("A1_no_US_row", "US" not in set(abbrs), float("US" in set(abbrs)), 0.0,
        "national aggregate row must be absent")
    rec("A2_49_units", len(abbrs) == N_ANALYSIS_UNITS, float(len(abbrs)),
        float(N_ANALYSIS_UNITS), "48 contiguous states + DC")
    rec("A2b_scope_excluded", not (set(OUT_OF_SCOPE_UNITS) & set(abbrs)),
        float(len(set(OUT_OF_SCOPE_UNITS) & set(abbrs))), 0.0,
        "AK/HI out of scope: isolated grids, no ISO coverage")
    rec("A3_year_range", panel["year"].min() == 2016 and panel["year"].max() == 2025,
        float(panel["year"].max()), 2025.0, f"min={panel['year'].min()}")

    x19 = panel.loc[panel["year"] == 2019]
    s19_sum = float(x19["share2019"].sum())
    s16_sum = float(x19["share2016"].sum())
    rec("A4_share2019_sums_to_1", abs(s19_sum - 1.0) < 1e-9, s19_sum, 1.0)
    rec("A5_share2016_sums_to_1", abs(s16_sum - 1.0) < 1e-9, s16_sum, 1.0)

    nat19 = float(x19["us_dc_gw"].iloc[0])
    rec("A6_national_2019_gw", abs(nat19 - REF_NATIONAL_2019_GW) < 1e-4, nat19,
        REF_NATIONAL_2019_GW, "sum over the 49-unit analysis universe")

    sh = x19.set_index("state_abbr")["share2019"]
    top8_ok, top8_worst = True, 0.0
    for st, ref in REF_SHARE2019_TOP8.items():
        d = abs(float(sh.get(st, np.nan)) - ref)
        top8_worst = max(top8_worst, d)
        top8_ok &= d < 1e-4
    rec("A7_share2019_top8", top8_ok, top8_worst, 1e-4,
        "max |computed - reference| over TX,VA,CA,IL,NC,GA,WA,IA")

    t3 = float(sh.nlargest(3).sum())
    t5 = float(sh.nlargest(5).sum())
    rec("A8_top3_sum", abs(t3 - REF_TOP3_SUM) < 1e-3, t3, REF_TOP3_SUM)
    rec("A9_top5_sum", abs(t5 - REF_TOP5_SUM) < 1e-3, t5, REF_TOP5_SUM)

    agg = panel.groupby("year").agg(us=("us_dc_gw", "first"), tot=("dc_gw", "sum"))
    max_dev = float((agg["us"] - agg["tot"]).abs().max())
    rec("A10_us_equals_sum_of_states", max_dev < 1e-6, max_dev, 1e-6,
        "per-year: us_dc_gw == sum_s dc_gw")

    dc = x19["census_division"].value_counts().to_dict()
    rec("A11_division_counts", dc == REF_DIVISION_COUNTS, float(len(dc)), 9.0, str(dc))

    # The primary leave-out is leave-one-division-out, so a division reduced to a
    # single unit would silently degenerate: that unit's shift would drop its own
    # entire division and the within-division comparison would vanish.  Assert it
    # cannot happen rather than trusting the counts above.
    smallest = min(dc.values())
    rec("A11b_no_singleton_division", smallest >= MIN_DIVISION_SIZE, float(smallest),
        float(MIN_DIVISION_SIZE),
        "smallest division must retain >= 3 units for leave-one-division-out")

    adj_ok = (len(adjacency) == N_ANALYSIS_UNITS
              and not (set(OUT_OF_SCOPE_UNITS) & set(adjacency))
              and all(nb for nb in adjacency.values())
              and all(a in adjacency.get(b, []) for a, nb in adjacency.items() for b in nb))
    rec("A12_adjacency", adj_ok, float(len(adjacency)), float(N_ANALYSIS_UNITS),
        "49 keys, symmetric, no out-of-scope unit, every unit has >= 1 neighbour")

    for name, ok, val, note in checks:
        flag = "PASS" if ok else "**FAIL**"
        print(f"  {name:<32s} {flag:>8s}   value={val!r}" + (f"   [{note}]" if note else ""))

    failures = [n for n, ok, _, _ in checks if not ok]
    n_pass = len(checks) - len(failures)
    print(f"\n  {n_pass}/{len(checks)} assertions passed.")
    if failures:
        print("\n  HARD ABORT: pre-flight failed -> the build is wrong. Do not estimate.")
        print(f"  Failed: {failures}")
    return n_pass, failures

# =============================================================================
# Instrument construction
# =============================================================================

def leave_out_groups(panel, adjacency, variant):
    """g(s) for each state.  g(s) ALWAYS includes s itself."""
    states = sorted(panel["state_abbr"].unique())
    div = panel.drop_duplicates("state_abbr").set_index("state_abbr")["census_division"]
    if variant == "state":
        return {s: [s] for s in states}
    if variant == "division":
        by_div = {}
        for s in states:
            by_div.setdefault(div[s], []).append(s)
        return {s: sorted(by_div[div[s]]) for s in states}
    if variant == "contig":
        return {s: sorted(set([s] + list(adjacency.get(s, [])))) for s in states}
    if variant == "union":
        # PRIMARY (SI sec. 6.3.3): union of the state, all contiguous states, and
        # the state's Census division.  Excludes every state that either the
        # standard regional partition or direct adjacency would flag, so the
        # exclusion set cannot be criticised as too narrow under either criterion.
        by_div = {}
        for s in states:
            by_div.setdefault(div[s], []).append(s)
        return {s: sorted(set([s] + list(adjacency.get(s, []))) | set(by_div[div[s]]))
                for s in states}
    raise ValueError(f"unknown leave-out variant: {variant}")

def build_instrument(panel, adjacency, variant, share_year):
    """
    Z_{s,t} = Share_s x (G^{-g(s)}_t - G^{-g(s)}_{base} ),    units GW.

    G^{-g(s)}_t = Capacity_national_t - sum_{j in g(s)} Capacity_{j,t}.

    The baseline year moves with the share year: a 2016 share is
    paired with a 2016 shift baseline, never with a 2019 one.  Z_{s,base} = 0 by
    construction for every s, mirroring the price design's base-year normalisation.
    No within-year interpolation: employment is annual, so unlike the daily price
    design there is no interpolation degree of freedom here.
    """
    groups = leave_out_groups(panel, adjacency, variant)
    wide = panel.pivot(index="year", columns="state_abbr", values="dc_gw")
    nat = panel.groupby("year")["us_dc_gw"].first()

    shift = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for s, grp in groups.items():
        shift[s] = nat.values - wide[grp].sum(axis=1).values

    share_col = f"share{share_year}"
    share = panel.drop_duplicates("state_abbr").set_index("state_abbr")[share_col]
    base = shift.loc[share_year]
    z = (shift - base).mul(share.reindex(shift.columns), axis=1)

    out = (z.stack().rename("bartik").reset_index()
             .merge(shift.stack().rename("shift").reset_index(),
                    on=["year", "state_abbr"], how="left"))
    return out[["state_abbr", "year", "shift", "bartik"]]

def attach_instruments(panel, adjacency, share_year):
    """Attach bartik_<variant> and shift_<variant> for all three variants."""
    df = panel.copy()
    for v in LEAVE_OUT_VARIANTS:
        z = build_instrument(panel, adjacency, v, share_year)
        z = z.rename(columns={"bartik": f"bartik_{v}", "shift": f"shift_{v}"})
        df = df.merge(z, on=["state_abbr", "year"], how="left")
    return df

# =============================================================================
# Linear algebra core: FWL residualisation + cluster-robust scalar sandwiches
# =============================================================================

def design_matrix(df, extra_fe=None, extra_controls=None):
    """
    W = [const, State FE, Year FE, (extra FE), (extra controls)].

    Dummies are built from the categories actually present in the estimation sample
    and the first level is dropped; the returned rank is the K used in the
    finite-sample correction G/(G-1) * (N-1)/(N-K).
    """
    n = len(df)
    blocks = [np.ones((n, 1))]
    blocks.append(pd.get_dummies(df["state_abbr"], drop_first=True, dtype=float).values)
    if extra_fe == "region_year":
        key = df["census_region"].astype(str) + "_" + df["year"].astype(str)
        blocks.append(pd.get_dummies(key, drop_first=True, dtype=float).values)
    elif extra_fe == "division_year":
        key = df["census_division"].astype(str) + "_" + df["year"].astype(str)
        blocks.append(pd.get_dummies(key, drop_first=True, dtype=float).values)
    else:
        blocks.append(pd.get_dummies(df["year"], drop_first=True, dtype=float).values)
    if extra_controls is not None and len(extra_controls) > 0:
        blocks.append(np.asarray(extra_controls, dtype=float).reshape(n, -1))
    W = np.hstack(blocks)
    K = int(np.linalg.matrix_rank(W))
    return W, K

class Residualiser:
    """Projects columns off W once; every estimator below then works on residuals."""

    def __init__(self, W):
        self.W = W
        self.Wp = np.linalg.pinv(W)

    def __call__(self, V):
        V = np.asarray(V, dtype=float)
        return V - self.W @ (self.Wp @ V)

def _cluster_sums(v, cl_idx, G):
    """sum_{i in g} v_i for each cluster g -> length-G vector."""
    out = np.zeros(G)
    np.add.at(out, cl_idx, v)
    return out

def fs_correction(N, K, G):
    """Cameron-Gelbach-Miller finite-sample correction."""
    return (G / (G - 1.0)) * ((N - 1.0) / (N - K))

def lin_fit(a, y, cl_idx, G, N, K):
    """
    Coefficient of y on a, both already residualised off W.
    Cluster-robust variance:  V = c * sum_g (a_g'u_g)^2 / (a'a)^2 .
    """
    den = float(a @ a)
    if den <= 0 or not np.isfinite(den):
        return dict(coef=np.nan, se=np.nan, var=np.nan, u=None, den=np.nan, r2=np.nan)
    coef = float(a @ y) / den
    u = y - coef * a
    sg = _cluster_sums(a * u, cl_idx, G)
    var = fs_correction(N, K, G) * float(sg @ sg) / den ** 2
    sst = float(y @ y)
    r2 = 1.0 - float(u @ u) / sst if sst > 0 else np.nan
    return dict(coef=coef, se=np.sqrt(var), var=var, u=u, den=den, r2=r2, score=a * u)

def iv_fit(z, x, y, cl_idx, G, N, K):
    """
    Just-identified IV in residualised space:  b = (z'y)/(z'x).
    Cluster-robust variance:  V = c * sum_g (z_g'u_g)^2 / (z'x)^2 , u = y - b*x.
    Identical to the 2SLS cluster sandwich for the coefficient of interest.
    """
    den = float(z @ x)
    if den == 0 or not np.isfinite(den):
        return dict(coef=np.nan, se=np.nan, var=np.nan, u=None, den=np.nan, r2=np.nan)
    coef = float(z @ y) / den
    u = y - coef * x
    sg = _cluster_sums(z * u, cl_idx, G)
    var = fs_correction(N, K, G) * float(sg @ sg) / den ** 2
    sst = float(y @ y)
    r2 = 1.0 - float(u @ u) / sst if sst > 0 else np.nan
    return dict(coef=coef, se=np.sqrt(var), var=var, u=u, den=den, r2=r2, score=z * u)

def cov_cluster(score_a, den_a, score_b, den_b, cl_idx, G, N, K):
    """Cross-equation cluster-robust covariance of two scalar ratio estimators."""
    sa = _cluster_sums(score_a, cl_idx, G)
    sb = _cluster_sums(score_b, cl_idx, G)
    return fs_correction(N, K, G) * float(sa @ sb) / (den_a * den_b)

def t_ci(coef, se, G, level=0.95):
    """Two-sided t_{G-1} CI and p-value (t, not normal)."""
    if not np.isfinite(coef) or not np.isfinite(se) or se <= 0:
        return np.nan, np.nan, np.nan, np.nan
    tstat = coef / se
    p = 2 * sps.t.sf(abs(tstat), df=G - 1)
    crit = sps.t.ppf(0.5 + level / 2, df=G - 1)
    return tstat, p, coef - crit * se, coef + crit * se

# =============================================================================
# Weak-instrument diagnostics
# =============================================================================

def effective_F(fs):
    """
    Olea & Pflueger (2013) effective first-stage F.

    General form:  F_eff = (pi' Z'Z pi) / tr(Omega_hat * Z'Z / N) ... but with a
    SINGLE excluded instrument (K = 1) the effective F reduces EXACTLY to the square
    of the cluster-robust first-stage t-statistic:

        F_eff = b_1st^2 / Var_cluster(b_1st)

    (Olea-Pflueger 2013, sec. 3; Andrews, Stock & Sun 2019, eq. 4.)  It is
    implemented that way rather than importing a general-K routine
    Compared against the OP critical value 23.1 for K = 1 at 10% worst-case bias --
    NOT the Staiger-Stock 10.
    """
    if not np.isfinite(fs["coef"]) or not np.isfinite(fs["var"]) or fs["var"] <= 0:
        return np.nan
    return fs["coef"] ** 2 / fs["var"]

def anderson_rubin_ci(z, x, y, cl_idx, G, N, K, units, level=0.95, max_widen=3):
    """
    Anderson-Rubin (1949) weak-IV-robust confidence set by grid inversion.

    For each candidate b0 regress (y - b0*x) on z (both residualised off W) and test
    the coefficient = 0 with a cluster-robust variance.  Collect every b0 not
    rejected at 5%.  AR is exact under ANY instrument strength, so it is the honest
    interval whenever the sec. 6.1 gate trips.  The set may be unbounded or
    disconnected -- both are reported literally, never as a point.

    CONTROL FLOW.  kind="empty" must not be returned the moment the fixed
    +/-20,000 grid accepts nothing, before testing whether the accepted set
    touched a boundary and before the grid is widened.  "Empty" asserts that
    the parameter is rejected at every value; "nothing on this grid" asserts only
    that the grid is in the wrong place.  Reporting the two as one produces
    incoherent rows: a cell can then carry a 2SLS interval entirely above zero
    alongside an AR set recorded as empty with ar_excludes_zero = False.

    An empty grid is therefore a reason to WIDEN, exactly like touching a boundary, and
    the widening budget is the same `max_widen` for both.  Only after the budget is
    spent is a verdict returned, and if the set is still not bracketed the function
    RAISES -- it never reports a set it did not find.

    Why a raise and not a NaN.  With a single instrument the design is exactly
    identified, so at b0 = beta_2SLS the moment z'(y - b0 x) is zero by
    construction, the AR statistic is exactly 0, and b0 is always accepted.  A
    genuinely empty AR set therefore CANNOT occur here: emptiness after widening to
    +/-1,280,000 jobs per gigawatt means the point estimate is outside that range or
    the inputs are degenerate, and both are pathologies the caller must see.  The
    one non-pathological way to accept nothing is a variance that is zero or
    non-finite at every grid point, which no widening can fix; that returns
    kind="undefined" rather than raising, and is still never called "empty".

    Grid geometry is unchanged: `half` and `step` are both multiplied by 4 per
    widening, so a widened set is reported on a coarser grid, as the caption of
    SI tab:emp_iv_inference states.  The widening notes are returned to the
    caller and written into the diagnostics rows, so which
    rows are on which grid is visible in the shipped file rather than only in the
    run log.
    """
    half, step = AR_GRID.get(units, AR_GRID["jobs_per_GW"])
    c = fs_correction(N, K, G)
    crit = sps.t.ppf(0.5 + level / 2, df=G - 1) ** 2
    zz = float(z @ z)
    zy, zx = float(z @ y), float(z @ x)
    notes = []
    accept = None
    for widen in range(max_widen + 1):
        grid = np.arange(-half, half + step / 2, step)
        # coefficient of z on (y - b0 x) is linear in b0
        coefs = (zy - grid * zx) / zz
        # residuals u(b0) = (y - b0 x) - coef(b0) * z  -> vectorised over the grid
        base = np.outer(y, np.ones_like(grid)) - np.outer(x, grid) - np.outer(z, coefs)
        sc = z[:, None] * base
        sg = np.zeros((G, grid.size))
        np.add.at(sg, cl_idx, sc)
        var = c * (sg ** 2).sum(axis=0) / zz ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = coefs ** 2 / var
        finite = np.isfinite(stat)
        if not finite.any():
            # Degenerate variance at every candidate: the AR statistic does not
            # exist on this sample.  Widening cannot help, and this is NOT emptiness.
            return np.nan, np.nan, "undefined", ["AR_undefined_variance"] + notes
        accept = finite & (stat <= crit)
        if accept.any() and not (accept[0] or accept[-1]):
            lo_i = int(np.argmax(accept))
            hi_i = int(len(accept) - 1 - np.argmax(accept[::-1]))
            gaps = np.diff(np.flatnonzero(accept))
            kind = "bounded" if (gaps == 1).all() else "disconnected"
            if kind == "disconnected":
                notes.append("AR_disconnected")
            return float(grid[lo_i]), float(grid[hi_i]), kind, notes
        if widen == max_widen:
            break
        # Widen for EITHER reason.  The note keeps the historical token unchanged
        # for the boundary case so existing greps and existing files still match,
        # and marks the empty case distinctly because it never occurred before.
        tok = f"AR_widened_x{4 ** (widen + 1)}"
        notes.append(tok if accept.any() else tok + "_empty_grid")
        half *= 4.0
        step *= 4.0
    if accept is not None and accept.any():
        # Still touching a boundary after the full budget: the set is unbounded.
        return -np.inf, np.inf, "unbounded", ["AR_unbounded"] + notes
    raise RuntimeError("Anderson-Rubin grid failed to bracket the confidence set after "
        f"{max_widen} widenings: nothing accepted on the final grid +/-{half:,.0f} "
        f"{units} at step {step:,.4g}. beta_2SLS = {zy / zx if zx else float('nan'):,.4f}, "
        "which the set must contain in a just-identified design. This is a "
        "pathology, not an empty confidence set, and it is raised rather than "
        "reported as 'empty'.")

def wcr_bootstrap(a, y, res, cl_idx, G, N, K, rng, n_boot=N_BOOT):
    """
    Wild cluster bootstrap-t, RESTRICTED (WCR), Rademacher weights
    (Cameron, Gelbach & Miller 2008).  Applied to ordinary-OLS regressions only --
    the reduced form and the first stage -- where WCR is unambiguous.

    Under H0: beta = 0 the restricted model is "dep on W only", so in residualised
    space the restricted residual IS y (the caller passes FWL-residualised inputs) and
    the restricted fit is W*b_r.  Each replication draws w_g in {-1,+1} per cluster and
    forms y*_orig = W b_r + w_{g(i)} * ytilde_i; projecting that off W leaves
    M_W(w o ytilde), which is what `res` computes.  The Wald t is then recomputed.
    With 50 clusters there are 2^50 possible draws, so no enumeration issue.

    p = (1 + #{|t*| >= |t|}) / (B + 1);  symmetric-percentile CI = b +/- q_.95(|t*|) * se.
    """
    base = lin_fit(a, y, cl_idx, G, N, K)
    if not np.isfinite(base["se"]) or base["se"] <= 0:
        return dict(p=np.nan, ci_low=np.nan, ci_high=np.nan, se=np.nan, t=np.nan)
    t_obs = base["coef"] / base["se"]
    w = rng.choice(np.array([-1.0, 1.0]), size=(G, n_boot))
    Ystar = res(w[cl_idx, :] * y[:, None])     # (N, B), re-projected off W
    den = float(a @ a)
    coefs = (Ystar.T @ a) / den                # (B,)
    resid = Ystar - np.outer(a, coefs)         # (N, B)
    sc = resid * a[:, None]
    sg = np.zeros((G, n_boot))
    np.add.at(sg, cl_idx, sc)
    var = fs_correction(N, K, G) * (sg ** 2).sum(axis=0) / den ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        tstar = coefs / np.sqrt(var)
    tstar = tstar[np.isfinite(tstar)]
    if tstar.size == 0:
        return dict(p=np.nan, ci_low=np.nan, ci_high=np.nan, se=np.nan, t=t_obs)
    p = (1 + int((np.abs(tstar) >= abs(t_obs)).sum())) / (tstar.size + 1)
    q = float(np.quantile(np.abs(tstar), 0.95))
    return dict(p=p, ci_low=base["coef"] - q * base["se"],
                ci_high=base["coef"] + q * base["se"],
                se=base["se"] * q / sps.t.ppf(0.975, df=G - 1), t=t_obs)

# =============================================================================
# Sample construction
# =============================================================================

def build_sample(df, drop_states=(), drop_partial=False, years=(IV_YEAR_MIN, YEAR_MAX)):
    """
  Years IV_YEAR_MIN-2024, employment observed, states with >= 3 observed
    years.  WV (share2019 == 0) is KEPT: it is a legitimate zero-exposure unit and
    it is precisely the state whose spillover motivated the division leave-out.
    """
    s = df[(df["year"] >= years[0]) & (df["year"] <= years[1])].copy()
    s = s[s["emp_available"] == 1]
    if drop_partial:
        s = s[s["partial_year"] != 1]
    cnt = s.groupby("state_abbr")["year"].nunique()
    singletons = sorted(cnt[cnt < MIN_OBS_YEARS].index.tolist())
    s = s[~s["state_abbr"].isin(singletons)]
    if drop_states:
        s = s[~s["state_abbr"].isin(list(drop_states))]
    s = s.sort_values(["state_abbr", "year"]).reset_index(drop=True)
    return s, singletons

_PRE_VOL_CACHE = {}

def pre_volatility(states, path=None):
    """
    Pre-period employment volatility, one value per state: the standard deviation
    of the annual change in log NAICS 5182 employment over 2005-2015.  This is the
    characteristic the balance test finds imbalanced against the 2019 exposure
    share, and it is pre-determined with respect to the 2016-2024 estimation
    window.  Returns a float array aligned to `states`.
    """
    key = str(path or DATA_PATHS["pre_volatility"])
    if key not in _PRE_VOL_CACHE:
        v = pd.read_csv(key)
        _PRE_VOL_CACHE[key] = dict(zip(v["state_abbr"],
                                       v["vol_pre_5182"].astype(float)))
    m = _PRE_VOL_CACHE[key]
    missing = sorted(set(states) - set(m))
    if missing:
        raise KeyError(f"pre-period volatility unavailable for {missing}")
    return np.asarray([m[s] for s in states], dtype=float)

def cluster_codes(df, col="state_abbr"):
    codes, uniques = pd.factorize(df[col], sort=True)
    return codes.astype(int), len(uniques)

# =============================================================================
# The estimation cell: one (outcome, variant, baseline, spec) combination
# =============================================================================

def estimate_cell(sample, ycol, zcol, extra_fe=None, extra_controls=None,
                  do_ar=True, do_wcr=True, rng=None, n_boot=N_BOOT):
    """
    Run OLS_FE, FirstStage, ReducedForm, IV2SLS and Wald on one sample.
    Returns a dict of every quantity the output contract needs.
    """
    df = sample
    y = df[ycol].to_numpy(float)
    x = df["dc_gw"].to_numpy(float)
    z = df[zcol].to_numpy(float)
    cl_idx, G = cluster_codes(df)
    N = len(df)

    W, K_W = design_matrix(df, extra_fe=extra_fe, extra_controls=extra_controls)
    # K is the parameter count of the FULL model: the absorbed terms in W PLUS the one
    # regressor of interest.  Getting this wrong silently shrinks every SE by ~0.1%.
    K = K_W + 1
    res = Residualiser(W)
    M = res(np.column_stack([y, x, z]))
    yt, xt, zt = M[:, 0], M[:, 1], M[:, 2]

    out = dict(n_obs=N, n_clusters=G, n_states=df["state_abbr"].nunique(),
               n_years=df["year"].nunique(), K=K, notes=[])

    ols = lin_fit(xt, yt, cl_idx, G, N, K)
    fs = lin_fit(zt, xt, cl_idx, G, N, K)
    rf = lin_fit(zt, yt, cl_idx, G, N, K)
    iv = iv_fit(zt, xt, yt, cl_idx, G, N, K)

    # Classic homoskedastic first-stage F = t^2 with the homoskedastic variance
    # (reported only for continuity with SI Table 9; the OP effective F governs).
    ssr = float(fs["u"] @ fs["u"]) if fs["u"] is not None else np.nan
    sigma2 = ssr / (N - K) if N > K else np.nan
    var_hom = sigma2 / fs["den"] if np.isfinite(fs["den"]) and fs["den"] > 0 else np.nan
    f_classic = fs["coef"] ** 2 / var_hom if np.isfinite(var_hom) and var_hom > 0 else np.nan
    f_eff = effective_F(fs)
    ok = bool(np.isfinite(f_eff) and f_eff >= OP_F_CRIT)
    if not ok:
        out["notes"].append("weak_iv")

    out.update(ols=ols, fs=fs, rf=rf, iv=iv, f_classic=f_classic, f_eff=f_eff,
               first_stage_ok=ok, yt=yt, xt=xt, zt=zt, cl_idx=cl_idx)

    # Wald ratio + delta-method variances.  Cov(b_RF, b_1st) is the
    # analytic cluster cross-covariance here; the bootstrap version is attached later
    # by the caller for the primary variant.
    wald = rf["coef"] / fs["coef"] if fs["coef"] not in (0, np.nan) else np.nan
    cov_rf_fs = cov_cluster(rf["score"], rf["den"], fs["score"], fs["den"],
                            cl_idx, G, N, K) if rf["u"] is not None else np.nan
    var_full = (rf["var"] + wald ** 2 * fs["var"] - 2 * wald * cov_rf_fs) / fs["coef"] ** 2
    var_simp = rf["var"] / fs["coef"] ** 2
    out.update(wald=wald, cov_rf_fs_analytic=cov_rf_fs,
               wald_se_delta_full=np.sqrt(var_full) if var_full > 0 else np.nan,
               wald_se_delta_simplified=np.sqrt(var_simp))

    units = outcome_units(ycol)
    if do_ar:
        lo, hi, kind, ar_notes = anderson_rubin_ci(zt, xt, yt, cl_idx, G, N, K, units)
        # ar_notes is kept as its own field as well as folded into the cell notes,
        # so the diagnostics rows can record which grid the endpoints are on
        # so that the widening flag is not discarded there.
        out.update(ar_low=lo, ar_high=hi, ar_kind=kind, ar_notes=list(ar_notes))
        out["notes"] += ar_notes
    else:
        out.update(ar_low=np.nan, ar_high=np.nan, ar_kind="not_run", ar_notes=[])

    if do_wcr and rng is not None:
        out["wcr_rf"] = wcr_bootstrap(zt, yt, res, cl_idx, G, N, K, rng, n_boot)
        out["wcr_fs"] = wcr_bootstrap(zt, xt, res, cl_idx, G, N, K, rng, n_boot)
    else:
        out["wcr_rf"] = out["wcr_fs"] = None

    return out

def resolve_benchmarks(sample, primary, bench_outcome="emp_5182"):
    """
    Resolve PANEL_FE_BENCHMARK and LOCAL_SHARE_MEDIAN from THIS run.

    The panel FE benchmark is the OLS_FE coefficient of dc_gw on `bench_outcome`
    with state and year fixed effects on the estimation sample -- the identical
    regression that SI tab:panel_fe_emp reports and that this script's own
    results_main OLS_FE row carries.  It is computed here, before the main loop,
    because the interpretation branches of are evaluated cell by cell
    inside that loop and the primary variant is not necessarily estimated first.

    The extra fit is cheap (no AR grid, no bootstrap) and consumes no random
    numbers, so the WCR and permutation streams are bit-identical to a run without
    it.  main() asserts afterwards that the coefficient equals the OLS_FE
    coefficient of the corresponding full cell exactly.

    LOCAL_SHARE_MEDIAN is the median local share over the UNITS of the estimation
    sample (one value per unit, local_share is unit-invariant).  The literal it
    replaces, 0.637, was the median over all 51 PwC rows including Alaska and
    Hawaii; over the 49 units of the analysis universe it is 0.644458.
    """
    global PANEL_FE_BENCHMARK, LOCAL_SHARE_MEDIAN, BENCHMARK_PROVENANCE
    if bench_outcome not in sample.columns:
        raise RuntimeError(f"cannot resolve PANEL_FE_BENCHMARK: {bench_outcome} is not in the "
            "estimation sample. The benchmark is computed, never assumed.")
    cell = estimate_cell(sample, bench_outcome, f"bartik_{primary}",
                         do_ar=False, do_wcr=False, rng=None)
    PANEL_FE_BENCHMARK = float(cell["ols"]["coef"])
    per_unit = sample.groupby("state_abbr")["local_share"].first()
    if per_unit.isna().all():
        raise RuntimeError("cannot resolve LOCAL_SHARE_MEDIAN: local_share is all-NaN.")
    LOCAL_SHARE_MEDIAN = float(per_unit.median())
    BENCHMARK_PROVENANCE = {
        "panel_fe_benchmark": (f"computed in this run: OLS_FE coefficient of dc_gw on {bench_outcome}, "
            f"state + year FE, cluster-robust, N={cell['n_obs']} G={cell['n_clusters']}, "
            f"leave-out variant '{primary}' (the OLS FE column is invariant to the "
            f"instrument). SE {cell['ols']['se']:.6g}."),
        "local_share_median": (f"computed in this run: median of local_share over the "
            f"{int(per_unit.notna().sum())} units of the estimation sample"),
    }
    return dict(benchmark=PANEL_FE_BENCHMARK, benchmark_se=float(cell["ols"]["se"]),
                local_share_median=LOCAL_SHARE_MEDIAN,
                n_obs=cell["n_obs"], n_clusters=cell["n_clusters"])

def cell_rows(cell, outcome, spec, variant, baseline, extra_note=""):
    """Turn one estimated cell into the five contract rows."""
    G = cell["n_clusters"]
    notes = list(dict.fromkeys(cell["notes"] + ([extra_note] if extra_note else [])))
    common = dict(outcome=outcome, outcome_units=outcome_units(outcome), specification=spec,
        leave_out_variant=variant, share_baseline=baseline, se_type="cluster_state",
        n_obs=cell["n_obs"], n_clusters=G, n_states=cell["n_states"],
        n_years=cell["n_years"], fs_coef=cell["fs"]["coef"], fs_se=cell["fs"]["se"],
        fs_F=cell["f_classic"], fs_F_effective=cell["f_eff"], fs_F_crit=OP_F_CRIT,
        first_stage_ok=cell["first_stage_ok"],
    )
    rows = []
    for est, r, r2 in [("OLS_FE", cell["ols"], cell["ols"]["r2"]),
                       ("FirstStage", cell["fs"], cell["fs"]["r2"]),
                       ("ReducedForm", cell["rf"], cell["rf"]["r2"]),
                       ("IV2SLS", cell["iv"], cell["iv"]["r2"])]:
        t, p, lo, hi = t_ci(r["coef"], r["se"], G)
        ci_type = "cluster_t"
        row_notes = list(notes)
        # gate: when the first stage is weak the AR set BECOMES the
        # reported inference for the 2SLS row; the point estimate is still written
        # but is flagged and must not be quoted.
        if est == "IV2SLS" and not cell["first_stage_ok"] and np.isfinite(cell["ar_low"]):
            lo, hi, ci_type = cell["ar_low"], cell["ar_high"], "AR"
            row_notes.append("AR_is_reported_inference")
        rows.append(dict(common, estimator=est, coef=r["coef"], se=r["se"], t_stat=t,
                         p_value=p, ci_low=lo, ci_high=hi, ci_type=ci_type,
                         r2_within=r2, mde_80=MDE_MULT * r["se"],
                         notes=";".join(row_notes)))
    # Wald: headline SE is the 2SLS clustered SE, NOT the delta-method one
    t, p, lo, hi = t_ci(cell["wald"], cell["iv"]["se"], G)
    rows.append(dict(common, estimator="Wald", coef=cell["wald"], se=cell["iv"]["se"],
                     t_stat=t, p_value=p, ci_low=lo, ci_high=hi, ci_type="cluster_t",
                     r2_within=cell["iv"]["r2"], mde_80=MDE_MULT * cell["iv"]["se"],
                     notes=";".join(notes + ["se_is_2SLS_clustered_not_delta"])))
    return rows

# =============================================================================
# Rotemberg weights and state influence
# =============================================================================

def rotemberg_year_weights(sample, ycol, zcol, share_col, shift_col, extra_fe=None):
    """
    Rotemberg (1983) weights in the Goldsmith-Pinkham, Sorkin & Swift (2020) sense,
    taken over YEAR-SHOCKS rather than industries.

    The canonical GPSS decomposition is over industries k.  This design has a single
    sector, so the canonical decomposition collapses to a weight of 1 -- it is
    degenerate, and requires saying so instead of printing a fake
    industry table.  The substitute treats the annual national increments as the
    shocks:

        Bartik_{s,t} = sum_tau 1{t=tau} * Share_s * Gbar_tau ,
        Z^{(tau)}_{s,t} = 1{t=tau} * Share_s ,        (two-way demeaned -> Ztil)

        alpha_tau = Gbar_tau (Ztil^tau ' Xtil) / sum_nu Gbar_nu (Ztil^nu ' Xtil)
        beta_tau  = (Ztil^tau ' Ytil) / (Ztil^tau ' Xtil)
        beta_Bartik = sum_tau alpha_tau beta_tau

    Gbar_tau is the share-weighted mean leave-out shift increment in year tau (the
    leave-out construction makes the shift mildly state-specific; the share-weighted
    mean is the natural scalar summary).  The 9 demeaned Z^(tau) have rank 8 because
    sum_tau Z^(tau) = Share_s is collinear with the state FE -- expected, not a bug.
    """
    df = sample
    W, K = design_matrix(df, extra_fe=extra_fe)
    res = Residualiser(W)
    y = df[ycol].to_numpy(float)
    x = df["dc_gw"].to_numpy(float)
    z = df[zcol].to_numpy(float)
    yt, xt, zt = res(y), res(x), res(z)

    years = np.sort(df["year"].unique())
    share = df[share_col].to_numpy(float)
    shift = df[shift_col].to_numpy(float)

    # Gbar_tau: share-weighted mean of (shift_t - shift_base) within year tau.
    # bartik = share * (shift - shift_base) so (shift - shift_base) = bartik/share
    # wherever share > 0; use the share-weighted mean, which is sum_s share_s * G_s.
    gbar = {}
    for tau in years:
        m = df["year"].to_numpy() == tau
        w = share[m]
        g_s = np.divide(z[m], w, out=np.zeros_like(w), where=w > 0)
        gbar[tau] = float((w * g_s).sum() / w.sum()) if w.sum() > 0 else 0.0

    Zsub = np.column_stack([(df["year"].to_numpy() == tau).astype(float) * share
                            for tau in years])
    Zsub_t = res(Zsub)
    rank = int(np.linalg.matrix_rank(Zsub_t))

    num = np.array([gbar[tau] * float(Zsub_t[:, i] @ xt) for i, tau in enumerate(years)])
    denom_zx = np.array([float(Zsub_t[:, i] @ xt) for i in range(len(years))])
    alpha = num / num.sum() if num.sum() != 0 else np.full(len(years), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_tau = np.array([float(Zsub_t[:, i] @ yt) / denom_zx[i]
                             if denom_zx[i] != 0 else np.nan
                             for i in range(len(years))])

    beta_bartik = float(zt @ yt) / float(zt @ xt)
    recon = float(np.nansum(alpha * beta_tau))
    sum_alpha = float(np.nansum(alpha))
    rel_err = abs(recon - beta_bartik) / abs(beta_bartik) if beta_bartik != 0 else np.nan
    return dict(years=years, alpha=alpha, beta_tau=beta_tau, gbar=gbar, rank=rank,
                sum_alpha=sum_alpha, recon=recon, beta_bartik=beta_bartik,
                rel_err=rel_err,
                n_negative=int((alpha < 0).sum()),
                sum_negative=float(alpha[alpha < 0].sum()) if (alpha < 0).any() else 0.0,
                valid=bool(abs(sum_alpha - 1) < 1e-6 and np.isfinite(rel_err) and rel_err < 0.05))

def state_influence(sample, ycol, zcol, extra_fe=None):
    """
    State shares of the first-stage and reduced-form identifying moments:

        infl_FS_s = sum_t Ztil_{s,t} Xtil_{s,t} / sum_{s'} sum_t Ztil Xtil
        infl_RF_s = sum_t Ztil_{s,t} Ytil_{s,t} / sum_{s'} sum_t Ztil Ytil

    This is the decomposition that answers which units drive the
    estimate.  It also exposes the design's real fragility: the numerator and the
    denominator of the Wald ratio can be driven by DIFFERENT states.
    """
    df = sample
    W, K = design_matrix(df, extra_fe=extra_fe)
    res = Residualiser(W)
    yt, xt, zt = res(df[ycol].to_numpy(float)), res(df["dc_gw"].to_numpy(float)), \
        res(df[zcol].to_numpy(float))
    g = pd.DataFrame({"state_abbr": df["state_abbr"].values,
                      "zx": zt * xt, "zy": zt * yt}).groupby("state_abbr").sum()
    infl_fs = g["zx"] / g["zx"].sum()
    infl_rf = g["zy"] / g["zy"].sum()
    rho = sps.spearmanr(infl_fs.values, infl_rf.values).statistic
    o = infl_fs.sort_values(ascending=False)
    return dict(infl_fs=infl_fs, infl_rf=infl_rf, spearman=float(rho),
                top1=float(o.iloc[:1].sum()), top3=float(o.iloc[:3].sum()),
                top5=float(o.iloc[:5].sum()), hhi=float((infl_fs ** 2).sum()))

# =============================================================================
# Share exogeneity: pre-trends and balance (mirrors r02)
# =============================================================================

def _hc1_ols(y, X):
    """OLS with HC1 standard errors -- exactly what the price analysis used."""
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    u = y - X @ b
    n, k = X.shape
    k = int(np.linalg.matrix_rank(X))
    meat = (X * (u ** 2)[:, None]).T @ X
    V = XtXi @ meat @ XtXi * (n / (n - k))
    return b, np.sqrt(np.diag(V)), n, k

def pretrend_test(panel_full, ycol, share_col):
    """
    Delta Y_{s,tau} = Intercept_tau + DivisionFE_s + gamma_tau * Share_s + eps.
    One observation per state per year-pair, HC1 SEs, DivisionFE in place of the
    price analysis's ISOFE.  Pre-period pairs are 2016-17, 2017-18, 2018-19 (only
    three are available because the capacity series starts in 2016 -- versus four in
    the price analysis; forbids padding this).
    """
    rows = []
    d = panel_full[(panel_full["year"] >= YEAR_MIN) & (panel_full["year"] <= YEAR_MAX)]
    d = d[d["emp_available"] == 1]
    for tau in range(YEAR_MIN + 1, YEAR_MAX + 1):
        a = d[d["year"] == tau - 1][["state_abbr", "census_division", share_col, ycol]]
        b = d[d["year"] == tau][["state_abbr", ycol]]
        m = a.merge(b, on="state_abbr", suffixes=("_0", "_1")).dropna()
        if len(m) < 12:
            continue
        dy = (m[f"{ycol}_1"] - m[f"{ycol}_0"]).to_numpy(float)
        div = pd.get_dummies(m["census_division"], drop_first=True, dtype=float).values
        X = np.hstack([np.ones((len(m), 1)), m[[share_col]].to_numpy(float), div])
        bb, se, n, k = _hc1_ols(dy, X)
        t = bb[1] / se[1] if se[1] > 0 else np.nan
        p = 2 * sps.t.sf(abs(t), df=n - k) if np.isfinite(t) else np.nan
        rows.append(dict(pair=f"{tau-1}_{tau}", gamma=float(bb[1]), se=float(se[1]),
                         p=float(p), n=n, is_pre=tau <= 2019))
    return rows

def balance_test(panel_full, share_col):
    """
    Characteristic_s = Intercept + DivisionFE_s + gamma * Share_s + eps, HC1 SEs.
    All characteristics are measured at or before 2019, so they are genuinely
    pre-determined.  gdp_pc_2024 is EXCLUDED: it is a 2024 measurement, therefore
    post-treatment, and putting it in a "pre-determined characteristics" table would
    be wrong.

    Pre-committed reading (written here BEFORE the numbers are seen): emp_5182_2016
    will almost certainly load on share2019 -- states with more 2019 capacity had
    more 2016 data-center employment.  That is mechanical and is absorbed by state FE
    in the estimating equation.  The rows that MATTER are the growth and dispersion
    rows: emp_5182_growth_1619, emp_5182_sd_pre.
    """
    d = panel_full[panel_full["emp_available"] == 1]
    pre = d[(d["year"] >= 2016) & (d["year"] <= 2019)]
    base = d.drop_duplicates("state_abbr").set_index("state_abbr")
    feat = pd.DataFrame(index=sorted(d["state_abbr"].unique()))
    y16 = d[d["year"] == 2016].set_index("state_abbr")
    y19 = d[d["year"] == 2019].set_index("state_abbr")
    for c in ["emp_5182", "emp_517", "emp_23", "emp_5415", "emp_51"]:
        feat[f"{c}_2016"] = y16[c]
    feat["emp_5182_mean_pre"] = pre.groupby("state_abbr")["emp_5182"].mean()
    feat["emp_5182_sd_pre"] = pre.groupby("state_abbr")["emp_5182"].std()
    feat["emp_517_mean_pre"] = pre.groupby("state_abbr")["emp_517"].mean()
    feat["earn_5182_2016"] = y16["earn_5182"]
    feat["earn_517_2016"] = y16["earn_517"]
    feat["emp_5182_growth_1619"] = (y19["emp_5182"] - y16["emp_5182"]) / y16["emp_5182"]
    feat["emp_517_growth_1619"] = (y19["emp_517"] - y16["emp_517"]) / y16["emp_517"]
    feat["n_neighbors"] = base["n_neighbors"]
    feat["share"] = base[share_col]
    feat["census_division"] = base["census_division"]

    rows = []
    for v in BALANCE_VARS:
        m = feat[[v, "share", "census_division"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(m) < 12:
            continue
        div = pd.get_dummies(m["census_division"], drop_first=True, dtype=float).values
        X = np.hstack([np.ones((len(m), 1)), m[["share"]].to_numpy(float), div])
        bb, se, n, k = _hc1_ols(m[v].to_numpy(float), X)
        t = bb[1] / se[1] if se[1] > 0 else np.nan
        p = 2 * sps.t.sf(abs(t), df=n - k) if np.isfinite(t) else np.nan
        rows.append(dict(var=v, gamma=float(bb[1]), se=float(se[1]), p=float(p), n=n))
    return rows

# =============================================================================
# Placebos
# =============================================================================

def permutation_placebo(sample, ycol, zcol, share_col, shift_col, base_year,
                        rng, n_perm=N_PERM):
    """
    Share-permutation placebo.  Permute Share_s across states, holding each state's
    leave-out shift path and everything else fixed; rebuild the instrument; recompute
    the reduced form and the Wald/2SLS coefficient.

    p = (1 + #{|beta_perm| >= |beta_obs|}) / (n_perm + 1).

    This tests whether the estimate reflects the ACTUAL spatial distribution of
    exposure or merely the national time path -- which is a stronger test than any
    sectoral placebo, and the supplied QWI extract contains no clean placebo sector
    anyway (every NAICS in it is either a claimed-effect sector or a superset of one).
    """
    df = sample
    cl_idx, G = cluster_codes(df)
    N = len(df)
    W, K_W = design_matrix(df)
    K = K_W + 1
    res = Residualiser(W)
    yt = res(df[ycol].to_numpy(float))
    xt = res(df["dc_gw"].to_numpy(float))
    zt = res(df[zcol].to_numpy(float))
    b_rf_obs = float(zt @ yt) / float(zt @ zt)
    b_iv_obs = float(zt @ yt) / float(zt @ xt)

    states = df.drop_duplicates("state_abbr")["state_abbr"].to_numpy()
    share = df.drop_duplicates("state_abbr").set_index("state_abbr")[share_col]
    # G_{s,t} = shift_{s,t} - shift_{s,base}: the state's own leave-out shift path
    base = df[df["year"] == base_year].set_index("state_abbr")[shift_col]
    gpath = df[shift_col].to_numpy(float) - df["state_abbr"].map(base).to_numpy(float)
    smap = {s: i for i, s in enumerate(states)}
    sidx = df["state_abbr"].map(smap).to_numpy(int)

    rf_perm, iv_perm = np.empty(n_perm), np.empty(n_perm)
    sh = share.reindex(states).to_numpy(float)
    for b in range(n_perm):
        zp = sh[rng.permutation(len(sh))][sidx] * gpath
        zpt = res(zp)
        dz = float(zpt @ zpt)
        dx = float(zpt @ xt)
        rf_perm[b] = float(zpt @ yt) / dz if dz > 0 else np.nan
        iv_perm[b] = float(zpt @ yt) / dx if dx != 0 else np.nan
    rf_perm = rf_perm[np.isfinite(rf_perm)]
    iv_perm = iv_perm[np.isfinite(iv_perm)]
    return dict(rf_obs=b_rf_obs, iv_obs=b_iv_obs,
        rf_p=(1 + int((np.abs(rf_perm) >= abs(b_rf_obs)).sum())) / (rf_perm.size + 1),
        iv_p=(1 + int((np.abs(iv_perm) >= abs(b_iv_obs)).sum())) / (iv_perm.size + 1),
        rf_mean=float(rf_perm.mean()), iv_mean=float(iv_perm.mean()),
        rf_q025=float(np.quantile(rf_perm, 0.025)), rf_q975=float(np.quantile(rf_perm, 0.975)),
        iv_q025=float(np.quantile(iv_perm, 0.025)), iv_q975=float(np.quantile(iv_perm, 0.975)),
        n_perm=int(min(rf_perm.size, iv_perm.size)))

def lead_placebo(panel_with_z, sample, ycol, zcol, lead=2):
    """
    Lead-instrument placebo: reduced form of Y_{s,t} on Bartik_{s,t+2} WHILE
    controlling for Bartik_{s,t}, State FE and Year FE.  A significant lead means
    future shocks predict current employment, i.e. anticipation or a differential
    trend.  The lead is available only where t+2 is inside the capacity panel, so N
    shrinks; the reduced N is reported.
    """
    lead_df = panel_with_z[["state_abbr", "year", zcol]].copy()
    lead_df["year"] = lead_df["year"] - lead
    lead_df = lead_df.rename(columns={zcol: "z_lead"})
    d = sample.merge(lead_df, on=["state_abbr", "year"], how="left").dropna(subset=["z_lead"])
    if len(d) < 30:
        return None
    cl_idx, G = cluster_codes(d)
    N = len(d)
    W, K_W = design_matrix(d, extra_controls=d[[zcol]].to_numpy(float))
    K = K_W + 1                      # + the lead instrument itself
    res = Residualiser(W)
    yt = res(d[ycol].to_numpy(float))
    zlt = res(d["z_lead"].to_numpy(float))
    r = lin_fit(zlt, yt, cl_idx, G, N, K)
    t, p, lo, hi = t_ci(r["coef"], r["se"], G)
    return dict(coef=r["coef"], se=r["se"], p=p, ci_low=lo, ci_high=hi, n=N, G=G)

# =============================================================================
# Net information-sector effect
# =============================================================================

def iv_cluster_full(y, Xall, Zall, cl_idx, G):
    """
    General cluster-robust 2SLS on an explicit design matrix.  Used only to build
    the STACKED / SUR-IV system so that its coefficients can be verified against the
    single-equation ones (Method A verification requirement).
    """
    ZZi = np.linalg.pinv(Zall.T @ Zall)
    ZX = Zall.T @ Xall
    A = ZX.T @ ZZi @ ZX
    Ai = np.linalg.pinv(A)
    b = Ai @ (ZX.T @ ZZi @ (Zall.T @ y))
    u = y - Xall @ b
    Xhat = Zall @ (ZZi @ ZX)
    sc = Xhat * u[:, None]
    k = Xall.shape[1]
    meat = np.zeros((k, k))
    for g in range(G):
        m = cl_idx == g
        s = sc[m].sum(axis=0)
        meat += np.outer(s, s)
    N = len(y)
    K = int(np.linalg.matrix_rank(Xall))
    V = Ai @ meat @ Ai * fs_correction(N, K, G)
    return b, V

def net_effect_stacked(sample, y1col, y2col, zcol):
    """
    Method A (primary): stacked / SUR-IV with state clustering.

    Long-stack the panel into 2N rows with an equation indicator D.  EVERY term --
    instrument, endogenous regressor, state FE, year FE -- is interacted with D, so
    the two equations are algebraically separate and each coefficient is numerically
    identical to its single-equation counterpart.  Clustering by state makes a
    cluster span BOTH equations, so the off-diagonal block of the cluster sandwich is
    a direct analytic estimate of Cov(beta_5182, beta_517).
    """
    df = sample
    n = len(df)
    cl_idx, G = cluster_codes(df)
    W, _ = design_matrix(df)
    x = df["dc_gw"].to_numpy(float)
    z = df[zcol].to_numpy(float)
    y1 = df[y1col].to_numpy(float)
    y2 = df[y2col].to_numpy(float)

    Z0 = np.zeros((n, W.shape[1]))
    z0 = np.zeros(n)
    Xall = np.vstack([
        np.hstack([x[:, None], np.zeros((n, 1)), W, Z0]),
        np.hstack([np.zeros((n, 1)), x[:, None], Z0, W]),
    ])
    Zall = np.vstack([
        np.hstack([z[:, None], np.zeros((n, 1)), W, Z0]),
        np.hstack([np.zeros((n, 1)), z[:, None], Z0, W]),
    ])
    yall = np.concatenate([y1, y2])
    cl_stack = np.concatenate([cl_idx, cl_idx])   # a cluster spans BOTH equations
    b, V = iv_cluster_full(yall, Xall, Zall, cl_stack, G)
    b1, b2 = float(b[0]), float(b[1])
    v11, v22, v12 = float(V[0, 0]), float(V[1, 1]), float(V[0, 1])
    return dict(b1=b1, b2=b2, v11=v11, v22=v22, v12=v12,
                net=b1 + b2, se_net=np.sqrt(max(v11 + v22 + 2 * v12, 0.0)),
                cov=v12, n_obs=2 * n, n_clusters=G)

def pairs_cluster_bootstrap(sample, ycols, zcol, rng, n_boot=N_BOOT):
    """
    Pairs cluster bootstrap over STATES (Method B, sec. 4.7).

    Resample the G states with replacement, keeping ALL years of a sampled state.
    Duplicated states are given fresh identifiers so they receive their own fixed
    effect and their own cluster -- the standard treatment.  Each replication
    returns b_1st and, for every outcome, b_RF and b_IV, so a single bootstrap
    supplies (i) Cov(b_RF, b_1st) for the delta method, (ii) Cov(b_5182, b_517) for
    the net effect, and (iii) percentile CIs.

    Replications whose effective first-stage F falls below 5 are DROPPED and counted
    -- never silently discarded.
    """
    states = sorted(sample["state_abbr"].unique())
    G0 = len(states)
    by_state = {s: sample[sample["state_abbr"] == s] for s in states}
    keep = {"fs": [], "rf": {c: [] for c in ycols}, "iv": {c: [] for c in ycols}}
    n_dropped = 0
    for b in range(n_boot):
        pick = rng.integers(0, G0, size=G0)
        frames = []
        for j, i in enumerate(pick):
            d = by_state[states[i]].copy()
            d["state_abbr"] = f"{states[i]}__{j}"
            frames.append(d)
        d = pd.concat(frames, ignore_index=True)
        cl_idx, G = cluster_codes(d)
        N = len(d)
        try:
            W, K_W = design_matrix(d)
            K = K_W + 1
            res = Residualiser(W)
            M = res(np.column_stack([d["dc_gw"].to_numpy(float),
                                     d[zcol].to_numpy(float)]
                                    + [d[c].to_numpy(float) for c in ycols]))
        except np.linalg.LinAlgError:
            n_dropped += 1
            continue
        xt, zt = M[:, 0], M[:, 1]
        fs = lin_fit(zt, xt, cl_idx, G, N, K)
        if not np.isfinite(fs["se"]) or fs["se"] <= 0 or effective_F(fs) < WEAK_F_FOR_BOOTSTRAP_DROP:
            n_dropped += 1
            continue
        keep["fs"].append(fs["coef"])
        for j, c in enumerate(ycols):
            yt = M[:, 2 + j]
            dz, dx = float(zt @ zt), float(zt @ xt)
            keep["rf"][c].append(float(zt @ yt) / dz)
            keep["iv"][c].append(float(zt @ yt) / dx)
    out = dict(n_kept=len(keep["fs"]), n_dropped=n_dropped,
               fs=np.array(keep["fs"]),
               rf={c: np.array(v) for c, v in keep["rf"].items()},
               iv={c: np.array(v) for c, v in keep["iv"].items()})
    return out

# =============================================================================
# Output assembly
# =============================================================================

def add_diag(diag, **kw):
    row = dict(diagnostic="", outcome="", leave_out_variant="", share_baseline="",
               subject="", statistic="", value=np.nan, se=np.nan, p_value=np.nan,
               n_obs=np.nan, threshold=np.nan, pass_fail="", notes="")
    row.update(kw)
    diag.append(row)

def write_table(rows, columns, path):
    df = pd.DataFrame(rows)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    df = df[columns]
    sort_cols = [c for c in ["outcome", "specification", "leave_out_variant", "estimator"]
                 if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")
    df.to_csv(path, index=False, float_format="%.6g", na_rep="", encoding="utf-8")
    return df

def print_cell(tag, outcome, cell):
    iv, ols, rf, fs = cell["iv"], cell["ols"], cell["rf"], cell["fs"]
    star = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    t_iv, p_iv, lo_iv, hi_iv = t_ci(iv["coef"], iv["se"], cell["n_clusters"])
    t_ol, p_ol, _, _ = t_ci(ols["coef"], ols["se"], cell["n_clusters"])
    t_rf, p_rf, _, _ = t_ci(rf["coef"], rf["se"], cell["n_clusters"])
    print(f"  {tag:<22s} {outcome:<22s} N={cell['n_obs']:<4d} G={cell['n_clusters']:<3d}"
          f" F_eff={cell['f_eff']:>8.1f} {'OK' if cell['first_stage_ok'] else 'WEAK'}")
    print(f"      first stage  b={fs['coef']:>12.4f} (se {fs['se']:.4f})")
    print(f"      OLS_FE       b={ols['coef']:>12.2f} (se {ols['se']:.2f}) p={p_ol:.4f}{star(p_ol)}")
    print(f"      ReducedForm  b={rf['coef']:>12.2f} (se {rf['se']:.2f}) p={p_rf:.4f}{star(p_rf)}")
    print(f"      IV2SLS       b={iv['coef']:>12.2f} (se {iv['se']:.2f}) p={p_iv:.4f}{star(p_iv)}"
          f"  95% CI [{lo_iv:.1f}, {hi_iv:.1f}]")
    if outcome_units(outcome) == "jobs_per_GW":
        inside = (lo_iv <= _benchmark() <= hi_iv)
        print(f"      vs panel FE benchmark {_benchmark():.0f} jobs/GW: "
              f"CI {'CONTAINS' if inside else 'EXCLUDES'} it;  "
              f"CI {'contains' if lo_iv <= 0 <= hi_iv else 'excludes'} 0;  "
              f"MDE_80 = {MDE_MULT * iv['se']:.0f}")

# =============================================================================
# Interpretation branches -- pre-committed, applied mechanically
# =============================================================================

def branch_5182(coef, se, lo, hi, p):
    if not np.isfinite(coef):
        return "NOT_ESTIMATED: no coefficient"
    sig = p < 0.05
    if sig and coef < 0:
        return "F_negative"
    if sig and coef > 0:
        if lo > _benchmark():
            return "A_IV_above_FE"
        if hi < _benchmark():
            return "C_IV_below_FE"
        return "B_IV_agrees_FE"
    contains_0 = lo <= 0 <= hi
    contains_fe = lo <= _benchmark() <= hi
    if contains_0 and contains_fe:
        return "D_underpowered"
    if not contains_fe:
        return "E_precise_null"
    return "D_underpowered"

def branch_517(coef, p):
    if not np.isfinite(coef):
        return "NOT_ESTIMATED: no coefficient"
    if p < 0.05 and coef < 0:
        return "T_substitution_confirmed"
    if p < 0.05 and coef > 0:
        return "T_complementarity"
    return "T_no_causal_substitution"

def branch_net(net, se, b1, b2, G):
    if not np.isfinite(net) or not np.isfinite(se):
        return "NOT_ESTIMATED: no net estimate"
    t, p, lo, hi = t_ci(net, se, G)
    if p < 0.05:
        return "N_net_positive" if net > 0 else "N_net_negative"
    # insignificant: distinguish an informative reallocation result from a wide CI
    if lo < -_benchmark() and hi > _benchmark():
        return "N_uninformative"
    if b1 > 0 and b2 < 0:
        return "N_net_zero"
    return "N_uninformative"

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="State-level Bartik shift-share IV: DC capacity -> employment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (--tag is REQUIRED on every run that writes output):
    python3 r06_employment_bartik_iv.py --tag _union
    python3 r06_employment_bartik_iv.py --tag _union --outcomes headline
    python3 r06_employment_bartik_iv.py --tag division --outcomes employment --leave-out division
    python3 r06_employment_bartik_iv.py --tag headline2 --outcome-cols emp_5182 emp_517 --skip-robustness
    python3 r06_employment_bartik_iv.py --tag base2016 --share-baseline 2016
    python3 r06_employment_bartik_iv.py --list-outcomes
        """)
    parser.add_argument("--panel", default=str(DATA_PATHS["panel"]),
                        help="state-year panel CSV (default: %(default)s)")
    parser.add_argument("--adjacency", default=str(DATA_PATHS["adjacency"]))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--outcomes", default="all", choices=sorted(OUTCOME_SETS),
                        help="named outcome set (default: %(default)s)")
    parser.add_argument("--outcome-cols", nargs="+", default=None,
                        help="explicit outcome columns; overrides --outcomes")
    parser.add_argument("--leave-out", nargs="+", default=LEAVE_OUT_VARIANTS,
                        choices=LEAVE_OUT_VARIANTS,
                        help="leave-out variants for results_main (default: all three)")
    parser.add_argument("--primary-leave-out", default=PRIMARY_LEAVE_OUT,
                        choices=LEAVE_OUT_VARIANTS)
    parser.add_argument("--share-baseline", type=int, default=SHARE_BASE_PRIMARY,
                        choices=[2016, 2019])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="skip the pairs cluster bootstrap (debug only)")
    parser.add_argument("--skip-placebo", action="store_true")
    parser.add_argument("--skip-wcr", action="store_true")
    # --tag is REQUIRED, with no default, so that a re-run can never write into
    # an unsuffixed artefact name that an earlier run already holds.  The tag
    # must name the universe and the primary leave-out, e.g. "union", "contig".
    parser.add_argument("--tag", required=True,
                        help="REQUIRED non-empty suffix appended to every output "
                             "filename; name the universe/variant it encodes "
                             "(e.g. --tag union). There is deliberately no default.")
    parser.add_argument("--list-outcomes", action="store_true")

    # --list-outcomes is informational and must keep working without a tag, so it is
    # served before argparse can enforce the requirement.
    if "--list-outcomes" in sys.argv[1:]:
        for k, v in OUTCOME_SETS.items():
            print(f"{k:<12s} {v}")
        sys.exit(0)

    args = parser.parse_args()

    # Only whitespace is stripped.  The shipped run is `--tag _union`, which must keep
    # producing the `__union` double-underscore names already on disk and already cited
    # in the SI; stripping underscores here would silently rename every artefact.
    tag = args.tag.strip()
    if not tag.strip("_") or tag.strip("_").lower() in {"default", "none", "null",
                                                        "tmp", "test"}:
        parser.error("--tag must be a non-empty, meaningful suffix that names the "
                     "universe/variant of this run (got %r). Placeholder tags are "
                     "rejected so an artefact name can never be silently reused."
                     % (args.tag,))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}"
    tee = Tee(out_dir / f"run_log{sfx}.txt")
    sys.stdout = tee
    t_start = time.time()

    try:
        rng = np.random.default_rng(args.seed)

        outcomes = args.outcome_cols if args.outcome_cols else OUTCOME_SETS[args.outcomes]
        baseline = args.share_baseline
        primary = args.primary_leave_out
        variants = list(dict.fromkeys(args.leave_out))
        if primary not in variants:
            # every downstream diagnostic is indexed on the primary variant, so it must
            # actually be estimated; prepend it rather than fail late.
            variants = [primary] + variants

        banner("STATE-LEVEL BARTIK SHIFT-SHARE IV  --  employment ")
        print(f"  panel        : {relpath(args.panel)}")
        print(f"  outcomes     : {outcomes}")
        print(f"  leave-out    : {variants}   (primary = {primary})")
        print(f"  share baseline: {baseline}")
        print(f"  seed         : {args.seed}   n_boot={args.n_boot}  n_perm={args.n_perm}")
        print(f"  output       : {relpath(out_dir)}")

        # ------------------------------------------------------------------ [0]
        print("\n[0] Loading panel and adjacency...")
        panel = pd.read_csv(args.panel)
        with open(args.adjacency) as f:
            adjacency = json.load(f)
        print(f"  panel: {panel.shape[0]} rows x {panel.shape[1]} cols; "
              f"sha256={sha256_of(args.panel)[:16]}...")

        panel, adjacency, scope = apply_scope(panel, adjacency)

        # the scoped panel is asserted against the module that owns
        # the universe, not against a literal in this file.  Strict in both
        # directions -- an out-of-scope unit still present AND an in-scope unit
        # missing both raise.  A sample that retains AK and HI while dropping DC
        # is the exact complement of the intended universe on both ends, and a
        # one-sided check would pass it.
        check_universe(panel, "state_abbr", where="apply_scope output (r06 bartik)")
        check_universe(adjacency.keys(), where="scoped adjacency map")
        print(f"  universe check: PASS -- {universe_describe().splitlines()[0]}")

        diag = []
        add_diag(diag, diagnostic="sample_composition", subject="scope_excluded_units",
                 statistic="value", value=float(len(scope["dropped"])), n_obs=len(panel),
                 notes=";".join(scope["dropped"])
                       + "|reason=out_of_scope_isolated_grid_not_data_availability")
        add_diag(diag, diagnostic="sample_composition", subject="scope_excluded_pct_gw_2019",
                 statistic="value", value=float(scope["dropped_pct_of_2019"]),
                 n_obs=len(panel), notes="share of 2019 national capacity removed by scope")
        n_pass, failures = preflight(panel, adjacency, diag)
        if failures:
            # any pre-flight failure is a hard abort, no results file.
            sys.stdout = tee.terminal
            tee.close()
            print("ABORT: pre-flight assertions failed:", failures)
            sys.exit(2)

        # ------------------------------------------------------------------ [1]
        print("\n[1] Constructing the instrument (3 leave-out variants)...")
        pz = attach_instruments(panel, adjacency, baseline)
        pz16 = attach_instruments(panel, adjacency, 2016)
        pz19 = attach_instruments(panel, adjacency, 2019)

        # The exported columns are DERIVED from LEAVE_OUT_VARIANTS, never listed
        # by hand, so that adding a variant to LEAVE_OUT_VARIANTS cannot
        # silently fail to export it.
        inst_cols = (["state_abbr", "year", "share2016", "share2019", "dc_gw", "us_dc_gw"]
                     + [f"shift_{v}" for v in LEAVE_OUT_VARIANTS]
                     + [f"bartik_{v}" for v in LEAVE_OUT_VARIANTS])
        missing_cols = [c for c in inst_cols if c not in pz19.columns]
        if missing_cols:
            raise RuntimeError(f"instrument export is missing columns: {missing_cols}")
        inst_out = pz19[inst_cols]
        inst_path = out_dir / f"bartik_instrument_panel{sfx}.csv"
        inst_out.to_csv(inst_path, index=False, float_format="%.6g", na_rep="", encoding="utf-8")
        print(f"  wrote {inst_path.name}  ({len(inst_out)} rows x {len(inst_cols)} cols, "
              f"including the primary variant '{PRIMARY_LEAVE_OUT}')")

        for v in LEAVE_OUT_VARIANTS:
            zz = pz[f"bartik_{v}"]
            print(f"  bartik_{v:<9s} range [{zz.min():.4f}, {zz.max():.4f}] GW, "
                  f"mean {zz.mean():.4f}, zero at {baseline}: "
                  f"{np.allclose(pz.loc[pz['year'] == baseline, f'bartik_{v}'], 0)}")
        corr = pz[[f"bartik_{v}" for v in LEAVE_OUT_VARIANTS]].corr()
        print("\n  Instrument correlations across leave-out variants "
              "(near-collinear => the choice is numerically immaterial):")
        print(corr.round(4).to_string())
        # Every unordered pair, derived from LEAVE_OUT_VARIANTS rather than
        # listed by index, so no pair involving the primary variant is omitted.
        inst_corr = {f"{a}_{b}": float(corr.loc[f"bartik_{a}", f"bartik_{b}"])
                     for i, a in enumerate(LEAVE_OUT_VARIANTS)
                     for b in LEAVE_OUT_VARIANTS[i + 1:]}

        # ------------------------------------------------------------------ [2]
        print("\n[2] Building the estimation sample...")
        sample, singletons = build_sample(pz)
        # the ESTIMATION SAMPLE itself -- not merely the panel it was
        # cut from -- is asserted against the universe module at start-up.  This is
        # the check that catches a "50 state / 439 observation" sample, in which
        # AK and HI survive and DC is silently dropped.
        check_universe(sample, "state_abbr", where="primary estimation sample")
        print(f"  universe check on the estimation sample: PASS "
              f"({sample['state_abbr'].nunique()} units == N_ANALYSIS_UNITS)")
        cnt = sample.groupby("state_abbr")["year"].nunique()
        unbalanced = {s: int(c) for s, c in cnt.items() if c < sample["year"].nunique()}
        zero_share = sorted(sample.loc[sample[f"share{baseline}"] == 0, "state_abbr"].unique())
        partial_rows = sample.loc[sample["partial_year"] == 1, ["state_abbr", "year"]] \
                             .values.tolist()
        print(f"  N={len(sample)}  states={sample['state_abbr'].nunique()}  "
              f"years={sample['year'].nunique()} ({sample['year'].min()}-{sample['year'].max()})")
        print(f"  scope: contiguous US, {N_ANALYSIS_UNITS} units "
              f"(excluded by scope: {list(scope['dropped'])})")
        if singletons:
            print(f"  MIN_OBS_YEARS={MIN_OBS_YEARS} rule BINDS on: {singletons}")
        else:
            print(f"  MIN_OBS_YEARS={MIN_OBS_YEARS} rule does NOT bind: every one of the "
                  f"{sample['state_abbr'].nunique()} in-scope units has >= {MIN_OBS_YEARS} "
                  f"observed years (min = {int(cnt.min())})")
        print(f"  unbalanced states: {unbalanced}")
        print(f"  zero-share states (kept, contribute to FE not to identification): {zero_share}")
        print(f"  partial_year rows retained in the main spec: {partial_rows}")
        for k, v in [("n_obs", len(sample)), ("n_states", sample["state_abbr"].nunique()),
                     ("n_clusters", sample["state_abbr"].nunique()),
                     ("n_years", sample["year"].nunique())]:
            add_diag(diag, diagnostic="sample_composition", subject=k, statistic="value",
                     value=float(v), n_obs=len(sample))
        add_diag(diag, diagnostic="sample_composition", subject="states_dropped",
                 statistic="value", value=float(len(singletons)), n_obs=len(sample),
                 notes=";".join(singletons) + "|reason=fewer_than_3_observed_years")
        add_diag(diag, diagnostic="sample_composition", subject="zero_share_states",
                 statistic="value", value=float(len(zero_share)), n_obs=len(sample),
                 notes=";".join(zero_share))

        # design moments on the primary variant
        Wd, Kd = design_matrix(sample)
        rd = Residualiser(Wd)
        Md = rd(np.column_stack([sample[f"bartik_{primary}"].to_numpy(float),
                                 sample["dc_gw"].to_numpy(float),
                                 sample["emp_5182"].to_numpy(float)]))
        sd_z, sd_x, sd_y = Md.std(axis=0, ddof=1)
        corr_zx = float(np.corrcoef(Md[:, 0], Md[:, 1])[0, 1])
        print(f"\n  Design moments after two-way demeaning ({primary} leave-out):")
        print(f"    sd(instrument) = {sd_z:.4f} GW   sd(dc_gw) = {sd_x:.4f} GW   "
              f"sd(emp_5182) = {sd_y:,.0f} jobs   corr(Z,X|FE) = {corr_zx:.4f}")

        # ---- instrument correlations, written to a FILE --------------------
        # SI tab:leaveout_compare prints the six pairwise correlations of the
        # residualised instruments on the estimation sample.  Until now those six
        # numbers existed only in a stdout capture (si_tables_contig.txt) and in no
        # machine-written result file, so the published table could not be checked
        # against an artefact.  They are computed here, on the same object the SI
        # caption describes -- after absorbing state and year fixed effects, on the
        # estimation sample -- and written to results_diagnostics.
        zres = {v: rd(sample[f"bartik_{v}"].to_numpy(float)) for v in LEAVE_OUT_VARIANTS}
        inst_corr_within = {}
        for i, a in enumerate(LEAVE_OUT_VARIANTS):
            for b in LEAVE_OUT_VARIANTS[i + 1:]:
                r_ab = float(np.corrcoef(zres[a], zres[b])[0, 1])
                inst_corr_within[f"{a}_{b}"] = r_ab
                add_diag(diag, diagnostic="instrument_correlation",
                         leave_out_variant=f"{a}|{b}", share_baseline=baseline,
                         subject="within_state_year_fe", statistic="correlation",
                         value=r_ab, n_obs=len(sample),
                         notes="pairwise correlation of the two leave-out instruments "
                               "after absorbing state and year FE, on the estimation "
                               "sample; this is the object SI tab:leaveout_compare prints")
                add_diag(diag, diagnostic="instrument_correlation",
                         leave_out_variant=f"{a}|{b}", share_baseline=baseline,
                         subject="raw_panel_levels", statistic="correlation",
                         value=inst_corr[f"{a}_{b}"], n_obs=len(pz),
                         notes="same pair before residualisation, on the full panel")
        print("\n  Instrument correlations after state and year FE, estimation sample "
              "(the six pairs SI tab:leaveout_compare reports):")
        for k, v in inst_corr_within.items():
            print(f"    {k:<20s} {v:.4f}")
        print(f"    minimum pairwise correlation = {min(inst_corr_within.values()):.4f}")

        # ---- benchmarks resolved from THIS run, not declared -----------------
        bench = resolve_benchmarks(sample, primary)
        print(f"\n  Benchmarks resolved from this run (no literals):")
        print(f"    panel FE benchmark = {bench['benchmark']:.4f} jobs/GW "
              f"(SE {bench['benchmark_se']:.4f}, N={bench['n_obs']}, G={bench['n_clusters']}) "
              f"-- OLS_FE emp_5182, the regression SI tab:panel_fe_emp reports")
        print(f"    median local share = {bench['local_share_median']:.6f} over the "
              f"{sample['state_abbr'].nunique()} units of the estimation sample")
        add_diag(diag, diagnostic="sample_composition", subject="panel_fe_benchmark",
                 statistic="value", value=float(bench["benchmark"]), se=bench["benchmark_se"],
                 n_obs=bench["n_obs"],
                 notes=BENCHMARK_PROVENANCE["panel_fe_benchmark"])
        add_diag(diag, diagnostic="sample_composition", subject="local_share_median",
                 statistic="value", value=float(bench["local_share_median"]),
                 n_obs=len(sample), notes=BENCHMARK_PROVENANCE["local_share_median"])

        # ------------------------------------------------------------------ [3]
        banner("[3] MAIN ESTIMATES  (results_main.csv)")
        main_rows = []
        cells = {}
        for v in variants:
            zcol = f"bartik_{v}"
            print(f"\n  --- leave-out variant: {v} | share baseline {baseline} ---")
            for oc in outcomes:
                spec_label = "log" if oc.startswith("log_") else "level"
                try:
                    cell = estimate_cell(sample, oc, zcol, rng=rng,
                                         do_wcr=(not args.skip_wcr) and v == primary,
                                         n_boot=args.n_boot)
                except Exception as e:
                    print(f"    ERROR {oc} [{v}]: {e}")
                    main_rows.append(dict(outcome=oc, outcome_units=outcome_units(oc),
                                          specification=spec_label, leave_out_variant=v,
                                          share_baseline=baseline, estimator="IV2SLS",
                                          coef=np.nan, notes=f"exception:{e}"))
                    continue
                cells[(oc, v)] = cell
                main_rows += cell_rows(cell, oc, spec_label, v, baseline)
                print_cell(v, oc, cell)

                # -------- per-cell diagnostics --------
                add_diag(diag, diagnostic="first_stage_F", outcome=oc, leave_out_variant=v,
                         share_baseline=baseline, subject="classic_homoskedastic",
                         statistic="F", value=cell["f_classic"], n_obs=cell["n_obs"],
                         notes="reported for continuity with SI Table 9 only")
                add_diag(diag, diagnostic="first_stage_F_effective", outcome=oc,
                         leave_out_variant=v, share_baseline=baseline,
                         subject="olea_pflueger_K1", statistic="F", value=cell["f_eff"],
                         n_obs=cell["n_obs"], threshold=OP_F_CRIT,
                         pass_fail="PASS" if cell["first_stage_ok"] else "FAIL",
                         notes="F_eff = (cluster-robust first-stage t)^2, exact for K=1")
                # The AR notes carry the grid the endpoints are on; keeping them out
                # of these rows would leave a reader of the shipped file unable
                # to tell a 25-step endpoint from a 100-step one.
                ar_note = ";".join([cell["ar_kind"]] + cell.get("ar_notes", []))
                add_diag(diag, diagnostic="anderson_rubin_ci", outcome=oc,
                         leave_out_variant=v, share_baseline=baseline, subject="lower",
                         statistic="ci_low", value=cell["ar_low"], n_obs=cell["n_obs"],
                         notes=ar_note)
                add_diag(diag, diagnostic="anderson_rubin_ci", outcome=oc,
                         leave_out_variant=v, share_baseline=baseline, subject="upper",
                         statistic="ci_high", value=cell["ar_high"], n_obs=cell["n_obs"],
                         notes=ar_note)
                add_diag(diag, diagnostic="mde", outcome=oc, leave_out_variant=v,
                         share_baseline=baseline, subject="IV2SLS", statistic="value",
                         value=MDE_MULT * cell["iv"]["se"], n_obs=cell["n_obs"],
                         threshold=_benchmark(),
                         notes=f"MDE_80 = {MDE_MULT} x clustered SE; benchmark = panel FE "
                               f"{_benchmark():.0f} jobs/GW (computed in this run, "
                               f"not a literal)")
                add_diag(diag, diagnostic="wald_se_delta_full", outcome=oc,
                         leave_out_variant=v, share_baseline=baseline, subject="analytic",
                         statistic="value", value=cell["wald_se_delta_full"],
                         n_obs=cell["n_obs"],
                         notes="all three terms incl. Cov(b_RF,b_1st)")
                add_diag(diag, diagnostic="wald_se_delta_simplified", outcome=oc,
                         leave_out_variant=v, share_baseline=baseline,
                         subject="price_analysis_form", statistic="value",
                         value=cell["wald_se_delta_simplified"], n_obs=cell["n_obs"],
                         notes="Var(b_RF)/b_1st^2 -- the form the price analysis used; "
                               "reported as a DISCLOSURE, never as the headline SE")
                add_diag(diag, diagnostic="wald_cov_rf_fs", outcome=oc,
                         leave_out_variant=v, share_baseline=baseline, subject="analytic",
                         statistic="value", value=cell["cov_rf_fs_analytic"],
                         n_obs=cell["n_obs"], notes="cluster-robust cross-covariance")
                if cell["wcr_rf"] is not None:
                    for lbl, w in [("reduced_form", cell["wcr_rf"]), ("first_stage", cell["wcr_fs"])]:
                        add_diag(diag, diagnostic="wcr_bootstrap", outcome=oc,
                                 leave_out_variant=v, share_baseline=baseline, subject=lbl,
                                 statistic="pvalue", value=w["p"], p_value=w["p"],
                                 n_obs=cell["n_obs"],
                                 notes=f"WCR Rademacher B={args.n_boot} seed={args.seed}; "
                                       f"CI=[{w['ci_low']:.6g},{w['ci_high']:.6g}]")

        # The benchmark was estimated once before the loop so
        # branches could be applied cell by cell.  Assert it against the full cell
        # now that the loop has run: two independently constructed objects must
        # agree exactly, or the benchmark this run reported against is not the
        # regression this run published.  No check may compare a number only to
        # itself.
        bench_cell = cells.get(("emp_5182", primary))
        if bench_cell is not None:
            gap = abs(float(bench_cell["ols"]["coef"]) - _benchmark())
            if gap > 1e-9:
                raise RuntimeError(f"panel FE benchmark {_benchmark():.10f} disagrees with the OLS_FE "
                    f"coefficient of the emp_5182/{primary} cell "
                    f"{float(bench_cell['ols']['coef']):.10f} (gap {gap:.3g}).")
            print(f"\n  Benchmark cross-check: OLS_FE emp_5182 [{primary}] = "
                  f"{float(bench_cell['ols']['coef']):.6f} == resolved benchmark "
                  f"{_benchmark():.6f}  (max gap {gap:.3g})")

        # ------------------------------------------------------------------ [4]
        banner("[4] ROTEMBERG DECOMPOSITION AND STATE INFLUENCE")
        print("  NOTE: the canonical GPSS Rotemberg decomposition is over INDUSTRIES.")
        print("        This design has a single sector, so it collapses to a weight of 1.")
        print("        Substituted: (a) Rotemberg weights over YEAR-shocks, "
              "(b) state influence shares.\n")
        share_col = f"share{baseline}"
        for oc in outcomes:
            if (oc, primary) not in cells:
                continue
            rw = rotemberg_year_weights(sample, oc, f"bartik_{primary}", share_col,
                                        f"shift_{primary}")
            for i, tau in enumerate(rw["years"]):
                add_diag(diag, diagnostic="rotemberg_year_weight", outcome=oc,
                         leave_out_variant=primary, share_baseline=baseline,
                         subject=str(int(tau)), statistic="weight",
                         value=float(rw["alpha"][i]), n_obs=len(sample))
                add_diag(diag, diagnostic="rotemberg_year_beta", outcome=oc,
                         leave_out_variant=primary, share_baseline=baseline,
                         subject=str(int(tau)), statistic="coef",
                         value=float(rw["beta_tau"][i]), n_obs=len(sample))
            # This row must not be written with pass_fail="PASS": it is not a
            # test.
            # alpha is DEFINED as num/num.sum() eleven lines up, so sum(alpha) == 1
            # identically, for any data, including data that is completely wrong.
            # A check that cannot fail is worse than no check, because it occupies
            # the space where a real one would go.  The row is kept -- the value is
            # still worth recording, and a departure from 1 would signal a NaN in
            # `num` -- but it is now labelled IDENTITY and can never read PASS.
            # The informative companion is the reproduction_rel_error row below,
            # which compares sum(alpha*beta_tau) against the independently computed
            # 2SLS coefficient and CAN fail.
            _alpha_is_nan = not np.isfinite(rw["sum_alpha"])
            add_diag(diag, diagnostic="rotemberg_validation", outcome=oc,
                     leave_out_variant=primary, share_baseline=baseline,
                     subject="sum_alpha", statistic="value", value=rw["sum_alpha"],
                     threshold=1.0,
                     pass_fail="FAIL_NAN" if _alpha_is_nan else "IDENTITY",
                     n_obs=len(sample),
                     notes="NOT A TEST: alpha := num/num.sum(), so sum(alpha) == 1 "
                           "by construction. Recorded "
                           "for completeness; only a NaN in num can move it. The "
                           "falsifiable companion is reproduction_rel_error.")
            add_diag(diag, diagnostic="rotemberg_validation", outcome=oc,
                     leave_out_variant=primary, share_baseline=baseline,
                     subject="reproduction_rel_error", statistic="value",
                     value=rw["rel_err"], threshold=0.05,
                     pass_fail="PASS" if (np.isfinite(rw["rel_err"]) and rw["rel_err"] < 0.05) else "FAIL",
                     n_obs=len(sample),
                     notes=f"sum(alpha*beta)={rw['recon']:.6g} vs beta_Bartik={rw['beta_bartik']:.6g}; "
                           f"rank(Ztilde)={rw['rank']} of {len(rw['years'])}")
            add_diag(diag, diagnostic="rotemberg_validation", outcome=oc,
                     leave_out_variant=primary, share_baseline=baseline,
                     subject="negative_weights", statistic="value",
                     value=float(rw["n_negative"]), n_obs=len(sample),
                     notes=f"sum of negative alpha = {rw['sum_negative']:.6g}")
            if oc in HEADLINE_OUTCOMES:
                print(f"  Rotemberg year-shocks [{oc}]: rank={rw['rank']}/{len(rw['years'])}, "
                      f"sum(alpha)={rw['sum_alpha']:.6f}, negatives={rw['n_negative']}, "
                      f"reproduction rel.err={rw['rel_err']:.4f} "
                      f"({'VALID' if rw['valid'] else 'INVALID -- do not report'})")
                for i, tau in enumerate(rw["years"]):
                    print(f"      {int(tau)}  alpha={rw['alpha'][i]:>9.4f}  "
                          f"beta_tau={rw['beta_tau'][i]:>14.2f}")

        infl_ref_ok = None
        for oc in outcomes:
            if (oc, primary) not in cells:
                continue
            inf = state_influence(sample, oc, f"bartik_{primary}")
            for st, val in inf["infl_rf"].items():
                add_diag(diag, diagnostic="state_influence_rf", outcome=oc,
                         leave_out_variant=primary, share_baseline=baseline, subject=st,
                         statistic="share", value=float(val), n_obs=len(sample))
            if oc == outcomes[0]:
                for st, val in inf["infl_fs"].items():
                    add_diag(diag, diagnostic="state_influence_fs", outcome="",
                             leave_out_variant=primary, share_baseline=baseline, subject=st,
                             statistic="share", value=float(val), n_obs=len(sample),
                             notes="first-stage moment share; outcome-independent")
                for k in ["top1", "top3", "top5", "hhi"]:
                    add_diag(diag, diagnostic="influence_concentration", outcome="",
                             leave_out_variant=primary, share_baseline=baseline, subject=k,
                             statistic="share", value=inf[k], n_obs=len(sample),
                             notes="first-stage influence concentration")
                tx = float(inf["infl_fs"].get("TX", np.nan))
                va = float(inf["infl_fs"].get("VA", np.nan))
                infl_ref_ok = (abs(tx - REF_INFL_FS["TX"]) < 5e-3
                               and abs(va - REF_INFL_FS["VA"]) < 5e-3)
                print(f"\n  First-stage state influence ({primary}): "
                      f"TX={tx:.4f} (ref {REF_INFL_FS['TX']}), VA={va:.4f} "
                      f"(ref {REF_INFL_FS['VA']}), top3={inf['top3']:.4f}, HHI={inf['hhi']:.4f}"
                      f"   [reference match: {infl_ref_ok}]")
                print("  Top-8 first-stage influence:")
                for st, val in inf["infl_fs"].sort_values(ascending=False).head(8).items():
                    print(f"      {st}  infl_FS={val:>8.4f}   infl_RF={inf['infl_rf'][st]:>8.4f}")
            add_diag(diag, diagnostic="rf_fs_influence_divergence", outcome=oc,
                     leave_out_variant=primary, share_baseline=baseline,
                     subject="spearman_fs_vs_rf", statistic="corr", value=inf["spearman"],
                     n_obs=len(sample),
                     notes="numerator and denominator of the Wald ratio are driven by "
                           "different states when this is low")
            if oc in HEADLINE_OUTCOMES:
                print(f"  [{oc}] Spearman(infl_FS, infl_RF) = {inf['spearman']:.4f};  "
                      f"CA infl_RF={float(inf['infl_rf'].get('CA', np.nan)):.4f} vs "
                      f"infl_FS={float(inf['infl_fs'].get('CA', np.nan)):.4f}")

        # ------------------------------------------------------------------ [5]
        banner("[5] SHARE EXOGENEITY: PRE-TRENDS AND BALANCE")
        pretrend_fail_5182 = False
        for oc in outcomes:
            rows = pretrend_test(pz, oc, share_col)
            for r in rows:
                add_diag(diag, diagnostic="pretrend_gamma", outcome=oc,
                         leave_out_variant="", share_baseline=baseline, subject=r["pair"],
                         statistic="coef", value=r["gamma"], se=r["se"], p_value=r["p"],
                         n_obs=r["n"], threshold=0.05,
                         pass_fail=("PASS" if (r["p"] >= 0.05 or not r["is_pre"]) else "FAIL"),
                         notes=f"is_pre_period={r['is_pre']}; division FE, HC1")
                if oc == "emp_5182" and r["is_pre"] and r["p"] < 0.05:
                    pretrend_fail_5182 = True
            if oc in HEADLINE_OUTCOMES:
                print(f"\n  Pre-trend gamma_tau [{oc}] (division FE, HC1; only 3 pre-period "
                      f"pairs exist because the capacity series starts in 2016):")
                for r in rows:
                    mark = "PRE " if r["is_pre"] else "post"
                    star = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.10 else ""
                    print(f"      {mark} {r['pair']}  gamma={r['gamma']:>12.2f} "
                          f"(se {r['se']:>10.2f})  p={r['p']:.4f}{star}  N={r['n']}")
        print(f"\n  Pre-period share-exogeneity criterion for emp_5182: "
              f"{'FAIL -- promote share_linear_trend to the reported spec' if pretrend_fail_5182 else 'PASS'}")

        bal = balance_test(pz, share_col)
        print("\n  Balance of pre-determined characteristics on Share (division FE, HC1).")
        print("  Pre-committed reading: a significant emp_5182_2016 is MECHANICAL (levels are")
        print("  absorbed by state FE). The rows that matter are the growth/dispersion rows.")
        for r in bal:
            star = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.10 else ""
            print(f"      {r['var']:<24s} gamma={r['gamma']:>14.3f} (se {r['se']:>12.3f}) "
                  f"p={r['p']:.4f}{star}  N={r['n']}")
            add_diag(diag, diagnostic="balance_gamma", outcome="", leave_out_variant="",
                     share_baseline=baseline, subject=r["var"], statistic="coef",
                     value=r["gamma"], se=r["se"], p_value=r["p"], n_obs=r["n"],
                     threshold=0.05, pass_fail="PASS" if r["p"] >= 0.05 else "FAIL",
                     notes="gdp_pc_2024 excluded: 2024 measurement is post-treatment")

        # ------------------------------------------------------------------ [6]
        banner("[6] PLACEBOS")
        print("  No clean sectoral placebo exists in the supplied QWI extract: every NAICS in")
        print("  it (5182, 517, 23, 2362, 5415) is a claimed-effect sector, and 51 is a strict")
        print("  superset of two of them. Two data-free placebos are substituted instead.\n")
        placebo_outcomes = [o for o in HEADLINE_OUTCOMES if o in outcomes]
        if not args.skip_placebo:
            for oc in placebo_outcomes:
                pp = permutation_placebo(sample, oc, f"bartik_{primary}", share_col,
                                         f"shift_{primary}", baseline, rng, args.n_perm)
                print(f"  Share-permutation placebo [{oc}] ({pp['n_perm']} draws):")
                print(f"      RF  observed={pp['rf_obs']:>12.2f}  perm p={pp['rf_p']:.4f}  "
                      f"perm mean={pp['rf_mean']:.2f}  [2.5%,97.5%]=[{pp['rf_q025']:.1f},{pp['rf_q975']:.1f}]")
                print(f"      IV  observed={pp['iv_obs']:>12.2f}  perm p={pp['iv_p']:.4f}  "
                      f"perm mean={pp['iv_mean']:.2f}  [2.5%,97.5%]=[{pp['iv_q025']:.1f},{pp['iv_q975']:.1f}]")
                for lbl, key in [("reduced_form", "rf"), ("iv2sls", "iv")]:
                    add_diag(diag, diagnostic="placebo_permutation", outcome=oc,
                             leave_out_variant=primary, share_baseline=baseline,
                             subject=lbl, statistic="pvalue", value=pp[f"{key}_p"],
                             p_value=pp[f"{key}_p"], n_obs=pp["n_perm"], threshold=0.10,
                             pass_fail="PASS" if pp[f"{key}_p"] < 0.10 else "FAIL",
                             notes=(f"observed={pp[f'{key}_obs']:.6g}; perm mean={pp[f'{key}_mean']:.6g}; "
                                    f"perm 2.5%={pp[f'{key}_q025']:.6g}; perm 97.5%={pp[f'{key}_q975']:.6g}"))
                lp = lead_placebo(pz, sample, oc, f"bartik_{primary}")
                if lp:
                    print(f"  Lead-instrument placebo [{oc}]: b(Z_{{t+2}})={lp['coef']:.2f} "
                          f"(se {lp['se']:.2f}) p={lp['p']:.4f}  N={lp['n']} "
                          f"(controls for Z_t, State FE, Year FE)")
                    add_diag(diag, diagnostic="placebo_lead", outcome=oc,
                             leave_out_variant=primary, share_baseline=baseline,
                             subject="bartik_lead2", statistic="coef", value=lp["coef"],
                             se=lp["se"], p_value=lp["p"], n_obs=lp["n"], threshold=0.05,
                             pass_fail="PASS" if lp["p"] >= 0.05 else "FAIL",
                             notes="significant lead => anticipation or differential trend")
        else:
            print("  SKIPPED (--skip-placebo)")

        # ------------------------------------------------------------------ [7]
        banner("[7] NET INFORMATION-SECTOR EFFECT  b(5182) + b(517)")
        net_results = {}
        boot = None
        net_pairs = [("net_info_5182_plus_517", "emp_5182", "emp_517"),
                     ("net_info_earn_5182_plus_517", "earn_5182", "earn_517")]
        net_pairs = [p for p in net_pairs if p[1] in outcomes and p[2] in outcomes]

        if not args.skip_bootstrap and net_pairs:
            boot_cols = sorted({c for _, a, b in net_pairs for c in (a, b)})
            print(f"  Pairs cluster bootstrap over states: B={args.n_boot}, seed={args.seed} ...")
            boot = pairs_cluster_bootstrap(sample, boot_cols, f"bartik_{primary}",
                                           np.random.default_rng(args.seed), args.n_boot)
            print(f"    kept {boot['n_kept']} reps, dropped {boot['n_dropped']} "
                  f"(effective F < {WEAK_F_FOR_BOOTSTRAP_DROP})")

        for name, c1, c2 in net_pairs:
            st = net_effect_stacked(sample, c1, c2, f"bartik_{primary}")
            s1 = cells.get((c1, primary))
            s2 = cells.get((c2, primary))
            # verification: the stacked coefficients MUST reproduce the
            # single-equation ones. The stacked design is block-diagonal by construction,
            # so this is an algebraic identity -- but it is asserted, not assumed.
            ok1 = abs(st["b1"] - s1["iv"]["coef"]) / max(abs(s1["iv"]["coef"]), 1e-12) < 1e-6
            ok2 = abs(st["b2"] - s2["iv"]["coef"]) / max(abs(s2["iv"]["coef"]), 1e-12) < 1e-6
            G = st["n_clusters"]
            t, p, lo, hi = t_ci(st["net"], st["se_net"], G)
            print(f"\n  [{name}]  stacked / SUR-IV, clustered by state (a cluster spans both eqs)")
            print(f"      b({c1}) = {st['b1']:.2f}   b({c2}) = {st['b2']:.2f}")
            print(f"      reproduces single-equation estimates to 1e-6: {ok1} / {ok2}")
            print(f"      Cov = {st['cov']:.4g}  (corr = "
                  f"{st['cov'] / np.sqrt(st['v11'] * st['v22']):.4f})")
            print(f"      NET = {st['net']:.2f}  se = {st['se_net']:.2f}  p = {p:.4f}  "
                  f"95% CI [{lo:.1f}, {hi:.1f}]")
            note = "stacked SUR-IV; Cov estimated, never set to zero"
            if not (ok1 and ok2):
                note += ";STACKING_MISMATCH"
            main_rows.append(dict(outcome=name, outcome_units=outcome_units(c1), specification="level",
                leave_out_variant=primary, share_baseline=baseline,
                estimator="IV2SLS_stacked", coef=st["net"], se=st["se_net"],
                se_type="cluster_state", t_stat=t, p_value=p, ci_low=lo, ci_high=hi,
                ci_type="cluster_t", n_obs=st["n_obs"], n_clusters=G,
                n_states=sample["state_abbr"].nunique(), n_years=sample["year"].nunique(),
                fs_coef=s1["fs"]["coef"], fs_se=s1["fs"]["se"], fs_F=s1["f_classic"],
                fs_F_effective=s1["f_eff"], fs_F_crit=OP_F_CRIT,
                first_stage_ok=s1["first_stage_ok"], r2_within=np.nan,
                mde_80=MDE_MULT * st["se_net"], notes=note))
            net_results[name] = dict(stacked=st, b1=st["b1"], b2=st["b2"], p=p,
                                     ci=[lo, hi], ok=(ok1 and ok2))

            if boot is not None and boot["n_kept"] > 20:
                bs = boot["iv"][c1] + boot["iv"][c2]
                se_b = float(bs.std(ddof=1))
                lo_b, hi_b = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
                cov_b = float(np.cov(boot["iv"][c1], boot["iv"][c2])[0, 1])
                t_b, p_b, _, _ = t_ci(st["net"], se_b, G)
                ratio = abs(cov_b / st["cov"]) if st["cov"] != 0 else np.inf
                nb = "pairs cluster bootstrap over states, percentile CI"
                if not (0.5 <= ratio <= 2.0):
                    nb += ";COV_DISAGREEMENT_>2x_trust_bootstrap"
                print(f"      bootstrap: se={se_b:.2f}  pct CI [{lo_b:.1f}, {hi_b:.1f}]  "
                      f"Cov_boot={cov_b:.4g} (ratio to analytic {ratio:.2f})  "
                      f"kept={boot['n_kept']}")
                main_rows.append(dict(outcome=name, outcome_units=outcome_units(c1), specification="level",
                    leave_out_variant=primary, share_baseline=baseline,
                    estimator="IV2SLS_bootstrap", coef=st["net"], se=se_b,
                    se_type="cluster_state", t_stat=t_b, p_value=p_b, ci_low=lo_b,
                    ci_high=hi_b, ci_type="bootstrap_pct", n_obs=len(sample),
                    n_clusters=G, n_states=sample["state_abbr"].nunique(),
                    n_years=sample["year"].nunique(), fs_coef=float(boot["fs"].mean()),
                    fs_se=float(boot["fs"].std(ddof=1)), fs_F=s1["f_classic"],
                    fs_F_effective=s1["f_eff"], fs_F_crit=OP_F_CRIT,
                    first_stage_ok=s1["first_stage_ok"], r2_within=np.nan,
                    mde_80=MDE_MULT * se_b,
                    notes=nb + f";reps_kept={boot['n_kept']};reps_dropped={boot['n_dropped']}"))
                net_results[name].update(se_boot=se_b, cov_boot=cov_b,
                                         ci_boot=[lo_b, hi_b])

        # bootstrapped Cov(b_RF, b_1st) for the delta method
        if boot is not None:
            for c in boot["rf"]:
                cb = float(np.cov(boot["rf"][c], boot["fs"])[0, 1])
                add_diag(diag, diagnostic="wald_cov_rf_fs", outcome=c,
                         leave_out_variant=primary, share_baseline=baseline,
                         subject="bootstrap", statistic="value", value=cb,
                         n_obs=boot["n_kept"],
                         notes="pairs cluster bootstrap over states; NOT set to zero")

        # ------------------------------------------------------------------ [8]
        rob_rows = []
        if not args.skip_robustness:
            banner("[8] ROBUSTNESS RE-ESTIMATIONS")
            for tag in ROBUSTNESS_TAGS:
                drop_states, extra_fe, extra_ctrl_name, use_base, use_variant = (), None, None, baseline, primary
                oc_list, spec_label = outcomes, "level"
                drop_partial = False
                if tag == "leaveout_state":
                    use_variant = "state"
                elif tag == "leaveout_contig":
                    use_variant = "contig"
                elif tag == "leaveout_division":
                    # Without this branch the tag fell through to use_variant =
                    # primary, so the row labelled leaveout_division silently
                    # duplicated the primary union estimate.
                    use_variant = "division"
                elif tag == "share2016":
                    use_base = 2016
                elif tag == "drop_VA":
                    drop_states = ("VA",)
                elif tag == "drop_TX":
                    drop_states = ("TX",)
                elif tag == "drop_top3_share":
                    drop_states = ("TX", "VA", "CA")
                elif tag == "drop_partial_years":
                    drop_partial = True
                elif tag == "fe_region_year":
                    extra_fe = "region_year"
                elif tag == "fe_division_year":
                    extra_fe = "division_year"
                elif tag == "share_linear_trend":
                    extra_ctrl_name = "share_x_t"
                elif tag == "vol_linear_trend":
                    extra_ctrl_name = "vol_x_t"
                elif tag == "vol_year_fe":
                    extra_ctrl_name = "vol_x_year"
                elif tag == "log":
                    oc_list = [c for c in LOG_OUTCOMES] if args.outcome_cols is None else \
                        [f"log_{c}" for c in outcomes if f"log_{c}" in LOG_OUTCOMES]
                    spec_label = "log"

                src = pz16 if use_base == 2016 else pz
                s_tag, _ = build_sample(src, drop_states=drop_states, drop_partial=drop_partial)
                extra_ctrl = None
                if extra_ctrl_name == "share_x_t":
                    extra_ctrl = (s_tag[f"share{use_base}"].to_numpy(float)
                                  * (s_tag["year"].to_numpy(float) - YEAR_MIN)).reshape(-1, 1)
                elif extra_ctrl_name == "vol_x_t":
                    # one linear trend whose slope varies with pre-period volatility
                    vol = pre_volatility(s_tag["state_abbr"].tolist())
                    extra_ctrl = (vol * (s_tag["year"].to_numpy(float)
                                         - YEAR_MIN)).reshape(-1, 1)
                elif extra_ctrl_name == "vol_x_year":
                    # volatility interacted with the full set of year dummies: high-
                    # and low-volatility states get entirely separate year effects.
                    # The base-year interaction is collinear with the state FE and is
                    # dropped with the base year dummy; design_matrix takes K from the
                    # rank of W, so the finite-sample correction stays correct.
                    vol = pre_volatility(s_tag["state_abbr"].tolist())
                    yd = pd.get_dummies(s_tag["year"], drop_first=True,
                                        dtype=float).to_numpy(float)
                    extra_ctrl = vol.reshape(-1, 1) * yd
                zcol = f"bartik_{use_variant}"
                print(f"\n  --- {tag} --- (variant={use_variant}, baseline={use_base}, "
                      f"dropped={list(drop_states)}, extra_fe={extra_fe}, N={len(s_tag)})")
                for oc in oc_list:
                    if oc not in s_tag.columns:
                        continue
                    try:
                        cell = estimate_cell(s_tag, oc, zcol, extra_fe=extra_fe,
                                             extra_controls=extra_ctrl, do_wcr=False,
                                             rng=rng)
                    except Exception as e:
                        rob_rows.append(dict(outcome=oc, outcome_units=outcome_units(oc),
                                             specification=spec_label,
                                             leave_out_variant=use_variant,
                                             share_baseline=use_base, estimator="IV2SLS",
                                             coef=np.nan, robustness_tag=tag,
                                             dropped_units=";".join(drop_states),
                                             extra_fe=extra_fe or "",
                                             notes=f"exception:{e}"))
                        continue
                    for r in cell_rows(cell, oc, spec_label, use_variant, use_base):
                        r.update(robustness_tag=tag, dropped_units=";".join(drop_states),
                                 extra_fe=extra_fe or (extra_ctrl_name or ""))
                        rob_rows.append(r)
                    if oc in HEADLINE_OUTCOMES or oc in ("log_emp_5182", "log_emp_517"):
                        print_cell(tag, oc, cell)

        # ------------------------------------------------------------------ [9]
        banner("[9] WRITING OUTPUT")
        bad = {d["diagnostic"] for d in diag} - DIAGNOSTIC_NAMES
        if bad:
            raise RuntimeError(f"diagnostic names outside the closed enumeration: {bad}")

        main_df = write_table(main_rows, MAIN_COLUMNS, out_dir / f"results_main{sfx}.csv")
        print(f"  results_main{sfx}.csv          {len(main_df):>5d} rows x {len(MAIN_COLUMNS)} cols")
        if rob_rows:
            rob_df = write_table(rob_rows, ROBUST_COLUMNS, out_dir / f"results_robustness{sfx}.csv")
            print(f"  results_robustness{sfx}.csv    {len(rob_df):>5d} rows x {len(ROBUST_COLUMNS)} cols")
        diag_df = pd.DataFrame(diag)[DIAG_COLUMNS]
        diag_df.to_csv(out_dir / f"results_diagnostics{sfx}.csv", index=False,
                       float_format="%.6g", na_rep="", encoding="utf-8")
        print(f"  results_diagnostics{sfx}.csv   {len(diag_df):>5d} rows x {len(DIAG_COLUMNS)} cols")

        # ---- results.json -------------------------------------------------
        def head_block(oc):
            c = cells.get((oc, primary))
            if c is None:
                # The key must not bake a literal into a field NAME, which no
                # re-run can ever refresh.
                return {k: "NOT_ESTIMATED: outcome not in the requested set"
                        for k in ["beta_iv", "se", "ci", "p", "fs_F_effective",
                                  "first_stage_ok", "ci_contains_panel_fe_benchmark",
                                  "ci_contains_0"]}
            t, p, lo, hi = t_ci(c["iv"]["coef"], c["iv"]["se"], c["n_clusters"])
            if not c["first_stage_ok"] and np.isfinite(c["ar_low"]):
                lo, hi = c["ar_low"], c["ar_high"]
            return dict(beta_iv=float(c["iv"]["coef"]), se=float(c["iv"]["se"]),
                        ci=[float(lo), float(hi)], p=float(p),
                        fs_F_effective=float(c["f_eff"]),
                        first_stage_ok=bool(c["first_stage_ok"]),
                        ci_contains_panel_fe_benchmark=bool(lo <= _benchmark() <= hi),
                        panel_fe_benchmark=float(_benchmark()),
                        ci_contains_0=bool(lo <= 0 <= hi))

        h5182, h517 = head_block("emp_5182"), head_block("emp_517")
        net_key = "net_info_5182_plus_517"
        if net_key in net_results:
            nr = net_results[net_key]
            net_block = dict(beta=float(nr["stacked"]["net"]),
                             se_stacked=float(nr["stacked"]["se_net"]),
                             se_bootstrap=float(nr.get("se_boot", np.nan))
                             if np.isfinite(nr.get("se_boot", np.nan))
                             else "NOT_ESTIMATED: bootstrap skipped",
                             cov_5182_517=float(nr["stacked"]["cov"]),
                             ci=[float(nr["ci"][0]), float(nr["ci"][1])], p=float(nr["p"]))
            net_branch = branch_net(nr["stacked"]["net"], nr["stacked"]["se_net"],
                                    nr["b1"], nr["b2"], nr["stacked"]["n_clusters"])
        else:
            net_block = {k: "NOT_ESTIMATED: 5182/517 not both in the requested outcome set"
                         for k in ["beta", "se_stacked", "se_bootstrap", "cov_5182_517",
                                   "ci", "p"]}
            net_branch = "NOT_ESTIMATED: net effect not computed"

        b1_lab = (branch_5182(h5182["beta_iv"], h5182["se"], h5182["ci"][0], h5182["ci"][1],
                              h5182["p"]) if isinstance(h5182["beta_iv"], float)
                  else "NOT_ESTIMATED: emp_5182 not in the requested outcome set")
        b2_lab = (branch_517(h517["beta_iv"], h517["p"]) if isinstance(h517["beta_iv"], float)
                  else "NOT_ESTIMATED: emp_517 not in the requested outcome set")

        import numpy as _np
        try:
            import statsmodels as _sm
            sm_ver = _sm.__version__
        except Exception:
            sm_ver = "NOT_INSTALLED"
        try:
            import linearmodels as _lm
            lm_ver = _lm.__version__
        except Exception:
            lm_ver = "NOT_INSTALLED"

        se_hom = np.nan
        try:
            # Homoskedastic design-moment SE for the power block.
            #
            # se_hom = sigma / |b_1st| / sqrt(sum of squared residualised z).  sd_z is
            # computed with ddof = 1, so sum(z~^2) = sd_z^2 * (N - 1) and the divisor
            # is sqrt(N - 1), NOT sqrt(N).  Using sqrt(N) makes the homoskedastic
            # MDE 0.114 percent too small.  The 5.8x ratio to the clustered MDE is
            # unaffected either way.
            c0 = cells.get(("emp_5182", primary))
            if c0 is not None:
                u = c0["iv"]["u"]
                sig = np.sqrt(float(u @ u) / (c0["n_obs"] - c0["K"]))
                se_hom = sig / (abs(c0["fs"]["coef"]) * sd_z * np.sqrt(c0["n_obs"] - 1))
        except Exception:
            pass

        results = {
            "meta": {
                "spec_version": "1.0",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "script": relpath(__file__),
                "panel_source": relpath(args.panel),
                "panel_sha256": sha256_of(args.panel),
                "seed": args.seed,
                "python": platform.python_version(),
                "package_versions": {"numpy": _np.__version__, "pandas": pd.__version__,
                                     "statsmodels": sm_ver, "linearmodels": lm_ver},
                # no null may survive into the written file
                "cli": {k: ("" if v is None else
                            (relpath(v) if isinstance(v, str) and str(REPO_ROOT) in v else v))
                        for k, v in vars(args).items()},
            },
            "preflight": {
                "assertions_run": n_pass + len(failures), "assertions_passed": n_pass,
                "failures": failures,
                "national_2019_gw": float(panel.loc[panel.year == 2019, "us_dc_gw"].iloc[0]),
                "share2019_top8": {k: float(panel.loc[(panel.year == 2019) &
                                                      (panel.state_abbr == k), "share2019"].iloc[0])
                                   for k in REF_SHARE2019_TOP8},
                "share2019_top3_sum": float(panel.loc[panel.year == 2019, "share2019"].nlargest(3).sum()),
                "share2019_top5_sum": float(panel.loc[panel.year == 2019, "share2019"].nlargest(5).sum()),
            },
            "sample": {
                "n_obs": int(len(sample)), "n_states": int(sample["state_abbr"].nunique()),
                "n_years": int(sample["year"].nunique()),
                "year_min": int(sample["year"].min()), "year_max": int(sample["year"].max()),
                "states_dropped": singletons, "drop_reason": "fewer than 3 observed years",
                "unbalanced_states": unbalanced, "zero_share_states": zero_share,
                "partial_year_rows": [[s, int(y)] for s, y in partial_rows],
            },
            "design": {
                "primary_leave_out": primary, "share_baseline_year": int(baseline),
                # All six pairs, both objects.  `instrument_correlations`
                # is the raw panel correlation; `_within_fe` is the residualised
                # correlation on the estimation sample, which is what
                # SI tab:leaveout_compare prints.
                "instrument_correlations": inst_corr,
                "instrument_correlations_within_fe": inst_corr_within,
                "instrument_panel_columns_exported": inst_cols,
                "sd_instrument_within": float(sd_z), "sd_capacity_within": float(sd_x),
                "corr_instrument_capacity_within": float(corr_zx),
                "sd_emp_5182_within": float(sd_y),
            },
            "power": {
                "mde_80_homoskedastic": float(MDE_MULT * se_hom) if np.isfinite(se_hom)
                else "NOT_ESTIMATED: emp_5182 not estimated",
                "mde_80_cluster_x1_5": float(MDE_MULT * se_hom * 1.5) if np.isfinite(se_hom)
                else "NOT_ESTIMATED: emp_5182 not estimated",
                "mde_80_cluster_x2": float(MDE_MULT * se_hom * 2.0) if np.isfinite(se_hom)
                else "NOT_ESTIMATED: emp_5182 not estimated",
                "mde_80_cluster_x3": float(MDE_MULT * se_hom * 3.0) if np.isfinite(se_hom)
                else "NOT_ESTIMATED: emp_5182 not estimated",
                "mde_80_realised_emp_5182": float(MDE_MULT * cells[("emp_5182", primary)]["iv"]["se"])
                if ("emp_5182", primary) in cells else "NOT_ESTIMATED: emp_5182 not estimated",
                "panel_fe_benchmark": float(_benchmark()),
                "panel_fe_benchmark_source": BENCHMARK_PROVENANCE["panel_fe_benchmark"],
                "local_share_median": float(_local_share_median()),
                "local_share_median_source": BENCHMARK_PROVENANCE["local_share_median"],
                "panel_fe_benchmark_rescaled_cleaned":
                    float(_benchmark() * _local_share_median()),
                # The benchmark is interpolated, never hard-coded in this prose.
                "note": (f"MDE band straddles the {_benchmark():,.0f} jobs/GW panel FE "
                         f"benchmark; a null must be reported as 'cannot distinguish "
                         f"from 0 OR from {_benchmark():,.0f}', never as 'no effect'"),
            },
            "headline": {"emp_5182": h5182, "emp_517": h517, net_key: net_block},
            "interpretation": {
                "emp_5182_branch": b1_lab, "emp_517_branch": b2_lab, "net_branch": net_branch,
                "pretrend_emp_5182_pass": (not pretrend_fail_5182),
                "first_stage_influence_reference_match": (infl_ref_ok
                                                          if infl_ref_ok is not None
                                                          else "NOT_ESTIMATED: no outcome run"),
            },
            "limitations": [
                "no all-industry QWI denominator available",
                "no clean sectoral placebo available in the supplied NAICS set",
                "only 3 pre-period year-pairs (capacity series starts 2016)",
                "15 states have non-monotonic cumulative MW (upstream, out of scope)",
                "canonical GPSS Rotemberg weights are degenerate with a single sector; "
                "year-shock weights and state influence shares are substituted",
            ],
        }
        with open(out_dir / f"results{sfx}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  results{sfx}.json")

        banner("HEADLINE (pre-committed interpretation branches)")
        print(f"  emp_5182 : beta_IV = {h5182.get('beta_iv')}  CI = {h5182.get('ci')}")
        print(f"             branch  = {b1_lab}")
        print(f"  emp_517  : beta_IV = {h517.get('beta_iv')}  CI = {h517.get('ci')}")
        print(f"             branch  = {b2_lab}")
        print(f"  net      : {net_block.get('beta')}  branch = {net_branch}")
        print(f"\n  Elapsed: {time.time() - t_start:.1f}s")
        banner(f"All done. Results in {relpath(out_dir)}")

    finally:
        sys.stdout = tee.terminal
        tee.close()

if __name__ == "__main__":
    main()
