"""
Descriptive panel fixed-effects employment analysis on the 49-unit contiguous-US
analysis universe (48 contiguous states + DC).

Scope restriction mirrors r06_employment_bartik_iv.apply_scope():
  * OUT_OF_SCOPE_UNITS = ("AK", "HI") are dropped BEFORE anything is estimated;
  * the District of Columbia is IN scope.  QWI and the PwC multiplier file spell
    the District of Columbia as the literal string "DC" in their state-name
    column, not "District of Columbia", so any name map carrying only the long
    form drops it from the inner join.  NAME_TO_ABBR["DC"] = "DC" is required.

Samples estimated (all four, so the scope choice can be decomposed):
  us50   : 50 units, AK + HI in, DC out.
  us51   : us50 + DC.                       (isolates DC)
  contig48 : us50 - AK - HI, DC still out.    (isolates the scope drop)
  contig49   : PRIMARY. 48 contiguous states + DC.  (the two combined)

Outputs (all in this directory, suffix __contig49 on the primary):
  panel_fe_5182__contig49.csv        4 panel-FE specs x 4 samples
  panel_fe_related__contig49.csv     NAICS 23 / 2362 / 5415 x 4 samples
  state_ols_summary__contig49.csv    state-level OLS summary x 4 samples
  sample_composition__contig49.csv   unit / obs accounting
  panel__contig49.csv                the estimation panel itself
  run_log__contig49.txt              full console log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS

# Release-set path rule: no absolute paths.  Resolved from this file's location.
_DIR = Path(__file__).resolve().parent                    # employment
CODES = _DIR.parent                                       # 
OUT = _DIR / "../results/r6_employment"                                  # generated results
OUT.mkdir(parents=True, exist_ok=True)
SFX = "__contig49"

class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")

    def write(self, m):
        self.terminal.write(m)
        self.log.write(m)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

tee = Tee(OUT / f"run_log{SFX}.txt")
sys.stdout = tee

def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

# ---------------------------------------------------------------------------
# Scope constants.
# Imported from analysis_universe, never redeclared here.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(_DIR))
from analysis_universe import (ANALYSIS_UNITS, ALL_QWI_UNITS,  # noqa: E402
                               N_ANALYSIS_UNITS, OUT_OF_SCOPE_UNITS,
                               UNIVERSE_LABEL, check_universe)

# r05's map, VERBATIM.  Note it has 'District of Columbia' and no bare 'DC'.
STATE_ABBR_MAP = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}
STATE_ABBR_MAP_FIXED = dict(STATE_ABBR_MAP)
STATE_ABBR_MAP_FIXED['DC'] = 'DC'          # 04_build_panel.py line 298

YEARS = list(range(2016, 2025))

qwi = pd.read_csv(_DIR / 'qwi_all_naics_annual.csv', dtype={'naics': str})
dc = pd.read_csv(_DIR / 'dc_facilities_by_state_year.csv')
mult = pd.read_csv(_DIR / 'pwc_multipliers.csv')

banner("INPUT FILES")
print(f"  qwi_all_naics_annual.csv       {qwi.shape[0]} rows, "
      f"{qwi.state_name.nunique()} state labels, naics={sorted(qwi.naics.unique())}")
print(f"  dc_facilities_by_state_year.csv {dc.shape[0]} rows, "
      f"'US' aggregate row present = {'US' in set(dc.state_abbr)}")
print(f"  pwc_multipliers.csv            {mult.shape[0]} rows")
print("  QWI spells the District of Columbia as: "
      f"{[s for s in qwi.state_name.unique() if 'olumbia' in s or s == 'DC']}")

# ---------------------------------------------------------------------------
# National multiplier and spillover decomposition -- r05 section 2, VERBATIM.
# 04_build_panel.py computes this over all 51 PwC rows too (before scope), so
# the primary run keeps it unchanged.  A renormalised variant is reported below.
# ---------------------------------------------------------------------------
NAT_MULT = mult['emp_total_no_spillover_avg'].sum() / mult['emp_direct_avg'].sum()
mult['state_abbr'] = mult['state'].map(STATE_ABBR_MAP_FIXED)
assert mult['state_abbr'].notna().all(), mult.loc[mult.state_abbr.isna(), 'state']
mult['spillover_dc'] = mult['emp_spillover_avg'] / NAT_MULT
mult['local_share'] = mult['emp_direct_avg'] / (mult['emp_direct_avg'] + mult['spillover_dc'])

m49 = mult[~mult['state_abbr'].isin(OUT_OF_SCOPE_UNITS)]
NAT_MULT_49 = m49['emp_total_no_spillover_avg'].sum() / m49['emp_direct_avg'].sum()
print(f"\n  NAT_MULT over all 51 PwC units = {NAT_MULT:.6f}  (r05 primary, r06 mirrors)")
print(f"  NAT_MULT over the 49 in-scope units = {NAT_MULT_49:.6f}  (sensitivity only)")

local_share_dict = dict(zip(mult['state_abbr'], mult['local_share']))
local_mult_dict = dict(zip(mult['state_abbr'], mult['emp_multiplier_local']))
spillover_rate_dict = dict(zip(mult['state_abbr'], mult['emp_spillover_rate']))

# ---------------------------------------------------------------------------
# DC capacity panel -- r05 section 3, VERBATIM (the 'US' row is already dropped)
# ---------------------------------------------------------------------------
dc_states = dc[dc['state_abbr'] != 'US'].copy()
assert 'US' not in set(dc_states.state_abbr)
dc_panel = []
for _, row in dc_states.iterrows():
    for yr in YEARS:
        dc_panel.append({'state_abbr': row['state_abbr'], 'year': yr,
                         'dc_count': row[f'count_{yr}'],
                         'dc_capacity_GW': row[f'MW_{yr}'] / 1000})
dc_panel = pd.DataFrame(dc_panel)

def build_panel(map_used, drop_units, direct_universe="in_scope"):
    """r05's panel construction with a swappable name map and a scope drop.

    direct_universe controls the national NAICS 5182 direct total that enters
    the spec-2 "raw x multiplier + spillover" outcome:
      'qwi_all'  -- r05 VERBATIM: summed over all 51 QWI state labels, including
                    labels that never enter the estimation panel.  This is what
                    produced the published 13,102.
      'in_scope' -- summed over exactly the units in the estimation sample, which
                    is what r06's apply_scope() does for national capacity.
    """
    q = qwi[qwi['naics'] == '5182'].copy()
    q['state_abbr'] = q['state_name'].map(map_used)

    qk = q[q['state_abbr'].notna() & ~q['state_abbr'].isin(drop_units)]
    src = q if direct_universe == "qwi_all" else qk
    direct_by_year = src.groupby('year')['Emp_annual_avg'].sum().to_dict()

    p = qk[['state_abbr', 'state_name', 'year', 'Emp_annual_avg']].merge(
        dc_panel, on=['state_abbr', 'year'], how='inner')
    p = p.rename(columns={'Emp_annual_avg': 'emp_5182'})

    p['local_share'] = p['state_abbr'].map(local_share_dict)
    p['local_multiplier'] = p['state_abbr'].map(local_mult_dict)
    p['spillover_rate'] = p['state_abbr'].map(spillover_rate_dict)
    p['emp_raw_multiplied'] = (p['emp_5182'] * p['local_multiplier']
                               + (p['year'].map(direct_by_year) - p['emp_5182'])
                               * p['spillover_rate'])
    p['emp_cleaned_direct'] = p['emp_5182'] * p['local_share']
    p['emp_cleaned_multiplied'] = p['emp_cleaned_direct'] * p['local_multiplier']
    # State total; see 04_build_panel.py for the derivation.  The
    # spec-2 column above double-counts the cross-state spillover; this one
    # counts it once and contains no national aggregate.
    p['emp_state_total'] = p['emp_5182'] * (
        p['local_share'] * p['local_multiplier'] + 1 - p['local_share'])
    return p, direct_by_year

def build_related(map_used, drop_units, naics):
    q = qwi[qwi['naics'] == naics].copy()
    q['state_abbr'] = q['state_name'].map(map_used)
    q = q[q['state_abbr'].notna() & ~q['state_abbr'].isin(drop_units)]
    p = q[['state_abbr', 'state_name', 'year', 'Emp_annual_avg']].merge(
        dc_panel, on=['state_abbr', 'year'], how='inner')
    return p.rename(columns={'Emp_annual_avg': f'emp_{naics}'})

SAMPLES = {
    "us50":   dict(map=STATE_ABBR_MAP,       drop=(),                 du="qwi_all",
                       note="r05 EXACTLY AS PUBLISHED: AK+HI in, DC out, national 5182 total over all 51 QWI labels"),
    "us51":   dict(map=STATE_ABBR_MAP_FIXED, drop=(),                 du="in_scope",
                       note="DC name bug fixed only"),
    "contig48": dict(map=STATE_ABBR_MAP,       drop=OUT_OF_SCOPE_UNITS, du="in_scope",
                       note="scope drop only, DC still out"),
    "contig49":   dict(map=STATE_ABBR_MAP_FIXED, drop=OUT_OF_SCOPE_UNITS, du="in_scope",
                       note="PRIMARY: 48 contiguous + DC, national total over the 49 in-scope units"),
    "us50_inscope_total": dict(map=STATE_ABBR_MAP, drop=(),           du="in_scope",
                       note="diagnostic: r05 sample, but national 5182 total over the 50 units actually estimated"),
}

panels, direct_totals = {}, {}
for k, cfg in SAMPLES.items():
    panels[k], direct_totals[k] = build_panel(cfg["map"], cfg["drop"], cfg["du"])

# ---------------------------------------------------------------------------
# Sample composition
# ---------------------------------------------------------------------------
banner("SAMPLE COMPOSITION")
comp = []
for k, p in panels.items():
    cnt = p.groupby('state_abbr').size()
    row = dict(sample=k, note=SAMPLES[k]["note"], n_units=p['state_abbr'].nunique(),
               n_obs=len(p), n_units_lt3yr=int((cnt < 3).sum()),
               units_lt3yr=";".join(f"{s}({n})" for s, n in cnt[cnt < 3].items()),
               units_lt9yr=";".join(f"{s}({n})" for s, n in cnt[cnt < 9].items()),
               has_AK='AK' in set(p.state_abbr), has_HI='HI' in set(p.state_abbr),
               has_DC='DC' in set(p.state_abbr),
               gw2024_total=float(p.loc[p.year == 2024, 'dc_capacity_GW'].sum()),
               emp5182_2024_total=float(p.loc[p.year == 2024, 'emp_5182'].sum()))
    comp.append(row)
    print(f"  {k:<11s} units={row['n_units']:<3d} obs={row['n_obs']:<4d} "
          f"AK={row['has_AK']!s:<5s} HI={row['has_HI']!s:<5s} DC={row['has_DC']!s:<5s} "
          f"short-panel units: {row['units_lt9yr'] or 'none'}")
comp = pd.DataFrame(comp)
comp.to_csv(OUT / f"sample_composition{SFX}.csv", index=False)

p49 = panels["contig49"]
# two-sided set assertion against analysis_universe.py.
check_universe(p49, 'state_abbr', where='r08 panel-FE contig49 sample')
assert p49['state_abbr'].nunique() == N_ANALYSIS_UNITS, p49['state_abbr'].nunique()
assert not (set(OUT_OF_SCOPE_UNITS) & set(p49.state_abbr))
assert 'DC' in set(p49.state_abbr)
# this line used to compare against the literals 49
# and 438.  N=438 is not a constant of nature -- it is a property of the r06
# estimation sample -- so it is read from that sample's own result file, and the
# unit count comes from the owner module.  A literal here could fall out of date
# the first time the sample changed, silently, inside an f-string.
_r06_main = pd.read_csv(_DIR / "../results/r6_bartik" / "results_main__union.csv")
_q = _r06_main[(_r06_main["outcome"] == "emp_5182")
               & (_r06_main["estimator"] == "IV2SLS")
               & (_r06_main["leave_out_variant"] == "union")
               & (_r06_main["share_baseline"] == 2019)]
if "robustness_tag" in _r06_main.columns:
    _q = _q[_q["robustness_tag"].isna()]
assert len(_q) == 1, f"results_main__union.csv: {len(_q)} primary rows, expected 1"
_R06_N, _R06_G = int(_q.iloc[0]["n_obs"]), int(_q.iloc[0]["n_clusters"])
assert _R06_G == N_ANALYSIS_UNITS, (_R06_G, N_ANALYSIS_UNITS)
# The Bartik estimation window now opens at the 2019 exposure-share baseline, so the
# IV sample is a strict subset of the panel FE sample in years.  The panel FE design
# carries no exposure share and therefore no baseline, so it keeps the full 2016-2024
# window; but the estimator comparison of SI 6.3 must be like for like, so the matched
# subsample is built here as well and the guard is applied to THAT.
_IV_YEAR_MIN = int(_r06_main.loc[_q.index[0], "year_min"]) if "year_min" in _r06_main.columns else 2019
p49_matched = p49[p49["year"] >= _IV_YEAR_MIN].copy()
print(f"\n  PRE-FLIGHT contig49 full window: {p49['state_abbr'].nunique()} units, "
      f"{len(p49)} obs (2016-2024, panel FE has no share baseline)")
print(f"  PRE-FLIGHT contig49 matched to IV: {p49_matched['state_abbr'].nunique()} units, "
      f"{len(p49_matched)} obs -- matches the r06 Bartik estimation sample "
      f"({_R06_G} units, N={_R06_N}, read from results_main__union.csv): "
      f"{p49_matched['state_abbr'].nunique() == _R06_G and len(p49_matched) == _R06_N}")
assert p49_matched['state_abbr'].nunique() == _R06_G and len(p49_matched) == _R06_N, \
    (p49_matched['state_abbr'].nunique(), _R06_G, len(p49_matched), _R06_N)
p49_matched.to_csv(OUT / f"panel_matched{SFX}.csv", index=False)

# Register the matched subsample as its own sample so every specification below is
# estimated on it too.  14_panel_fe_table.py cross-checks this file against the
# OLS_FE rows of the Bartik module, and those now run on the IV window; the two
# implementations can only be compared on the same years.
panels["contig49_matched"] = p49_matched
SAMPLES["contig49_matched"] = dict(
    map=SAMPLES["contig49"]["map"], drop=SAMPLES["contig49"]["drop"],
    du=SAMPLES["contig49"]["du"],
    note=f"contig49 restricted to {_IV_YEAR_MIN}-2024, the Bartik estimation window",
    year_min=_IV_YEAR_MIN)
p49.to_csv(OUT / f"panel{SFX}.csv", index=False)

# ---------------------------------------------------------------------------
# Panel FE -- r05 section 6, VERBATIM
# ---------------------------------------------------------------------------
def run_panel_fe(df, x_col, y_col):
    pdf = df[['state_abbr', 'year', x_col, y_col]].dropna().copy()
    pdf = pdf.set_index(['state_abbr', 'year'])
    mod = PanelOLS(pdf[y_col], pdf[[x_col]], entity_effects=True, time_effects=True)
    return mod.fit(cov_type='clustered', cluster_entity=True)

def fe_row(sample, label, res, x='dc_capacity_GW'):
    b, se, p = float(res.params[x]), float(res.std_errors[x]), float(res.pvalues[x])
    return dict(sample=sample, spec=label, beta=b, se=se, t=b / se, p_value=p,
                ci95_lo=b - 1.96 * se, ci95_hi=b + 1.96 * se,
                r2_within=float(res.rsquared_within),
                n_obs=int(res.nobs), n_units=int(res.entity_info['total']))

SPECS_5182 = [('emp_5182', 'Raw NAICS 5182'),
              ('emp_cleaned_direct', 'Cleaned direct'),
              ('emp_raw_multiplied', 'Raw x multiplier + spillover'),
              # Balanced total, reported alongside the unbalanced spec-2 row
              # so that 14_panel_fe_table.py can compare both sides.
              ('emp_state_total', 'State total (local impact + spillover)'),
              ('emp_cleaned_multiplied', 'Cleaned x local multiplier')]

banner("PANEL FIXED EFFECTS -- NAICS 5182 FAMILY   (jobs per GW)")
rows = []
for k, p in panels.items():
    for y, lab in SPECS_5182:
        rows.append(fe_row(k, lab, run_panel_fe(p, 'dc_capacity_GW', y)))
fe5182 = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# The 'Raw x multiplier + spillover' specification is the only one here whose
# dependent variable contains a national aggregate: it adds to each unit's own
# multiplied employment a spillover term proportional to national NAICS 5182
# employment outside that unit.  build_panel() above computes that national
# total with a plain .sum() by year, which is unbalanced -- Michigan leaves QWI
# after 2021, so the 2022-2024 national totals lose ~8,570 jobs and the
# coefficient absorbs Michigan's exit from the data as a national decline.
# 09_ols_employment_dc.py computes the same aggregate over units observed in
# every year (panel_aggregates.balanced_sum_by_year), which is the convention
# the SI describes and the value reported there.  The unbalanced value is kept
# here as a diagnostic and labelled in the file itself, so a reader who opens
# panel_fe_5182__contig49.csv cannot mistake the row.  The other three
# specifications contain no national aggregate and are unaffected.
# ---------------------------------------------------------------------------
SPILLOVER_SPEC = 'Raw x multiplier + spillover'
fe5182["national_total_basis"] = np.where(
    fe5182["spec"] == SPILLOVER_SPEC, "unbalanced (plain sum by year)", "n/a")
fe5182["status"] = np.where(
    fe5182["spec"] == SPILLOVER_SPEC, "not_reported", "reported")
fe5182["replaced_by"] = np.where(
    fe5182["spec"] == SPILLOVER_SPEC,
    "09_ols_employment_dc.py -> panel_fe_employment.csv, "
    "spec 'Raw x Multiplier+Spillover' (balanced national total)", "")
fe5182.to_csv(OUT / f"panel_fe_5182{SFX}.csv", index=False)
print(f"\n  the '{SPILLOVER_SPEC}' rows of panel_fe_5182{SFX}.csv are "
      f"labelled not_reported: they use an unbalanced national NAICS 5182 "
      f"total. The balanced value is in panel_fe_employment.csv.")

for y, lab in SPECS_5182:
    print(f"\n  {lab}")
    for k in SAMPLES:
        r = fe5182[(fe5182["sample"] == k) & (fe5182.spec == lab)].iloc[0]
        print(f"    {k:<11s} beta={r.beta:>10,.1f}  SE={r.se:>9,.1f}  p={r.p_value:.4f}  "
              f"R2w={r.r2_within:.3f}  N={r.n_obs}/{r.n_units}")

banner("PANEL FIXED EFFECTS -- RELATED SECTORS")
rows = []
for k, cfg in SAMPLES.items():
    for naics, lab in [('23', 'NAICS 23 Construction'),
                       ('2362', 'NAICS 2362 Nonresidential building'),
                       ('5415', 'NAICS 5415 Computer systems design')]:
        pr = build_related(cfg["map"], cfg["drop"], naics)
        # build_related rebuilds from config and so bypasses any year restriction
        # carried by the sample; apply it here or the matched sample silently
        # reverts to the full window (it did, at N=438 against the IV's 291).
        if cfg.get("year_min"):
            pr = pr[pr["year"] >= cfg["year_min"]]
        rows.append(fe_row(k, lab, run_panel_fe(pr, 'dc_capacity_GW', f'emp_{naics}')))
ferel = pd.DataFrame(rows)
ferel.to_csv(OUT / f"panel_fe_related{SFX}.csv", index=False)
for lab in ferel.spec.unique():
    print(f"\n  {lab}")
    for k in SAMPLES:
        r = ferel[(ferel["sample"] == k) & (ferel.spec == lab)].iloc[0]
        print(f"    {k:<11s} beta={r.beta:>10,.1f}  SE={r.se:>9,.1f}  p={r.p_value:.4f}  "
              f"R2w={r.r2_within:.3f}  N={r.n_obs}/{r.n_units}")

# ---------------------------------------------------------------------------
# State-level time-series OLS -- r05 section 4, VERBATIM
# ---------------------------------------------------------------------------
def run_state_ols(pdf, x_col, y_cols):
    out = []
    for st in sorted(pdf['state_abbr'].unique()):
        d = pdf[pdf['state_abbr'] == st].dropna(subset=[x_col] + y_cols)
        n = len(d)
        x = d[x_col].values.astype(float)
        base = dict(state_abbr=st, n_obs=n)
        if n < 3 or np.std(x) < 1e-10:
            for yc in y_cols:
                base[f'{yc}_slope'] = np.nan
                base[f'{yc}_p'] = np.nan
        else:
            for yc in y_cols:
                sl, ic, r, pv, se = stats.linregress(x, d[yc].values)
                base[f'{yc}_slope'] = sl
                base[f'{yc}_p'] = pv
        out.append(base)
    return pd.DataFrame(out)

banner("STATE-LEVEL TIME-SERIES OLS (N = 9 years per state)")
srows = []
for k, p in panels.items():
    r = run_state_ols(p, 'dc_capacity_GW',
                      ['emp_5182', 'emp_cleaned_direct', 'emp_cleaned_multiplied'])
    v = r.dropna(subset=['emp_cleaned_direct_slope'])
    sig_pos = ((v['emp_cleaned_direct_p'] < 0.05) & (v['emp_cleaned_direct_slope'] > 0)).sum()
    srows.append(dict(sample=k, n_units=len(r), n_valid=len(v),
                      n_sig_p05_cleaned=int((v['emp_cleaned_direct_p'] < 0.05).sum()),
                      n_sig_pos_p05_cleaned=int(sig_pos),
                      n_sig_p05_raw=int((v['emp_5182_p'] < 0.05).sum()),
                      median_slope_raw=float(v['emp_5182_slope'].median()),
                      median_slope_cleaned=float(v['emp_cleaned_direct_slope'].median()),
                      median_slope_cleaned_mult=float(v['emp_cleaned_multiplied_slope'].median())))
    r.to_csv(OUT / f"state_ols_{k}{SFX}.csv", index=False)
sols = pd.DataFrame(srows)
sols.to_csv(OUT / f"state_ols_summary{SFX}.csv", index=False)
print(sols.to_string(index=False))

# ---------------------------------------------------------------------------
# Sensitivity: renormalise NAT_MULT over the 49 in-scope units
# ---------------------------------------------------------------------------
banner("SENSITIVITY -- NAT_MULT renormalised over the 49 in-scope units")
m2 = mult.copy()
m2['spillover_dc'] = m2['emp_spillover_avg'] / NAT_MULT_49
m2['local_share'] = m2['emp_direct_avg'] / (m2['emp_direct_avg'] + m2['spillover_dc'])
ls2 = dict(zip(m2['state_abbr'], m2['local_share']))
ps = p49.copy()
ps['local_share'] = ps['state_abbr'].map(ls2)
ps['emp_cleaned_direct'] = ps['emp_5182'] * ps['local_share']
ps['emp_cleaned_multiplied'] = ps['emp_cleaned_direct'] * ps['local_multiplier']
for y, lab in [('emp_cleaned_direct', 'Cleaned direct'),
               ('emp_cleaned_multiplied', 'Cleaned x local multiplier')]:
    r = fe_row('contig49_natmult49', lab, run_panel_fe(ps, 'dc_capacity_GW', y))
    print(f"  {lab:<28s} beta={r['beta']:>10,.1f}  SE={r['se']:>9,.1f}  p={r['p_value']:.4f}")

banner("DONE")
print(f"  outputs written to {OUT.resolve().relative_to(_DIR.parent)}")
sys.stdout = tee.terminal
tee.close()
print("done")
