#!/usr/bin/env python3
"""
panel_aggregates.py -- the single owner of "sum over a panel".

employment analysis.

WHY THIS FILE EXISTS
--------------------
A descriptive aggregate computed with .sum() over an unbalanced panel counts a
state leaving the data as an employment decline.  QWI is unbalanced in exactly
two places and both of them bite:

    Alaska   is observed in 2016 only          (out of scope anyway)
    Michigan is absent from 2022, 2023, 2024   (IN scope -- this is the one)

A bare ``groupby('year')['emp'].sum()`` therefore drops Michigan's ~8,570
NAICS 5182 jobs, ~15,000 NAICS 517 jobs and ~53,000 NAICS 51 jobs out of the
2022-2024 national totals, and every 2016->2024 change computed from those
totals carries Michigan's exit as if it were a real employment change.  In one
national series the sign of the 2016-2024 change turns on it.

This module makes the balanced restriction explicit, mandatory, and LOUD: every
call prints the per-year unit count of the raw series and of the balanced
series, so an unbalanced panel can never hide again.  It performs no I/O and
holds no estimation logic.

USAGE
-----
    import sys; sys.path.insert(0, ".../employment")
    from panel_aggregates import balanced_sum_by_year

    tot, meta = balanced_sum_by_year(df, "state_abbr", "year", "emp_5182",
                                     label="NAICS 5182 national total")
    # tot  -- pd.Series indexed by year, summed over meta["balanced_units"] only
    # meta -- n per year raw and balanced, the dropped units, the year span

A "balanced" unit is one observed with a non-null value in EVERY year of the
span the aggregate covers.  That is the only definition used here, and the
span is stated explicitly at every call site.
"""
from __future__ import annotations

__all__ = [
    "PanelBalanceError",
    "year_counts",
    "balanced_units",
    "balanced_sum_by_year",
    "format_balance_report",
]

class PanelBalanceError(AssertionError):
    """A panel aggregate could not be formed on a balanced sample."""

def year_counts(df, unit_col, year_col, value_col):
    """n distinct units with a non-null `value_col`, per year.  dict {year: n}."""
    sub = df[df[value_col].notna()]
    return {int(y): int(g[unit_col].nunique())
            for y, g in sub.groupby(year_col)}

def balanced_units(df, unit_col, year_col, value_col, years=None):
    """
    Units observed with a non-null `value_col` in EVERY year of `years`.

    `years` defaults to the full set of years present in `df`.
    Returns (sorted list of balanced units, sorted list of dropped units,
             sorted list of years).
    """
    sub = df[df[value_col].notna()]
    if years is None:
        years = sorted(int(y) for y in sub[year_col].unique())
    else:
        years = sorted(int(y) for y in years)
    yrs = set(years)

    seen = (sub[sub[year_col].isin(yrs)]
            .groupby(unit_col)[year_col]
            .apply(lambda s: set(int(v) for v in s)))

    keep = sorted(u for u, s in seen.items() if yrs.issubset(s))
    drop = sorted(u for u in df[unit_col].dropna().unique() if u not in keep)
    return keep, drop, years

def format_balance_report(meta, indent="  "):
    """Human-readable per-year n, raw and balanced.  Never abbreviates."""
    L = []
    L.append(f"{indent}[balance] {meta['label']}")
    L.append(f"{indent}  span                : {meta['years'][0]}-{meta['years'][-1]} "
             f"({len(meta['years'])} years)")
    L.append(f"{indent}  units, raw          : "
             + " ".join(f"{y}:{meta['n_raw'].get(y, 0)}" for y in meta['years']))
    L.append(f"{indent}  units, balanced     : "
             + " ".join(f"{y}:{meta['n_balanced'].get(y, 0)}" for y in meta['years']))
    L.append(f"{indent}  n balanced units    : {len(meta['balanced_units'])}")
    if meta["dropped_units"]:
        L.append(f"{indent}  DROPPED (unbalanced): {', '.join(meta['dropped_units'])}")
    else:
        L.append(f"{indent}  DROPPED (unbalanced): none -- panel is already balanced")
    return "\n".join(L)

def balanced_sum_by_year(df, unit_col, year_col, value_col, *, years=None,
                         label="", log=print, min_units=2):
    """
    Sum `value_col` by year over units observed in EVERY year of the span.

    Returns (pd.Series indexed by int year, meta dict).  Always logs the
    per-year unit count of the raw and of the balanced series.

    Raises PanelBalanceError if fewer than `min_units` units survive.
    """
    import pandas as pd

    label = label or value_col
    keep, drop, years = balanced_units(df, unit_col, year_col, value_col, years)

    if len(keep) < min_units:
        raise PanelBalanceError(
            f"{label}: only {len(keep)} unit(s) observed in every year of "
            f"{years[0]}-{years[-1]}; refusing to aggregate")

    sub = df[df[value_col].notna() & df[year_col].isin(years)]
    n_raw = {int(y): int(g[unit_col].nunique()) for y, g in sub.groupby(year_col)}

    bal = sub[sub[unit_col].isin(keep)]
    n_bal = {int(y): int(g[unit_col].nunique()) for y, g in bal.groupby(year_col)}

    total = bal.groupby(year_col)[value_col].sum()
    total.index = [int(i) for i in total.index]
    total = total.sort_index()

    meta = {
        "label": label,
        "years": years,
        "n_raw": n_raw,
        "n_balanced": n_bal,
        "balanced_units": keep,
        "dropped_units": drop,
        "unbalanced_total": sub.groupby(year_col)[value_col].sum().sort_index(),
    }

    if log is not None:
        log(format_balance_report(meta))

    # The balanced series must itself be balanced.  Cheap, and it has caught
    # a wrong `years=` argument once already.
    assert len(set(n_bal.values())) == 1, n_bal
    assert list(n_bal.values())[0] == len(keep), (n_bal, len(keep))

    return total, meta
