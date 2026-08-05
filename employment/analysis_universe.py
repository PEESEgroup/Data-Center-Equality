#!/usr/bin/env python3
"""
analysis_universe.py -- the single owner of the analysis universe.

employment analysis.

WHY THIS FILE EXISTS
--------------------
The analysis universe must be owned by one module rather than re-declared as a
literal in every script that needs it, so that nothing can drift silently.  This
module is the one place the universe is defined; everything else imports it.

THE UNIVERSE
------------
49 units: the 48 contiguous states plus the District of Columbia.

Alaska and Hawaii are out of scope BY DEFINITION, not by data availability.  Both
sit on isolated grids that none of the seven ISOs analysed in this paper reaches,
the four ISO transmission-planning portfolios used for the cost attribution do
not cover them, and the cross-state spillover and regional cost-sharing
mechanisms studied here cannot operate on an isolated grid.

Note that the first of those three grounds does not by itself discriminate:
eleven retained units are also outside all seven ISOs (WA, OR, ID, NV, UT, AZ,
CO, FL, GA, AL, SC).  The second and third grounds do discriminate correctly.

DC is IN scope.  A name map carrying only 'District of Columbia' -> 'DC' drops it
silently, because the source files spell the unit as the literal two-letter
string "DC", so the map returns NaN.  Any consumer of this module gets DC by
construction.

USAGE
-----
    import sys; sys.path.insert(0, ".../employment")
    from analysis_universe import ANALYSIS_UNITS, OUT_OF_SCOPE_UNITS, check_universe

    df = restrict_to_universe(df)          # drop out-of-scope rows
    check_universe(df)                     # raises UniverseError on any mismatch

`check_universe` is deliberately strict in BOTH directions: it raises on an
out-of-scope unit that is still present AND on an in-scope unit that is missing.
A "50 state" sample that retains AK and HI while dropping DC is the exact
complement of the intended universe on both ends, and a one-sided check would
pass it.

This module performs no I/O, has no dependency beyond pandas, and holds no
estimation logic.  It is safe to import from anywhere.
"""
from __future__ import annotations

__all__ = [
    "ANALYSIS_UNITS",
    "N_ANALYSIS_UNITS",
    "OUT_OF_SCOPE_UNITS",
    "ALL_QWI_UNITS",
    "UNIVERSE_LABEL",
    "UNIVERSE_NOUN",
    "UniverseError",
    "check_universe",
    "restrict_to_universe",
    "is_in_scope",
    "describe",
]

# --------------------------------------------------------------------------- #
# The universe.                                                               #
# --------------------------------------------------------------------------- #

#: The 49 analysis units: 48 contiguous states + the District of Columbia.
#: Sorted, immutable, and the only copy in this codebase.
ANALYSIS_UNITS: tuple[str, ...] = (
    "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
    "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME",
    "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
    "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
)

#: Units present in the raw QWI / capacity files that are OUT of scope.
OUT_OF_SCOPE_UNITS: tuple[str, ...] = ("AK", "HI")

#: Every unit label the raw source files carry (51 = 50 states + DC).
ALL_QWI_UNITS: tuple[str, ...] = tuple(sorted(ANALYSIS_UNITS + OUT_OF_SCOPE_UNITS))

N_ANALYSIS_UNITS: int = len(ANALYSIS_UNITS)

#: Use these strings in prose and in table captions.  The correct noun is
#: "units", not "states": the 49 include DC, which is not a state.
UNIVERSE_LABEL: str = "the 48 contiguous states plus the District of Columbia"
UNIVERSE_NOUN: str = "units"

_UNITS_SET = frozenset(ANALYSIS_UNITS)
_OUT_SET = frozenset(OUT_OF_SCOPE_UNITS)

# Self-checks at import.  A literal never raises; these do.
assert N_ANALYSIS_UNITS == 49, N_ANALYSIS_UNITS
assert len(_UNITS_SET) == 49, "duplicate unit in ANALYSIS_UNITS"
assert not (_UNITS_SET & _OUT_SET), "a unit is both in and out of scope"
assert "DC" in _UNITS_SET, "DC must be in scope"
assert len(ALL_QWI_UNITS) == 51, len(ALL_QWI_UNITS)
assert list(ANALYSIS_UNITS) == sorted(ANALYSIS_UNITS), "ANALYSIS_UNITS must be sorted"

class UniverseError(AssertionError):
    """A frame's units do not match the declared analysis universe."""

# --------------------------------------------------------------------------- #
# Helpers.                                                                    #
# --------------------------------------------------------------------------- #

def is_in_scope(unit: str) -> bool:
    """True iff `unit` is one of the 49 analysis units."""
    return unit in _UNITS_SET

def _extract_units(df, unit_col=None):
    """Pull the unit labels out of a DataFrame, a Series, or any iterable."""
    import pandas as pd

    if isinstance(df, pd.DataFrame):
        if unit_col is not None:
            if unit_col not in df.columns:
                raise UniverseError(
                    f"unit column {unit_col!r} not in frame; columns = "
                    f"{list(df.columns)[:20]}")
            col = unit_col
        else:
            for cand in ("state_abbr", "unit", "state", "abbr", "geo", "state_code"):
                if cand in df.columns:
                    col = cand
                    break
            else:
                raise UniverseError(
                    "cannot find a unit column; pass unit_col=. Tried "
                    "state_abbr/unit/state/abbr/geo/state_code; columns = "
                    f"{list(df.columns)[:20]}")
        vals = df[col]
    elif isinstance(df, pd.Series):
        vals = df
    else:
        vals = pd.Series(list(df), dtype=object)

    return set(vals.dropna().astype(str).str.strip().unique())

def check_universe(df, unit_col=None, *, allow_missing=(), where=""):
    """
    Raise `UniverseError` unless `df`'s units are exactly the 49 analysis units.

    Parameters
    ----------
    df : DataFrame, Series, or iterable of unit labels.
    unit_col : column name; auto-detected from the usual candidates if omitted.
    allow_missing : units permitted to be absent.  Use ONLY for a genuinely
        unbalanced source and say why at the call site -- e.g. MI leaves QWI
        after 2021, so a 2022-2024 slice legitimately has 48 units.  Passing a
        unit here is an assertion that its absence is understood, not an excuse
        to silence the check.
    where : free-text label included in the error message.

    Returns
    -------
    The set of units found, so the call can be chained or logged.

    Raises
    ------
    UniverseError, on an out-of-scope unit present OR an in-scope unit missing.
    Both directions matter: a sample that retains AK and HI while dropping DC
    would pass a one-sided check.
    """
    found = _extract_units(df, unit_col)
    allow = frozenset(allow_missing)

    unknown = allow - _UNITS_SET
    if unknown:
        raise UniverseError(
            f"allow_missing names {sorted(unknown)}, which are not analysis units")

    extra = sorted(found - _UNITS_SET)
    missing = sorted(_UNITS_SET - found - allow)

    if extra or missing:
        tag = f" [{where}]" if where else ""
        parts = [f"universe mismatch{tag}: found {len(found)} units, "
                 f"expected {N_ANALYSIS_UNITS} ({UNIVERSE_LABEL})"]
        if extra:
            out_of_scope = [u for u in extra if u in _OUT_SET]
            other = [u for u in extra if u not in _OUT_SET]
            if out_of_scope:
                parts.append(
                    f"  OUT-OF-SCOPE UNITS PRESENT: {out_of_scope} -- these sit on "
                    f"isolated grids and must be dropped before anything is estimated")
            if other:
                parts.append(f"  UNRECOGNISED UNITS: {other}")
        if missing:
            parts.append(f"  IN-SCOPE UNITS MISSING: {missing}")
            if "DC" in missing:
                parts.append(
                    "  DC is missing. The usual cause is a name map "
                    "that sends 'District of Columbia'->'DC' while the source file "
                    "already spells the unit 'DC', so the map returns NaN and the "
                    "inner join drops it silently.")
        if allow:
            parts.append(f"  (permitted absent: {sorted(allow)})")
        raise UniverseError("\n".join(parts))

    return found

def restrict_to_universe(df, unit_col=None, *, check=True, where=""):
    """
    Return `df` with out-of-scope rows removed.

    This drops rows only.  It does NOT renormalise exposure shares or recompute
    national totals -- those live in the estimation scripts' own `apply_scope`,
    because they depend on which column carries capacity.  If a caller needs
    renormalised shares it must use `r06_employment_bartik_iv.apply_scope`.

    With `check=True` (the default) the result is passed through
    `check_universe`, so a frame that was already missing DC fails here rather
    than three stages downstream.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise UniverseError("restrict_to_universe expects a DataFrame")

    if unit_col is None:
        for cand in ("state_abbr", "unit", "state", "abbr", "geo", "state_code"):
            if cand in df.columns:
                unit_col = cand
                break
        else:
            raise UniverseError(
                "cannot find a unit column; pass unit_col=. columns = "
                f"{list(df.columns)[:20]}")

    out = df[df[unit_col].astype(str).str.strip().isin(_UNITS_SET)].copy()
    if check:
        check_universe(out, unit_col, where=where or "restrict_to_universe")
    return out

def describe() -> str:
    """One-paragraph description, for run logs and for pasting into captions."""
    return (f"Analysis universe: {N_ANALYSIS_UNITS} {UNIVERSE_NOUN} "
            f"({UNIVERSE_LABEL}). Out of scope: "
            f"{', '.join(OUT_OF_SCOPE_UNITS)} -- isolated grids reached by none of "
            f"the ISOs analysed here, not covered by the transmission-planning "
            f"portfolios used for the cost attribution, and outside the "
            f"cross-state spillover mechanism studied. Exclusion is by "
            f"definition, not by data availability.")

if __name__ == "__main__":
    print(describe())
    print(f"\nANALYSIS_UNITS ({N_ANALYSIS_UNITS}):")
    for i in range(0, N_ANALYSIS_UNITS, 10):
        print("   " + " ".join(ANALYSIS_UNITS[i:i + 10]))
    print(f"\nOUT_OF_SCOPE_UNITS ({len(OUT_OF_SCOPE_UNITS)}): "
          f"{' '.join(OUT_OF_SCOPE_UNITS)}")
