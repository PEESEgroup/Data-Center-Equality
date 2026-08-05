#!/usr/bin/env python3
"""
Stream LEHD QWI state-level bulk files and filter to the 6 target NAICS codes,
"All Workers" demographic cell, all private ownership, unadjusted.

Mirrors exactly the Census API query used to build qwi_all_naics_annual.csv:
    ownercode=A05, firmage=0, firmsize=0, seasonadj=U, sex=0, agegrp=A00
(plus race=A0, ethnicity=A0, education=E0 which the API's `sa` endpoint implies)
"""
import gzip
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Release-set path rule: no absolute paths.  Everything is resolved from this
# file's own location, so the repository runs wherever it is cloned.  The raw
# LEHD downloads are bulk intermediates, not results: they go to a scratch
# directory inside the repository that step 02 reads and nothing else does.
EMP = os.path.dirname(os.path.abspath(__file__))          # employment/
SCRATCH = os.path.join(EMP, "_scratch")
OUT = os.path.join(SCRATCH, "raw")
os.makedirs(OUT, exist_ok=True)

BASE = "https://lehd.ces.census.gov/data/qwi/R2026Q1"

STATE_FIPS = {
    "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co",
    "09": "ct", "10": "de", "11": "dc", "12": "fl", "13": "ga", "15": "hi",
    "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks", "21": "ky",
    "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn",
    "28": "ms", "29": "mo", "30": "mt", "31": "ne", "32": "nv", "33": "nh",
    "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd", "39": "oh",
    "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd",
    "47": "tn", "48": "tx", "49": "ut", "50": "vt", "51": "va", "53": "wa",
    "54": "wv", "55": "wi", "56": "wy",
}

# file suffix -> set of industry codes to keep
FILE_INDUSTRIES = {
    "ns": {"23", "51"},
    "n3": {"517"},
    "n4": {"2362", "5182", "5415"},
}

# 0-indexed column positions in the QWI CSV
C = dict(seasonadj=1, geo_level=2, geography=3, ind_level=4, industry=5,
         ownercode=6, sex=7, agegrp=8, race=9, ethnicity=10, education=11,
         firmage=12, firmsize=13, year=14, quarter=15, agg_level=16,
         Emp=17, EmpEnd=18, EarnBeg=44, Payroll=48)

KEEP_COLS = ["geography", "ind_level", "industry", "year", "quarter",
             "agg_level", "Emp", "EmpEnd", "EarnBeg", "Payroll"]

def fetch_one(fips, st, suffix):
    dest = os.path.join(OUT, f"{st}_{suffix}.csv")
    if os.path.exists(dest) and os.path.getsize(dest) > 100:
        return (st, suffix, "cached", 0)
    url = f"{BASE}/{st}/qwi_{st}_sa_f_gs_{suffix}_op_u.csv.gz"
    inds = FILE_INDUSTRIES[suffix]
    n = 0
    for attempt in range(4):
        try:
            r = requests.get(url, stream=True, timeout=180)
            if r.status_code != 200:
                return (st, suffix, f"HTTP {r.status_code}", 0)
            gz = gzip.GzipFile(fileobj=r.raw)
            rows = []
            header_seen = False
            for raw in io.TextIOWrapper(gz, encoding="utf-8"):
                if not header_seen:
                    header_seen = True
                    hdr = raw.rstrip("\n").split(",")
                    # sanity check column positions once
                    for k, i in C.items():
                        if hdr[i] != k:
                            raise RuntimeError(
                                f"column mismatch {k}: expected at {i}, got {hdr[i]}")
                    continue
                f = raw.rstrip("\n").split(",")
                if f[C["industry"]] not in inds:
                    continue
                if (f[C["ownercode"]] != "A05" or f[C["sex"]] != "0"
                        or f[C["agegrp"]] != "A00" or f[C["race"]] != "A0"
                        or f[C["ethnicity"]] != "A0" or f[C["education"]] != "E0"
                        or f[C["firmage"]] != "0" or f[C["firmsize"]] != "0"
                        or f[C["seasonadj"]] != "U"):
                    continue
                rows.append(",".join(f[C[c]] for c in KEEP_COLS))
                n += 1
            with open(dest, "w") as fh:
                fh.write(",".join(KEEP_COLS) + "\n")
                fh.write("\n".join(rows) + "\n")
            return (st, suffix, "ok", n)
        except Exception as e:
            if attempt == 3:
                return (st, suffix, f"ERROR {type(e).__name__}: {e}", 0)
            time.sleep(3 * (attempt + 1))
    return (st, suffix, "unreachable", 0)

def main():
    jobs = [(fips, st, sfx) for fips, st in STATE_FIPS.items()
            for sfx in FILE_INDUSTRIES]
    done = 0
    fails = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_one, *j): j for j in jobs}
        for fut in as_completed(futs):
            st, sfx, status, n = fut.result()
            done += 1
            if status not in ("ok", "cached"):
                fails.append((st, sfx, status))
                print(f"[{done}/{len(jobs)}] {st}_{sfx}: {status}", flush=True)
            elif done % 15 == 0:
                print(f"[{done}/{len(jobs)}] ... {st}_{sfx} {status} n={n}", flush=True)
    print(f"DONE. failures={fails}")

if __name__ == "__main__":
    main()
