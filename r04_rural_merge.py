"""
Merges the county-level energy-poverty difference with USDA Rural-Urban Continuum Codes (RUCC).
Writes two versions:
  - counties with share_diff > 0 only, with a log10 column for plotting
  - all counties, including those with a zero difference
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ============ Paths ============
DIR_POVERTY  = Path("./rider/compare")
PATH_RUCC    = Path("./rider/rural/Ruralurban.xlsx")
OUT_DIR      = Path("./rider/rural")
YEARS        = [2025, 2030]

RUCC_BINS = [
    ((1, 2), 44),   # metropolitan
    ((3, 5), 36),   # small and medium metro
    ((6, 9), 28),   # rural
]


def process_year(year: int):
    pov_csv = DIR_POVERTY / f"energy_poverty_share_{year}.csv"
    if not pov_csv.exists():
        print(f"[skip] {pov_csv} does not exist")
        return

    df1 = pd.read_csv(pov_csv, dtype={"county_fips": str})
    df2 = pd.read_excel(PATH_RUCC, dtype={"FIPS": str})

    df1["county_fips"] = df1["county_fips"].str.zfill(5)
    df2["FIPS"]        = df2["FIPS"].str.zfill(5)

    base = df1[["county_fips", "share_diff_gt6", "share_diff_gt10"]].merge(
        df2[["FIPS", "RUCC_2023"]], left_on="county_fips", right_on="FIPS", how="left",
    ).drop(columns=["FIPS"])

    base["RUCC_2023"] = pd.to_numeric(base["RUCC_2023"], errors="coerce")

    condlist  = [base["RUCC_2023"].between(lo, hi) for (lo, hi), _ in RUCC_BINS]
    choicelist = [val for _, val in RUCC_BINS]
    base["rucc_group"] = np.select(condlist, choicelist, default=np.nan)

    # --- Version 1: diff > 0 only, with log ---
    pos = base[base["share_diff_gt6"] >= 1e-5].copy()
    pos["log_share_diff_gt6"] = np.log10(pos["share_diff_gt6"])
    out1 = OUT_DIR / f"county_rucc_{year}.xlsx"
    pos.to_excel(out1, index=False)
    print(f"[OK] {year} (diff>0) → {out1} ({len(pos)} counties)")

    # --- Version 2: all counties ---
    out2 = OUT_DIR / f"county_rucc_{year}_all.xlsx"
    base.to_excel(out2, index=False)
    print(f"[OK] {year} (all)    → {out2} ({len(base)} counties)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        process_year(y)
    print("done!")


if __name__ == "__main__":
    main()
