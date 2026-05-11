"""
高能源负担家庭的收入与 AMI 分层分析。
从 bench 和 dc 两组 burden 数据中提取 bench burden > 6% 的家庭，
计算户均收入、burden 差值，按 AMI150 分层输出。
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# ============ 路径配置 ============
DIR_BENCH = Path("./rider/bench")
DIR_DC    = Path("./rider/dc")
OUT_DIR   = Path("./rider/income")
YEARS     = [2025, 2030]

SEG_COLS: List[str] = ["AMI150", "TEN", "TEN-YBL6", "TEN-BLD", "TEN-HFL", "NAME"]

INCOME_LO = 1000
INCOME_HI = 200000


def _seg_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in SEG_COLS:
        if c not in df.columns:
            df[c] = "NA"
        df[c] = df[c].astype(str)
        df.loc[df[c].str.lower().isin(["nan", "none", "null", ""]), c] = "NA"
    df["seg_key"] = df[SEG_COLS].agg("||".join, axis=1)
    return df


def process_year(year: int):
    bench_csv = DIR_BENCH / f"energy_burden_{year}.csv"
    dc_csv    = DIR_DC    / f"energy_burden_{year}.csv"
    if not bench_csv.exists() or not dc_csv.exists():
        print(f"[跳过] {year}: 缺少输入文件")
        return

    b = pd.read_csv(bench_csv, low_memory=False)
    d = pd.read_csv(dc_csv, low_memory=False)

    b["county_fips"] = b["county_fips"].astype(str).str.zfill(5)
    d["county_fips"] = d["county_fips"].astype(str).str.zfill(5)
    b = _seg_key(b)
    d = _seg_key(d)

    b = b.rename(columns={"energy_burden_%": "burden_bench"})
    d = d.rename(columns={"energy_burden_%": "burden_dc"})

    keep_b = ["county_fips", "seg_key", "iso", "UNITS", "income_total",
              "burden_bench", "AMI150"]
    keep_d = ["county_fips", "seg_key", "burden_dc"]
    merged = b[[c for c in keep_b if c in b.columns]].merge(
        d[[c for c in keep_d if c in d.columns]],
        on=["county_fips", "seg_key"],
        how="inner",
    )

    merged["UNITS"]        = pd.to_numeric(merged["UNITS"], errors="coerce")
    merged["income_total"] = pd.to_numeric(merged["income_total"], errors="coerce")
    merged["burden_bench"] = pd.to_numeric(merged["burden_bench"], errors="coerce")
    merged["burden_dc"]    = pd.to_numeric(merged["burden_dc"], errors="coerce")

    valid = merged[
        merged["burden_bench"].between(0, 100)
        & merged["burden_dc"].between(0, 100)
        & (merged["burden_bench"] > 6)
        & merged["income_total"].between(INCOME_LO, INCOME_HI)
        & (merged["UNITS"] > 0)
    ].copy()

    valid["income_per_unit"] = valid["income_total"] / valid["UNITS"]

    valid = valid[
        valid["income_per_unit"].between(INCOME_LO, INCOME_HI)
    ].copy()

    valid["burden_diff_%"] = valid["burden_bench"] - valid["burden_dc"]

    out_df = valid[[
        "county_fips", "income_per_unit", "AMI150",
        "burden_bench", "burden_dc", "burden_diff_%",
    ]].copy()

    out_path = OUT_DIR / f"income_ami150_{year}.xlsx"
    out_df.to_excel(out_path, index=False)
    print(f"[OK] {year} → {out_path} ({len(out_df)} rows)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        process_year(y)
    print("完成!")


if __name__ == "__main__":
    main()
