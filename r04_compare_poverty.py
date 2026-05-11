"""
Energy poverty share comparison: bench vs dc (counterfactual).

Step 1 — 县级：按县统计两种情景下 energy burden >6% 和 >10% 的家庭占比及差值。
           share_diff = bench − dc，正值说明基线情景中更多家庭处于能源贫困。
Step 2 — 州级：将县级结果按人口加权汇总到州。
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# ============ 路径配置 ============
DIR_BENCH = Path("./rider/bench")
DIR_DC    = Path("./rider/dc")
OUT_DIR   = Path("./rider/compare")
YEARS     = list(range(2025, 2031))

SEG_COLS: List[str] = ["AMI150", "TEN", "TEN-YBL6", "TEN-BLD", "TEN-HFL", "NAME"]

STATE_FIPS_TO_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL",
    "13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME",
    "24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH",
    "34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
    "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI",
    "56":"WY","72":"PR",
}


# ============ 工具函数 ============
def _seg_key(df: pd.DataFrame) -> pd.DataFrame:
    """构造分层 key，用于行级对齐。"""
    df = df.copy()
    for c in SEG_COLS:
        if c not in df.columns:
            df[c] = "NA"
        df[c] = df[c].astype(str)
        df.loc[df[c].str.lower().isin(["nan", "none", "null", ""]), c] = "NA"
    df["seg_key"] = df[SEG_COLS].agg("||".join, axis=1)
    return df


# ============ Step 1: 县级 poverty share ============
def compute_county_year(bench_csv: Path, dc_csv: Path, year: int) -> pd.DataFrame:
    b = pd.read_csv(bench_csv, low_memory=False)
    d = pd.read_csv(dc_csv, low_memory=False)

    b["county_fips"] = b["county_fips"].astype(str).str.zfill(5)
    d["county_fips"] = d["county_fips"].astype(str).str.zfill(5)
    b = _seg_key(b)
    d = _seg_key(d)

    b = b.rename(columns={"energy_burden_%": "burden_bench"})
    d = d.rename(columns={"energy_burden_%": "burden_dc"})

    merged = b[["county_fips", "seg_key", "iso", "UNITS", "burden_bench"]].merge(
        d[["county_fips", "seg_key", "burden_dc"]],
        on=["county_fips", "seg_key"],
        how="inner",
    )

    merged["UNITS"]        = pd.to_numeric(merged["UNITS"], errors="coerce").fillna(0.0)
    merged["burden_bench"] = pd.to_numeric(merged["burden_bench"], errors="coerce")
    merged["burden_dc"]    = pd.to_numeric(merged["burden_dc"], errors="coerce")

    valid = merged[
        (merged["UNITS"] > 0)
        & merged["burden_bench"].between(0, 100)
        & merged["burden_dc"].between(0, 100)
    ].copy()

    out_rows = []
    for county, g in valid.groupby("county_fips", sort=False):
        tu = g["UNITS"].sum()
        if tu <= 0:
            continue

        iso = g["iso"].dropna().astype(str).iloc[0] if g["iso"].notna().any() else ""

        b6  = g.loc[g["burden_bench"] > 6,  "UNITS"].sum() / tu
        b10 = g.loc[g["burden_bench"] > 10, "UNITS"].sum() / tu
        d6  = g.loc[g["burden_dc"]    > 6,  "UNITS"].sum() / tu
        d10 = g.loc[g["burden_dc"]    > 10, "UNITS"].sum() / tu

        out_rows.append({
            "county_fips": county,
            "iso": iso,
            "year": year,
            "total_units_counted": float(tu),
            "share_bench_gt6":  b6,
            "share_bench_gt10": b10,
            "share_dc_gt6":     d6,
            "share_dc_gt10":    d10,
            "share_diff_gt6":   b6 - d6,
            "share_diff_gt10":  b10 - d10,
        })

    return pd.DataFrame(out_rows)


# ============ Step 2: 州级汇总 ============
def aggregate_to_state(county_df: pd.DataFrame) -> pd.DataFrame:
    """将县级 poverty share 按人口加权汇总到州级。"""
    df = county_df.copy()
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["county_fips"].str[:2]
    df["state_abbr"] = df["state_fips"].map(STATE_FIPS_TO_ABBR)

    df["extra_pop_gt6"]  = df["total_units_counted"] * df["share_diff_gt6"]
    df["extra_pop_gt10"] = df["total_units_counted"] * df["share_diff_gt10"]

    df["bench_pop_gt6"]  = df["total_units_counted"] * df["share_bench_gt6"]
    df["bench_pop_gt10"] = df["total_units_counted"] * df["share_bench_gt10"]
    df["dc_pop_gt6"]     = df["total_units_counted"] * df["share_dc_gt6"]
    df["dc_pop_gt10"]    = df["total_units_counted"] * df["share_dc_gt10"]

    group_cols = ["state_fips", "state_abbr", "year"]
    agg_cols = [
        "total_units_counted",
        "bench_pop_gt6", "bench_pop_gt10",
        "dc_pop_gt6", "dc_pop_gt10",
        "extra_pop_gt6", "extra_pop_gt10",
    ]

    state = (
        df.groupby(group_cols, as_index=False)[agg_cols]
        .sum()
        .rename(columns={"total_units_counted": "total_units_state"})
    )

    state["share_bench_gt6_state"]  = state["bench_pop_gt6"]  / state["total_units_state"]
    state["share_bench_gt10_state"] = state["bench_pop_gt10"] / state["total_units_state"]
    state["share_dc_gt6_state"]     = state["dc_pop_gt6"]     / state["total_units_state"]
    state["share_dc_gt10_state"]    = state["dc_pop_gt10"]    / state["total_units_state"]
    state["share_diff_gt6_state"]   = state["extra_pop_gt6"]  / state["total_units_state"]
    state["share_diff_gt10_state"]  = state["extra_pop_gt10"] / state["total_units_state"]

    state = state.drop(columns=[
        "bench_pop_gt6", "bench_pop_gt10",
        "dc_pop_gt6", "dc_pop_gt10",
        "extra_pop_gt6", "extra_pop_gt10",
    ])

    return state.sort_values(["year", "state_fips"]).reset_index(drop=True)


# ============ 主流程 ============
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: 县级 ---
    county_parts = []
    for y in YEARS:
        bc = DIR_BENCH / f"energy_burden_{y}.csv"
        dc = DIR_DC    / f"energy_burden_{y}.csv"
        if not bc.exists() or not dc.exists():
            print(f"[跳过] {y}: 缺少输入文件")
            continue

        res = compute_county_year(bc, dc, y)
        if res.empty:
            print(f"[警告] {y} 年没有可用记录")
            continue

        out_csv = OUT_DIR / f"energy_poverty_share_{y}.csv"
        res.to_csv(out_csv, index=False, encoding="utf-8-sig")
        county_parts.append(res)
        print(f"[县级] {y} → {out_csv} ({len(res)} counties)")

    if not county_parts:
        print("[错误] 没有任何县级结果，退出。")
        return

    county_all = pd.concat(county_parts, ignore_index=True)
    county_all_csv = OUT_DIR / "energy_poverty_share_all_years.csv"
    county_all.to_csv(county_all_csv, index=False, encoding="utf-8-sig")
    print(f"[县级] 合并总表 → {county_all_csv} ({len(county_all)} rows)")

    # --- Step 2: 州级 ---
    state_all = aggregate_to_state(county_all)

    for y in YEARS:
        sy = state_all[state_all["year"] == y]
        if sy.empty:
            continue
        out_state = OUT_DIR / f"state_poverty_share_{y}.csv"
        sy.to_csv(out_state, index=False, encoding="utf-8-sig")
        print(f"[州级] {y} → {out_state} ({len(sy)} states)")

    state_all_csv = OUT_DIR / "state_poverty_share_all_years.csv"
    state_all.to_csv(state_all_csv, index=False, encoding="utf-8-sig")
    print(f"[州级] 合并总表 → {state_all_csv} ({len(state_all)} rows)")


if __name__ == "__main__":
    main()
