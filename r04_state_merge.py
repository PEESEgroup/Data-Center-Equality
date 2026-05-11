"""
State-level merge: 合并 AI 使用率、GDP、DC 容量和能源贫困差值到一张宽表。
仅保留四个数据源共同存在的州（内连接）。
"""
from pathlib import Path
import pandas as pd

# ============ 路径配置 ============
PATH_AI       = Path("./econ_and_ai/state_accept/state_ai.xlsx")
PATH_BB       = Path("./econ_and_ai/state_accept/state_bb.xlsx")
PATH_GDP      = Path("./econ_and_ai/state_accept/state_gdp.xlsx")
PATH_CAPACITY = Path("./econ_and_ai/state_accept/state_dc_capacity.xlsx")
DIR_POVERTY   = Path("./rider/compare")
OUT_DIR       = Path("./rider/compare")
YEARS         = [2025, 2030]

# ============ 州名 → 缩写 映射 ============
STATE_NAME_TO_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
    "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
    "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
    "North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR",
    "Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
    "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
}


def _map_name_to_abbr(df: pd.DataFrame, name_col: str = "State") -> pd.DataFrame:
    df = df.copy()
    df["abbr"] = df[name_col].map(STATE_NAME_TO_ABBR)
    return df.dropna(subset=["abbr"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 读取 4 个州级数据源 ---
    df_ai = _map_name_to_abbr(pd.read_excel(PATH_AI))
    df_ai = df_ai.rename(columns={"AI Tool Usage (%)": "ai_tool_usage_pct"})

    df_bb = _map_name_to_abbr(pd.read_excel(PATH_BB))
    df_bb = df_bb.rename(columns={"AI Tool Usage (%)": "broadband_pct"})

    df_gdp = _map_name_to_abbr(pd.read_excel(PATH_GDP))

    df_cap = pd.read_excel(PATH_CAPACITY).rename(columns={"State": "abbr"})

    # --- 逐年合并 ---
    for y in YEARS:
        pov_csv = DIR_POVERTY / f"state_poverty_share_{y}.csv"
        if not pov_csv.exists():
            print(f"[跳过] {pov_csv} 不存在")
            continue

        df_pov = pd.read_csv(pov_csv, dtype={"state_fips": str})
        df_pov = df_pov.rename(columns={"state_abbr": "abbr"})

        # 取五个表共同的州（内连接）
        common = (
            set(df_ai["abbr"])
            & set(df_bb["abbr"])
            & set(df_gdp["abbr"])
            & set(df_cap["abbr"])
            & set(df_pov["abbr"].dropna())
        )
        print(f"{y}: 共同州 {len(common)} 个")

        merged = (
            df_ai.loc[df_ai["abbr"].isin(common), ["abbr", "ai_tool_usage_pct"]]
            .merge(df_bb.loc[df_bb["abbr"].isin(common), ["abbr", "broadband_pct"]], on="abbr")
            .merge(df_gdp.loc[df_gdp["abbr"].isin(common), ["abbr", "Nominal_GDP_per_capita_2024_USD"]], on="abbr")
            .merge(df_cap.loc[df_cap["abbr"].isin(common), ["abbr", "Capacity_2025", "Capacity_2030"]], on="abbr")
            .merge(df_pov.loc[df_pov["abbr"].isin(common),
                              ["abbr", "total_units_state",
                               "share_bench_gt6_state", "share_bench_gt10_state",
                               "share_dc_gt6_state", "share_dc_gt10_state",
                               "share_diff_gt6_state", "share_diff_gt10_state"]],
                   on="abbr")
        )

        merged = merged.sort_values("abbr").reset_index(drop=True)

        out_path = OUT_DIR / f"merged_states_{y}.xlsx"
        merged.to_excel(out_path, index=False)
        print(f"[OK] {y} → {out_path} ({len(merged)} states)")

    print("完成!")


if __name__ == "__main__":
    main()
