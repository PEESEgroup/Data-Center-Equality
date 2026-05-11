"""
Counterfactual energy burden: 计及数据中心影响的情景。
unit_price = gen − DC_cumulate + tre × (RU_attr_ratio / RU_alloc_ratio) + dse
无 sankey 数据的区域: unit_price = total − DC_cumulate
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# ===================== 路径配置 =====================
PATH_MAP           = Path("./rider/burden/county_to_iso_or_city.csv")
DIR_LEAD_2022      = Path("./econ_and_ai/LEAD/")
PATH_INCOME_GROWTH = Path("./econ_and_ai/macroeconomic.xlsx")
INCOME_SHEET       = 0
DIR_ISO_PRICES     = Path("./rider/EIA/")
DIR_SANKEY         = Path("./sankey/")
PATH_GAS_OIL       = Path("./econ_and_ai/fuel/state_fuel_price.xlsx")
OUT_DIR            = Path("./rider/dc")
YEARS              = list(range(2025, 2031))
CSV_CHUNK_ROWS     = 2_000_000

SANKEY_ISOS        = ["CAISO", "ERCOT", "MISO", "PJM"]

STATE_FIPS_TO_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL",
    "13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME",
    "24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH",
    "34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
    "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI",
    "56":"WY","72":"PR",
}


# ===================== 通用工具 =====================
def clean_zone(z: str, iso: str | None = None) -> str:
    if pd.isna(z) or z is None:
        return ""
    z0 = str(z).strip()
    if iso and isinstance(iso, str) and len(iso) > 0:
        pattern = r"^\s*{}[\s_\-:/]+".format(re.escape(iso))
        z0 = re.sub(pattern, "", z0, flags=re.IGNORECASE)
    z0 = re.sub(r"[^A-Za-z0-9]+", "", z0)
    return z0.upper()


def clean_iso_name_from_filename(p: Path) -> str:
    name = p.stem
    m = re.match(r"([A-Za-z\-]+)", name)
    iso = m.group(1) if m else name
    return iso.upper().replace("-", "")


def split_and_clean_zones(series_zones, iso):
    tokens = []
    for z in series_zones.astype(str):
        parts = re.split(r"\s*\|\s*", z)
        for p in parts:
            if p and p.lower() != "nan":
                tokens.append(clean_zone(p, iso))
    return sorted({t for t in tokens if t})


# ===================== 1) 县→ISO/Zone 映射 =====================
def load_county_map(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype={"county_fips": str})
    else:
        df = pd.read_excel(path, dtype={"county_fips": str})
    need_cols = ["county_fips", "iso", "zone"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"映射表缺少列：{miss}")
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["iso"] = df["iso"].astype(str).str.strip().str.upper()
    return df[need_cols]


# ===================== 2) LEAD 2022 基准 =====================
def load_lead_2022(dir_path: Path) -> pd.DataFrame:
    records = []
    for csv in sorted(dir_path.glob("* AMI Counties 2022.csv")):
        try:
            df = pd.read_csv(csv, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(csv, low_memory=False, encoding="latin1")

        df.columns = [c.strip() for c in df.columns]
        need_val = ["HINCP*UNITS", "ELEP*UNITS", "GASP*UNITS", "FULP*UNITS"]
        for c in need_val:
            if c not in df.columns:
                raise KeyError(f"{c} 缺失于 {csv.name}")

        units_col = "UNITS" if "UNITS" in df.columns else ("FREQUENCY" if "FREQUENCY" in df.columns else None)
        if units_col is None:
            raise KeyError(f"既无 UNITS 也无 FREQUENCY：{csv.name}")
        seg_candidates = ["AMI150", "TEN", "TEN-YBL6", "TEN-BLD", "TEN-HFL", "NAME"]
        seg_cols = [c for c in seg_candidates if c in df.columns]

        if "FIP" not in df.columns:
            raise KeyError(f"FIP 缺失于 {csv.name}")
        df["county_fips"] = df["FIP"].astype(str).str.zfill(5)
        df["state_abbr"] = df["county_fips"].str[:2].map(STATE_FIPS_TO_ABBR)
        cols = ["county_fips", "state_abbr", units_col] + seg_cols + need_val
        tmp = df[cols].copy()
        tmp = tmp.rename(columns={units_col: "UNITS"})
        tmp["UNITS"] = pd.to_numeric(tmp["UNITS"], errors="coerce").fillna(0.0)
        for c in need_val:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)
        records.append(tmp)

    return pd.concat(records, ignore_index=True)


# ===================== 3) 收入增长系数 (2022=1) =====================
def load_income_growth(path: Path, sheet=0, years: List[int] = YEARS) -> Dict[int, float]:
    df = pd.read_excel(path, sheet_name=sheet)
    row_idx = None
    if df.columns.size >= 2:
        first_col = df.columns[0]
        mask = df[first_col].astype(str).str.contains("Real Dispos", case=False, na=False)
        if mask.any():
            row_idx = df[mask].index[-1]
    if row_idx is None:
        row_idx = df.dropna(how="all", axis=0).index[-1]
    row = df.loc[row_idx]
    growth = {}
    for y in years + [2022]:
        if y in df.columns:
            growth[y] = float(row[y])
        else:
            raise KeyError(f"收入增长系数表缺少年份列 {y}")
    if abs(growth[2022] - 1.0) > 1e-6:
        base = growth[2022]
        for k in list(growth.keys()):
            growth[k] = growth[k] / base
    return {y: growth[y] for y in years}


# ===================== 4) Sankey 传输调节因子 =====================
def load_sankey_factors(sankey_dir: Path, target_years: list[int]) -> dict:
    """
    Returns {ISO: {zone_clean: {year: factor}}}
    factor = RU_attr_ratio / RU_alloc_ratio
    超出 sankey 数据年限的年份用已有年份均值填充。
    """
    result: dict = {}
    for iso in SANKEY_ISOS:
        attr_path = sankey_dir / f"{iso}_attribution.csv"
        alloc_path = sankey_dir / f"{iso}_allocation.csv"
        if not attr_path.exists() or not alloc_path.exists():
            continue

        attr = pd.read_csv(attr_path)
        alloc = pd.read_csv(alloc_path)
        merged = attr[["Year", "Zone", "RU_ratio"]].merge(
            alloc[["Year", "Zone", "RU_ratio"]],
            on=["Year", "Zone"], suffixes=("_attr", "_alloc"),
        )

        iso_upper = iso.upper()
        iso_dict: dict = {}
        for zone, grp in merged.groupby("Zone"):
            zc = clean_zone(zone, iso_upper)
            yearly: dict[int, float] = {}
            for _, r in grp.iterrows():
                y = int(r["Year"])
                ra, rl = float(r["RU_ratio_attr"]), float(r["RU_ratio_alloc"])
                if rl > 0:
                    yearly[y] = min(1.0, ra / rl)
            if not yearly:
                continue
            mean_f = float(np.mean(list(yearly.values())))
            for y in target_years:
                if y not in yearly:
                    yearly[y] = mean_f
            iso_dict[zc] = yearly
        result[iso_upper] = iso_dict
    return result


# ===================== 5) 反事实电价倍率 =====================
def _pick_row(df_idx, names):
    for nm in names:
        if nm in df_idx.index:
            return df_idx.loc[nm]
    for nm in names:
        m = df_idx.index.to_series().str.contains(re.escape(nm), case=False, na=False)
        if m.any():
            return df_idx.loc[m].iloc[0]
    return None


def _val(series, year):
    if series is None:
        return None
    v = series.get(year, np.nan)
    v = pd.to_numeric(v, errors="coerce")
    return float(v) if pd.notna(v) and np.isfinite(float(v)) else None


def load_counterfactual_ratios(
    dir_path: Path, sankey: dict, years: list[int],
) -> dict:
    """
    Read gen/tre/dse/total/DC_cumulate, compute adjusted unit price.
    Returns {ISO: {zone_clean: {year: ratio}}}
    ratio = adjusted_price / base_total
    有 sankey:  adjusted = gen − dc + tre × factor + dse
    无 sankey:  adjusted = total − dc
    """
    ratios: dict = {}

    for xlsx in sorted(dir_path.glob("*_zone_prices.xlsx")):
        iso = clean_iso_name_from_filename(xlsx)
        ratios.setdefault(iso, {})
        sankey_iso = sankey.get(iso, {})

        try:
            xf = pd.ExcelFile(xlsx)
        except Exception:
            continue

        for sheet in xf.sheet_names:
            try:
                df = pd.read_excel(xlsx, sheet_name=sheet, engine="openpyxl")
            except Exception:
                continue
            if df is None or df.empty:
                continue

            col_map = {}
            for c in df.columns:
                cs = str(c).strip()
                if re.fullmatch(r"\d{4}", cs):
                    col_map[c] = int(cs)
            df = df.rename(columns=col_map)

            item_col = None
            for c in df.columns:
                if str(c).strip().lower() == "item":
                    item_col = c
                    break
            if item_col is None:
                item_col = df.columns[0]
            df[item_col] = df[item_col].astype(str).str.strip().str.lower()
            df = df.set_index(item_col)

            s_gen   = _pick_row(df, ["gen (generation)", "gen"])
            s_tre   = _pick_row(df, ["tre (transmission)", "tre"])
            s_dse   = _pick_row(df, ["dse (distribution)", "dse"])
            s_total = _pick_row(df, ["total"])
            s_dc    = _pick_row(df, ["dc_cumulate"])

            if s_total is None:
                continue

            base_total = None
            for yc in sorted(k for k in s_total.index if isinstance(k, int)):
                v = pd.to_numeric(s_total.get(yc), errors="coerce")
                if pd.notna(v) and np.isfinite(v) and v > 0:
                    base_total = float(v)
                    break
            if base_total is None or base_total == 0:
                continue

            zc = clean_zone(sheet, iso)
            if not zc:
                continue

            has_sankey = zc in sankey_iso
            zone_ratios: dict[int, float] = {}

            for y in years:
                gen   = _val(s_gen, y)
                tre   = _val(s_tre, y)
                dse   = _val(s_dse, y)
                total = _val(s_total, y)
                dc    = _val(s_dc, y) or 0.0

                if has_sankey and gen is not None and tre is not None and dse is not None:
                    factor = sankey_iso[zc].get(y, 1.0)
                    adjusted = gen - dc + tre * factor + dse
                elif total is not None:
                    adjusted = total - dc
                else:
                    continue

                zone_ratios[y] = adjusted / base_total

            if zone_ratios:
                ratios[iso][zc] = zone_ratios

    return ratios


# ===================== 6) 州燃料价格倍率 (2022=1) =====================
def load_state_fuel_price_ratios(path: Path, years: List[int] = YEARS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gas_ratios = {}
    oil_ratios = {}
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        idx_col = df.columns[0]
        df[idx_col] = df[idx_col].astype(str).str.strip()
        df = df.set_index(idx_col)

        def pick_row_exact(cand):
            for nm in cand:
                if nm in df.index:
                    return df.loc[nm]
            for nm in cand:
                m = df.index.to_series().str.contains(re.escape(nm), case=False, na=False)
                if m.any():
                    return df.loc[m].iloc[0]
            return None

        s_ng = pick_row_exact(["NG (Natural Gas)", "Natural Gas"])
        s_dfo = pick_row_exact(["DFO (Distillate Fuel Oil)", "Distillate Fuel Oil"])
        if s_ng is None or s_dfo is None:
            continue
        base_ng = float(s_ng[2022])
        base_dfo = float(s_dfo[2022])
        if base_ng == 0 or base_dfo == 0:
            continue
        gas_ratios[sheet] = {y: float(s_ng[y]) / base_ng for y in years if y in s_ng.index}
        oil_ratios[sheet] = {y: float(s_dfo[y]) / base_dfo for y in years if y in s_dfo.index}
    gas_df = pd.DataFrame.from_dict(gas_ratios, orient="index").sort_index()
    oil_df = pd.DataFrame.from_dict(oil_ratios, orient="index").sort_index()
    gas_df.index.name = "state_abbr"
    oil_df.index.name = "state_abbr"
    return gas_df, oil_df


# ===================== 主流程 =====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("加载县映射表...")
    m = load_county_map(PATH_MAP)
    iso_price_files = list(DIR_ISO_PRICES.glob("*_zone_prices.xlsx"))
    iso_valid = set(clean_iso_name_from_filename(p) for p in iso_price_files)
    m = m[m["iso"].isin(iso_valid)].copy()

    print("加载 LEAD 2022 基准数据...")
    lead = load_lead_2022(DIR_LEAD_2022)
    base = lead.merge(m, on="county_fips", how="inner")

    print("加载收入增长系数...")
    income_growth = load_income_growth(PATH_INCOME_GROWTH, INCOME_SHEET, YEARS)

    print("加载 sankey 传输调节因子...")
    sankey_factors = load_sankey_factors(DIR_SANKEY, YEARS)
    for iso, zmap in sankey_factors.items():
        print(f"  {iso}: {len(zmap)} zones")

    print("加载反事实电价倍率...")
    price_ratios = load_counterfactual_ratios(DIR_ISO_PRICES, sankey_factors, YEARS)

    print("加载燃料价格倍率...")
    gas_ratio_df, oil_ratio_df = load_state_fuel_price_ratios(PATH_GAS_OIL, YEARS)

    seg_cols = [c for c in ["AMI150", "TEN", "TEN-YBL6", "TEN-BLD", "TEN-HFL", "NAME"] if c in base.columns]

    results_by_year = {y: [] for y in YEARS}
    n_counties = base["county_fips"].nunique()
    rows_appended = 0
    skipped_no_zone = 0

    print(f"开始处理 {n_counties} 个县...")
    for county, g in base.groupby("county_fips", sort=False):
        iso = g["iso"].iloc[0]
        state_abbr = g["state_abbr"].iloc[0]

        zones = split_and_clean_zones(g["zone"], iso)
        zone_ratios = [price_ratios.get(iso, {}).get(z, None) for z in zones]
        zone_ratios = [r for r in zone_ratios if r]
        if not zone_ratios:
            skipped_no_zone += 1
            continue

        def avg_ratio(year):
            vals = [r.get(year) for r in zone_ratios if r.get(year) is not None]
            return float(np.mean(vals)) if vals else None

        for _, row in g.iterrows():
            inc0 = float(row["HINCP*UNITS"])
            elec0 = float(row["ELEP*UNITS"])
            gas0 = float(row["GASP*UNITS"])
            fuel0 = float(row["FULP*UNITS"])

            for y in YEARS:
                r_elec = avg_ratio(y)
                if r_elec is None:
                    continue

                income = inc0 * float(income_growth[y])
                elec_cost = elec0 * r_elec

                try:
                    gas_r = float(gas_ratio_df.loc[state_abbr, y])
                    oil_r = float(oil_ratio_df.loc[state_abbr, y])
                except Exception:
                    continue
                gas_cost = gas0 * gas_r
                fuel_cost = fuel0 * oil_r

                if income <= 0:
                    burden = np.nan
                else:
                    burden = 100.0 * (elec_cost + gas_cost + fuel_cost) / income

                out_row = {
                    "county_fips": county,
                    "iso": iso,
                    "zone_list": ";".join(zones),
                    "state": state_abbr,
                    "UNITS": float(row["UNITS"]),
                    "income_total": income,
                    "elec": elec_cost,
                    "gas": gas_cost,
                    "fuel": fuel_cost,
                    "energy_burden_%": burden,
                }
                for c in seg_cols:
                    out_row[c] = row[c]

                results_by_year[y].append(out_row)
                rows_appended += 1

    print(f"处理完成: counties={n_counties}, skipped={skipped_no_zone}, rows={rows_appended}")

    for y in YEARS:
        dfy = pd.DataFrame(results_by_year[y])
        if dfy.empty:
            continue
        n_before = len(dfy)
        dfy = dfy[dfy["energy_burden_%"].between(0, 100)].copy()
        print(f"  {y}: 过滤 [0,100] 后 {n_before} -> {len(dfy)} rows (去除 {n_before - len(dfy)})")
        out_csv = OUT_DIR / f"energy_burden_{y}.csv"
        n = len(dfy)
        if n <= CSV_CHUNK_ROWS:
            dfy.to_csv(out_csv, index=False, encoding="utf-8-sig")
        else:
            first = True
            for s in range(0, n, CSV_CHUNK_ROWS):
                e = min(s + CSV_CHUNK_ROWS, n)
                dfy.iloc[s:e].to_csv(
                    out_csv,
                    mode="w" if first else "a",
                    header=first,
                    index=False,
                    encoding="utf-8-sig",
                )
                first = False
        print(f"  {out_csv.name}: {n} rows")

    print(f"完成! CSV 已写入: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
