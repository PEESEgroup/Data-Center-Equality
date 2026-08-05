#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute transmission cost Attribution (load-growth responsibility) and
Allocation (existing-load share) for RU / DC / Other across CAISO, ERCOT,
MISO, PJM zones.  Generate three-column Sankey diagrams:
    Left (RU/DC/Other attribution) → Zones → Right (RU/DC/Other allocation)

Outputs
-------
  ./sankey/{ISO}_attribution.csv, {ISO}_allocation.csv, {ISO}_results.xlsx
  ./figures/03/{ISO}_sankey_{year}.html/.png
  ./figures/03/{ISO}_sankey_all.html/.png
"""

import warnings, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

warnings.filterwarnings("ignore")

# ======================== paths ========================
BASE    = Path(__file__).resolve().parent
DATA    = BASE / "load_and_costs"
DC_PATH = BASE / "tables" / "dc_cumulative_by_iso.xlsx"
SK_DIR  = BASE / "results" / "sankey"
FIG_DIR = BASE / "figures" / "03"

# ======================== zone config ========================
ZONES = {
    "CAISO": ["PGE", "SCE", "SDGE"],
    "ERCOT": ["Houston", "North", "South", "West"],
    "MISO":  ["LRZ 1", "LRZ 2,7", "LRZ 3,5", "LRZ 6", "LRZ 8,9,10"],
    "PJM":   ["AECO","AEP","APS","ATSI","BGE","COMED","DAY","DEOK",
              "DOM","DPL","DUQ","EKPC","JCPL","METED","PECO",
              "PENELEC","PEPCO","PPL","PSEG","RECO"],
}

# canonical zone → project-file column
PROJ_COL = {
    "MISO": {"LRZ 1":"MISO_LRZ1","LRZ 2,7":"MISO_LRZ2_7",
             "LRZ 3,5":"MISO_LRZ3_5","LRZ 6":"MISO_LRZ6",
             "LRZ 8,9,10":"MISO_LRZ8_9_10"},
    "PJM":  {"EKPC":"EKPC PJM"},
}

# canonical zone → load-sheet name
LOAD_SH = {
    "MISO": {"LRZ 1":"MISO_LRZ1","LRZ 2,7":"MISO_LRZ2_7",
             "LRZ 3,5":"MISO_LRZ3_5","LRZ 6":"MISO_LRZ6",
             "LRZ 8,9,10":"MISO_LRZ8_9_10"},
    "PJM":  {"EKPC":"EKPC PJM"},
}

# canonical zone → dc_cumulative column(s)
DC_COL = {
    "MISO": {"LRZ 1":"LRZ 1","LRZ 3,5":"LRZ 3, 5",
             "LRZ 2,7":"LRZ 2, 7","LRZ 6":"LRZ 6",
             "LRZ 8,9,10":"LRZ 8, 9, 10"},
    "PJM":  {"EKPC":"EKPC PJM"},
}

DC_MERGE = {}

DISPLAY = {
    "CAISO": {"PGE":"PGE (CA)","SCE":"SCE (CA)","SDGE":"SDGE (CA)"},
    "ERCOT": {"Houston":"Houston (TX)","North":"North (TX)",
              "South":"South (TX)","West":"West (TX)"},
    "MISO":  {"LRZ 1":"LRZ 1 (MN)","LRZ 3,5":"LRZ 3,5 (IA)",
              "LRZ 2,7":"LRZ 2,7 (MI,WI)","LRZ 6":"LRZ 6 (IN)",
              "LRZ 8,9,10":"LRZ 8,9,10 (LA,MS)"},
    "PJM":   {"AECO":"AECO (NJ)","AEP":"AEP (VA,OH)","APS":"APS (PA)",
              "ATSI":"ATSI (OH)","BGE":"BGE (MD)","COMED":"COMED (IL)",
              "DAY":"DAY (OH)","DEOK":"DEOK (OH)","DOM":"DOM (VA)",
              "DPL":"DPL (DE,MD)","DUQ":"DUQ (PA)","EKPC":"EKPC (KY)",
              "JCPL":"JCPL (NJ)","METED":"METED (PA)","PECO":"PECO (PA)",
              "PENELEC":"PENELEC (PA)","PEPCO":"PEPCO (MD)","PPL":"PPL (PA)",
              "PSEG":"PSEG (NJ)","RECO":"RECO (NJ)"},
}

MAX_PROJ_YEAR = 2025

# ======================== helpers ========================

def _canon(s):
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()

def _find_col(df, name):
    if name in df.columns:
        return name
    cn = _canon(name)
    for c in df.columns:
        if _canon(c) == cn:
            return c
    return None

# ======================== data readers ========================

def read_dc_cap(iso):
    df = pd.read_excel(DC_PATH, sheet_name=iso)
    df.rename(columns={df.columns[0]: "Year"}, inplace=True)
    df["Year"] = df["Year"].astype(int)
    return df

def get_dc_gwh(dc_df, zone, iso, years):
    merge = DC_MERGE.get(iso, {})
    cmap  = DC_COL.get(iso, {})
    cols  = merge[zone] if zone in merge else [cmap.get(zone, zone)]
    out = np.zeros(len(years))
    for c in cols:
        ac = _find_col(dc_df, c)
        if ac is None:
            continue
        for i, y in enumerate(years):
            r = dc_df[dc_df["Year"] == y]
            if len(r):
                v = pd.to_numeric(r[ac].iloc[0], errors="coerce")
                if np.isfinite(v):
                    out[i] += v * 8760 * 0.8 / 1000.0
    return out

def read_load_classes(iso, zone, years):
    """Return {class_name: ndarray(GWh)} for every non-Total class."""
    path = DATA / iso / "02_load_by_zone_and_class.xlsx"
    sh   = LOAD_SH.get(iso, {}).get(zone, zone)
    xls  = pd.ExcelFile(path)
    if sh not in xls.sheet_names:
        return {}
    df = pd.read_excel(xls, sheet_name=sh)
    df["Year"] = df["Year"].astype(int)
    gwh = [c for c in df.columns if "(GWh)" in str(c) and "Total" not in str(c)]
    out = {}
    for col in gwh:
        cls = str(col).split("(")[0].strip()
        arr = np.full(len(years), np.nan)
        for i, y in enumerate(years):
            r = df[df["Year"] == y]
            if len(r):
                arr[i] = pd.to_numeric(r[col].iloc[0], errors="coerce")
        out[cls] = arr
    return out

def read_plr(iso, zone):
    """Peak-to-load ratios → {class_name: float}."""
    path = DATA / iso / "01_load_factor.xlsx"
    df   = pd.read_excel(path)
    idx  = df.columns[0]
    df[idx] = df[idx].astype(str).str.strip()
    cmap = DC_COL.get(iso, {})
    cn   = cmap.get(zone, zone)
    ac   = _find_col(df, cn)
    if ac is None:
        return {}
    return {str(r[idx]).strip(): float(r[ac])
            for _, r in df.iterrows()
            if np.isfinite(pd.to_numeric(r[ac], errors="coerce"))}

def read_share(iso):
    """Return {zone: {years, ru_ratio, ind_ratio}}."""
    path = DATA / iso / f"00_{iso}_share.xlsx"
    xls  = pd.ExcelFile(path)
    out  = {}
    for sh in xls.sheet_names:
        df  = pd.read_excel(xls, sheet_name=sh)
        ic  = df.columns[0]
        df[ic] = df[ic].astype(str).str.strip()
        ru = df[df[ic] == "RU"]
        dc = df[df[ic] == "DC"]
        if len(ru) == 0 or len(dc) == 0:
            continue
        yc = [c for c in df.columns[1:] if isinstance(c, (int, float, np.integer))]
        yrs = np.array([int(c) for c in yc])
        out[sh] = {
            "years":     yrs,
            "ru_ratio":  pd.to_numeric(ru.iloc[-1][yc], errors="coerce").to_numpy(float),
            "ind_ratio": pd.to_numeric(dc.iloc[-1][yc], errors="coerce").to_numpy(float),
        }
    return out

def read_proj_amounts(iso, sheet):
    """Per-zone per-year cost amounts ($M) from 03_projects.xlsx.
    Auto-detects whether zone columns are percentages (sum≈100) or
    fractions (sum≈1) and normalises to fractions."""
    path = DATA / iso / "03_projects.xlsx"
    df   = pd.read_excel(path, sheet_name=sheet)
    cc   = next((c for c in df.columns if "Cost" in str(c) and "$M" in str(c)), None)
    if cc is None:
        raise ValueError(f"No cost col in {iso}/{sheet}")
    df[cc]    = pd.to_numeric(df[cc], errors="coerce").fillna(0)
    df["Year"] = df["Year"].astype(int)
    df = df[df["Year"] <= MAX_PROJ_YEAR]
    pm = PROJ_COL.get(iso, {})

    # detect % vs fraction: sum all zone columns per row
    zcols = [pm.get(z, z) for z in ZONES[iso] if pm.get(z, z) in df.columns]
    if zcols:
        row_sums = df[zcols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        scale = 0.01 if row_sums.mean() > 10 else 1.0
    else:
        scale = 1.0

    recs = []
    for yr in sorted(df["Year"].unique()):
        yd = df[df["Year"] == yr]
        row = {"Year": yr}
        for z in ZONES[iso]:
            pc = pm.get(z, z)
            if pc in yd.columns:
                ratios = pd.to_numeric(yd[pc], errors="coerce").fillna(0).values * scale
                row[z] = float(np.nansum(ratios * yd[cc].values))
            else:
                row[z] = 0.0
        recs.append(row)
    return pd.DataFrame(recs)

# ======================== attribution ========================

def compute_attribution(iso):
    zones   = ZONES[iso]
    dc_df   = read_dc_cap(iso)
    z_amts  = read_proj_amounts(iso, "Table1_Responsibility")
    pyears  = sorted(z_amts["Year"].unique())
    minY, maxY = min(pyears), max(pyears)
    ext = np.arange(minY, maxY + 6)          # +5 window +1 for delta

    rows = []
    for zone in zones:
        loads = read_load_classes(iso, zone, ext)
        plr   = read_plr(iso, zone)
        dcg   = get_dc_gwh(dc_df, zone, iso, ext)

        if not loads or not plr:
            for y in pyears:
                a = float(z_amts.loc[z_amts["Year"]==y, zone].sum())
                rows.append(dict(Year=y, Zone=zone,
                    RU_attr=0, DC_attr=0, Other_attr=a,
                    RU_ratio=0, DC_ratio=0, Zone_amount=a))
            continue

        n = len(ext)

        # per-class responsibility
        cls_resp = {}
        for cls, vals in loads.items():
            p = plr.get(cls, 1.0)
            r = np.zeros(n)
            for i in range(n - 1):
                if np.isfinite(vals[i]) and np.isfinite(vals[i+1]):
                    r[i] = max((vals[i+1] - vals[i]) / 8760 * p, 0)
            cls_resp[cls] = r

        # DC responsibility (uses Industrial PLR)
        ip = plr.get("Industrial", 1.0)
        dc_resp = np.zeros(n)
        for i in range(n - 1):
            dc_resp[i] = max((dcg[i+1] - dcg[i]) / 8760 * ip, 0)

        res_resp = cls_resp.get("Residential", np.zeros(n))

        # total = all original classes (DC NOT added)
        tot_resp = np.zeros(n)
        for r in cls_resp.values():
            tot_resp += r

        for y in pyears:
            ix  = y - minY
            end = min(ix + 5, n)
            ru_s = float(np.nansum(res_resp[ix:end]))
            dc_s = float(np.nansum(dc_resp[ix:end]))
            to_s = float(np.nansum(tot_resp[ix:end]))

            if to_s > 0:
                rur = ru_s / to_s
                dcr = dc_s / to_s
                if rur + dcr > 1:
                    dcr = 1 - rur
            else:
                rur = dcr = 0.0
            otr = max(1 - rur - dcr, 0)

            a = float(z_amts.loc[z_amts["Year"]==y, zone].sum())
            rows.append(dict(Year=y, Zone=zone,
                RU_attr=a*rur, DC_attr=a*dcr, Other_attr=a*otr,
                RU_ratio=rur, DC_ratio=dcr, Zone_amount=a))

    return pd.DataFrame(rows)

# ======================== allocation ========================

def compute_allocation(iso):
    zones  = ZONES[iso]
    dc_df  = read_dc_cap(iso)
    z_amts = read_proj_amounts(iso, "Table2_ShareRatio")
    pyears = sorted(z_amts["Year"].unique())
    sdata  = read_share(iso)

    rows = []
    for zone in zones:
        sd = sdata.get(zone)
        if sd is None:
            for y in pyears:
                a = float(z_amts.loc[z_amts["Year"]==y, zone].sum())
                rows.append(dict(Year=y, Zone=zone,
                    RU_alloc=0, DC_alloc=0, Other_alloc=a,
                    RU_ratio=0, DC_ratio=0, Zone_amount=a))
            continue

        sy = sd["years"]
        ru_r = sd["ru_ratio"]
        in_r = sd["ind_ratio"]

        dcg = get_dc_gwh(dc_df, zone, iso, sy)

        # industrial / commercial GWh
        lsh = LOAD_SH.get(iso, {}).get(zone, zone)
        lpath = DATA / iso / "02_load_by_zone_and_class.xlsx"
        xls = pd.ExcelFile(lpath)
        ind_g = np.full(len(sy), np.nan)
        com_g = np.full(len(sy), np.nan)
        if lsh in xls.sheet_names:
            ldf = pd.read_excel(xls, sheet_name=lsh)
            ldf["Year"] = ldf["Year"].astype(int)
            ic = next((c for c in ldf.columns if "Industrial" in str(c) and "GWh" in str(c)), None)
            cc = next((c for c in ldf.columns if "Commercial" in str(c) and "GWh" in str(c)), None)
            for i, y in enumerate(sy):
                r = ldf[ldf["Year"]==y]
                if len(r):
                    if ic: ind_g[i] = pd.to_numeric(r[ic].iloc[0], errors="coerce")
                    if cc: com_g[i] = pd.to_numeric(r[cc].iloc[0], errors="coerce")

        p = read_plr(iso, zone)
        ip = p.get("Industrial", 1.0)
        cp = p.get("Commercial", 1.0)

        # DC allocation ratio (adjusted from industrial ratio)
        dc_ar = np.zeros(len(sy))
        for i in range(len(sy)):
            if not np.isfinite(ind_g[i]) or ind_g[i] <= 0:
                continue
            if not np.isfinite(in_r[i]):
                continue
            if dcg[i] <= ind_g[i]:
                dc_ar[i] = (dcg[i] / ind_g[i]) * in_r[i]
            else:
                cv = com_g[i] if np.isfinite(com_g[i]) else 0
                dn = ind_g[i] * ip + cv * cp
                if dn > 0:
                    dc_ar[i] = (1 - ru_r[i]) * dcg[i] * ip / dn
                else:
                    dc_ar[i] = in_r[i]

        for y in pyears:
            mask = sy == y
            yi = int(np.where(mask)[0][0]) if mask.any() else int(np.argmin(np.abs(sy - y)))
            rr = float(ru_r[yi]) if np.isfinite(ru_r[yi]) else 0
            dr = float(dc_ar[yi]) if np.isfinite(dc_ar[yi]) else 0
            otr = max(1 - rr - dr, 0)
            a = float(z_amts.loc[z_amts["Year"]==y, zone].sum())
            rows.append(dict(Year=y, Zone=zone,
                RU_alloc=a*rr, DC_alloc=a*dr, Other_alloc=a*otr,
                RU_ratio=rr, DC_ratio=dr, Zone_amount=a))

    return pd.DataFrame(rows)

# ======================== sankey ========================

C_RU  = "#2ca02c"
C_DC  = "#4a90d9"
C_OT  = "#999999"
C_ZN  = "#f29091"
ALPHA = 0.35

def _rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def _blend(h, n, lo=0.05, hi=0.6):
    h = h.lstrip("#")
    r0,g0,b0 = int(h[:2],16), int(h[2:4],16), int(h[4:6],16)
    out = []
    for i in range(n):
        t = lo + (hi-lo)*i/max(n-1,1)
        out.append("#%02x%02x%02x" % (
            int(r0*(1-t)+255*t), int(g0*(1-t)+255*t), int(b0*(1-t)+255*t)))
    return out

def _ypos(n, pad=0.02):
    """Evenly space n nodes in [pad, 1-pad]."""
    if n == 1:
        return [0.5]
    return [pad + i * (1 - 2*pad) / (n - 1) for i in range(n)]

def compute_cross_zone_flows(iso):
    """Per-project redistribution: zone(responsibility) → zone(allocation).
    Returns {year: {resp_zone: {alloc_zone: $M}}}.
    Shares are normalised so tracked-zone totals balance."""
    path = DATA / iso / "03_projects.xlsx"
    t1 = pd.read_excel(path, sheet_name="Table1_Responsibility")
    t2 = pd.read_excel(path, sheet_name="Table2_ShareRatio")
    cc = next(c for c in t1.columns if "Cost" in str(c) and "$M" in str(c))
    for df in (t1, t2):
        df[cc] = pd.to_numeric(df[cc], errors="coerce").fillna(0)
        df["Year"] = df["Year"].astype(int)
    t1 = t1[t1["Year"] <= MAX_PROJ_YEAR]
    t2 = t2[t2["Year"] <= MAX_PROJ_YEAR]

    zones = ZONES[iso]
    nZ = len(zones)
    pm = PROJ_COL.get(iso, {})

    def _sc(df):
        zc = [pm.get(z, z) for z in zones if pm.get(z, z) in df.columns]
        if zc:
            rs = df[zc].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            return 0.01 if rs.mean() > 10 else 1.0
        return 1.0

    s1, s2 = _sc(t1), _sc(t2)
    years = sorted(set(t1["Year"]).intersection(t2["Year"]))
    result = {}

    for yr in years:
        y1 = t1[t1["Year"] == yr].reset_index(drop=True)
        y2 = t2[t2["Year"] == yr].reset_index(drop=True)
        n = min(len(y1), len(y2))
        costs = y1[cc].iloc[:n].values.astype(float)

        resp = np.zeros((n, nZ))
        asc  = np.zeros((n, nZ))
        for zi, z in enumerate(zones):
            c = pm.get(z, z)
            if c in y1.columns:
                resp[:, zi] = pd.to_numeric(
                    y1[c].iloc[:n], errors="coerce").fillna(0).values * s1
            if c in y2.columns:
                asc[:, zi] = pd.to_numeric(
                    y2[c].iloc[:n], errors="coerce").fillna(0).values * s2

        for mat in (resp, asc):
            rs = mat.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1
            mat /= rs

        fm = (resp * costs[:, None]).T @ asc
        result[yr] = {zones[i]: {zones[j]: float(fm[i, j])
                                  for j in range(nZ)}
                      for i in range(nZ)}
    return result


def _build_year_payload(iso, attr, alloc, cross, year):
    """Build link arrays + labelled node names for one year."""
    zones = ZONES[iso]
    disp  = DISPLAY.get(iso, {})
    nZ    = len(zones)
    zc    = _blend(C_ZN, nZ)

    iML = lambda k: 3 + k
    iMR = lambda k: 3 + nZ + k
    iR  = lambda k: 3 + 2 * nZ + k

    a  = attr[attr["Year"] == year]
    b  = alloc[alloc["Year"] == year]
    cf = cross.get(year, {z: {z2: 0 for z2 in zones} for z in zones})

    src, tgt, val, lc = [], [], [], []
    lt = [0.0, 0.0, 0.0]
    rt = [0.0, 0.0, 0.0]

    for i, z in enumerate(zones):
        za = a[a["Zone"] == z]
        for v, k, c in [(float(za["RU_attr"].sum()), 0, C_RU),
                         (float(za["DC_attr"].sum()), 1, C_DC),
                         (float(za["Other_attr"].sum()), 2, C_OT)]:
            lt[k] += v
            if v > 1e-3:
                src.append(k); tgt.append(iML(i)); val.append(v)
                lc.append(_rgba(c, ALPHA))

    for i, z1 in enumerate(zones):
        for j, z2 in enumerate(zones):
            v = cf.get(z1, {}).get(z2, 0.0)
            if v > 1e-3:
                src.append(iML(i)); tgt.append(iMR(j)); val.append(v)
                lc.append(_rgba(zc[i], ALPHA))

    for j, z in enumerate(zones):
        zone_in = sum(cf.get(z1, {}).get(z, 0.0) for z1 in zones)
        zb = b[b["Zone"] == z]
        rr = float(zb["RU_ratio"].iloc[0]) if len(zb) else 0
        dr = float(zb["DC_ratio"].iloc[0]) if len(zb) else 0
        otr = max(1 - rr - dr, 0)
        for v, k, c in [(zone_in * rr, 0, C_RU),
                         (zone_in * dr, 1, C_DC),
                         (zone_in * otr, 2, C_OT)]:
            rt[k] += v
            if v > 1e-3:
                src.append(iMR(j)); tgt.append(iR(k)); val.append(v)
                lc.append(_rgba(c, ALPHA))

    left  = ["RU Attribution", "DC Attribution", "Other Attribution"]
    mid_l = [disp.get(z, z) for z in zones]
    mid_r = [disp.get(z, z) for z in zones]
    right = ["RU Allocation", "DC Allocation", "Other Allocation"]
    labels = left + mid_l + mid_r + right
    for k in range(3):
        labels[k] += f"<br>(${lt[k]:.1f}M)"
    for k in range(3):
        labels[iR(k)] += f"<br>(${rt[k]:.1f}M)"

    return labels, dict(source=src, target=tgt, value=val, color=lc)


def make_sankey_interactive(iso, attr, alloc, cross):
    """All-years HTML with year-dropdown (like 507 notebooks)."""
    zones = ZONES[iso]
    nZ    = len(zones)
    years = sorted(set(attr["Year"]).intersection(alloc["Year"]))
    zc    = _blend(C_ZN, nZ)
    colors = [C_RU, C_DC, C_OT] + zc + zc + [C_RU, C_DC, C_OT]

    y0 = years[0]
    labels0, links0 = _build_year_payload(iso, attr, alloc, cross, y0)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels0, color=colors, pad=20, thickness=18,
                  line=dict(color="rgba(0,0,0,0)", width=0)),
        link=dict(**links0,
                  hovertemplate="%{source.label} → %{target.label}"
                                "<br>$%{value:.2f}M<extra></extra>"),
    ))

    if len(years) > 1:
        buttons = []
        for y in years:
            lab_y, lnk_y = _build_year_payload(iso, attr, alloc, cross, y)
            buttons.append(dict(
                label=str(y), method="restyle",
                args=[{"node.label": [lab_y],
                       "link.source": [lnk_y["source"]],
                       "link.target": [lnk_y["target"]],
                       "link.value":  [lnk_y["value"]],
                       "link.color":  [lnk_y["color"]]}, [0]]))
        fig.update_layout(updatemenus=[dict(
            type="dropdown", buttons=buttons,
            x=1.02, y=1.0, xanchor="left", yanchor="top", showactive=True)])

    fig.update_layout(
        font=dict(size=12, family="Arial"),
        margin=dict(l=10, r=140, t=10, b=10),
        width=933, height=max(350, 25 * nZ + 100) // (1 if nZ > 10 else 2))
    return fig


def make_sankey_static(iso, attr, alloc, cross, year):
    """Single-year figure sized for half-A4 paper (like 508 notebooks)."""
    zones = ZONES[iso]
    nZ    = len(zones)
    zc    = _blend(C_ZN, nZ)
    colors = [C_RU, C_DC, C_OT] + zc + zc + [C_RU, C_DC, C_OT]

    labels, links = _build_year_payload(iso, attr, alloc, cross, year)

    HALF_A4_PX = int(4.135 * 96 * 2 / 3)
    H_PX = max(160, 12 * nZ + 50) // (1 if nZ > 10 else 2)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=colors, pad=12, thickness=12,
                  line=dict(color="rgba(0,0,0,0)", width=0)),
        link=dict(**links,
                  hovertemplate="%{source.label} → %{target.label}"
                                "<br>$%{value:.2f}M<extra></extra>"),
    ))
    fig.update_layout(
        font=dict(size=7, family="Arial"),
        width=HALF_A4_PX, height=H_PX,
        margin=dict(l=5, r=5, t=5, b=5),
        hovermode=False)
    return fig

# ======================== main ========================

if __name__ == "__main__":
    import plotly.io as pio

    SK_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for iso in ["CAISO", "ERCOT", "MISO", "PJM"]:
        print(f"\n{'='*60}\n  {iso}\n{'='*60}")

        attr  = compute_attribution(iso)
        alloc = compute_allocation(iso)
        cross = compute_cross_zone_flows(iso)
        print(f"  Attribution: {len(attr)} rows  |  Allocation: {len(alloc)} rows")

        attr.to_csv(SK_DIR / f"{iso}_attribution.csv", index=False)
        alloc.to_csv(SK_DIR / f"{iso}_allocation.csv", index=False)
        with pd.ExcelWriter(SK_DIR / f"{iso}_results.xlsx") as w:
            attr.to_excel(w, sheet_name="Attribution", index=False)
            alloc.to_excel(w, sheet_name="Allocation", index=False)

        # one interactive HTML with year-dropdown
        fig = make_sankey_interactive(iso, attr, alloc, cross)
        fig.write_html(str(FIG_DIR / f"{iso}_sankey.html"),
                       include_plotlyjs="cdn")
        print(f"    interactive html ✓")

        # static SVG for 2020 & 2025 (half-A4, 7pt font)
        for y in [2020, 2025]:
            if y in set(attr["Year"]).intersection(alloc["Year"]):
                fig = make_sankey_static(iso, attr, alloc, cross, y)
                svg_path = str(FIG_DIR / f"{iso}_sankey_{y}.svg")
                pio.write_image(fig, svg_path, format="svg",
                                width=fig.layout.width,
                                height=fig.layout.height, scale=1)
                print(f"    {y} svg ✓")

    print("\nDone!")
