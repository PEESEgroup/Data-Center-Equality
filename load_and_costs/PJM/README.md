# PJM Load and Cost Data

Data on load forecasts, load factors, and transmission projects for PJM Interconnection.

## Compiled Data (used by scripts)

| File | Description |
|------|-------------|
| `00_PJM_share.xlsx` | Data center load share by PJM transmission zone (compiled) |
| `00_load_factor_raw.xlsx` | Raw load factor data from utility filings |
| `01_load_factor.xlsx` | Processed load factors by zone and customer class |
| `02_load_by_zone_and_class.xlsx` | Annual load (MWh) by zone and customer class |
| `03_projects.xlsx` | Transmission project costs and timelines |

## Source Data

### `LTLP/` — Long-Term Load Projections

PJM zone-level load forecast data and sector models:

| File | Description |
|------|-------------|
| `2026load.xlsx` | PJM 2026 load forecast by transmission zone |
| `sector-model-residential.xlsx` | Residential sector load model |
| `sector-model-commercial.xlsx` | Commercial sector load model |
| `sector-model-industrial.xlsx` | Industrial sector load model |
| `name_map.xlsx` | Zone name mapping reference |
| `owner_map.xlsx` | Transmission owner mapping reference |

Source: PJM Load Forecast Development Process (Manual 19).

### `Projects/` — RTEP Transmission Project Data

PJM Regional Transmission Expansion Plan (RTEP) project data:

| File | Description |
|------|-------------|
| `ProjectConstructionUpgrades.xlsx` | PJM project construction and upgrade list (raw) |
| `ProjectConstructionUpgrades_merged.xlsx` | Merged with zone-level allocation |
| `ProjectConstructionUpgrades_final.xlsx` | Final processed version |
| `reports/` | 62 TEAC board whitepapers and RTEP window review & recommendation reports (2020-2026). Include baseline upgrades, network upgrades, and load-driven project approvals. |

Source: PJM TEAC (Transmission Expansion Advisory Committee) and Manual 14B.
