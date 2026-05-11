# MISO Load and Cost Data

Data on load forecasts, load factors, and transmission projects for the Midcontinent Independent System Operator.

## Compiled Data (used by scripts)

| File | Description |
|------|-------------|
| `00_MISO_share.xlsx` | Data center load share by MISO Local Resource Zone (compiled) |
| `00_load_factor_raw.xlsx` | Raw load factor data from utility filings |
| `01_load_factor.xlsx` | Processed load factors by zone and customer class |
| `02_load_by_zone_and_class.xlsx` | Annual load (MWh) by zone and customer class |
| `03_projects.xlsx` | Transmission project costs and timelines |

## Source Data

### `LTLF/` — Long-Term Load Forecast

| File | Description |
|------|-------------|
| `LTLF2026.xlsx` | MISO 2026 Long-Term Load Forecast (raw) |
| `LTLF2026_processed.xlsx` | Processed forecast data |
| `combined_miso_load_tables.xlsx` | Consolidated load tables |
| `hourly_load_split.xlsx` | Hourly load profile splits by zone |
| `zone_category_loads_v2.xlsx` | Load by zone and customer category |
| `miso_utility_pop_proportions.csv` | Utility-to-LRZ population proportions for load disaggregation |

### `Projects/` — Transmission Project Data

| Subdirectory / File | Contents |
|-------------|----------|
| `MTEP{yy} Appendix A*.xlsx` | MISO Transmission Expansion Plan new project lists (MTEP20 through MTEP25) |
| `MTEP_Consolidated_Projects.xlsx` | All MTEP projects consolidated |
| `MTEP_LG_Projects_by_LRZ.xlsx` | Load-growth-driven projects disaggregated by Local Resource Zone |
| `zones.xlsx` | Zone mapping reference |
| `Sch6/` | MISO Schedule 6 zonal transmission rates and billing determinants (monthly filings from OASIS, 2021-2025) |
| `report/` | Supporting planning reports |
