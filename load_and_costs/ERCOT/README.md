# ERCOT Load and Cost Data

Data on load forecasts, load factors, and transmission projects for the Electric Reliability Council of Texas.

## Compiled Data (used by scripts)

| File | Description |
|------|-------------|
| `00_ERCOT_share.xlsx` | Data center load share by ERCOT weather/load zone (compiled) |
| `00_load_factor_raw.xlsx` | Raw load factor data from PUCT rate case filings |
| `01_load_factor.xlsx` | Processed load factors by zone and customer class |
| `02_load_by_zone_and_class.xlsx` | Annual load (MWh) by zone and customer class |
| `03_projects.xlsx` | Transmission project costs and timelines |
| `03_projects_classification.xlsx` | Project classification (load-growth vs reliability vs economic) |

## Source Data

### `LF/` — Load Factor Source Filings

PUCT rate case filings from Texas T&D utilities, containing billing determinants and cost-of-service schedules:

| Subdirectory | Contents |
|-------------|----------|
| `AEP/` | AEP Texas rate case filings (2019, 2024 dockets) |
| `CenterPoint/` | CenterPoint Energy Houston rate case filings (2019, 2024) |
| `ONCOR/` | Oncor Electric Delivery rate case filings (2022, 2025). Each case includes Schedules A-M (rate base, O&M, functionalization, class allocation, rate design, etc.) |

### `LTLF/` — Long-Term Load Forecast

ERCOT system-wide and zonal load forecasts:

| File | Description |
|------|-------------|
| `Load_2025.xlsb` | ERCOT 2025 Long-Term Load Forecast (raw) |
| `combined_ercot_load_tables.xlsx` | Consolidated load tables from ERCOT CDR |
| `load_forecast_by_load_zone.csv` | Forecast by ERCOT load zone |
| `load_forecast_by_weather_zone.csv` | Forecast by ERCOT weather zone |
| `ercot_utility_pop_proportions.csv` | Utility-to-zone population proportions for load disaggregation |

### `Projects/` — Transmission Project Data

| Subdirectory | Contents |
|-------------|----------|
| `4CP/` | Four Coincident Peak data (2020-2025): zone-level 4CP shares and utility mappings used for ERCOT transmission cost allocation |
| `ERCOT-RPG/` | Regional Planning Group project data: `RPG_Projects.xlsx` (compiled), annual reports (2020-2025), and meeting materials |
| `ERCOT-ROS/` | Regional Operations Studies: system planning ROS documents (2020-2025) with raw DOCX files and summaries |
| `Weather Zone/` | Weather zone to load zone mapping reference |
