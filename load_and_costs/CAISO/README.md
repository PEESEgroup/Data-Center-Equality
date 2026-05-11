# CAISO Load and Cost Data

Data on load forecasts, load factors, and transmission projects for the California Independent System Operator.

## Compiled Data (used by scripts)

| File | Description |
|------|-------------|
| `00_CAISO_share.xlsx` | Data center load share by CAISO pricing zone (compiled from utility filings) |
| `00_load_factor_raw.xlsx` | Raw load factor data from CAISO/utility reports |
| `01_load_factor.xlsx` | Processed load factors by zone and customer class |
| `02_load_by_zone_and_class.xlsx` | Annual load (MWh) by zone and customer class (residential, commercial, industrial) |
| `03_projects.xlsx` | Transmission project costs, timelines, and categorization |

## Source Data

### `LTLP/` — Long-Term Load Projections

California Energy Demand Updated (CEDU) 2024 Baseline Forecasts from the California Energy Commission, disaggregated by utility service territory:

- `CEDU 2024 Baseline Forecast - PGE.xlsx` — Pacific Gas & Electric
- `CEDU 2024 Baseline Forecast - SCE.xlsx` — Southern California Edison
- `CEDU 2024 Baseline Forecast - SDGE.xlsx` — San Diego Gas & Electric

Source: CEC Integrated Energy Policy Report (IEPR), January 2024.

### `Projects/` — Transmission Access Charge (TAC) Data

Annual Transmission Access Charge filings and project-level data used to compute transmission cost attribution.

| Subdirectory | Contents |
|-------------|----------|
| `TAC/` | Annual TAC rate PDFs from CAISO (2020-2025) |
| `raw_data/{year}/` | Per-year source files: TAC workbook (`{year}TAC.xlsx`), CAISO transmission plan (`CAISO{year}.pdf`), project appendix (`appendix{year}.pdf`) |
| `results/` | Processed outputs: `CAISO_load_growth_projects.xlsx`, `load_growth_projects.csv`, `tac_allocation_by_year.csv`, per-year analysis CSVs |
