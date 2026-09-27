# South Lhonak Lake satellite dataset (Sentinel-2)

Contains modified Copernicus Sentinel data (2015-2026), processed in Google Earth Engine
(`COPERNICUS/S2_SR_HARMONIZED`, Level-2A surface reflectance). Terrain filter: SRTM
(`USGS/SRTMGL1_003`). Built by `build_satellite_dataset.py` (v2.1).

## What is in the dataset
- `scenes.csv`: 69 acquisition dates, open-water season (20 Aug - 5 Nov) each year.
  Columns: date, phase (pre_breach / immediate_post_breach / post_breach, breach = night of
  3-4 Oct 2023), qc, cloud_over_basin, ice_snow_frac_lake, water_lower_km2, water_upper_km2,
  edge_upper_km2, edge_pct_of_upper, and file names of the review images / GeoTIFF chip.
- `images/`: true-colour PNG per date, plus an overlay (cyan = lower-bound water, yellow =
  additional upper-bound water, red outline = measurement basin).
- `chips/`: 6-band GeoTIFF per date (B2, B3, B4, B8, B11, SCL), 10 m, EPSG:32645.
- `lake_outline_pre_breach_2022.geojson`, `measurement_basin.geojson`.
- `yearly_summary.csv`, `area_history_lists.json`.

## Method
Water is not identified by NDWI/MNDWI alone: both flag snow and glacier ice as water.
A pixel must also be dark in NIR, on gentle terrain, at lake elevation:
- lower bound: NDWI > 0 and NIR (B8) < 1500 (certain open water)
- upper bound: lower-bound water OR (NDWI > 0.1 and NIR < 2600), gaps filled (adds turbid /
  ice-laden water); so lower <= upper always
- both: slope < 15 deg, elevation 5000-5500 m; clouds (SCL 8, 9, 10) masked.
Area = all water pixels inside one fixed basin polygon (the pre-breach lake outline from
October 2022 scenes plus a 150 m margin). Thresholds were set from spectral probes at
the lake; published areas were used only as consistency checks, not for fitting.

## Quality flags (`qc`), checked in this order
- `ice_or_snow_covered`: >= 10% of the lake surface (pre-breach outline
  eroded 30 m) is bright in NIR (frozen / snow-covered lake, or heavy floating ice).
- `cloud_affected`: >= 10% cloud over the basin (area under-counted by up to
  that fraction).
- `check_basin_edge`: water in the inner 30 m band of the basin exceeds
  2% of the upper-bound area (possible truncation if the lake grew beyond
  the pre-breach outline).
- `good`: none of the above.
Only `good` scenes enter `yearly_summary.csv` and `area_history_lists.json`.

## Counts
By qc: {"good": 44, "cloud_affected": 20, "ice_or_snow_covered": 5}
By phase: {"pre_breach": 40, "immediate_post_breach": 11, "post_breach": 18}

| year | good scenes | lower km2 (median) | upper km2 (median) |
|---|---|---|---|
| 2016 | 2 | 1.203 | 1.319 |
| 2017 | 1 | 1.28 | 1.35 |
| 2018 | 3 | 1.271 | 1.369 |
| 2019 | 1 | 1.321 | 1.404 |
| 2020 | 3 | 1.382 | 1.461 |
| 2022 | 8 | 1.409 | 1.574 |
| 2023 | 11 | 1.142 | 1.401 |
| 2024 | 8 | 1.325 | 1.39 |
| 2025 | 6 | 1.376 | 1.443 |
| 2026 | 1 | 1.408 | 1.467 |

## Consistency checks against published values (not used for fitting)
About 1.31 km2 (a 2016 study); 1.14 km2 (2010) and 1.66 km2 (2023, pre-breach) in a
published area series; about 1.63 km2 on 17 Sep 2023 and 1.67 km2 on 28 Sep 2023 (ISRO/NRSC).
Post-breach published figures differ (about 107 ha and 60.3 ha), so immediate post-breach
rows are reported as a bracket.

## Limitations
- Optical imagery cannot see the lake under cloud, or when it is snow / ice covered.
- Monsoon (Jun - mid Aug) and winter are not covered by this dataset.
- Post-breach water is turbid and ice-laden; treat lower/upper as a bracket, not a measurement.
- 2024-2026 areas are our own measurements, not independently confirmed.
- SRTM (2000) terrain is used for slope/elevation screening only.
