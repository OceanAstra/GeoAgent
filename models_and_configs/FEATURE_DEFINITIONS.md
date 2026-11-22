# Feature Definitions (13 Model Inputs + 5 Excluded Leakage Fields)

Class order: `CORROSION, EQUIPMENT_FAILURE, EXTERNAL_DAMAGE, MATERIAL_DEFECT, OPERATIONAL, OTHER`.

## 13 Model Inputs (`MAIN_13`, exact order fed to CatBoost)

| # | Field | Type | Unit | Meaning | Availability |
|---|---|---|---|---|---|
| 1 | `PIPE_DIAMETER` | numeric | inch | Nominal pipe diameter (NPS) | pre-incident |
| 2 | `PIPE_WALL_THICKNESS` | numeric | inch | Nominal wall thickness | pre-incident |
| 3 | `PIPE_SMYS` | numeric | psi | Specified minimum yield strength | pre-incident |
| 4 | `PIPE_SPECIFICATION` | categorical | — | Pipe specification / material standard | pre-incident |
| 5 | `PIPE_SEAM_TYPE` | categorical | — | Longitudinal seam type | pre-incident |
| 6 | `PIPE_COATING_TYPE` | categorical | — | External coating type | pre-incident |
| 7 | `PIPE_TYPE` | categorical | — | Pipe type (body / seam) | pre-incident |
| 8 | `ACCIDENT_PSIG` | numeric | psig | Pressure at the time of the incident | at diagnosis (post) |
| 9 | `MOP_PSIG` | numeric | psig | Maximum allowable operating pressure | pre-incident |
| 10 | `INTERNAL_INSPECTION_IND` | numeric (0/1) | — | Internal / in-line inspection indicator | see missingness note |
| 11 | `PIPE_AGE_AT_ACCIDENT` | numeric (derived) | year | Incident year minus installation year | pre-incident |
| 12 | `MANUFACTURE_AGE_AT_ACCIDENT` | numeric (derived) | year | Incident year minus manufacturing year | pre-incident |
| 13 | `PRESSURE_RATIO` | numeric (derived) | — | `ACCIDENT_PSIG / MOP_PSIG` | at diagnosis (post) |

`ACCIDENT_PSIG`, `MOP_PSIG`, and `PRESSURE_RATIO` are available at post-incident diagnosis time, matching the retrospective six-class cause diagnosis task.

## Four Binary Rule Indicators (Rule View)

Computed in `g2_common.build_views` with no fitted parameters:

| Dim | Definition | Threshold |
|---|---|---|
| `coat` | `PIPE_COATING_TYPE in {NONE, UNKNOWN, N/A}` | string match |
| `age` | `PIPE_AGE_AT_ACCIDENT <= 10` | 10 years |
| `mfg` | `MANUFACTURE_AGE_AT_ACCIDENT <= 5` | 5 years |
| `pr` | `PRESSURE_RATIO > 1.2` | 1.2 |

## Five Excluded Fields (Label-Adjacent Leakage Prevention)

| Field | Reason Excluded |
|---|---|
| `UNDER_CATHODIC_PROTECTION_IND` | Recorded exclusively inside the corrosion cause-reporting block |
| `CP_ANNUAL_SURVEY_IND` | Same |
| `CLOSE_INTERVAL_SURVEY_IND` | Same |
| `EXTERNALLY_COATED` | Same |
| `PRIOR_DAMAGE` | Same |

Diagnostic check: non-zero entries across all five fields fall 100% into `CORROSION`, so excluding them prevents trivial cause-block leakage.

## Missing-Value Contract & Missingness Rates

- **Numeric fields:** Imputed using the training (`fit`) split median only.
- **Categorical fields:** Missing entries are encoded with the fixed token `NOT_RECORDED` and handled natively by CatBoost.
- **Class balance:** Natural class frequencies are retained without synthetic oversampling or label-driven class weights.

Full missingness rates on `dev_confirm` ($n = 6,417$) are recorded in `DATA_DICTIONARY.csv`. Four fields exceed 88% missingness:
- `PIPE_TYPE`: **95.11%** (categorical, coded as `NOT_RECORDED`)
- `INTERNAL_INSPECTION_IND`: **93.28%** (numeric)
- `MANUFACTURE_AGE_AT_ACCIDENT`: **89.15%** (numeric)
- `PIPE_SEAM_TYPE`: **88.92%** (categorical, coded as `NOT_RECORDED`)

Accordingly, as noted in Section 8 of the manuscript, TreeSHAP contributions for highly unrecorded structural fields (particularly `PIPE_TYPE` and `PIPE_SEAM_TYPE`) reflect reporting completeness patterns alongside recorded physical attributes.
