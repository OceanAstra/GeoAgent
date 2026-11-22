# Dataset construction and eligibility

PHMSA source records were mapped into six incident-cause categories, then retained only when at least one of the **13 model input fields** was genuinely observed. A missing numeric field does not count. For categorical features, `NOT_RECORDED`, `NOT_COLLECTED`, and an empty value are non-observations; answer tokens such as `NO DATA`, `UNKNOWN`, and `NONE` were distinct recorded responses and are counted as observed values in the recovered rule.

| Stage | Records retained | Number excluded at stage |
|---|---:|---:|
| 15 source files read | 49,676 | — |
| Six-class mapping | 46,504 | 3,172 |
| ≥1 observed model attribute | 43,834 | 2,670 |
| Development data (through 2023) | 42,769 | 1,065 later-period records kept separate |

The final development splits are 29,941 fit / 6,411 calibration / 6,417 development-confirmation. These were all checked in the archived development-only manifest; accident groups do not overlap between splits. The row-set match for the 2,670-row filter was recovered in the later server-side lineage audit. Re-extraction of all 15 raw PHMSA source files from scratch was not independently repeated when preparing this compact public deposit.

**Excluded-row identities and cause labels are deliberately not distributed**, particularly those associated with the post-2023 period. The non-identifying stage counts are exactly the table above; the fuller audit table (`GeOAD_Data_Construction_Audit.csv`) was produced on the authors' build machine and is **not** part of this release, so only these aggregate counts are available here.
