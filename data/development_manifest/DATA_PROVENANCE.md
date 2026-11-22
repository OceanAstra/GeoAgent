# Dataset provenance

**Authoritative public source:** the U.S. Pipeline and Hazardous Materials Safety
Administration (PHMSA) pipeline incident and hazardous-liquid accident files:
https://www.phmsa.dot.gov/data-and-statistics/pipeline/source-data

The repository does **not** redistribute the complete original PHMSA raw dumps.
The United States Department of Transportation generally supports public reuse
and redissemination of government-produced data; these original government data
are not claimed as exclusive intellectual property of the GeoAgent team.
Third-party and restricted datasets should not be assumed to have identical rights.
The project team has chosen to share its curated derivative research records to
support independent reproduction; these records are **not** an official PHMSA
release and are not automatically governed by the MIT software license.

The original data processing uses source-revision field harmonization, the
six-cause crosswalk (`label_crosswalk_s2.csv`), derived numeric/categorical
inputs, reconstructed accident-group isolation, and the explicit
>=1-observed-attribute eligibility rule. For stage counts and precise
missingness-token semantics see `DATA_CONSTRUCTION_SUMMARY.md` under
`data/development_manifest/`.

The shared development-only manifest contains **42,769** records with 13 model
inputs, six-class labels, accident-group IDs, and the three split roles. The
**1,065 later-period sealed records** and other post-2023 sealed row-level
labels are **not included**. Other retrospective input/evaluation rows are
separately located in `data/feedback_replay/`, with their own year scopes.

A full byte-for-byte rebuild from newly downloaded PHMSA source snapshots was
**not executed** in the release audit. Consult `docs/DATA_AND_BASELINE_RIGHTS.md`
for complete source, rights and redistribution-scope notes.
