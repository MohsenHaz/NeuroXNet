# CLAIM 2024 checklist (Tejani et al., Radiology: AI 2024) - location of each item

| # | Item | Where |
|---|---|---|
| 1-2 | Title / abstract identify AI, study design | Manuscript title, abstract |
| 3-5 | Background, objectives, study design (retrospective development + external validation) | Sections 1, 3 |
| 6 | Data source, eligibility, pre-processing | Section 3.2, `data/README_data.md`, `scripts/01_preprocess.py` |
| 7-9 | Data elements, de-identification, missing data | Section 3.9 (CQ500 note, external cohort), `docs/DEVIATIONS.md` |
| 10-13 | Reference standard: definition, rater qualifications, annotation tools, inter-rater variability | Annotation protocol (Section 3.9), `data/labels/annotation_protocol.pdf`, kappa 0.85 computed from `reader1,reader2` columns |
| 14-15 | Partitioning, level of partition, sample sizes | `data/splits/*.csv`, Tables 1-2 |
| 16-19 | Model, software, initialisation, training details, selection, hyper-parameters | `configs/*.yaml`, Table 4, `src/neuroxnet/training/train.py` |
| 20-22 | Metrics, statistical measures of uncertainty, robustness | `src/neuroxnet/eval/*`, Wilson CIs, bootstrap AUC CIs, McNemar, subgroups (Table 11) |
| 23 | Methods for explainability | `src/neuroxnet/explain/*`, Table 13 |
| 24-27 | Data flow, performance, failure analysis, external validation | Figures 3-6, Tables 5-11, Section 4.12 |
| 28-30 | Limitations, implications, registration | Section 4.14; not registered |
| 31-33 | Availability of code, weights, data | README, Data Availability statement, Zenodo DOI |
