# Required data artifacts — not included in this archive

The following inputs are required before the end-to-end experiments can run:

1. `labels/cq500_severity_labels.csv` — study-level reader 1, reader 2, consensus severity labels and Evans/qualitative annotation fields. This is derived research data and is not part of the public CQ500 release. Obtain it from the authors through the stated access process.
2. `splits/cq500_split_seed42.csv` — immutable patient-level 344/74/73 assignment.
3. `splits/cq500_cv5_folds.csv` — immutable five-fold assignment.
4. CQ500 DICOM data — download separately from the source named in the manuscript, subject to its terms.
5. External cohort — raw DICOM is institution-restricted. The manuscript says de-identified predictions are included in the intended deposit, but they are not present in the supplied ZIP.
6. Grad-CAM masks — the manuscript's quantitative ROI analysis requires the 100-study ROI subset and masks; none are present in the supplied ZIP.

Do not create synthetic substitutes for these files and do not infer study labels from model predictions.
