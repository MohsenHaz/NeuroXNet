# Deviations and unresolved source evidence

1. The supplied ZIP did not contain source DICOM, reader-level labels, adjudicated labels, ROI masks, actual frozen split/fold ID files, per-study prediction CSVs, trained checkpoints, result tables/figures, or training logs. These have not been fabricated.
2. The CSVs created in this release are header-only schemas. They do not represent the 491 actual studies.
3. `annotation_protocol.pdf` is a placeholder and must be replaced with the approved protocol used by the readers.
4. The split/fold generator enforces manuscript-level target sizes/counts but cannot reconstruct the original historical ID assignment. A generated assignment is not evidence that the original experiment used those exact IDs.
5. The manuscript reports a single-reader external cohort; this remains a limitation and must be described in the paper.
6. Baseline and ablation identities must be checked against their actual implementations. In particular, adapted classification architectures must not be described as canonical published segmentation implementations unless the full original architecture and protocol were used.
7. Parameter counts and MSAM details must be regenerated from the frozen code and reconciled with the manuscript before release.
8. No statistical superiority claim is supported unless paired per-study predictions are available and the prespecified paired analyses are run.
9. Preprocessing implementation uses streaming one-study-at-a-time processing; confirm that the resize and interpolation operations exactly match the final manuscript protocol.
10. Gradient checkpointing and conservative slice/study chunking are memory-management changes. Validate numerical equivalence and runtime on the intended hardware before treating them as the exact original training implementation.
