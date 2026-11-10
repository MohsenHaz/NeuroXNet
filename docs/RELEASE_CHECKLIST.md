# Release checklist

- [ ] Replace schema-only split/fold files with the actual frozen ID assignments and verify patient-level separation.
- [ ] Verify 344/74/73 partition sizes and manuscript class counts; confirm test counts 44/12/10/7.
- [ ] Verify five fold sizes 99/98/98/99/97 and positive counts 39/38/39/39/38.
- [ ] Supply approved label file, reader-level annotation file, annotation protocol, and ROI subset/masks where release is permitted.
- [ ] Supply actual external-cohort de-identified predictions, after governance approval.
- [ ] Run preprocessing and retain normalisation logs; verify physical spacing/orientation requirements against manuscript.
- [ ] Train NeuroXNet, all seven baselines, all ablations, and five-fold CV; retain checkpoints, configs, logs, git hash, GPU details.
- [ ] Generate one-row-per-study prediction CSVs for every model/split/fold.
- [ ] Generate result tables and figures only from saved per-study predictions.
- [ ] Validate all reported rates against integer counts and sample sizes; validate confusion matrices, CIs, AUCs, macro-F1, Dice/IoU identities, and probability/prediction consistency.
- [ ] Validate memory and runtime on the stated A100 40 GB system.
- [ ] Pin the final environment, build Docker image, run tests, calculate SHA256SUMS, archive release, and mint DOI.
- [ ] Update manuscript Data Availability Statement with the actual DOI and explicit restricted items.
