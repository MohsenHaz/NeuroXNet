# Deviations between the submitted manuscript and this package

This file is the single place where anything that differs from the manuscript is recorded. It must be updated every
time the package and the paper diverge.

1. **Baseline, ablation and cross-validation predictions.** The per-study predictions and probability scores of the
   baseline runs, the ablation runs and the cross-validation folds reported in the submitted manuscript were not retained.
   Scripts 03, 04 and 05 rerun those experiments with logging; the tables of the resubmission are the outputs of
   `07_evaluate_all.py` on the new prediction files, and every number of the old tables that cannot be regenerated is replaced.
2. **Ablation split.** The submitted ablation used its own internal split. In this package the ablation is run on the
   identical 73-study held-out test split (seed 42), and `predictions/ablation/full_model.csv` is a copy of
   `predictions/internal_test/neuroxnet.csv`, so the Full Model row equals the NeuroXNet rows of Tables 5-7 by construction.
3. **Parameter counts.** Counts come from `scripts/_param_counts.py` (feature extractor, ImageNet head excluded, timm
   `features_only`), not from the values typed in the submitted manuscript (19.4 M / 17.5 M). The manuscript must be updated to the script output.
4. **MSAM convolutions.** Eq. (6) is implemented with k x k depthwise convolutions followed by a shared 1 x 1 projection
   (separable k x k) so that the attention and pyramid modules stay within about 2 M parameters; the manuscript equation and
   Section 3.4 must state this.
5. **Schematic figures.** Figures 7, 8 and 10 of the submitted manuscript are schematic renderings. `08_gradcam_eval.py`
   writes real Grad-CAM panels on actual studies to `results/figures/fig8_gradcam_real_cases.png`; those replace the schematics.
6. **Figure 6 (decision curve).** Removed from the resubmission; regenerated here by `10_make_tables_figures.py` from the
   stored scores with the caption formula, so it can be reinstated only from that output.
7. **External cohort.** Labelled by a single reader (no second reader, no adjudication). Raw images and the institution
   name are not released; only `data/external/external_predictions_deidentified.csv`.
8. **Latency.** CPU timings are not reported. GPU latency in Table 14 is the median of 100 timed runs after 20 warm-up runs,
   per complete 128-slice study including preprocessing, measured by `09_benchmark_latency.py` on one device for all models.
9. **Tools that cannot be rerun.** None once scripts 00-10 have been executed; if any step is skipped, list it here with the reason.
