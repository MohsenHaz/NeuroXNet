# Implementation notes and deviations

These notes record where the released implementation differs from, or is more specific than, the manuscript text.

1. **Artifacts not stored in the repository.** DICOM data, reader-level and adjudicated labels, ROI masks, the frozen
   split and fold files, per-study predictions, trained weights and training logs are deposited in the archive described in the
   Data Availability Statement. The CSV files under `data/` and `predictions/` are header-only schemas.
2. **Split and fold generator.** `scripts/make_frozen_splits.py` reproduces the stated partition sizes (344/74/73;
   test classes 44/12/10/7; fold sizes 99/98/98/99/97; positives 39/38/39/39/38) from a populated label table. A generated
   assignment is not the historical assignment; the frozen files of the reported experiments are the authoritative ones.
3. **Preprocessing.** Windowing (WL 40 HU, WW 80 HU), selection of 128 slices centred on the mid-volume index with zero
   padding, and in-plane resizing to 256 x 256 are applied. There is no through-plane resampling (manuscript Section 3.2).
   The normalisation constants are computed incrementally over the selected and resized training slices and are logged;
   the constants used for the reported experiments are listed in Table 4.
4. **MSAM.** The multi-scale module uses k x k depthwise convolutions followed by a shared 1 x 1 point-wise convolution
   (manuscript Section 3.4), which keeps the attention and pyramid modules within the parameter budget of Table 13.
5. **Baselines.** The five image backbones are ImageNet networks with the NeuroXNet task heads. TransUNet and Attention
   U-Net are classification variants (ResNet50 encoder with a 12-layer, 768-dimensional ViT bottleneck; ResNet34 encoder
   with attention gates and one up-sampling fusion) without segmentation decoders, as stated in manuscript Section 3.13.
6. **Effective batch.** Table 4 specifies 8 studies (1,024 slices) per optimiser step. The code realises this with one study
   per forward pass, eight gradient-accumulation steps and activation checkpointing over chunks of eight slices.
   Batch-normalisation statistics are therefore computed per chunk and not over 1,024 slices; numerical equivalence with a
   single 8-study batch is not claimed.
7. **Checkpoint selection.** The checkpoint is selected on the sum of the validation AUROC (detection) and validation macro-F1
   (severity); early stopping uses the validation loss with patience 15 (Table 4).
8. **Ablation.** Configurations are provided in `configs/ablation/` and `scripts/04_train_ablation.sh`. The manuscript reports
   no ablation results and makes no claim about the individual contribution of the components.
9. **Statistical comparison.** The evaluation script computes paired McNemar tests and bootstrap AUC differences only when
   aligned per-study predictions exist for every compared model. The manuscript reports descriptive comparisons only.
10. **External cohort.** The external cohort was graded by a single reader; this is stated as a limitation in the manuscript.
