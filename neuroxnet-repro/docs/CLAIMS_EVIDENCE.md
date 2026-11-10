# Claims-to-evidence map

| Manuscript claim/contribution | Evidence reported in manuscript | Required reproducibility evidence | Qualification |
|---|---|---|---|
| One network performs detection and four-tier grading | Shared EfficientNetB4 backbone and parallel heads; internal/external results | Model source, config, checkpoint, study-level probabilities, labels | Architecture combines established components; novelty is in task/configuration and evaluation |
| Components are evaluated jointly as the NeuroXNet configuration | Held-out and external results of the complete model | Model source, config, checkpoint, study-level probabilities | The manuscript makes no claim about the individual contribution of the components |
| Clinically meaningful severity stratification | Four Evans' index tiers with qualitative descriptors; reader κ=0.85 | Annotation protocol, independent reader labels, consensus labels, adjudication log | 0.35/0.40 are operational study-specific thresholds, not guideline cutoffs |
| Interpretability via Grad-CAM | Mean IoU 0.779 and Dice 0.875 | ROI masks, selected slice IDs, heatmaps, threshold code, per-study metrics | Attention overlap is not proof of causal explanation or clinical validity |
| External generalization | 182-study external cohort; 95.60% detection accuracy and 91.21% severity accuracy | External label counts, de-identified per-study predictions, protocol, scanner subgroup definitions | External single-reader labels; raw scans restricted; one external institution |
| Efficient inference | 21.6 ms per 128-slice study on A100 | Benchmark script, raw timing logs, hardware/software metadata, preprocessing policy | Must state whether timing includes DICOM decoding and exactly how slices are batched |
| Better than baselines | Higher descriptive point estimates on 73-study test set | Aligned per-study predictions for every model; paired tests if claiming inferential superiority | Manuscript says no paired test; do not state statistically significant superiority |
| Reproducibility/transparency | CLAIM and TRIPOD+AI referenced; method details | Complete data/label access pathway, checkpoints, split files, predictions, pinned environment | Artifacts are deposited in the archive described in the Data Availability Statement |

## Manuscript-reported primary numbers

- Internal binary: 71/73 correct; 97.26% accuracy; 96.55% sensitivity; 97.73% specificity; AUROC 0.9891.
- Internal severity: 69/73 correct; 94.52% accuracy; macro-F1 0.9201.
- External binary: 174/182 correct; 95.60% accuracy; sensitivity 96.51%; specificity 94.79%; AUROC 0.9734.
- External severity: 166/182 correct; 91.21% accuracy; macro-F1 0.8876.
- Grad-CAM: mean IoU 0.779, Dice 0.875.
- Latency: 21.6 ms per complete 128-slice study on A100 40 GB.

These values are manuscript transcriptions, not recomputed from the missing prediction files.
