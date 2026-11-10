# NeuroXNet model card

## Intended use
Research evaluation of hydrocephalus detection and four-level severity grading from non-contrast head CT. Not for autonomous clinical diagnosis or treatment decisions.

## Model summary
EfficientNet-B4 backbone with MSAM, CSAG, FPN, binary detection and four-class severity heads, as described in the manuscript. Study-level predictions aggregate slice logits before probability conversion.

## Data
The manuscript describes 491 CQ500 studies and an external cohort of 182 studies. The archive supplied for this repository did not include the underlying images, adjudicated labels, external labels, or per-study predictions. Do not interpret the manuscript aggregate metrics as validation of this code release.

## Limitations
External cohort labels were single-reader according to the manuscript. Dataset shift, retrospective sampling, derived severity labels, and lack of prospective clinical validation limit generalisability.

## Release status
Code and schemas are provided. Trained weights and prediction artifacts are absent until generated from authorized source data.
