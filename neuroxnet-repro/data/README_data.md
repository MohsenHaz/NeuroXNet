# Data and label governance

This repository does not redistribute CQ500 DICOM files, derived study labels, ROI masks, or external-cohort records. Obtain CQ500 through its official access route and place the extracted study folders under `data/raw/cq500/`. Do not commit DICOM images or identifiable information.

The CSV files in `splits/`, `labels/`, `external/`, and `../predictions/` are schema templates only when they contain headers and no records. They are not the manuscript's actual study-level data. Populate them only from the frozen source records and annotation database, then validate and archive their checksums. Do not infer patient IDs, reader scores, Evans' indices, ROI masks, or external predictions from aggregate tables in the manuscript.

The derived severity labels are not native CQ500 labels. Release them only if permitted by the ethics approval, data-use agreement, and institutional policy. External predictions must be genuinely de-identified and approved for release. Keep the original linkage key outside this repository.

Expected input: one folder per CQ500 study, with DICOM series readable by `src/neuroxnet/data/dicom_io.py`. Expected labels: `study_id,reader1,reader2,adjudicated,final_label,evans_index,temporal_horn,periventricular`. `final_label` must be numeric 0=normal, 1=mild, 2=moderate, 3=severe; `reader1`, `reader2`, and `adjudicated` must preserve the source annotation encoding and be documented in the approved protocol.
