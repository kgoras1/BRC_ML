# Split verification

## Summary
| Check | Result |
|---|---|
| Patients shared between train (883) and test (221) | **0** |
| Patients shared between any two of train / val / cal / test, all 40 one-vs-rest classifiers | **0** |
| Split strategy recorded by the original training runs, all 40 classifiers | **patient-level** |
| Fallback calls (`_slide_level_split()` or caught `ValueError`), all 40 classifiers | **0** |
| Replay identical to the original training record (split strategy, negative/positive slides and patients per partition) | **40 of 40** |

Source: `original_split_records.txt` / `.json`, `verify_fallbacks_output.txt` / `.json` and
`split_report/split_report_seed_42.txt` (section "SPLIT INTEGRITY").

## Train/test split
`split_report/` contains the report written by `../data_split/Datasplit_train_test.py`
(patients and WSIs per partition, subtype × cohort stratum and patient). The split is made
with `sklearn.model_selection.train_test_split` on the list of unique patient IDs, stratified
on subtype × cohort, so all WSIs of a patient are in the same partition.

## Train/validation/calibration splits and fallback branches
Within the training patients, `patient_stratified_train_val_cal_split()` makes a patient-level
80/10/10 train/val/cal split in `../MIL_train_eval/Attention_based_MIL.py` and an 80/20
train/cal split in `../Slide_Level_LR_kNN_train_eval/LR_MeanPooling_OvR.py`. Both contain
fallback branches (`_slide_level_split()` and the `except ValueError` blocks), written as
safeguards for early pipeline tests on small data subsets, where a class can have too few
patients for a stratified patient-level split.

### Record written by the original training runs
At training time, both scripts write for every one-vs-rest classifier a `split_strategy` field
to `training_summary.json`: `patient-level`, or `slide-level (fallback)` whenever
`_slide_level_split()` was used. The same entry holds the negative/positive slide counts and
the number of patients of each partition. `extract_original_split_records.py` collects these
entries from the reported runs (MIL 2026-06-06, slide-level LR 2026-06-08) into
`original_split_records.txt` / `.json`: all 40 classifiers record `patient-level`, and every
calibration set is non-empty (89 patients for MIL, 177 for LR), so the `except ValueError`
branch that drops the calibration set was not taken either.

```bash
python extract_original_split_records.py \
    --mil_dir    path/to/MIL_training/seed_42 \
    --mil_os_dir path/to/MIL_training_oversample/seed_42 \
    --lr_root    path/to/LR_MeanPooling_Case_Level_results_V4
```

### Deterministic replay
`verify_fallbacks.py` re-executes the split function for every reported run and one-vs-rest
classifier, with the same arguments and seed. For each classifier it reports the number of
patients in train / val / cal / test, the number of patients shared between any two of these
partitions, the number of fallback calls, and whether the replayed split strategy and
negative/positive slide and patient counts of each partition are identical to the original
training record:

```bash
python verify_fallbacks.py --mean_features_pkl path/to/mean_features_train_test_seed_42.pkl
```

The replay is identical to the original record for all 40 classifiers. In the full dataset
every class has well over 100 patients in the training set, so the fallback branches are never
reached.

## Run timestamps
`run_timestamps.txt` lists when the patient-level split was created and when each reported
model was trained. All reported models were trained after, and on, this split.
