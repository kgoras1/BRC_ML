# Split verification

## Summary
| Check | Result |
|---|---|
| Patients shared between train (883) and test (221) | **0** |
| Patients shared between any two of train / val / cal / test, all 40 one-vs-rest classifiers | **0** |
| Fallback calls (`_slide_level_split()` or caught `ValueError`), all 40 classifiers | **0** |

Source: `verify_fallbacks_output.txt` / `.json` and `split_report/split_report_seed_42.txt`
(section "SPLIT INTEGRITY").

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

`verify_fallbacks.py` re-executes this function for every reported run and one-vs-rest
classifier, with the same arguments and seed. For each classifier it reports the number of
patients in train / val / cal / test, the number of patients shared between any two of these
partitions, and the number of fallback calls:

```bash
python verify_fallbacks.py --mean_features_pkl path/to/mean_features_train_test_seed_42.pkl
```

The partition sizes it reproduces (MIL 705/89/89, LR 706/177 patients) match the training
logs. In the full dataset every class has well over 100 patients in the training set, so the
fallback branches are never reached.

## Run timestamps
`run_timestamps.txt` lists when the patient-level split was created and when each reported
model was trained. All reported models were trained after, and on, this split.
