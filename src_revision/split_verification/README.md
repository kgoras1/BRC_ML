# Split verification

## Train/test split
`split_report/` contains the report written by `../data_split/Datasplit_train_test.py`
(patients and WSIs per partition, subtype × cohort stratum and patient).

## Fallback branches
`patient_stratified_train_val_cal_split()` in `../MIL_train_eval/Attention_based_MIL.py` and
`../Slide_Level_LR_kNN_train_eval/LR_MeanPooling_OvR.py` contains fallback branches
(`_slide_level_split()` and the `except ValueError` blocks). They were written as safeguards
for the early pipeline tests on small data subsets, where a class can have too few patients
for a stratified patient-level split.

`verify_fallbacks.py` re-executes this function for every reported run and one-vs-rest
classifier, with the same arguments and seed, and counts each fallback call and each caught
`ValueError`:

```bash
python verify_fallbacks.py --mean_features_pkl path/to/mean_features_train_test_seed_42.pkl
```

Output: `verify_fallbacks_output.txt` / `.json` — 0 fallbacks for all classifiers.

Example of a case where the fallback *would* apply: a subset containing only 2 HER2 patients
cannot be split into stratified train/validation/calibration sets at patient level, and the
script reports one `_slide_level_split()` call. In the full dataset every class has well over
100 patients in the training set.

## Run timestamps
`run_timestamps.txt` lists when the patient-level split was created and when each reported
model was trained. All reported models were trained after, and on, this split.
