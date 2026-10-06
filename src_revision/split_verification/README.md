# Split verification

Evidence that all reported results use a patient-level train/test split, and that
no fallback branch in the splitting code was used.

## Contents

| Path | What it is |
|---|---|
| `split_report/split_report_seed_42.{json,txt}` | Report written by `../data_split/Datasplit_train_test.py`: patients and WSIs per partition, per subtype × cohort stratum, and per patient. |
| `run_records/Attention_MIL_weighted_ce/`, `run_records/Attention_MIL_oversample/` | `training_summary.json` and per-class `debug_splits.json` written by `../MIL_train_eval/Attention_based_MIL.py`. |
| `run_records/LR_mean_pooling/` | `training_summary.json` written by `../Slide_Level_LR_kNN_train_eval/LR_MeanPooling_OvR.py`, one per imbalance/calibration setting. |
| `replay_splits.py` | Re-executes the published splitting functions with the seeds and arguments of the reported runs. |
| `replay_splits_output.json` | Output of `replay_splits.py`. |

Run records are copied unedited from the training runs (absolute paths refer to our HPC).

## How the checks work

**Train/test split.** `Datasplit_train_test.py` splits unique patients (not WSIs) with a
single stratified `train_test_split` on a subtype × cohort key, assigns all WSIs of a patient
to that patient's partition, and asserts that no patient appears in both partitions. Its only
fallback (label-only stratification) applies when a subtype × cohort stratum has fewer than
2 patients; the smallest stratum in the data has 14.

**Internal splits.** For every one-vs-rest classifier, the MIL and slide-level LR scripts record
`split_strategy` (`patient-level` or `slide-level (fallback)`) and the number of patients in each
partition. Patient counts are only populated on the patient-level path, and an empty calibration
set would show as 0 patients. All recorded runs report `patient-level` with populated calibration sets.

**Replay.** `replay_splits.py` wraps the fallback functions to count every call, re-runs all
internal splits (MIL 80/10/10, slide-level LR 80/20, tile-level LR 80/20), and checks that:
- no fallback function is called;
- partitions are patient-disjoint and disjoint from the test set;
- the replayed patient counts equal those in `run_records/`.

```bash
python replay_splits.py --mean_features_pkl path/to/mean_features_train_test_seed_42.pkl
```

The k-NN classifier performs no internal split.
