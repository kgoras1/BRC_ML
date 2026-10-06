#!/usr/bin/env python3
"""
Count how often the fallback branches of the patient-level split were used.

Covers the two scripts in which they are defined:
  - MIL_train_eval/Attention_based_MIL.py               (80/10/10 train/val/cal)
  - Slide_Level_LR_kNN_train_eval/LR_MeanPooling_OvR.py (80/20 train/cal)

For every reported run and one-vs-rest classifier, the split function
patient_stratified_train_val_cal_split() is re-executed with the arguments and
seed of that run, and two things are counted:
  - slide_level_fallback : calls to _slide_level_split()
                           (no patient IDs, < 3 patients per class, failed
                            validation split, or single-class val/cal subset)
  - value_error_caught   : ValueErrors raised by train_test_split() and caught
                           inside the split function (the except-branches)

Usage:
  python verify_fallbacks.py --mean_features_pkl path/to/mean_features_train_test_seed_42.pkl
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "MIL_train_eval"))
sys.path.insert(0, os.path.join(SRC, "Slide_Level_LR_kNN_train_eval"))

import Attention_based_MIL as mil          # noqa: E402
import LR_MeanPooling_OvR as lr            # noqa: E402

SEED = 42
# Split arguments of the reported runs (see job scripts); the split does not depend
# on the imbalance strategy, loss or calibration flag, which act after splitting.
RUNS = {
    "Attention_MIL_weighted_ce": (mil, dict(val_size=0.10, cal_size=0.10)),
    "Attention_MIL_oversample":  (mil, dict(val_size=0.10, cal_size=0.10)),
    **{f"LR_mean_pooling/imbalance_{imb}_{cal}": (lr, dict(cal_size=0.20))
       for imb in ("none", "oversample", "undersample", "smote") for cal in ("nocal", "cal")},
}
COUNTS = {"slide_level_fallback": 0, "value_error_caught": 0}


def instrument(module):
    """Count fallback calls and ValueErrors raised by train_test_split inside `module`."""
    slide_level_split = module._slide_level_split
    train_test_split = module.train_test_split

    def counted_slide_level_split(*a, **k):
        COUNTS["slide_level_fallback"] += 1
        return slide_level_split(*a, **k)

    def counted_train_test_split(*a, **k):
        try:
            return train_test_split(*a, **k)
        except ValueError:
            COUNTS["value_error_caught"] += 1
            raise

    module._slide_level_split = counted_slide_level_split
    module.train_test_split = counted_train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean_features_pkl", required=True,
                    help="PKL with train_ids / train_labels / class_names of the training set")
    ap.add_argument("--output", default=os.path.join(HERE, "verify_fallbacks_output.json"))
    args = ap.parse_args()

    instrument(mil)
    instrument(lr)

    d = pickle.load(open(args.mean_features_pkl, "rb"))
    slide_ids = list(d["train_ids"])
    classes = list(d["class_names"])
    y = np.array([classes.index(l) for l in d["train_labels"]])

    results, total = [], 0
    for run, (module, split_args) in RUNS.items():
        for k, cls in enumerate(classes):
            COUNTS.update(slide_level_fallback=0, value_error_caught=0)
            out = module.patient_stratified_train_val_cal_split(
                list(range(len(slide_ids))), (y == k).astype(int), slide_ids, seed=SEED, **split_args)
            n_fallbacks = COUNTS["slide_level_fallback"] + COUNTS["value_error_caught"]
            total += n_fallbacks
            results.append({
                "run": run, "classifier": f"{cls} vs rest",
                "split": "patient-level" if out[6] is not None else "slide-level (fallback)",
                **COUNTS, "fallbacks_total": n_fallbacks,
            })

    print(f"{'run':42s} {'classifier':16s} {'split':14s} fallbacks")
    for r in results:
        print(f"{r['run']:42s} {r['classifier']:16s} {r['split']:14s} {r['fallbacks_total']}")
    print(f"\nTotal fallbacks across {len(results)} classifiers: {total}")

    json.dump({"total_fallbacks": total, "per_classifier": results}, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
