#!/usr/bin/env python3
"""
Deterministic replay of all data splits used in the revised experiments.

Re-executes the splitting functions of the published scripts with the same
arguments and seed used for the reported runs, and verifies that:

  1. Train/test split: no patient is shared, and the label-only stratification
     fallback of Datasplit_train_test.py could not be triggered
     (every subtype x cohort stratum has >= 2 patients).
  2. Internal splits (attention-MIL 80/10/10, slide-level LR 80/20,
     tile-level LR 80/20): every OvR classifier takes the patient-level path,
     no fallback function is called, partitions are patient-disjoint, and the
     replayed patient counts equal those recorded in run_records/.

Usage:
  python replay_splits.py \
      --mean_features_pkl path/to/mean_features_train_test_seed_42.pkl \
      --split_report split_report/split_report_seed_42.json \
      --output replay_splits_output.json

Author: Konstantinos Papagoras
"""
import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
import sklearn.model_selection as skms

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for sub in ("MIL_train_eval", "Slide_Level_LR_kNN_train_eval", "LR_tile_train_eval"):
    sys.path.insert(0, os.path.join(SRC, sub))

import Attention_based_MIL as mil          # noqa: E402
import LR_MeanPooling_OvR as lrm           # noqa: E402

SEED = 42
FALLBACK_CALLS = Counter()


def count_calls(module, name):
    """Wrap a fallback function so every call is counted."""
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        FALLBACK_CALLS[f"{module.__name__}.{name}"] += 1
        return original(*args, **kwargs)
    setattr(module, name, wrapper)


def recorded_counts(pattern, key_val, key_cal):
    """Per-class recorded patient counts from run_records/*.json."""
    out = {}
    for f in glob.glob(os.path.join(HERE, "run_records", pattern)):
        for c in json.load(open(f))["per_class"]:
            out.setdefault(c["class_name"], set()).add(
                (c["split_strategy"], c["train_counts"]["n_patients"],
                 c.get(key_val, {}).get("n_patients"), c[key_cal]["n_patients"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean_features_pkl", required=True)
    ap.add_argument("--split_report", default=os.path.join(HERE, "split_report", "split_report_seed_42.json"))
    ap.add_argument("--output", default=os.path.join(HERE, "replay_splits_output.json"))
    args = ap.parse_args()

    count_calls(mil, "_slide_level_split")
    count_calls(lrm, "_slide_level_split")

    d = pickle.load(open(args.mean_features_pkl, "rb"))
    ids = list(d["train_ids"])
    classes = list(d["class_names"])
    y_mc = np.array([classes.index(l) for l in d["train_labels"]])
    pid = np.array([mil.get_patient_id(s) for s in ids])
    test_pids = {mil.get_patient_id(s) for s in d["test_ids"]}

    # ---- 1. train/test split --------------------------------------------------
    rep = json.load(open(args.split_report))
    rep_train = {r["patient_id"] for r in rep["train"]["patient_detail"]}
    rep_test = {r["patient_id"] for r in rep["test"]["patient_detail"]}
    assert rep_train == set(pid) and rep_test == test_pids, "PKL and split report disagree"
    strata = Counter()
    for part in ("train", "test"):
        for k, v in rep[part]["by_label_cohort"].items():
            strata[k] += v["n_patients"]
    result = {
        "train_test": {
            "train_patients": len(rep_train), "test_patients": len(rep_test),
            "shared_patients": len(rep_train & rep_test),
            "smallest_label_x_cohort_stratum": min(strata.items(), key=lambda kv: kv[1]),
            "label_only_fallback_possible": any(v < 2 for v in strata.values()),
        },
        "attention_MIL": {}, "LR_mean_pooling": {}, "LR_tile_level": {},
    }

    rec_mil = recorded_counts("Attention_MIL_*/training_summary.json", "val_counts", "calib_counts")
    rec_lr = recorded_counts("LR_mean_pooling/*/training_summary.json", "val_counts", "cal_counts")

    def pats(idx):
        return set(pid[np.asarray(idx, dtype=int)]) if len(idx) else set()

    rows = list(range(len(ids)))
    for k, c in enumerate(classes):
        y_bin = (y_mc == k).astype(int)

        # ---- 2a. attention-MIL: 80/10/10 (defaults of Attention_based_MIL.py)
        tr, va, ca, *_, n_tr, n_va, n_ca = mil.patient_stratified_train_val_cal_split(
            rows, y_bin, ids, val_size=0.10, cal_size=0.10, seed=SEED)
        P = [pats(tr), pats(va), pats(ca)]
        replayed = ("patient-level" if n_tr is not None else "slide-level (fallback)", n_tr, n_va, n_ca)
        result["attention_MIL"][c] = {
            "split_strategy": replayed[0], "n_patients_train_val_cal": [n_tr, n_va, n_ca],
            "patient_disjoint": not (P[0] & P[1] or P[0] & P[2] or P[1] & P[2]),
            "disjoint_from_test": not ((P[0] | P[1] | P[2]) & test_pids),
            "matches_recorded_runs": rec_mil.get(c) == {replayed},
        }

        # ---- 2b. slide-level LR: 80/20 train/calibration (--cal_size 0.20)
        tr, _, ca, *_, n_tr, _, n_ca = lrm.patient_stratified_train_val_cal_split(
            rows, y_bin, ids, cal_size=0.20, seed=SEED)
        P = [pats(tr), pats(ca)]
        replayed = ("patient-level" if n_tr is not None else "slide-level (fallback)", n_tr, None, n_ca)
        result["LR_mean_pooling"][c] = {
            "split_strategy": replayed[0], "n_patients_train_cal": [n_tr, n_ca],
            "patient_disjoint": not (P[0] & P[1]),
            "matches_recorded_runs": rec_lr.get(c) == {replayed},
        }

        # ---- 2c. tile-level LR: stratified 80/20 patient split must not raise
        patient_rows = {}
        for i, p in enumerate(pid):
            patient_rows.setdefault(p, []).append(i)
        patients = sorted(patient_rows)
        p_lab = np.array([Counter(int(y_mc[i] == k) for i in patient_rows[p]).most_common(1)[0][0]
                          for p in patients])
        try:
            skms.train_test_split(patients, test_size=0.20, random_state=SEED, stratify=p_lab)
            branch = "stratified patient-level (no fallback)"
        except ValueError:
            branch = "FALLBACK (unstratified patient-level)"
        result["LR_tile_level"][c] = {"split": branch, "positive_patients": int(p_lab.sum())}

    result["fallback_function_calls"] = dict(FALLBACK_CALLS) or "none"
    json.dump(result, open(args.output, "w"), indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
