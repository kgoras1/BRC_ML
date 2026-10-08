#!/usr/bin/env python3
"""
Extract the split records written by the original training runs.

Attention_based_MIL.py and LR_MeanPooling_OvR.py write, at training time, one entry per
one-vs-rest classifier into training_summary.json, containing:
  - split_strategy : "patient-level", or "slide-level (fallback)" when
                     _slide_level_split() was used (every fallback path returns no
                     patient counts, which sets this field)
  - train/val/calib (MIL) or train/cal (LR) counts: negative and positive slides and
                     number of patients

This script collects these entries for every reported run into
original_split_records.json / .txt. verify_fallbacks.py compares its replay against them.

Usage:
  python extract_original_split_records.py \
      --mil_dir      path/to/MIL_training/seed_42 \
      --mil_os_dir   path/to/MIL_training_oversample/seed_42 \
      --lr_root      path/to/LR_MeanPooling_Case_Level_results_V4
"""
import argparse
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LR_RUNS = [f"imbalance_{imb}_{cal}"
           for imb in ("none", "oversample", "undersample", "smote") for cal in ("nocal", "cal")]


def counts(c):
    return None if c is None else {k: c[k] for k in ("neg", "pos", "n_patients")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mil_dir", required=True)
    ap.add_argument("--mil_os_dir", required=True)
    ap.add_argument("--lr_root", required=True)
    ap.add_argument("--output", default=os.path.join(HERE, "original_split_records.json"))
    args = ap.parse_args()

    # Run names as in verify_fallbacks.py
    runs = {"Attention_MIL_weighted_ce": args.mil_dir,
            "Attention_MIL_oversample": args.mil_os_dir,
            **{f"LR_mean_pooling/{r}": os.path.join(args.lr_root, r) for r in LR_RUNS}}

    records = []
    for run, run_dir in runs.items():
        path = os.path.join(run_dir, "training_summary.json")
        written = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        summary = json.load(open(path))
        for c in summary["per_class"]:
            records.append({
                "run": run, "classifier": f"{c['class_name']} vs rest",
                "summary_written": written,
                "split_strategy": c["split_strategy"],
                "train": counts(c["train_counts"]),
                "val": counts(c.get("val_counts")),           # MIL only
                "cal": counts(c.get("calib_counts", c.get("cal_counts"))),
            })

    json.dump(records, open(args.output, "w"), indent=2)

    def fmt(c):
        return "-" if c is None else f"{c['neg']}/{c['pos']} ({c['n_patients']})"

    lines = ["Split records written by the original training runs (training_summary.json)",
             "Counts: negative/positive slides (patients)", "",
             f"{'run':44s} {'classifier':14s} {'written':16s} {'split_strategy':14s} "
             f"{'train':18s} {'val':16s} {'cal':16s}"]
    for r in records:
        lines.append(f"{r['run']:44s} {r['classifier']:14s} {r['summary_written']:16s} "
                     f"{r['split_strategy']:14s} {fmt(r['train']):18s} {fmt(r['val']):16s} {fmt(r['cal']):16s}")
    n_patient = sum(r["split_strategy"] == "patient-level" for r in records)
    lines += ["", f"patient-level split: {n_patient} of {len(records)} classifiers"]
    txt = "\n".join(lines) + "\n"
    open(os.path.splitext(args.output)[0] + ".txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
