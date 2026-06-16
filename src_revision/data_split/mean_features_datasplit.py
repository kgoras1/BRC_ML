"""
Mean Features Data Split
========================
Replicates the exact patient-level train/test split from the MIL PKL — but
for the mean-pooled feature matrix — by reading the already-generated split
report JSON (avoids loading the 76 GB MIL PKL).

Strategy
--------
1. Load the split report JSON  →  extract train / test patient-ID sets.
2. Load the clean mean features PKL.
3. Derive each slide's patient ID with the same get_patient_id() used in
   Datasplit_train_test.py.
4. Assign each slide to train or test based on which patient set it belongs to.
5. Verify:
     a) Every mean-feature slide maps to exactly one known patient.
     b) Patient-level integrity: zero patients in both splits.
     c) Slide counts per label/cohort match the split report.
6. Save to Mean_Features_Dict/  as
       mean_features_train_test_seed_42.pkl

Output PKL schema
-----------------
{
  "train_features" : np.ndarray (N_train, D),
  "train_labels"   : list[str],
  "train_ids"      : list[str],
  "test_features"  : np.ndarray (N_test,  D),
  "test_labels"    : list[str],
  "test_ids"       : list[str],
  "class_names"    : list[str],   # sorted unique labels
}

Author : Konstantinos Athanasios Papagoras
Date   : 2026
"""

import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
REPORT_JSON = (
    "/home/projects2/WSI_project/PhD_WSI/BRC-WSI-Revision/"
    "Datasplits_MIL/split_report_seed_42.json"
)
MEAN_FEAT_PKL = (
    "/home/projects2/WSI_project/PhD_WSI/BRC-WSI-Revision/"
    "Mean_Features_Dict/mean_features_clean.pkl"
)
OUTPUT_DIR = (
    "/home/projects2/WSI_project/PhD_WSI/BRC-WSI-Revision/Mean_Features_Dict"
)
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "mean_features_train_test_seed_42.pkl")


# ── patient ID extractor (identical to Datasplit_train_test.py) ────────────────
def get_patient_id(slide_id: str) -> str:
    if "TCGA" in slide_id:
        parts = slide_id.split("-")
        return "-".join(parts[:3]) if len(parts) >= 3 else slide_id
    if "CPTAC" in slide_id:
        parts = slide_id.split("_", 2)
        if len(parts) >= 3:
            return parts[2].split("-")[0]
        return slide_id
    if "Warwick" in slide_id:
        if "_score_" in slide_id:
            return slide_id.rsplit("_score_", 1)[0]
        return slide_id
    return slide_id


def get_cohort(slide_id: str) -> str:
    if "TCGA" in slide_id:
        return "TCGA"
    if "CPTAC" in slide_id:
        return "CPTAC"
    if "Warwick" in slide_id:
        return "Warwick"
    return "unknown"


# ── 1. Load split report ───────────────────────────────────────────────────────
print(f"Loading split report: {REPORT_JSON}")
with open(REPORT_JSON) as f:
    report = json.load(f)

train_patient_ids = {r["patient_id"] for r in report["train"]["patient_detail"]}
test_patient_ids  = {r["patient_id"] for r in report["test"]["patient_detail"]}

assert train_patient_ids.isdisjoint(test_patient_ids), "Report itself has patient leakage!"

print(f"  Train patients from report: {len(train_patient_ids)}")
print(f"  Test  patients from report: {len(test_patient_ids)}")
print(f"  Expected train slides: {report['train']['n_slides']}")
print(f"  Expected test  slides: {report['test']['n_slides']}")

# ── 2. Load clean mean features ────────────────────────────────────────────────
print(f"\nLoading mean features: {MEAN_FEAT_PKL}")
with open(MEAN_FEAT_PKL, "rb") as f:
    mf = pickle.load(f)

names  = [str(n) for n in mf["slide_names"]]
labels = list(mf["slide_labels"])
feats  = mf["slide_features"]  # (N, D)

print(f"  Slides: {len(names)}, shape: {feats.shape}, dtype: {feats.dtype}")
print(f"  Label dist: {Counter(labels)}")

# ── 3. Assign each slide to train / test via patient ID ────────────────────────
train_idx, test_idx, unassigned = [], [], []

for i, sid in enumerate(names):
    pid = get_patient_id(sid)
    if pid in train_patient_ids:
        train_idx.append(i)
    elif pid in test_patient_ids:
        test_idx.append(i)
    else:
        unassigned.append((sid, pid))

if unassigned:
    print(f"\nWARNING: {len(unassigned)} slides have no matching patient in the report:")
    for sid, pid in unassigned[:10]:
        print(f"  slide='{sid}'  patient_id='{pid}'")

print(f"\nAssigned -> train: {len(train_idx)}  test: {len(test_idx)}  "
      f"unassigned: {len(unassigned)}")

# ── 4. Build split arrays ──────────────────────────────────────────────────────
train_feats  = feats[train_idx]
train_labels = [labels[i] for i in train_idx]
train_ids    = [names[i]  for i in train_idx]

test_feats   = feats[test_idx]
test_labels  = [labels[i] for i in test_idx]
test_ids     = [names[i]  for i in test_idx]

class_names = sorted(set(train_labels + test_labels))

# ── 5. Verification ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

# 5a. Patient leakage check
train_pids = {get_patient_id(s) for s in train_ids}
test_pids  = {get_patient_id(s) for s in test_ids}
overlap    = train_pids & test_pids
print(f"\n[a] Patient leakage: {len(overlap)} shared patients — "
      + ("PASSED" if len(overlap) == 0 else f"FAILED! {sorted(overlap)[:5]}"))

# 5b. Slide count vs. report
exp_tr = report["train"]["n_slides"]
exp_te = report["test"]["n_slides"]
tr_ok  = len(train_idx) == exp_tr
te_ok  = len(test_idx)  == exp_te
print(f"\n[b] Slide counts:")
print(f"    Train: got {len(train_idx):4d}  expected {exp_tr:4d}  — "
      + ("MATCH" if tr_ok else "MISMATCH"))
print(f"    Test:  got {len(test_idx):4d}  expected {exp_te:4d}  — "
      + ("MATCH" if te_ok else "MISMATCH"))

# 5c. Label distribution vs. report
print("\n[c] Label distribution per split:")
for split_name, got_labels, rep_key in [
    ("Train", train_labels, "train"),
    ("Test",  test_labels,  "test"),
]:
    got  = Counter(got_labels)
    exp  = {cls: d["n_slides"] for cls, d in report[rep_key]["by_label"].items()}
    all_ok = True
    print(f"  {split_name}:")
    for cls in sorted(class_names):
        g, e = got.get(cls, 0), exp.get(cls, 0)
        match = "OK" if g == e else "MISMATCH"
        if g != e:
            all_ok = False
        print(f"    {cls:<8}  got={g:4d}  expected={e:4d}  {match}")
    print(f"  => {'ALL MATCH' if all_ok else 'SOME MISMATCHES'}")

# 5d. Patient count vs. report
print("\n[d] Patient counts per split:")
for split_name, pids, rep_key in [
    ("Train", train_pids, "train"),
    ("Test",  test_pids,  "test"),
]:
    exp = report[rep_key]["n_patients"]
    match = "MATCH" if len(pids) == exp else "MISMATCH"
    print(f"  {split_name}: got {len(pids)}  expected {exp}  — {match}")

# 5e. Cohort distribution vs. report
print("\n[e] Cohort distribution per split:")
for split_name, ids, rep_key in [
    ("Train", train_ids, "train"),
    ("Test",  test_ids,  "test"),
]:
    got = Counter(get_cohort(s) for s in ids)
    exp = {coh: d["n_slides"] for coh, d in report[rep_key]["by_cohort"].items()}
    all_ok = True
    print(f"  {split_name}:")
    for coh in sorted(set(list(got.keys()) + list(exp.keys()))):
        g, e = got.get(coh, 0), exp.get(coh, 0)
        match = "OK" if g == e else "MISMATCH"
        if g != e:
            all_ok = False
        print(f"    {coh:<10}  got={g:4d}  expected={e:4d}  {match}")
    print(f"  => {'ALL MATCH' if all_ok else 'SOME MISMATCHES'}")

# 5f. Feature sanity
print(f"\n[f] Feature sanity:")
print(f"    NaN in train features: {np.isnan(train_feats).sum()}")
print(f"    Inf in train features: {np.isinf(train_feats).sum()}")
print(f"    NaN in test  features: {np.isnan(test_feats).sum()}")
print(f"    Inf in test  features: {np.isinf(test_feats).sum()}")

print("\n" + "=" * 60)

# ── 6. Save ────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
payload = {
    "train_features": train_feats,
    "train_labels":   train_labels,
    "train_ids":      train_ids,
    "test_features":  test_feats,
    "test_labels":    test_labels,
    "test_ids":       test_ids,
    "class_names":    class_names,
}
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(payload, f, protocol=4)

size_mb = os.path.getsize(OUTPUT_PKL) / 1e6
print(f"\nSaved -> {OUTPUT_PKL}  ({size_mb:.1f} MB)")
print(f"  train_features shape: {train_feats.shape}")
print(f"  test_features  shape: {test_feats.shape}")
print(f"  class_names: {class_names}")
