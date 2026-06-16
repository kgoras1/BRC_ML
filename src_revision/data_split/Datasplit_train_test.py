"""
Reproducible Patient-Level Stratified Train/Test Split for MIL WSI Data

Workflow:
  1. Load the existing split PKL (mil_data_seed_42_TCGA_Warwick_Clean.pkl)
  2. Merge train + test back into a single unified pool
  3. Clean: drop NaN labels, normal/benign slides, keep only the 4 BRCA subtypes
  4. Derive patient IDs and cohort per slide
  5. Patient-level stratified split (stratified on label x cohort composite key)
     — no patient can appear in both train and test
  6. Save new PKL in the same schema as the input file
  7. Write a thorough JSON + plain-text report

Run this on compute01 (login node cannot fit the 76 GB file in RAM):
    ssh compute01 "python3 /net/well/pool/projects2/WSI_project/PhD_WSI/ \\
        BRC-WSI-Revision/scripts_preprocessing/Datasplit_train_test.py"

Author : Konstantinos Athanasios Papagoras
Date   : 2026
"""

import json
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
TEST_SIZE = 0.2

INPUT_PKL = (
    "/net/well/pool/projects2/WSI_project/PhD_WSI/"
    "feature_extraction/ml_attention_pooling/"
    "end_2_end_MIL_splits/mil_data/"
    "mil_data_seed_42_TCGA_Warwick_Clean.pkl"
)

OUTPUT_DIR = (
    "/net/well/pool/projects2/WSI_project/PhD_WSI/"
    "BRC-WSI-Revision/Datasplits_MIL"
)

OUTPUT_PKL = os.path.join(
    OUTPUT_DIR,
    f"mil_data_seed_{SEED}_TCGA_Warwick_CPTAC_Resplit.pkl",
)
OUTPUT_REPORT_JSON = os.path.join(OUTPUT_DIR, f"split_report_seed_{SEED}.json")
OUTPUT_REPORT_TXT = os.path.join(OUTPUT_DIR, f"split_report_seed_{SEED}.txt")

# Labels that are considered "normal" and must be excluded
NORMAL_LABELS = {"Normal", "BRCA_Normal", "NORMAL", "Benign", "benign", "normal"}

# The four classes of interest (string names, as stored in class_names list)
KNOWN_CLASSES = {"Basal", "HER2", "LumA", "LumB"}


# ──────────────────────────────────────────────────────────────────────────────
# Patient / cohort utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_cohort(slide_id: str) -> str:
    """Return cohort label for a slide ID."""
    if "TCGA" in slide_id:
        return "TCGA"
    if "CPTAC" in slide_id:
        return "CPTAC"
    if "Warwick" in slide_id:
        return "Warwick"
    return "unknown"


def get_patient_id(slide_id: str) -> str:
    """
    Extract patient-level ID from slide ID so that all slides from the same
    patient share an identical patient key.

    TCGA   : TCGA-XX-XXXX-01Z-...  →  TCGA-XX-XXXX   (first 3 dash-parts)
    CPTAC  : CPTAC_Label_PATID-UUID → PATID            (part after label prefix)
    Warwick: HER2_Warwick_Subset_N_score_M → HER2_Warwick_Subset_N
             (drop trailing _score_* suffix; verified: each patient = 1 slide)
    """
    if "TCGA" in slide_id:
        parts = slide_id.split("-")
        return "-".join(parts[:3]) if len(parts) >= 3 else slide_id

    if "CPTAC" in slide_id:
        # CPTAC_LumA_01BR015-UUID  →  split on '_' up to 3 parts
        parts = slide_id.split("_", 2)
        if len(parts) >= 3:
            return parts[2].split("-")[0]   # '01BR015'
        return slide_id

    if "Warwick" in slide_id:
        # HER2_Warwick_Training_40_score_2  →  HER2_Warwick_Training_40
        if "_score_" in slide_id:
            return slide_id.rsplit("_score_", 1)[0]
        return slide_id

    return slide_id


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_and_merge(pkl_path: str):
    """
    Load the existing split PKL and merge train + test into flat lists.

    Returns
    -------
    bags            : list[np.ndarray]  — one per slide
    labels_int      : list[int]         — integer-encoded label index
    slide_ids       : list[str]
    instance_names  : list[list[str]]
    instance_labels : list
    instance_paths  : list[list[str]]
    class_names     : list[str]
    """
    print(f"[load] Opening {pkl_path} ...")
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)

    class_names = list(data["class_names"])
    print(f"[load] class_names = {class_names}")

    bags = list(data["train_bags"]) + list(data["test_bags"])
    labels_int = list(data["train_labels"]) + list(data["test_labels"])
    slide_ids = list(data["train_ids"]) + list(data["test_ids"])
    instance_names = list(data["train_instance_names"]) + list(data["test_instance_names"])
    instance_labels = list(data["train_instance_labels"]) + list(data["test_instance_labels"])
    instance_paths = list(data["train_instance_paths"]) + list(data["test_instance_paths"])

    print(
        f"[load] Merged: {len(bags)} slides "
        f"(was train={len(data['train_bags'])}, test={len(data['test_bags'])})"
    )
    return (
        bags, labels_int, slide_ids,
        instance_names, instance_labels, instance_paths,
        class_names,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean(bags, labels_int, slide_ids, inst_names, inst_labels, inst_paths, class_names):
    """
    Remove slides whose decoded label is NaN, normal/benign, or outside the
    four known BRCA subtype classes.  Returns the same tuple, filtered.
    """
    kept = []
    removed = Counter()

    for i, (lbl_idx, sid) in enumerate(zip(labels_int, slide_ids)):
        # Decode to string
        try:
            lbl_str = class_names[int(lbl_idx)]
        except (IndexError, ValueError, TypeError):
            removed["nan_or_invalid"] += 1
            continue

        if lbl_str in NORMAL_LABELS:
            removed["normal"] += 1
            continue

        if lbl_str not in KNOWN_CLASSES:
            removed[f"unknown_label:{lbl_str}"] += 1
            continue

        kept.append(i)

    n_before = len(labels_int)
    n_after = len(kept)
    print(f"[clean] {n_before} -> {n_after} slides  (removed {n_before - n_after})")
    if removed:
        for reason, cnt in removed.items():
            print(f"  removed {cnt} slides: {reason}")

    bags_c = [bags[i] for i in kept]
    labels_c = [labels_int[i] for i in kept]
    ids_c = [slide_ids[i] for i in kept]
    inames_c = [inst_names[i] for i in kept]
    ilabels_c = [inst_labels[i] for i in kept]
    ipaths_c = [inst_paths[i] for i in kept]
    return bags_c, labels_c, ids_c, inames_c, ilabels_c, ipaths_c


# ──────────────────────────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────────────────────────

def patient_stratified_split(bags, labels_int, slide_ids,
                              inst_names, inst_labels, inst_paths,
                              class_names, seed, test_size):
    """
    Patient-level stratified train/test split.

    Stratification key = label x cohort  (e.g. "Basal_TCGA")
    Every slide from the same patient goes into the same partition.
    Falls back to label-only stratification if any label x cohort stratum
    has fewer than 2 patients (sklearn requires >= 2 per class).
    """
    # ── 1. Build per-patient summary ──────────────────────────────────────────
    patient_slides = defaultdict(list)          # patient_id -> [slide_indices]
    for idx, sid in enumerate(slide_ids):
        pid = get_patient_id(sid)
        patient_slides[pid].append(idx)

    patients = sorted(patient_slides.keys())
    n_patients = len(patients)
    print(f"[split] {len(slide_ids)} slides  |  {n_patients} unique patients")

    # Dominant label and cohort per patient
    patient_label = {}
    patient_cohort = {}
    for pid in patients:
        idxs = patient_slides[pid]
        lbls = [labels_int[i] for i in idxs]
        dominant = Counter(lbls).most_common(1)[0][0]
        patient_label[pid] = dominant
        patient_cohort[pid] = get_cohort(slide_ids[idxs[0]])

    # Composite stratification key
    strat_keys = [
        f"{class_names[patient_label[p]]}_{patient_cohort[p]}"
        for p in patients
    ]
    strat_counts = Counter(strat_keys)
    print("[split] Stratification key distribution (patients):")
    for k, v in sorted(strat_counts.items()):
        print(f"  {k}: {v} patients")

    # Fall back to label-only stratification if any stratum < 2
    use_label_only = any(v < 2 for v in strat_counts.values())
    if use_label_only:
        print(
            "[split] WARNING: some label x cohort strata have < 2 patients; "
            "falling back to label-only stratification."
        )
        strat_keys = [class_names[patient_label[p]] for p in patients]

    # ── 2. Split patients ──────────────────────────────────────────────────────
    train_pats, test_pats = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        stratify=strat_keys,
    )
    train_pat_set = set(train_pats)
    test_pat_set = set(test_pats)

    # ── 3. Map patients -> slide indices ───────────────────────────────────────
    train_idx = [i for pid in train_pats for i in patient_slides[pid]]
    test_idx = [i for pid in test_pats for i in patient_slides[pid]]

    def subset(idx_list):
        return (
            [bags[i] for i in idx_list],
            [labels_int[i] for i in idx_list],
            [slide_ids[i] for i in idx_list],
            [inst_names[i] for i in idx_list],
            [inst_labels[i] for i in idx_list],
            [inst_paths[i] for i in idx_list],
        )

    tr = subset(train_idx)
    te = subset(test_idx)

    print(
        f"[split] Train: {len(tr[0])} slides / {len(train_pats)} patients  |  "
        f"Test: {len(te[0])} slides / {len(test_pats)} patients"
    )

    # ── 4. Verify: no patient in both partitions ───────────────────────────────
    overlap = train_pat_set & test_pat_set
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} patients in both splits!"
    print("[split] Patient leakage check: PASSED (0 shared patients)")

    return tr, te, patient_slides, patient_label, patient_cohort


# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_stats(bags, labels_int, slide_ids, class_names,
                 patient_slides, patient_label, patient_cohort,
                 split_name):
    """Build a rich stats dict for one split partition."""
    cohorts = sorted({get_cohort(s) for s in slide_ids})
    classes = sorted(KNOWN_CLASSES)

    slide_by_label_cohort = Counter()
    for sid, lbl in zip(slide_ids, labels_int):
        slide_by_label_cohort[(class_names[lbl], get_cohort(sid))] += 1

    pids_in_split = {get_patient_id(sid) for sid in slide_ids}
    pat_by_label_cohort = Counter()

    # Count how many slides per patient ended up in this split
    pid_to_slide_count = Counter(get_patient_id(sid) for sid in slide_ids)

    for pid in pids_in_split:
        lbl_str = class_names[patient_label[pid]]
        coh = patient_cohort[pid]
        pat_by_label_cohort[(lbl_str, coh)] += 1

    patient_detail = []
    for pid in sorted(pids_in_split):
        patient_detail.append({
            "patient_id": pid,
            "cohort": patient_cohort[pid],
            "label": class_names[patient_label[pid]],
            "n_slides_in_split": pid_to_slide_count[pid],
        })

    tile_counts = [
        bag.shape[0] if hasattr(bag, "shape") else len(bag) for bag in bags
    ]

    stats = {
        "split": split_name,
        "n_slides": len(bags),
        "n_patients": len(pids_in_split),
        "total_tiles": int(np.sum(tile_counts)),
        "mean_tiles_per_slide": float(np.mean(tile_counts)),
        "by_cohort": {},
        "by_label": {},
        "by_label_cohort": {},
        "patient_detail": patient_detail,
    }

    for coh in cohorts:
        mask = [get_cohort(s) == coh for s in slide_ids]
        stats["by_cohort"][coh] = {
            "n_slides": sum(mask),
            "n_patients": sum(1 for pid in pids_in_split if patient_cohort[pid] == coh),
        }

    for cls in classes:
        mask = [class_names[l] == cls for l in labels_int]
        stats["by_label"][cls] = {
            "n_slides": sum(mask),
            "n_patients": sum(
                1 for pid in pids_in_split
                if class_names[patient_label[pid]] == cls
            ),
        }

    for (cls, coh), cnt in sorted(slide_by_label_cohort.items()):
        key = f"{cls}_{coh}"
        stats["by_label_cohort"][key] = {
            "label": cls,
            "cohort": coh,
            "n_slides": cnt,
            "n_patients": pat_by_label_cohort.get((cls, coh), 0),
        }

    return stats


def build_report(
    all_bags, all_labels, all_ids, class_names,
    tr_bags, tr_labels, tr_ids,
    te_bags, te_labels, te_ids,
    patient_slides, patient_label, patient_cohort,
    seed, test_size,
):
    now = datetime.now().isoformat()
    classes = sorted(KNOWN_CLASSES)
    cohorts = sorted({get_cohort(s) for s in all_ids})

    tr_stats = _split_stats(
        tr_bags, tr_labels, tr_ids, class_names,
        patient_slides, patient_label, patient_cohort, "train"
    )
    te_stats = _split_stats(
        te_bags, te_labels, te_ids, class_names,
        patient_slides, patient_label, patient_cohort, "test"
    )

    all_pids = {get_patient_id(s) for s in all_ids}
    overall_tile_counts = [
        bag.shape[0] if hasattr(bag, "shape") else len(bag) for bag in all_bags
    ]

    report = {
        "meta": {
            "created": now,
            "seed": seed,
            "test_size": test_size,
            "input_pkl": INPUT_PKL,
            "output_pkl": OUTPUT_PKL,
            "strategy": "patient-level split, stratified on label x cohort",
        },
        "overall": {
            "n_slides": len(all_bags),
            "n_patients": len(all_pids),
            "total_tiles": int(np.sum(overall_tile_counts)),
            "mean_tiles_per_slide": float(np.mean(overall_tile_counts)),
            "class_names": class_names,
            "by_label": {
                cls: sum(1 for l in all_labels if class_names[l] == cls)
                for cls in classes
            },
            "by_cohort": {
                coh: sum(1 for s in all_ids if get_cohort(s) == coh)
                for coh in cohorts
            },
        },
        "train": tr_stats,
        "test": te_stats,
    }
    return report


def report_to_text(report: dict) -> str:
    """Render the report dict as a human-readable text."""
    lines = []
    sep = "=" * 72

    def h(title):
        lines.append("")
        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)

    def row(label, value, indent=2):
        lines.append(f"{'':>{indent}}{label:<44}{value}")

    meta = report["meta"]
    h("DATASET SPLIT REPORT")
    row("Date", meta["created"])
    row("Random seed", meta["seed"])
    row("Test fraction", meta["test_size"])
    row("Strategy", meta["strategy"])
    row("Input PKL", meta["input_pkl"])
    row("Output PKL", meta["output_pkl"])

    # ── Overall ──────────────────────────────────────────────────────────────
    h("OVERALL (merged, before split)")
    ov = report["overall"]
    row("Total slides", ov["n_slides"])
    row("Total patients", ov["n_patients"])
    row("Total tiles", ov["total_tiles"])
    row("Mean tiles / slide", f"{ov['mean_tiles_per_slide']:.1f}")
    lines.append("")
    lines.append("  Slides per label:")
    for cls, cnt in sorted(ov["by_label"].items()):
        pct = 100.0 * cnt / ov["n_slides"]
        lines.append(f"    {cls:<12} {cnt:>5}  ({pct:.1f}%)")
    lines.append("")
    lines.append("  Slides per cohort:")
    for coh, cnt in sorted(ov["by_cohort"].items()):
        pct = 100.0 * cnt / ov["n_slides"]
        lines.append(f"    {coh:<12} {cnt:>5}  ({pct:.1f}%)")

    # ── Per split ─────────────────────────────────────────────────────────────
    for split_key in ("train", "test"):
        sp = report[split_key]
        h(f"{split_key.upper()} SET")
        row("Slides", sp["n_slides"])
        row("Patients", sp["n_patients"])
        row("Total tiles", sp["total_tiles"])
        row("Mean tiles / slide", f"{sp['mean_tiles_per_slide']:.1f}")

        lines.append("")
        lines.append("  Slides per cohort:")
        for coh, d in sorted(sp["by_cohort"].items()):
            pct = 100.0 * d["n_slides"] / sp["n_slides"]
            lines.append(
                f"    {coh:<12}  slides={d['n_slides']:>4}  ({pct:4.1f}%)  "
                f"patients={d['n_patients']:>4}"
            )

        lines.append("")
        lines.append("  Slides per label:")
        for cls, d in sorted(sp["by_label"].items()):
            pct = 100.0 * d["n_slides"] / sp["n_slides"]
            lines.append(
                f"    {cls:<12}  slides={d['n_slides']:>4}  ({pct:4.1f}%)  "
                f"patients={d['n_patients']:>4}"
            )

        lines.append("")
        lines.append("  Slides per label x cohort:")
        for key, d in sorted(sp["by_label_cohort"].items()):
            pct = 100.0 * d["n_slides"] / sp["n_slides"]
            lines.append(
                f"    {d['label']:<8} x {d['cohort']:<8}  "
                f"slides={d['n_slides']:>4}  ({pct:4.1f}%)  "
                f"patients={d['n_patients']:>4}"
            )

        lines.append("")
        lines.append(
            "  Per-patient breakdown  "
            "(patient_id | cohort | label | n_slides):"
        )
        lines.append(
            f"  {'patient_id':<45} {'cohort':<10} {'label':<8} {'slides':>6}"
        )
        lines.append("  " + "-" * 72)
        for row_d in sorted(
            sp["patient_detail"],
            key=lambda r: (r["cohort"], r["label"], r["patient_id"])
        ):
            lines.append(
                f"  {row_d['patient_id']:<45} "
                f"{row_d['cohort']:<10} "
                f"{row_d['label']:<8} "
                f"{row_d['n_slides_in_split']:>6}"
            )

    h("SPLIT INTEGRITY")
    tr_pids = {r["patient_id"] for r in report["train"]["patient_detail"]}
    te_pids = {r["patient_id"] for r in report["test"]["patient_detail"]}
    overlap = tr_pids & te_pids
    row("Patients in train", len(tr_pids))
    row("Patients in test", len(te_pids))
    row("Shared patients (leakage)", len(overlap))
    if overlap:
        lines.append(f"  WARNING — leaked patients: {sorted(overlap)[:10]}")
    else:
        lines.append("  Patient leakage check: PASSED")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────────────────────────────────────

def save_split_pkl(path, tr, te, class_names):
    """Save new split PKL in the same schema as the input file."""
    tr_bags, tr_labels, tr_ids, tr_inames, tr_ilabels, tr_ipaths = tr
    te_bags, te_labels, te_ids, te_inames, te_ilabels, te_ipaths = te

    payload = {
        "train_bags": tr_bags,
        "train_labels": tr_labels,
        "train_ids": tr_ids,
        "train_instance_names": tr_inames,
        "train_instance_labels": tr_ilabels,
        "train_instance_paths": tr_ipaths,
        "test_bags": te_bags,
        "test_labels": te_labels,
        "test_ids": te_ids,
        "test_instance_names": te_inames,
        "test_instance_labels": te_ilabels,
        "test_instance_paths": te_ipaths,
        "class_names": class_names,
    }
    print(f"[save] Writing split PKL -> {path}")
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    size_gb = os.path.getsize(path) / 1e9
    print(f"[save] Done  ({size_gb:.1f} GB)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*72}")
    print(f"  MIL DATA SPLIT  --  seed={SEED}  test_size={TEST_SIZE}")
    print(f"{'='*72}\n")

    # 1. Load and merge existing split
    (bags, labels_int, slide_ids,
     inst_names, inst_labels, inst_paths,
     class_names) = load_and_merge(INPUT_PKL)

    # 2. Clean
    bags, labels_int, slide_ids, inst_names, inst_labels, inst_paths = clean(
        bags, labels_int, slide_ids, inst_names, inst_labels, inst_paths, class_names
    )

    # 3. Patient-level stratified split
    tr, te, patient_slides, patient_label, patient_cohort = patient_stratified_split(
        bags, labels_int, slide_ids,
        inst_names, inst_labels, inst_paths,
        class_names, seed=SEED, test_size=TEST_SIZE,
    )
    tr_bags, tr_labels, tr_ids = tr[0], tr[1], tr[2]
    te_bags, te_labels, te_ids = te[0], te[1], te[2]

    # 4. Save split PKL
    save_split_pkl(OUTPUT_PKL, tr, te, class_names)

    # 5. Build report
    report = build_report(
        bags, labels_int, slide_ids, class_names,
        tr_bags, tr_labels, tr_ids,
        te_bags, te_labels, te_ids,
        patient_slides, patient_label, patient_cohort,
        seed=SEED, test_size=TEST_SIZE,
    )

    # 6. Save report files
    with open(OUTPUT_REPORT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[report] JSON -> {OUTPUT_REPORT_JSON}")

    txt = report_to_text(report)
    with open(OUTPUT_REPORT_TXT, "w") as fh:
        fh.write(txt)
    print(f"[report] TXT  -> {OUTPUT_REPORT_TXT}")

    print("\n" + txt)


if __name__ == "__main__":
    main()
