#!/usr/bin/env python3
"""Stage 6: Retrain classifier on user corrections + regenerate contact sheets.

Usage:
  python 06_retrain.py              # retrain + regenerate HTML
  python 06_retrain.py --no-regen   # retrain only (skip HTML regeneration)
  python 06_retrain.py --dry-run    # show stats, write nothing

Training strategy:
  Strong labels: corrections.csv (explicit user ground truth, weight=2x)
  Weak labels:   original assignments with confidence >= WEAK_THRESHOLD
                 (uncorrected, no -unsure, no Thrash)

After training:
  - Predicts new album + confidence for every photo
  - conf >= 0.80 → main album
  - conf 0.50-0.80 → <album>-unsure
  - conf < 0.50 → Diverses
  Corrected photos always get conf=1.0 and their corrected album (no -unsure).

Writes: metadata/assignments.csv (backup: assignments.csv.bak)
Then:   regenerates all review/*.html via 04_contact_sheets.py
"""
import argparse
import csv
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path.home() / "photo-sort"
ASSIGN_CSV = ROOT / "metadata" / "assignments.csv"
CORRECTIONS_CSV = ROOT / "metadata" / "corrections.csv"
EMB_NPY = ROOT / "embeddings" / "clip_vitb32.npy"
EMB_IDX = ROOT / "embeddings" / "clip_vitb32_uuids.json"

# Confidence thresholds
CONF_MAIN = 0.80     # >= this → main album
CONF_UNSURE = 0.50   # >= this but < CONF_MAIN → <album>-unsure
# below CONF_UNSURE → Diverses

# Minimum corrections per class to include as valid ground-truth labels
MIN_CORR = 3

# Minimum confidence in original assignment to include as a weak label
WEAK_THRESHOLD = 0.92

# Albums that are local-only / noise — never used as training labels
SKIP_ALBUMS = {"Thrash", "_test_"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("retrain")


def load_corrections() -> dict[str, str]:
    """uuid -> corrected_album (most recent correction wins)."""
    result = {}
    if not CORRECTIONS_CSV.exists():
        return result
    with CORRECTIONS_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["new_album"] not in SKIP_ALBUMS:
                result[r["uuid"]] = r["new_album"]
    return result


def load_assignments() -> dict[str, tuple[str, float, str]]:
    """uuid -> (album, confidence, source)."""
    result = {}
    with ASSIGN_CSV.open() as f:
        for r in csv.DictReader(f):
            result[r["uuid"]] = (r["album"], float(r["confidence"]), r["source"])
    return result


def build_training_set(corrections, assigned, emb, uuid_to_idx):
    """Build X, y arrays for training."""
    corr_counts = Counter(corrections.values())
    valid_corr_albums = {a for a, n in corr_counts.items() if n >= MIN_CORR}
    skipped_albums = {a for a, n in corr_counts.items() if n < MIN_CORR}
    if skipped_albums:
        log.info(f"  Skipping low-count correction albums (<{MIN_CORR}): {skipped_albums}")

    X, y, weights = [], [], []

    # Strong labels: corrections (weight = 2)
    strong = 0
    for uid, album in corrections.items():
        if album not in valid_corr_albums:
            continue
        if uid not in uuid_to_idx:
            continue
        X.append(emb[uuid_to_idx[uid]])
        y.append(album)
        weights.append(2.0)
        strong += 1

    # Weak labels: original high-confidence assignments (weight = 1)
    weak = 0
    for uid, (album, conf, src) in assigned.items():
        if uid in corrections:
            continue  # already have correction
        if conf < WEAK_THRESHOLD:
            continue
        if album.endswith("-unsure") or album in SKIP_ALBUMS:
            continue
        if uid not in uuid_to_idx:
            continue
        X.append(emb[uuid_to_idx[uid]])
        y.append(album)
        weights.append(1.0)
        weak += 1

    log.info(f"  Training set: {strong} strong + {weak} weak = {len(X)} total samples")
    log.info(f"  Classes from corrections: {sorted(valid_corr_albums)}")

    return np.array(X, dtype=np.float32), np.array(y), np.array(weights, dtype=np.float32)


def normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def train_classifier(X, y, weights):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        log.error("scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(2)

    X_norm = normalize(X)
    clf = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
        verbose=0,
    )
    clf.fit(X_norm, y, sample_weight=weights)
    log.info(f"  Classifier trained: {len(clf.classes_)} classes")
    return clf


def write_assignments(clf, emb, uuids_emb, corrections, dry_run=False):
    all_norm = normalize(emb.astype(np.float32))
    probs = clf.predict_proba(all_norm)      # (N, n_classes)
    predicted = clf.predict(all_norm)
    confidences = probs.max(axis=1)
    classes = clf.classes_

    # Count outcomes
    stats = Counter()
    rows = []
    for uid, pred, conf in zip(uuids_emb, predicted, confidences):
        if uid in corrections and corrections[uid] not in SKIP_ALBUMS:
            # Ground-truth override — always goes to corrected album at full confidence
            # SKIP_ALBUMS (e.g. Thrash) are NOT baked into assignments.csv;
            # they live only in corrections.csv and are applied via current_album()
            album = corrections[uid]
            final_conf = 1.0
            source = "correction"
        elif conf >= CONF_MAIN:
            album = pred
            final_conf = float(conf)
            source = "retrain"
        elif conf >= CONF_UNSURE:
            album = f"{pred}-unsure"
            final_conf = float(conf)
            source = "retrain"
        else:
            album = "Diverses"
            final_conf = float(conf)
            source = "retrain"

        rows.append((uid, album, f"{final_conf:.4f}", source))
        stats[album] += 1

    log.info(f"  Predicted {len(rows)} photos across {len(stats)} albums")

    # Top albums by count
    for album, cnt in sorted(stats.items(), key=lambda x: -x[1])[:15]:
        log.info(f"    {cnt:5d}  {album}")

    if dry_run:
        log.info("[DRY RUN] Not writing assignments.csv")
        return stats

    # Backup
    backup = ASSIGN_CSV.with_suffix(".csv.bak")
    backup.write_bytes(ASSIGN_CSV.read_bytes())
    log.info(f"  Backup → {backup.name}")

    with ASSIGN_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "album", "confidence", "source"])
        for row in rows:
            w.writerow(row)

    log.info(f"  Wrote {ASSIGN_CSV}")
    return stats


def regenerate_sheets():
    log.info("Regenerating contact sheets (04_contact_sheets.py)…")
    rc = subprocess.run(
        [sys.executable, str(ROOT / "04_contact_sheets.py")],
        capture_output=False,   # stream output live
    )
    if rc.returncode != 0:
        log.error(f"04_contact_sheets.py failed (exit {rc.returncode})")
        sys.exit(rc.returncode)
    log.info("Contact sheets regenerated.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-regen", action="store_true", help="Skip HTML regeneration")
    ap.add_argument("--dry-run", action="store_true", help="Show stats only, write nothing")
    args = ap.parse_args()

    log.info("=== Stage 6: Retrain ===")

    # Load data
    log.info("Loading corrections…")
    corrections = load_corrections()
    corr_counts = Counter(corrections.values())
    log.info(f"  {len(corrections)} corrections across {len(corr_counts)} albums")

    log.info("Loading original assignments…")
    assigned = load_assignments()
    log.info(f"  {len(assigned)} photos in assignments.csv")

    log.info("Loading CLIP embeddings…")
    emb = np.load(EMB_NPY)
    uuids_emb = json.loads(EMB_IDX.read_text())
    uuid_to_idx = {u: i for i, u in enumerate(uuids_emb)}
    log.info(f"  {emb.shape} embeddings")

    # Build training set
    log.info("Building training set…")
    X, y, weights = build_training_set(corrections, assigned, emb, uuid_to_idx)

    if len(X) == 0:
        log.error("No training data! Add more corrections first.")
        sys.exit(1)

    # Train
    log.info("Training Logistic Regression…")
    clf = train_classifier(X, y, weights)

    # Predict + write
    log.info("Predicting all photos…")
    stats = write_assignments(clf, emb, uuids_emb, corrections, dry_run=args.dry_run)

    if not args.dry_run and not args.no_regen:
        regenerate_sheets()

    # Record retrain checkpoint so the UI can show "N new since last retrain"
    if not args.dry_run:
        import datetime as _dt
        checkpoint = {
            "timestamp": _dt.datetime.utcnow().isoformat(),
            "corrections_count": len(corrections),
        }
        (ROOT / "metadata" / "last_retrain.json").write_text(
            json.dumps(checkpoint, indent=2)
        )
        log.info(f"  Checkpoint saved ({len(corrections)} corrections used)")

    log.info("=== Retrain complete ===")
    return stats


if __name__ == "__main__":
    main()
