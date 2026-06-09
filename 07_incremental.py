#!/usr/bin/env python3
"""Stage 7: Detect and process new photos added since last scan.

Modes:
  --detect         Scan Photos.app for new UUIDs not in inventory.csv.
                   Writes metadata/inbox.csv. Fast (~30s).
  --process        Embed + classify photos in inbox.csv.
                   Appends to inventory.csv, assignments.csv, embeddings.
                   Clears inbox.csv on success. Slow (~2-3 min).
  --all            --detect then --process.

After --detect: server reloads inbox paths so thumbnails are immediately
  accessible and Inbox.html can be browsed manually.
After --process: photos appear in their assigned albums on next retrain
  or are directly added to assignments.csv with ML predictions.
"""
import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.home() / "photo-sort"
INV_CSV = ROOT / "metadata" / "inventory.csv"
INBOX_CSV = ROOT / "metadata" / "inbox.csv"
ASSIGN_CSV = ROOT / "metadata" / "assignments.csv"
CORRECTIONS_CSV = ROOT / "metadata" / "corrections.csv"
EMB_NPY = ROOT / "embeddings" / "clip_vitb32.npy"
EMB_IDX = ROOT / "embeddings" / "clip_vitb32_uuids.json"
OSX_PYTHON = Path.home() / ".local" / "pipx" / "venvs" / "osxphotos" / "bin" / "python"

CONF_MAIN = 0.80
CONF_UNSURE = 0.50
SKIP_ALBUMS = {"Thrash", "_test_"}
MIN_CORR = 3
WEAK_THRESHOLD = 0.92

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("incremental")

# ── detect helper (runs in osxphotos venv) ────────────────────────────────────

_DETECT_HELPER_SRC = r'''#!/usr/bin/env python3
"""Scan Photos.app for UUIDs not yet in inventory.csv. Writes inbox.csv."""
import csv, sys
from pathlib import Path
import osxphotos

root = Path(sys.argv[1])
inv = root / "metadata" / "inventory.csv"
out = root / "metadata" / "inbox.csv"

known = set()
if inv.exists():
    with inv.open() as f:
        for r in csv.DictReader(f):
            known.add(r["uuid"])

print(f"Known UUIDs in inventory: {len(known)}", flush=True)
print("Loading PhotosDB…", flush=True)
db = osxphotos.PhotosDB()
all_photos = db.photos(images=True, intrash=False)
total = len(all_photos)
print(f"Total photos in library: {total}", flush=True)

FIELDS = ["uuid", "original_filename", "date", "derivative_path", "has_local_derivative"]
new_photos = []
for idx, p in enumerate(all_photos):
    if p.uuid in known:
        continue
    path = p.path
    if not path:
        continue
    new_photos.append({
        "uuid": p.uuid,
        "original_filename": p.original_filename or "",
        "date": str(p.date.date()) if p.date else "",
        "derivative_path": path,
        "has_local_derivative": "True",
    })
    if idx % 1000 == 0:
        print(f"PROGRESS:{idx}:{total}", flush=True)
print(f"PROGRESS:{total}:{total}", flush=True)

with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(new_photos)

print(f"RESULT:{len(new_photos)}")
'''


def run_detect():
    log.info("=== Phase 1: Detect new photos ===")
    helper = ROOT / "_inbox_detect_helper.py"
    helper.write_text(_DETECT_HELPER_SRC)
    proc = subprocess.Popen(
        [str(OSX_PYTHON), str(helper), str(ROOT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    count = 0
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log.info(f"  {line}")
            if line.startswith("RESULT:"):
                count = int(line.split(":")[1])
    proc.wait()
    if proc.returncode != 0:
        log.error(f"Detection helper failed (exit {proc.returncode})")
        sys.exit(proc.returncode)
    log.info(f"=== Detect complete: {count} new photos written to inbox.csv ===")
    return count


# ── process (embed + classify) ────────────────────────────────────────────────

def run_process():
    log.info("=== Phase 2: Process inbox photos ===")

    if not INBOX_CSV.exists():
        log.error("inbox.csv not found — run --detect first")
        sys.exit(1)

    inbox_rows = list(csv.DictReader(INBOX_CSV.open()))
    todo = [r for r in inbox_rows if r.get("has_local_derivative") == "True" and r.get("derivative_path")]
    log.info(f"  {len(todo)} photos to embed")
    if not todo:
        log.info("  Nothing to do.")
        return

    # Load CLIP model
    try:
        import torch
        import open_clip
        from PIL import Image
    except ImportError as e:
        log.error(f"Missing dependency: {e}. Run inside the venv.")
        sys.exit(2)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"  Loading CLIP (device={device})…")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    log.info("  CLIP ready")

    # Embed new photos
    BATCH = 64
    new_embs = []
    new_uuids = []
    failed = 0
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        imgs, uids = [], []
        for r in batch:
            try:
                img = preprocess(Image.open(r["derivative_path"]).convert("RGB"))
                imgs.append(img)
                uids.append(r["uuid"])
            except Exception as e:
                log.warning(f"  skip {r['uuid'][:8]}: {e}")
                failed += 1
        if not imgs:
            continue
        with torch.no_grad():
            t = torch.stack(imgs).to(device)
            emb = model.encode_image(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            new_embs.append(emb.cpu().float().numpy())
            new_uuids.extend(uids)
        if (i // BATCH) % 5 == 0:
            elapsed = time.time() - t0
            log.info(f"  embedded {len(new_uuids)}/{len(todo)} ({elapsed:.0f}s, {failed} failed)")

    if not new_embs:
        log.error("No embeddings generated")
        sys.exit(1)
    new_emb_arr = np.vstack(new_embs)
    log.info(f"  Embedded {len(new_uuids)} photos ({failed} failed)")

    # Load existing embeddings + append
    log.info("  Appending to embeddings…")
    old_emb = np.load(EMB_NPY)
    old_uuids = json.loads(EMB_IDX.read_text())
    combined_emb = np.vstack([old_emb, new_emb_arr])
    combined_uuids = old_uuids + new_uuids
    np.save(EMB_NPY, combined_emb)
    EMB_IDX.write_text(json.dumps(combined_uuids))
    log.info(f"  Embeddings: {len(old_uuids)} → {len(combined_uuids)}")

    # Train classifier on existing corrections + assignments
    log.info("  Training classifier…")
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        log.error("scikit-learn not installed")
        sys.exit(2)

    corrections = {}
    if CORRECTIONS_CSV.exists():
        with CORRECTIONS_CSV.open() as f:
            for r in csv.DictReader(f):
                if r["new_album"] and r["new_album"] not in SKIP_ALBUMS:
                    corrections[r["uuid"]] = r["new_album"]

    assigned = {}
    with ASSIGN_CSV.open() as f:
        for r in csv.DictReader(f):
            assigned[r["uuid"]] = (r["album"], float(r["confidence"]))

    uuid_to_idx = {u: i for i, u in enumerate(combined_uuids)}

    # Build training set
    from collections import Counter
    corr_counts = Counter(corrections.values())
    valid_corr = {a for a, n in corr_counts.items() if n >= MIN_CORR}

    X_tr, y_tr, w_tr = [], [], []
    for uid, alb in corrections.items():
        if alb not in valid_corr or uid not in uuid_to_idx:
            continue
        X_tr.append(combined_emb[uuid_to_idx[uid]])
        y_tr.append(alb)
        w_tr.append(2.0)
    for uid, (alb, conf) in assigned.items():
        if uid in corrections or conf < WEAK_THRESHOLD or alb.endswith("-unsure") or alb in SKIP_ALBUMS:
            continue
        if uid not in uuid_to_idx:
            continue
        X_tr.append(combined_emb[uuid_to_idx[uid]])
        y_tr.append(alb)
        w_tr.append(1.0)

    if not X_tr:
        log.warning("No training data — using Diverses as fallback for all new photos")
        clf = None
    else:
        def _norm(X):
            n = np.linalg.norm(X, axis=1, keepdims=True)
            return X / np.where(n == 0, 1.0, n)
        X_arr = _norm(np.array(X_tr, dtype=np.float32))
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                 class_weight="balanced", n_jobs=-1)
        clf.fit(X_arr, y_tr, sample_weight=np.array(w_tr))
        log.info(f"  Classifier: {len(clf.classes_)} classes, {len(X_tr)} training samples")

    # Classify new photos
    new_emb_norm = new_emb_arr / np.linalg.norm(new_emb_arr, axis=1, keepdims=True)
    if clf is not None:
        probs = clf.predict_proba(new_emb_norm)
        preds = clf.predict(new_emb_norm)
        confs = probs.max(axis=1)
    else:
        preds = ["Diverses"] * len(new_uuids)
        confs = [0.0] * len(new_uuids)

    # Append to inventory.csv and assignments.csv
    uid_set = {r["uuid"] for r in inbox_rows}
    inv_rows = [r for r in inbox_rows if r["uuid"] in set(new_uuids)]

    # inventory.csv: append new rows
    inv_fields = ["uuid", "original_filename", "date", "ismissing",
                  "has_local_original", "has_local_derivative", "derivative_path",
                  "derivative_size_bytes", "uti_original", "isphoto", "ismovie"]
    with INV_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=inv_fields, extrasaction="ignore")
        for r in inv_rows:
            r.setdefault("ismissing", "False")
            r.setdefault("has_local_original", "False")
            r.setdefault("derivative_size_bytes", "0")
            r.setdefault("uti_original", "")
            r.setdefault("isphoto", "True")
            r.setdefault("ismovie", "False")
            w.writerow(r)

    # assignments.csv: append new rows
    added = 0
    with ASSIGN_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        for uid, pred, conf in zip(new_uuids, preds, confs):
            if conf >= CONF_MAIN:
                album = pred
            elif conf >= CONF_UNSURE:
                album = f"{pred}-unsure"
            else:
                album = "Diverses"
            w.writerow([uid, album, f"{conf:.4f}", "incremental"])
            added += 1

    log.info(f"  Appended {added} photos to assignments.csv")

    # Clear inbox.csv
    INBOX_CSV.write_text("uuid,original_filename,date,derivative_path,has_local_derivative\n")
    log.info("  inbox.csv cleared")

    # Regenerate contact sheets
    log.info("  Regenerating contact sheets…")
    subprocess.run([sys.executable, str(ROOT / "04_contact_sheets.py")], check=True)
    log.info("=== Process complete ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--process", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all or (not args.detect and not args.process):
        run_detect()
        run_process()
    elif args.detect:
        run_detect()
    elif args.process:
        run_process()


if __name__ == "__main__":
    main()
