#!/usr/bin/env python3
"""Stage 2: NSFW score every photo with NudeNet.

Outputs metadata/nsfw_scores.csv with per-photo classification:
- nsfw_label: "explicit" | "nude" | "safe"
- nsfw_confidence: 0..1
- detected_classes: comma-list of NudeNet classes seen with conf>0.3

NudeNet v3 API: NudeDetector().detect(image_path) → list of {class, score, box}.
"""
import csv
import logging
import sys
import time
from pathlib import Path

from PIL import Image
from nudenet import NudeDetector

ROOT = Path.home() / "photo-sort"
CSV_IN = ROOT / "metadata" / "inventory.csv"
CSV_OUT = ROOT / "metadata" / "nsfw_scores.csv"
LOG_PATH = ROOT / "logs" / "02_nsfw.log"
ERR_PATH = ROOT / "logs" / "02_nsfw_errors.log"

CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("nsfw")

# NudeNet class semantics
# "explicit" classes — sexual / genitalia / anus exposed
EXPLICIT = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",  # debatable — keep explicit
}
# "nude" classes — non-genital nudity
NUDE = {
    "FEMALE_BREAST_EXPOSED",
    "MALE_BREAST_EXPOSED",  # usually shirtless men — debatable, but let's see
}

log.info("Loading NudeDetector...")
t0 = time.time()
detector = NudeDetector()
log.info(f"  ready in {time.time()-t0:.1f}s")

log.info(f"Reading inventory {CSV_IN}")
with CSV_IN.open() as f:
    rows = [r for r in csv.DictReader(f) if r["has_local_derivative"] == "True"]
log.info(f"  {len(rows)} photos to scan")

# Resume support
done_uuids = set()
if CSV_OUT.exists():
    with CSV_OUT.open() as f:
        done_uuids = {r["uuid"] for r in csv.DictReader(f)}
    log.info(f"  resume: {len(done_uuids)} already scored")

mode = "a" if done_uuids else "w"
err_f = ERR_PATH.open("a")
out_f = CSV_OUT.open(mode, newline="")
w = csv.writer(out_f)
if not done_uuids:
    w.writerow(["uuid", "nsfw_label", "nsfw_confidence", "explicit_max", "nude_max", "detected"])

t_start = time.time()
todo = [r for r in rows if r["uuid"] not in done_uuids]
log.info(f"  to scan: {len(todo)}")
fails = 0
for i, row in enumerate(todo):
    try:
        # Quick filter: tiny thumbnails (<100px) often fail or are useless for NSFW
        path = row["derivative_path"]
        detections = detector.detect(path)
        explicit_max = 0.0
        nude_max = 0.0
        detected_strs = []
        for det in detections:
            cls = det.get("class", "")
            sc = float(det.get("score", 0.0))
            if sc >= 0.3:
                detected_strs.append(f"{cls}:{sc:.2f}")
            if cls in EXPLICIT and sc > explicit_max:
                explicit_max = sc
            elif cls in NUDE and sc > nude_max:
                nude_max = sc

        if explicit_max >= nude_max and explicit_max > 0:
            label, conf = "explicit", explicit_max
        elif nude_max > 0:
            label, conf = "nude", nude_max
        else:
            label, conf = "safe", 0.0

        w.writerow([row["uuid"], label, f"{conf:.4f}", f"{explicit_max:.4f}", f"{nude_max:.4f}", ",".join(detected_strs)])
    except Exception as e:
        fails += 1
        err_f.write(f"{row['uuid']}\t{row['derivative_path']}\t{e}\n")
        err_f.flush()
        w.writerow([row["uuid"], "error", "0", "0", "0", str(e)[:80]])

    if (i + 1) % 1000 == 0:
        out_f.flush()
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta_min = (len(todo) - i - 1) / max(rate, 1e-9) / 60
        log.info(f"  progress {i+1}/{len(todo)}  rate {rate:.1f} img/s  eta {eta_min:.0f} min  fails {fails}")

out_f.close()
err_f.close()
elapsed = time.time() - t_start
log.info(f"DONE. scanned {len(todo)}, fails {fails}, {elapsed/60:.1f} min")
