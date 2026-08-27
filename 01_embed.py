#!/usr/bin/env python3
"""Stage 1: CLIP-embed all photos. Saves embeddings + uuid_index.

Robustness:
- Periodic flush every 2000 images
- Skip failed loads, log to embed_errors.log
- Resume if partial output exists (skip already-embedded UUIDs)
"""
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image

ROOT = Path(os.environ.get("PIXEL_ROOT", str(Path.home() / "photo-sort")))
CSV_PATH = ROOT / "metadata" / "inventory.csv"
EMB_PATH = ROOT / "embeddings" / "clip_vitb32.npy"
IDX_PATH = ROOT / "embeddings" / "clip_vitb32_uuids.json"
LOG_PATH = ROOT / "logs" / "01_embed.log"
ERR_PATH = ROOT / "logs" / "01_embed_errors.log"

BATCH_SIZE = 64
FLUSH_EVERY = 2000

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
EMB_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("embed")

device = "mps" if torch.backends.mps.is_available() else "cpu"
log.info(f"START. device={device} batch={BATCH_SIZE}")

log.info("Loading CLIP ViT-B-32 (openai)...")
t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device).eval()
log.info(f"  model ready in {time.time()-t0:.1f}s")

# Load inventory
log.info(f"Reading inventory {CSV_PATH}")
with CSV_PATH.open() as f:
    rows = [r for r in csv.DictReader(f) if r["has_local_derivative"] == "True"]
log.info(f"  {len(rows)} photos with derivatives")

# Resume support
done_uuids = set()
existing_emb = None
existing_idx = []
if EMB_PATH.exists() and IDX_PATH.exists():
    existing_idx = json.loads(IDX_PATH.read_text())
    done_uuids = set(existing_idx)
    existing_emb = np.load(EMB_PATH)
    log.info(f"  resume: {len(done_uuids)} already embedded")

# Filter
todo = [r for r in rows if r["uuid"] not in done_uuids]
log.info(f"  to embed: {len(todo)}")

err_f = ERR_PATH.open("a")
collected_emb = [existing_emb] if existing_emb is not None else []
collected_uuids = list(existing_idx)
buffer_imgs = []
buffer_uuids = []
done_since_flush = 0
total_failed = 0
t_start = time.time()


def flush_buffer():
    global collected_emb, collected_uuids, buffer_imgs, buffer_uuids, done_since_flush
    if not buffer_imgs:
        return
    batch = torch.stack(buffer_imgs).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    collected_emb.append(feats.cpu().numpy().astype(np.float32))
    collected_uuids.extend(buffer_uuids)
    buffer_imgs.clear()
    buffer_uuids.clear()


def persist():
    if not collected_emb:
        return
    arr = np.concatenate(collected_emb, axis=0)
    np.save(EMB_PATH, arr)
    IDX_PATH.write_text(json.dumps(collected_uuids))


for i, row in enumerate(todo):
    try:
        img = Image.open(row["derivative_path"]).convert("RGB")
        buffer_imgs.append(preprocess(img))
        buffer_uuids.append(row["uuid"])
    except Exception as e:
        total_failed += 1
        err_f.write(f"{row['uuid']}\t{row['derivative_path']}\t{e}\n")
        err_f.flush()

    if len(buffer_imgs) >= BATCH_SIZE:
        flush_buffer()

    done_since_flush += 1
    if done_since_flush >= FLUSH_EVERY:
        flush_buffer()
        persist()
        elapsed = time.time() - t_start
        done_total = len(collected_uuids)
        rate = (done_total - len(done_uuids)) / max(elapsed, 1e-9)
        remaining = len(todo) - (i + 1)
        eta_min = remaining / max(rate, 1e-9) / 60
        log.info(f"  progress {i+1}/{len(todo)}  rate {rate:.1f} img/s  eta {eta_min:.0f} min  fail {total_failed}")
        done_since_flush = 0

flush_buffer()
persist()
err_f.close()

elapsed = time.time() - t_start
log.info(f"DONE. embedded {len(collected_uuids)} (this run +{len(todo)-total_failed}), failed {total_failed}, {elapsed/60:.1f} min")
log.info(f"  embeddings: {EMB_PATH}  ({EMB_PATH.stat().st_size/1e6:.1f} MB)")
