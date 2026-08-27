#!/usr/bin/env python3
"""Stage 1: CLIP-embed all photos. Saves embeddings + uuid_index.

Robustness:
- Periodic flush every 2000 images
- Skip failed loads, log to embed_errors.log
- Resume if partial output exists (skip already-embedded UUIDs)
"""
import csv
import itertools
import json
import logging
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

# Decode parallelism. The embed loop used to be strictly serial (decode N,
# then infer N): PIL decode+preprocess ran single-threaded at ~52 img/s
# while MPS inference managed ~123 img/s, so the GPU idled ~74% of the time
# and the observed rate was ~32 img/s (measured 2026-08-27 on a synthetic
# 1,024-photo 640x480 library, see #39). PIL's JPEG decode and the tensor
# ops release the GIL, so a thread pool overlaps decode with inference.
NUM_DECODE_WORKERS = int(os.environ.get("PIXEL_DECODE_WORKERS", "8"))
# Bounded prefetch keeps memory flat (~0.6 MB per preprocessed tensor):
# at most PREFETCH decoded tensors are in flight ahead of inference.
PREFETCH = BATCH_SIZE * 4

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


def decode_one(row):
    """Decode + preprocess one photo in a worker thread. Never raises: a
    failed decode is returned as an error tuple so it is logged per-file on
    the main thread and cannot kill a worker or a batch."""
    try:
        img = Image.open(row["derivative_path"]).convert("RGB")
        return row, preprocess(img), None
    except Exception as e:  # noqa: BLE001 — any decode failure is a data error
        return row, None, e


# Sliding window of decode futures: workers decode ahead (bounded by
# PREFETCH) while the main thread batches tensors and runs MPS inference.
# Results are consumed in submission order, so resume semantics and the
# FLUSH_EVERY persist cadence are identical to the serial version.
pool = ThreadPoolExecutor(max_workers=NUM_DECODE_WORKERS)
todo_iter = iter(todo)
pending = deque()
for row in itertools.islice(todo_iter, PREFETCH):
    pending.append(pool.submit(decode_one, row))

i = -1
while pending:
    i += 1
    row, tensor, err = pending.popleft().result()
    nxt = next(todo_iter, None)
    if nxt is not None:
        pending.append(pool.submit(decode_one, nxt))

    if err is not None:
        total_failed += 1
        err_f.write(f"{row['uuid']}\t{row['derivative_path']}\t{err}\n")
        err_f.flush()
    else:
        buffer_imgs.append(tensor)
        buffer_uuids.append(row["uuid"])

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

pool.shutdown(wait=True)
flush_buffer()
persist()
err_f.close()

elapsed = time.time() - t_start
n_this_run = len(todo) - total_failed
rate = n_this_run / max(elapsed, 1e-9)
log.info(f"DONE. embedded {len(collected_uuids)} (this run +{n_this_run}), failed {total_failed}, "
         f"{elapsed:.1f}s ({rate:.1f} img/s)")
log.info(f"  embeddings: {EMB_PATH}  ({EMB_PATH.stat().st_size/1e6:.1f} MB)")
