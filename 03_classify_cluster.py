#!/usr/bin/env python3
"""Stage 3: Classify against seed categories + cluster remainder.

Output: metadata/assignments.csv with per-photo:
  uuid, album, confidence, source ("nsfw"|"clip-seed"|"cluster"), cluster_id

Albums target ≤20 total. Seed albums:
  * Motorrad-Werkstatt
  * Aquarium
  * Garten
  * Pornografie  (from NSFW "explicit")
  * Nacktheit    (from NSFW "nude")
Each may have an `-unsure` variant for 0.5..0.8 confidence.
Remaining photos → HDBSCAN clusters (top-K by size).
"""
import csv
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import open_clip

import model_tag
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(os.environ.get("PIXEL_ROOT", str(Path.home() / "photo-sort")))
EMB_PATH = ROOT / "embeddings" / "clip_vitb32.npy"
IDX_PATH = ROOT / "embeddings" / "clip_vitb32_uuids.json"
NSFW_CSV = ROOT / "metadata" / "nsfw_scores.csv"
INV_CSV = ROOT / "metadata" / "inventory.csv"
OUT_CSV = ROOT / "metadata" / "assignments.csv"
CLUSTER_META = ROOT / "metadata" / "cluster_meta.json"
LOG_PATH = ROOT / "logs" / "03_classify.log"

CONF_MAIN = 0.80
CONF_UNSURE = 0.50
N_CLUSTERS = 10  # K-Means K; total budget = 5 seeds + 5 unsures + 10 clusters + Unsortiert = 21
LOGIT_SCALE = 100.0  # CLIP standard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("classify")

# ---------- Load embeddings ----------
log.info("Loading CLIP embeddings...")
emb = np.load(EMB_PATH).astype(np.float32)
uuids = json.loads(IDX_PATH.read_text())
log.info(f"  embeddings: {emb.shape}  uuids: {len(uuids)}")
assert emb.shape[0] == len(uuids)

# Normalize (should already be)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

# ---------- NSFW lookup ----------
log.info("Loading NSFW scores...")
nsfw = {}
if NSFW_CSV.exists():
    with NSFW_CSV.open() as f:
        for r in csv.DictReader(f):
            nsfw[r["uuid"]] = (r["nsfw_label"], float(r["nsfw_confidence"]))
    log.info(f"  nsfw entries: {len(nsfw)}")
else:
    log.warning("  no NSFW CSV found — Pornografie/Nacktheit albums will be empty")

# ---------- CLIP text encodings ----------
log.info("Encoding seed text prompts via CLIP...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Text embeddings get compared against the stored image space below --
# hard-fail if the stored space came from a different model config (#38).
model_tag.check_tag(EMB_PATH)
model, _, _ = open_clip.create_model_and_transforms(
    model_tag.MODEL_NAME, pretrained=model_tag.PRETRAINED
)
tokenizer = open_clip.get_tokenizer(model_tag.MODEL_NAME)
model = model.to(device).eval()

seed_groups = {
    "Motorrad-Werkstatt": [
        "a photo of a motorcycle", "a photo of motorcycles parked",
        "a photo of motorcycle parts", "a photo of a workshop with tools",
        "a photo of mechanic tools and equipment", "a photo of vehicle repair",
        "a photo of a motorcycle engine", "a photo of a garage workshop",
    ],
    "Aquarium": [
        "a photo of an aquarium with fish", "a photo of fish in a tank",
        "a photo of aquatic plants underwater", "a photo of a fish tank",
        "a photo of tropical fish", "a photo of underwater aquarium scene",
    ],
    "Garten": [
        "a photo of a garden", "a photo of plants and flowers",
        "a photo of a backyard with greenery", "a photo of a vegetable garden",
        "a photo of flowers blooming", "a photo of outdoor plants",
        "a photo of a patio with potted plants",
    ],
}
# Background / distractor categories — gives softmax something to compete with
distractor_prompts = [
    "a screenshot of a phone screen", "a screenshot of a computer screen",
    "a photo of a document or receipt", "a selfie of a person",
    "a portrait photo of people", "a photo of food on a plate",
    "a photo of a building or architecture", "a photo of a landscape",
    "a photo of a vehicle, car or truck", "a photo of pets or animals",
    "a photo of an event or party", "a photo of clothing or fashion",
    "an indoor photo of a room", "a photo of the sky or clouds",
    "a photo of a beach or water", "a photo of a meal or restaurant",
]

with torch.no_grad():
    seed_embs = {}
    for name, prompts in seed_groups.items():
        toks = tokenizer(prompts).to(device)
        feats = model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        seed_embs[name] = feats.mean(0).cpu().numpy().astype(np.float32)
        seed_embs[name] /= np.linalg.norm(seed_embs[name]) + 1e-9
    dist_toks = tokenizer(distractor_prompts).to(device)
    dist_feats = model.encode_text(dist_toks)
    dist_feats = dist_feats / dist_feats.norm(dim=-1, keepdim=True)
    dist_emb = dist_feats.cpu().numpy().astype(np.float32)  # (D, 512)

seed_names = list(seed_embs.keys())
seed_matrix = np.stack([seed_embs[n] for n in seed_names], axis=0)  # (S, 512)
all_text = np.concatenate([seed_matrix, dist_emb], axis=0)  # (S+D, 512)

# ---------- Per-photo classification ----------
log.info("Classifying all photos...")
t0 = time.time()
# Compute similarities in chunks (to avoid memory blowup)
CHUNK = 10000
sims = np.zeros((emb.shape[0], all_text.shape[0]), dtype=np.float32)
for i in range(0, emb.shape[0], CHUNK):
    sims[i:i+CHUNK] = emb[i:i+CHUNK] @ all_text.T
# Softmax with logit scale
logits = sims * LOGIT_SCALE
m = logits.max(axis=1, keepdims=True)
exp = np.exp(logits - m)
probs = exp / exp.sum(axis=1, keepdims=True)  # (N, S+D)
seed_probs = probs[:, :len(seed_names)]
seed_top_idx = seed_probs.argmax(axis=1)
seed_top_p = seed_probs[np.arange(emb.shape[0]), seed_top_idx]

# ---------- Build assignments ----------
log.info("Assigning buckets...")
assignments = {}  # uuid -> (album, conf, source, cluster_id_or_-1)
nsfw_assigned = 0
clip_main = 0
clip_unsure = 0
nsfw_main = 0
nsfw_unsure = 0
unassigned = 0
unassigned_uuids = []
unassigned_idx = []

for i, uid in enumerate(uuids):
    # First: NSFW takes precedence
    nlabel, nconf = nsfw.get(uid, ("safe", 0.0))
    if nlabel == "explicit":
        if nconf >= CONF_MAIN:
            assignments[uid] = ("Pornografie", nconf, "nsfw", -1); nsfw_main += 1; continue
        if nconf >= CONF_UNSURE:
            assignments[uid] = ("Pornografie-unsure", nconf, "nsfw", -1); nsfw_unsure += 1; continue
    elif nlabel == "nude":
        if nconf >= CONF_MAIN:
            assignments[uid] = ("Nacktheit", nconf, "nsfw", -1); nsfw_main += 1; continue
        if nconf >= CONF_UNSURE:
            assignments[uid] = ("Nacktheit-unsure", nconf, "nsfw", -1); nsfw_unsure += 1; continue

    # CLIP seed
    p = seed_top_p[i]
    if p >= CONF_MAIN:
        assignments[uid] = (seed_names[seed_top_idx[i]], float(p), "clip-seed", -1); clip_main += 1; continue
    if p >= CONF_UNSURE:
        assignments[uid] = (f"{seed_names[seed_top_idx[i]]}-unsure", float(p), "clip-seed", -1); clip_unsure += 1; continue

    unassigned += 1
    unassigned_uuids.append(uid)
    unassigned_idx.append(i)

log.info(f"  NSFW main: {nsfw_main}  NSFW unsure: {nsfw_unsure}")
log.info(f"  CLIP main: {clip_main}   CLIP unsure: {clip_unsure}")
log.info(f"  unassigned (→cluster): {unassigned}")
log.info(f"  classification {time.time()-t0:.1f}s")

# ---------- K-Means on the unassigned ----------
cluster_meta = {}
if unassigned >= N_CLUSTERS * 50:
    log.info(f"MiniBatchKMeans (k={N_CLUSTERS}) on {unassigned} embeddings...")
    sub_emb = emb[unassigned_idx]
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, random_state=42, batch_size=4096,
        n_init=5, max_iter=200, reassignment_ratio=0.02,
    )
    labels = km.fit_predict(sub_emb)
    log.info(f"  KMeans done in {time.time()-t0:.1f}s")

    label_counts = Counter(labels)
    log.info(f"  cluster sizes: {dict(sorted(label_counts.items(), key=lambda x: -x[1]))}")

    label_terms_list = list(seed_groups.keys()) + distractor_prompts
    label_term_embs = all_text  # (S+D, 512)
    for cid in range(N_CLUSTERS):
        idx_in_sub = np.where(labels == cid)[0]
        if len(idx_in_sub) == 0:
            continue
        cluster_orig_idx = [unassigned_idx[i] for i in idx_in_sub]
        centroid = emb[cluster_orig_idx].mean(0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims_lbl = centroid @ label_term_embs.T
        top_lbl_idx = np.argsort(-sims_lbl)[:5]
        suggestions = [(label_terms_list[k] if k < len(label_terms_list) else f"prompt_{k}", float(sims_lbl[k])) for k in top_lbl_idx]
        cluster_embs = emb[cluster_orig_idx]
        rep_sims = cluster_embs @ centroid
        rep_order = np.argsort(-rep_sims)[:8]
        rep_uuids = [uuids[cluster_orig_idx[k]] for k in rep_order]
        cluster_meta[str(cid)] = {
            "size": int(label_counts[cid]),
            "suggestions": suggestions,
            "rep_uuids": rep_uuids,
        }

    for sub_i, lab in enumerate(labels):
        orig_i = unassigned_idx[sub_i]
        uid = uuids[orig_i]
        assignments[uid] = (f"Cluster-{lab:02d}", 1.0, "cluster", int(lab))

    log.info(f"  all {N_CLUSTERS} clusters kept (KMeans has no noise category)")
else:
    log.warning(f"  too few unassigned for clustering; all → 'Unsortiert'")
    for uid in unassigned_uuids:
        assignments[uid] = ("Unsortiert", 0.0, "fallback", -1)

# ---------- Write outputs ----------
log.info(f"Writing {OUT_CSV}")
with OUT_CSV.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["uuid", "album", "confidence", "source", "cluster_id"])
    for uid in uuids:
        a, c, s, cid = assignments[uid]
        w.writerow([uid, a, f"{c:.4f}", s, cid])

CLUSTER_META.write_text(json.dumps(cluster_meta, indent=2))

# Summary
album_counts = Counter(a for a, _, _, _ in assignments.values())
log.info("Album sizes:")
for a, n in album_counts.most_common():
    log.info(f"  {a:30}  {n:6}")
log.info("DONE.")
