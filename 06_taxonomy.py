#!/usr/bin/env python3
"""Stage 6: Build Yves' actual taxonomy.

Rules:
  Person-based (multi-album, from Photos face data):
    Yves              = "Yves Vogl" in persons
    Coven Call        = Yves Vogl + (Jason Merse | Moritz Bleif | Linda Hennen | Joel Hösterey)
                        OR Yves Vogl + CLIP-detected "concert stage / performing band"
    Bolle             = "Bolle"
    Yuna              = "Yuna"
    Ferro             = "Ferro"
    Family & Friends  = any other named person (not Yves, not band, not pet)

  NSFW (from existing NudeNet scores):
    Pornografie, Pornografie-unsure
    Nacktheit, Nacktheit-unsure
    Homeporn          = empty (seed-only, Yves moves manually)

  CLIP zero-shot albums (with new prompts):
    Aquarium, Terrarium, Garten, Astro, Kochen, Urlaub,
    Konzerte & Festivals, Gitarre & Co, Unterwegs, Zuhause,
    Andere Motorräder, Tattoo Ideen, Diverses

  Empty seed-driven albums (Yves populates via UI):
    Homeporn, Meine Motorräder, Tattoos (Yves), Tattoo Ideen
    (created in albums list, no photos auto-assigned)

Output:
  metadata/assignments.csv         — uuid, album, confidence, source  (primary, one per uuid)
  metadata/multi_assignments.csv   — uuid, album, source             (all memberships)
  metadata/cluster_meta.json       — empty (no clusters in new taxonomy)
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

ROOT = Path(os.environ.get("PIXEL_ROOT", str(Path.home() / "photo-sort")))
EMB_PATH = ROOT / "embeddings" / "clip_vitb32.npy"
IDX_PATH = ROOT / "embeddings" / "clip_vitb32_uuids.json"
NSFW_CSV = ROOT / "metadata" / "nsfw_scores.csv"
PERSONS_JSON = ROOT / "metadata" / "persons_per_photo.json"
OUT_CSV = ROOT / "metadata" / "assignments.csv"
MULTI_CSV = ROOT / "metadata" / "multi_assignments.csv"
CLUSTER_META = ROOT / "metadata" / "cluster_meta.json"
LOG_PATH = ROOT / "logs" / "06_taxonomy.log"

CONF_MAIN = 0.80
CONF_UNSURE = 0.50
LOGIT_SCALE = 100.0

# People classification
COVEN_CALL_BAND = {"Jason Merse", "Moritz Bleif", "Linda Hennen", "Joel Hösterey"}
PETS = {"Bolle", "Yuna", "Ferro"}
YVES = "Yves Vogl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("taxonomy")

# ---------- Load data ----------
log.info("Loading embeddings + persons...")
emb = np.load(EMB_PATH).astype(np.float32)
uuids = json.loads(IDX_PATH.read_text())
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
persons_map = json.loads(PERSONS_JSON.read_text())

nsfw = {}
with NSFW_CSV.open() as f:
    for r in csv.DictReader(f):
        nsfw[r["uuid"]] = (r["nsfw_label"], float(r["nsfw_confidence"]))

# ---------- CLIP text encoding ----------
log.info("Loading CLIP...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Text embeddings get compared against the stored image space loaded
# above -- hard-fail if that space came from a different model config (#38).
model_tag.check_tag(EMB_PATH)
model, _, _ = open_clip.create_model_and_transforms(
    model_tag.MODEL_NAME, pretrained=model_tag.PRETRAINED
)
tokenizer = open_clip.get_tokenizer(model_tag.MODEL_NAME)
model = model.to(device).eval()

# CLIP seed groups
clip_groups = {
    "Aquarium": [
        "a photo of an aquarium with fish", "a photo of a fish tank",
        "a photo of tropical fish swimming", "a photo of aquatic plants underwater",
    ],
    "Terrarium": [
        "a photo of a terrarium with reptiles", "a photo of a snake or lizard",
        "a photo of a vivarium habitat", "a photo of an indoor reptile enclosure",
    ],
    "Garten": [
        "a photo of a garden", "a photo of plants and flowers outdoors",
        "a photo of a vegetable garden", "a photo of a backyard with greenery",
    ],
    "Astro": [
        "a photo of stars at night", "a photo of the moon",
        "an astrophotography image of nebulae or galaxies", "a photo of the milky way",
        "a long-exposure photo of the night sky",
    ],
    "Kochen": [
        "a photo of cooking in a kitchen", "a photo of food preparation",
        "a photo of ingredients being chopped", "a photo of someone cooking at the stove",
    ],
    "Urlaub": [
        "a tourist photo of a vacation destination", "a photo of a famous landmark",
        "a photo of a beach with palm trees", "a photo of mountains during travel",
        "a sightseeing photo from a trip",
    ],
    "Konzerte & Festivals": [
        "a photo of a concert stage with lights", "a photo of a live music performance",
        "a photo of a festival crowd at night", "a photo of a band performing on stage",
        "a photo of stage lighting and musicians",
    ],
    "Gitarre & Co": [
        "a close-up photo of an electric guitar", "a photo of guitar pedals and amplifiers",
        "a photo of musical instruments in a studio", "a photo of a guitar collection",
    ],
    "Unterwegs": [
        "a photo from a moving car or train", "a photo of a street scene from a sidewalk",
        "a photo of public transport interior", "a photo while walking outside in a city",
    ],
    "Zuhause": [
        "a photo inside a living room", "a photo of an indoor home interior",
        "a photo of a cozy bedroom", "a photo of a kitchen at home",
    ],
    "Andere Motorräder": [
        "a photo of a motorcycle parked", "a photo of motorcycles at a meet",
        "a photo of a sport bike or cruiser", "a photo of a motorbike from the side",
    ],
    "Werkstatt": [
        "a photo of a workshop with tools", "a photo of mechanic tools and equipment",
        "a photo of vehicle repair in progress", "a photo of a garage workbench",
        "a photo of disassembled motorcycle parts", "a photo of a motorcycle engine being worked on",
    ],
    "Tattoo Ideen": [
        "a flash sheet of tattoo designs", "a photo of tattoo artwork on paper",
        "a tattoo design reference image", "a sketch of a tattoo design",
    ],
    "Screenshots": [
        "a screenshot of a phone screen", "a screenshot of a computer screen",
        "a screenshot of a chat or messaging app", "a screenshot of a web page",
        "a screenshot of a mobile app interface", "a screenshot showing UI elements and text",
        "a screen capture of a smartphone", "a screen recording or screenshot with status bar",
    ],
}

# Distractors — gives softmax something to compete with so threshold makes sense
distractor_prompts = [
    "a photo of a document, paper or receipt",
    "a selfie portrait close-up",
    "a photo of a meeting or office work",
    "a photo of pets indoors",
    "a generic photo of a person",
    "a photo of clothes or fashion",
    "a photo of food on a plate at a restaurant",
    "a photo of a building or architecture",
    "a photo of clouds or weather",
]

log.info("Encoding text prompts...")
with torch.no_grad():
    group_embs = {}
    for name, prompts in clip_groups.items():
        toks = tokenizer(prompts).to(device)
        feats = model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        group_embs[name] = feats.mean(0).cpu().numpy().astype(np.float32)
        group_embs[name] /= np.linalg.norm(group_embs[name]) + 1e-9

    dist_toks = tokenizer(distractor_prompts).to(device)
    dist_feats = model.encode_text(dist_toks)
    dist_feats = dist_feats / dist_feats.norm(dim=-1, keepdim=True)
    dist_emb = dist_feats.cpu().numpy().astype(np.float32)

    # Coven Call stage prompt
    stage_prompts = [
        "a photo of a person performing on a concert stage with lights",
        "a photo of someone singing or playing guitar on stage",
        "a band member performing live at a concert",
    ]
    stage_toks = tokenizer(stage_prompts).to(device)
    stage_feats = model.encode_text(stage_toks)
    stage_feats = stage_feats / stage_feats.norm(dim=-1, keepdim=True)
    stage_emb = stage_feats.mean(0).cpu().numpy().astype(np.float32)
    stage_emb /= np.linalg.norm(stage_emb) + 1e-9

group_names = list(group_embs.keys())
group_matrix = np.stack([group_embs[n] for n in group_names], axis=0)
all_text = np.concatenate([group_matrix, dist_emb], axis=0)

# ---------- Per-photo CLIP classification ----------
log.info("Classifying with CLIP...")
t0 = time.time()
CHUNK = 10000
sims = np.zeros((emb.shape[0], all_text.shape[0]), dtype=np.float32)
for i in range(0, emb.shape[0], CHUNK):
    sims[i:i+CHUNK] = emb[i:i+CHUNK] @ all_text.T
logits = sims * LOGIT_SCALE
m = logits.max(axis=1, keepdims=True)
exp = np.exp(logits - m)
probs = exp / exp.sum(axis=1, keepdims=True)
group_probs = probs[:, :len(group_names)]
top_idx = group_probs.argmax(axis=1)
top_p = group_probs[np.arange(emb.shape[0]), top_idx]

# Coven Call stage similarity per image
stage_sim = emb @ stage_emb  # cosine in [-1, 1] but ~0..0.35

log.info(f"  CLIP classification {time.time()-t0:.1f}s")

# ---------- Assignment logic ----------
log.info("Building assignments...")
memberships = defaultdict(list)  # uuid -> list of (album, source, confidence)

for i, uid in enumerate(uuids):
    persons = persons_map.get(uid, [])
    persons_set = set(persons)
    nlabel, nconf = nsfw.get(uid, ("safe", 0.0))

    # Person-based memberships
    has_yves = YVES in persons_set
    band_members_present = persons_set & COVEN_CALL_BAND
    pets_present = persons_set & PETS
    other_named = persons_set - {YVES} - COVEN_CALL_BAND - PETS

    if has_yves:
        memberships[uid].append(("Yves", "face", 1.0))

    # Coven Call: Yves + band, OR Yves + stage CLIP (sim > 0.22)
    if has_yves and (band_members_present or stage_sim[i] > 0.22):
        memberships[uid].append(("Coven Call", "face+context", float(stage_sim[i]) if not band_members_present else 1.0))

    for pet in pets_present:
        memberships[uid].append((pet, "face", 1.0))

    if other_named:
        memberships[uid].append(("Family & Friends", "face", 1.0))

    # NSFW
    if nlabel == "explicit":
        if nconf >= CONF_MAIN:
            memberships[uid].append(("Pornografie", "nsfw", nconf))
        elif nconf >= CONF_UNSURE:
            memberships[uid].append(("Pornografie-unsure", "nsfw", nconf))
    elif nlabel == "nude":
        if nconf >= CONF_MAIN:
            memberships[uid].append(("Nacktheit", "nsfw", nconf))
        elif nconf >= CONF_UNSURE:
            memberships[uid].append(("Nacktheit-unsure", "nsfw", nconf))

    # CLIP topical (only if no strong NSFW)
    p = float(top_p[i])
    if p >= CONF_MAIN:
        memberships[uid].append((group_names[top_idx[i]], "clip", p))
    elif p >= CONF_UNSURE:
        memberships[uid].append((f"{group_names[top_idx[i]]}-unsure", "clip", p))

# Photos with zero memberships → "Diverses"
for uid in uuids:
    if not memberships[uid]:
        memberships[uid].append(("Diverses", "fallback", 0.0))

# Empty seed albums (always created, populated by user via UI)
EMPTY_SEEDS = ["Homeporn", "Meine Motorräder", "Tattoos (Yves)"]

# ---------- Pick primary per photo ----------
# Priority order for primary album (highest first)
PRIORITY = {
    "Pornografie": 100, "Pornografie-unsure": 99,
    "Nacktheit": 95, "Nacktheit-unsure": 94,
    "Coven Call": 90,
    "Bolle": 85, "Yuna": 85, "Ferro": 85,
    "Yves": 80,
    "Family & Friends": 70,
    "Aquarium": 60, "Terrarium": 60, "Garten": 60, "Astro": 60,
    "Konzerte & Festivals": 55,
    "Gitarre & Co": 55, "Kochen": 55, "Urlaub": 55,
    "Andere Motorräder": 55, "Werkstatt": 55, "Tattoo Ideen": 55,
    "Screenshots": 53,
    "Unterwegs": 50, "Zuhause": 50,
    "Diverses": 5,
}


def prio(album: str) -> int:
    if album in PRIORITY:
        return PRIORITY[album]
    if album.endswith("-unsure"):
        return PRIORITY.get(album[:-len("-unsure")], 30) - 5
    return 10


# Write outputs
log.info("Writing assignments + multi_assignments...")

primary_counts = Counter()
membership_counts = Counter()

with OUT_CSV.open("w", newline="") as fpri, MULTI_CSV.open("w", newline="") as fmul:
    wp = csv.writer(fpri)
    wp.writerow(["uuid", "album", "confidence", "source", "cluster_id"])
    wm = csv.writer(fmul)
    wm.writerow(["uuid", "album", "confidence", "source"])
    for uid in uuids:
        mems = memberships[uid]
        # write all to multi
        for a, s, c in mems:
            wm.writerow([uid, a, f"{c:.4f}", s])
            membership_counts[a] += 1
        # pick primary
        primary = max(mems, key=lambda m: (prio(m[0]), m[2]))
        wp.writerow([uid, primary[0], f"{primary[2]:.4f}", primary[1], -1])
        primary_counts[primary[0]] += 1

# Make sure empty seed albums exist in album list (server uses assignments to derive list);
# write 0 photos but include in primary CSV via a "ghost row" approach? Better: write into a separate file.
EMPTY_ALBUMS = ROOT / "metadata" / "empty_albums.txt"
EMPTY_ALBUMS.write_text("\n".join(EMPTY_SEEDS) + "\n")

# No clusters in this taxonomy
CLUSTER_META.write_text("{}")

log.info("PRIMARY album counts:")
for a, n in primary_counts.most_common():
    log.info(f"  {a:30}  {n:6}")
log.info("MULTI-membership counts:")
for a, n in membership_counts.most_common():
    log.info(f"  {a:30}  {n:6}")
log.info(f"Empty seed albums: {EMPTY_SEEDS}")
log.info("DONE.")
