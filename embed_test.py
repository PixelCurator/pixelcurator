#!/usr/bin/env python3
"""Mini test: embed 200 random photos with CLIP, time it, also do zero-shot on seed categories."""
import csv
import random
import time
from pathlib import Path

import torch
import open_clip
from PIL import Image
import numpy as np

CSV_PATH = Path.home() / "photo-sort" / "metadata" / "inventory.csv"
N_SAMPLE = 200
BATCH_SIZE = 32

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

print("Loading CLIP ViT-B-32 (OpenAI)...")
t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()
print(f"  loaded in {time.time()-t0:.1f}s")

print(f"Reading inventory...")
with open(CSV_PATH) as f:
    rows = [r for r in csv.DictReader(f) if r["has_local_derivative"] == "True"]
random.seed(42)
sample = random.sample(rows, N_SAMPLE)
paths = [r["derivative_path"] for r in sample]

# Seed categories — zero-shot text prompts
prompts = {
    "motorcycle": "a photo of a motorcycle, motorcycles, or motorcycle parts",
    "workshop": "a photo of a workshop, mechanic tools, or vehicle repair",
    "aquarium": "a photo of an aquarium with fish or aquatic plants",
    "garden": "a photo of a garden with plants, flowers, or outdoor greenery",
    "nudity": "a photo of nude or unclothed people, nudity",
    "porn": "explicit sexual content, pornography",
    "people": "a photo of one or more people",
    "food": "a photo of food or a meal",
    "screenshot": "a screenshot of a phone or computer screen",
    "document": "a photo of a document, paper, or receipt",
}

print(f"\nEncoding {len(prompts)} text prompts...")
with torch.no_grad():
    text_tokens = tokenizer(list(prompts.values())).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

print(f"\nEmbedding {N_SAMPLE} images in batches of {BATCH_SIZE}...")
t0 = time.time()
all_feats = []
failed = 0
loaded = 0
for i in range(0, len(paths), BATCH_SIZE):
    batch_paths = paths[i:i+BATCH_SIZE]
    imgs = []
    for p in batch_paths:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        except Exception as e:
            failed += 1
    if not imgs:
        continue
    loaded += len(imgs)
    batch = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    all_feats.append(feats.cpu().numpy())

embeddings = np.concatenate(all_feats, axis=0)
elapsed = time.time() - t0
print(f"  embedded {loaded} images in {elapsed:.1f}s = {loaded/elapsed:.1f} img/s")
print(f"  failed: {failed}")
print(f"  embedding shape: {embeddings.shape}, dtype: {embeddings.dtype}")

# Zero-shot classification — sim matrix
print(f"\nTop-3 categories per image (sample of 10):")
sims = embeddings @ text_feats.cpu().numpy().T
labels = list(prompts.keys())
for i in random.sample(range(loaded), min(10, loaded)):
    top = np.argsort(-sims[i])[:3]
    print(f"  {Path(paths[i]).name[:30]:32}  " + ", ".join(f"{labels[j]}({sims[i,j]:.2f})" for j in top))

# Distribution per category (threshold = top-1)
print(f"\nDominant category counts (top-1 per image, n={loaded}):")
top1 = sims.argmax(axis=1)
from collections import Counter
for cat, n in Counter([labels[j] for j in top1]).most_common():
    print(f"  {cat:15}  {n:4}  ({n*100/loaded:.1f}%)")

# Project total time
n_total = 81555
projected_sec = elapsed * n_total / loaded
print(f"\nProjection for {n_total} photos: ~{projected_sec/60:.0f} min ({projected_sec/3600:.1f} h)")
