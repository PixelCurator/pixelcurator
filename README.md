# PixelCurator

<!-- OpenSSF Scorecard badge — the api.scorecard.dev badge only resolves for
     PUBLIC repos. Uncomment this once the repo is made public (and flip
     publish_results to true in .github/workflows/scorecard.yml):
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/PixelCurator/pixelcurator/badge)](https://scorecard.dev/viewer/?uri=github.com/PixelCurator/pixelcurator)
-->

On-device AI photo library organizer for macOS Photos.app. Classifies tens of thousands of photos into user-defined albums using CLIP + NudeNet + Apple's face recognition, all running locally — no cloud uploads.

## Status

**Prototype** (Python + JS, runs in browser via local HTTP server). Production target: native macOS app (SwiftUI + embedded Python + WKWebView).

## What it does

1. **Inventory** the Photos library via [osxphotos](https://github.com/RhetTbull/osxphotos)
2. **Embed** every photo with OpenAI CLIP ViT-B-32 (QuickGELU config, Apple Silicon MPS)
3. **NSFW-score** every photo via NudeNet (ONNX, on-device)
4. **Extract** Apple's face recognition (people + pets) from the Photos DB
5. **Classify** against user's album taxonomy:
   - Person-based (Yves, family, pets, band) — exact match from face data
   - NSFW (Pornografie, Nacktheit) — NudeNet probabilities
   - CLIP zero-shot (Aquarium, Konzerte & Festivals, Werkstatt, …) — user-defined names with English prompts internally
6. **Review UI** — browser-based contact sheets with click-to-reassign modal + "find similar" via CLIP cosine
7. **Write back** to Photos.app via `photoscript` (creates albums, never modifies originals)

Confidence convention: ≥80% → main album, 50–80% → `-unsure` album, <50% → catch-all `Diverses`.

## Layout

```
PixelCurator/
├── inventory.py             # Photos.app DB scan → inventory.csv
├── 01_embed.py              # CLIP embeddings → embeddings/clip_vitb32.npy
├── 02_nsfw.py               # NudeNet → metadata/nsfw_scores.csv
├── 03_classify_cluster.py   # Legacy seed+HDBSCAN/KMeans (kept for reference)
├── 06_taxonomy.py           # Current classifier: persons + CLIP + NSFW
├── 04_contact_sheets.py     # HTML+JS contact sheets with click modal
├── 05_write_albums.py       # photoscript helper to write Photos.app albums
├── server.py                # Local HTTP server: sheets, images, reassign API
├── run_pipeline.sh          # Orchestrator: embed → NSFW → classify → sheets
├── embeddings/              # CLIP numpy arrays + UUID index (~167 MB)
├── metadata/                # CSV/JSON state (inventory, NSFW, assignments, corrections)
├── review/                  # Generated HTML contact sheets
└── logs/                    # All run logs per stage
```

## Run from scratch

```bash
cd PixelCurator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# pipx + osxphotos for Photos.app access (needs Full Disk Access)
pipx install osxphotos
pipx inject osxphotos photoscript

# stage by stage
python inventory.py                # ~1 min
python 01_embed.py                 # ~25 min (Apple Silicon MPS)
python 02_nsfw.py                  # ~40 min (CPU)
python 06_taxonomy.py              # ~3 min
python 04_contact_sheets.py        # <30 sec

# launch review UI
python server.py
# open http://127.0.0.1:8765/
```

## Requirements

- macOS 14+ (Sonoma) for the modern Photos DB schema
- Apple Silicon strongly recommended (Intel works but 5–10× slower for embeddings)
- **Full Disk Access** granted to the terminal / Claude.app launching the scripts
- Photos.app library local (iCloud derivatives are sufficient — full originals not required)
- ~200–300 MB free disk for models + embeddings

## Architecture notes

Productization roadmap (decisions on file):
- Decision: native macOS app (not web — iCloud doesn't expose OAuth for Photos)
- Hybrid migration path: SwiftUI shell + embedded Python + WKWebView for the existing review UI
- Monetization: €39 one-time + free tier up to ~2 000 photos, no time trial
- Distribution: notarized DMG outside Mac App Store

## Origin

Built during a sorting session on a personal library of 81 559 photos (June 2026). The prototype lives in `~/photo-sort/` as the active working copy; this directory is the canonical archived copy for the productization effort.
