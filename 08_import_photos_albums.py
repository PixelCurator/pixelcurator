#!/usr/bin/env python3
"""Stage 8: Import album assignments from Photos.app.

Reads all photos and their album memberships from Photos.app via osxphotos.
Matches Photos.app album names (case-insensitive) against our existing albums.
Writes matching assignments as corrections (source="photos_app").
Unknown Photos.app album names are imported as new albums (unless --skip-new).

Does NOT overwrite existing manual corrections (source=manual/confirm/similar-batch).

Usage:
  venv/bin/python 08_import_photos_albums.py [--dry-run] [--skip-new] [--regen]

  --dry-run    Show what would be imported; write nothing
  --skip-new   Only process Photos.app albums that already exist in our system
  --regen      Regenerate contact sheets after writing corrections
"""
import argparse
import csv
import datetime as dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import corrections_log

ROOT = Path(os.environ.get("PIXEL_ROOT", str(Path.home() / "photo-sort")))
ASSIGN_CSV = ROOT / "metadata" / "assignments.csv"
CORRECTIONS_CSV = ROOT / "metadata" / "corrections.csv"
OSX_PYTHON = Path.home() / ".local" / "pipx" / "venvs" / "osxphotos" / "bin" / "python"
DUMP_FILE = ROOT / "metadata" / "_photos_albums_dump.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("import_albums")

# Known system/smart album names to exclude
SYSTEM_ALBUMS = {
    "All Photos", "Recents", "Last Import", "Favorites", "Recently Deleted",
    "Videos", "Selfies", "Portrait", "Live Photos", "Slow-Mo", "Time-lapse",
    "Bursts", "Screenshots", "Screen Recordings", "Imports", "Animated", "Hidden",
    "RAW+JPEG", "Long Exposure", "Loop", "Bounce", "Slo-mo",
}

# Photos.app album name → canonical album name in our system.
# Use this to merge renamed/rebranded albums into an existing one.
ALBUM_ALIASES: dict[str, str] = {
    "𝕏": "Twitter",   # rebranded from Twitter
}

# ── dump helper (runs in osxphotos venv) ──────────────────────────────────────

_DUMP_HELPER_SRC = r'''#!/usr/bin/env python3
"""Dump UUID -> album names from Photos.app. Writes JSON to metadata/_photos_albums_dump.json."""
import csv, sys, json
from pathlib import Path
import osxphotos

SYSTEM_ALBUMS = {
    "All Photos", "Recents", "Last Import", "Favorites", "Recently Deleted",
    "Videos", "Selfies", "Portrait", "Live Photos", "Slow-Mo", "Time-lapse",
    "Bursts", "Screenshots", "Screen Recordings", "Imports", "Animated", "Hidden",
    "RAW+JPEG", "Long Exposure", "Loop", "Bounce", "Slo-mo",
}

root = Path(sys.argv[1])
out_file = root / "metadata" / "_photos_albums_dump.json"

print("Loading PhotosDB…", flush=True)
db = osxphotos.PhotosDB()
all_photos = db.photos(images=True, intrash=False)
total = len(all_photos)
print(f"Total photos in library: {total}", flush=True)

uuid_albums = {}
album_counts = {}
for idx, p in enumerate(all_photos):
    albums = [a.title for a in p.album_info
              if a.title and a.title not in SYSTEM_ALBUMS]
    if albums:
        uuid_albums[p.uuid] = albums
        for a in albums:
            album_counts[a] = album_counts.get(a, 0) + 1
    if idx % 5000 == 0 and idx > 0:
        print(f"PROGRESS:{idx}:{total}", flush=True)

out_file.write_text(json.dumps(
    {"uuid_albums": uuid_albums, "album_counts": album_counts},
    separators=(",", ":"),
))
print(f"Photos with album info: {len(uuid_albums)}", flush=True)
print(f"Unique Photos.app album names: {len(album_counts)}", flush=True)
for name, count in sorted(album_counts.items(), key=lambda x: -x[1])[:25]:
    print(f"  {count:>6}  {name}", flush=True)
print(f"RESULT:{len(uuid_albums)}", flush=True)
'''


def run_dump() -> dict:
    """Run helper in osxphotos venv; return parsed dump dict."""
    log.info("=== Phase 1: Dump Photos.app album assignments ===")
    helper = ROOT / "_photos_albums_dump_helper.py"
    helper.write_text(_DUMP_HELPER_SRC)
    proc = subprocess.Popen(
        [str(OSX_PYTHON), str(helper), str(ROOT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log.info("  %s", line)
    proc.wait()
    if proc.returncode != 0:
        log.error("Dump helper failed (exit %d)", proc.returncode)
        sys.exit(proc.returncode)
    if not DUMP_FILE.exists():
        log.error("Dump file not created: %s", DUMP_FILE)
        sys.exit(1)
    data = json.loads(DUMP_FILE.read_text())
    log.info("=== Dump done: %d photos with album info ===", len(data["uuid_albums"]))
    return data


# ── import (match + write corrections) ───────────────────────────────────────

def run_import(dump: dict, dry_run: bool, skip_new: bool) -> dict:
    """Match Photos.app albums → our albums; write corrections.

    Returns the stats counters (skipped_manual / matched / new_album /
    no_match / already_correct) so the acceptance tests can assert on
    them directly instead of re-implementing the matching logic."""
    log.info("=== Phase 2: Import album assignments ===")
    uuid_albums: dict[str, list[str]] = dump["uuid_albums"]
    pa_album_counts: dict[str, int] = dump["album_counts"]

    # ── Load our current state ──────────────────────────────────────────────
    our_assigned: dict[str, str] = {}
    with ASSIGN_CSV.open() as f:
        for r in csv.DictReader(f):
            our_assigned[r["uuid"]] = r["album"]
    log.info("  Our assignments: %d photos, %d albums",
             len(our_assigned), len(set(our_assigned.values())))

    # Load corrections via the shared replay; manual ones are protected
    # (sources in corrections_log.MANUAL_SOURCES are never overridden).
    existing_corrections, manual_corrections = (
        corrections_log.replay_with_manual_protection(CORRECTIONS_CSV))
    log.info("  Manual corrections (protected): %d", len(manual_corrections))

    # ── Build album lookup ──────────────────────────────────────────────────
    all_our = set(our_assigned.values()) | set(existing_corrections.values())
    our_main = {a for a in all_our
                if not a.endswith("-unsure") and a not in {"Thrash", "_test_"}}
    our_lower: dict[str, str] = {a.lower(): a for a in our_main}
    log.info("  Our main albums: %d", len(our_main))

    # Classify Photos.app albums
    matched_pa: set[str] = set()
    new_pa: set[str] = set()
    for name in pa_album_counts:
        (matched_pa if name.lower() in our_lower else new_pa).add(name)

    log.info("  Photos.app albums matching ours: %d", len(matched_pa))
    log.info("  New Photos.app albums (unknown to us): %d", len(new_pa))
    if new_pa and not skip_new:
        sample = sorted(new_pa)[:10]
        log.info("  Will create new albums: %s%s",
                 ", ".join(sample), " …" if len(new_pa) > 10 else "")
    elif new_pa and skip_new:
        log.info("  --skip-new: ignoring unknown Photos.app albums")

    # ── Process photos ──────────────────────────────────────────────────────
    to_write: list[tuple[str, str]] = []
    stats = dict(skipped_manual=0, matched=0, new_album=0, no_match=0, already_correct=0)

    for uuid, pa_albums in uuid_albums.items():
        if uuid in manual_corrections:
            stats["skipped_manual"] += 1
            continue

        # Find first matching album (direct match or alias)
        target: str | None = None
        is_new = False
        for pa_name in pa_albums:
            # 1. Direct case-insensitive match
            canonical = our_lower.get(pa_name.lower())
            if canonical:
                target = canonical
                break
            # 2. Alias match (e.g. 𝕏 → Twitter)
            alias_target = ALBUM_ALIASES.get(pa_name)
            if alias_target:
                canonical = our_lower.get(alias_target.lower())
                if canonical:
                    target = canonical
                    break
        if target is None and not skip_new:
            for pa_name in pa_albums:
                if pa_name not in SYSTEM_ALBUMS:
                    # Apply alias even for new-album creation
                    target = ALBUM_ALIASES.get(pa_name, pa_name)
                    is_new = target not in our_main
                    break

        if target is None:
            stats["no_match"] += 1
            continue

        # Skip if already assigned to the same album (modulo -unsure)
        current = existing_corrections.get(uuid, our_assigned.get(uuid, ""))
        current_main = current.removesuffix("-unsure") if current.endswith("-unsure") else current
        if current_main.lower() == target.lower():
            stats["already_correct"] += 1
            continue

        stats["new_album" if is_new else "matched"] += 1
        to_write.append((uuid, target))

    log.info("  ─── Summary ───────────────────────────────────────────")
    log.info("  %8d  Photos.app photos had album info", len(uuid_albums))
    log.info("  %8d  skipped (manual corrections protected)", stats["skipped_manual"])
    log.info("  %8d  already correctly assigned (no change needed)", stats["already_correct"])
    log.info("  %8d  → matched existing albums", stats["matched"])
    log.info("  %8d  → new albums to create", stats["new_album"])
    log.info("  %8d  no Photos.app album matched", stats["no_match"])
    log.info("  %8d  corrections to write", len(to_write))
    log.info("  ───────────────────────────────────────────────────────")

    if dry_run:
        log.info("=== DRY RUN — nothing written ===")
        for uuid, album in to_write[:20]:
            old = our_assigned.get(uuid, "(unknown)")
            log.info("  %s…  %s → %s", uuid[:8], old, album)
        if len(to_write) > 20:
            log.info("  … and %d more", len(to_write) - 20)
        return stats

    if not to_write:
        log.info("Nothing to write — all Photos.app albums already match our assignments.")
        return stats

    ts = dt.datetime.now(dt.UTC).isoformat()
    with CORRECTIONS_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        for uuid, album in to_write:
            w.writerow([uuid, album, ts, "photos_app"])
    log.info("Wrote %d corrections to %s", len(to_write), CORRECTIONS_CSV)
    log.info("=== Import complete ===")
    return stats


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview what would be imported; write nothing")
    ap.add_argument("--skip-new", action="store_true",
                    help="Only import albums already present in our system")
    ap.add_argument("--regen", action="store_true",
                    help="Regenerate contact sheets after import")
    args = ap.parse_args()

    dump = run_dump()
    run_import(dump, dry_run=args.dry_run, skip_new=args.skip_new)

    if args.regen and not args.dry_run:
        log.info("Regenerating contact sheets…")
        subprocess.run([sys.executable, str(ROOT / "04_contact_sheets.py")], check=True)
        log.info("Done.")


if __name__ == "__main__":
    main()
