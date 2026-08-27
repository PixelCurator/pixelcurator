"""The ONE replay implementation for metadata/corrections.csv.

corrections.csv is an append-only event log: one row per user decision,
where the LAST row per uuid wins and an empty new_album is a tombstone
(written by empty-thrash / restore / undo-to-inbox) that removes any
earlier correction for that uuid. Replay order is file order — the
timestamp column is informational only.

This replay loop used to be hand-copied across five consumers (server.py
boot + test-reset, 04_contact_sheets.py, 06_retrain.py,
07_incremental.py twice, 08_import_photos_albums.py). One of the copies
had silently diverged — 06_retrain filtered row-by-row instead of
replaying, so tombstones and later Thrash corrections did not remove
earlier entries and retrain resurrected albums the user had explicitly
un-corrected (#34). The same drift was latent in 07_incremental's
classifier copy until it was routed through this module. All consumers now
import from here; per-consumer post-filters live below as explicit,
individually tested helpers.

Deliberately dependency-free (csv + pathlib only) so the fast CI job stays
lean.
"""
import csv
from pathlib import Path

# Albums excluded from ML training: Thrash entries wait for Empty Trash,
# _test_ is the manual-QA album. (Exclusion from TRAINING only — plain
# replay keeps them, e.g. 04_contact_sheets needs the Thrash entries to
# remove those photos from the album sheets.)
SKIP_ALBUMS = {"Thrash", "_test_"}

# Correction sources that count as explicit user decisions and must never be
# overwritten by a bulk import (08_import_photos_albums).
MANUAL_SOURCES = {"manual", "confirm", "similar-batch"}


def replay(path: Path) -> dict[str, str]:
    """uuid -> currently corrected album. Pure replay, no post-filtering.

    Last entry per uuid wins; an entry with an empty new_album is a
    tombstone removing any earlier correction. A missing file replays to
    an empty log.
    """
    result: dict[str, str] = {}
    if not path.exists():
        return result
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("new_album"):
                result[r["uuid"]] = r["new_album"]
            else:
                result.pop(r["uuid"], None)  # tombstone
    return result


def replay_excluding_skip_albums(path: Path) -> dict[str, str]:
    """replay(), then drop uuids whose FINAL album is in SKIP_ALBUMS.

    This is the training view (06_retrain, 07_incremental's classifier):
    the filter applies to the final replayed state, NOT row-by-row — a
    later Thrash correction must remove the earlier album entry, and a
    correction after a Thrash entry is active again.
    """
    return {u: a for u, a in replay(path).items() if a not in SKIP_ALBUMS}


def replay_with_manual_protection(path: Path) -> tuple[dict[str, str], set[str]]:
    """Returns (active, protected) for the Photos.app import (08).

    active: same as replay(path).
    protected: uuids that have at least one entry from a MANUAL_SOURCES
    source and have not been tombstoned since. Note the asymmetry, kept
    from the original 08 implementation: a manual correction followed by a
    non-manual one (e.g. photos_app) still protects the uuid — only a
    tombstone lifts the protection.
    """
    active: dict[str, str] = {}
    protected: set[str] = set()
    if not path.exists():
        return active, protected
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("new_album"):
                active[r["uuid"]] = r["new_album"]
                if r.get("source") in MANUAL_SOURCES:
                    protected.add(r["uuid"])
            else:
                active.pop(r["uuid"], None)
                protected.discard(r["uuid"])
    return active, protected
