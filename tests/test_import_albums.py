"""Unit tests: 08_import_photos_albums.py business logic.

Tests for alias resolution, manual-correction protection,
already-correct skipping, and new-album creation.
"""
import csv
import datetime as dt
from pathlib import Path


# ── Replicate run_import() logic for isolated testing ─────────────────────────

SYSTEM_ALBUMS = {
    "All Photos", "Recents", "Last Import", "Favorites", "Recently Deleted",
    "Videos", "Selfies", "Portrait", "Live Photos", "Slow-Mo", "Time-lapse",
    "Bursts", "Screenshots", "Screen Recordings", "Imports", "Animated", "Hidden",
}

ALBUM_ALIASES = {"𝕏": "Twitter"}


def simulate_run_import(
    uuid_albums: dict[str, list[str]],
    our_assignments: dict[str, str],
    existing_corrections: dict[str, str],
    manual_uuids: set[str],
    skip_new: bool = False,
) -> tuple[list[tuple[str, str]], dict]:
    """
    Simulate 08_import_photos_albums.run_import() with in-memory data.
    Returns (to_write, stats).
    """
    all_our = set(our_assignments.values()) | set(existing_corrections.values())
    our_main = {a for a in all_our
                if not a.endswith("-unsure") and a not in {"Thrash", "_test_"}}
    our_lower: dict[str, str] = {a.lower(): a for a in our_main}

    to_write: list[tuple[str, str]] = []
    stats = dict(skipped_manual=0, matched=0, new_album=0,
                 no_match=0, already_correct=0)

    for uuid, pa_albums in uuid_albums.items():
        if uuid in manual_uuids:
            stats["skipped_manual"] += 1
            continue

        target: str | None = None
        is_new = False

        for pa_name in pa_albums:
            # Direct match
            canonical = our_lower.get(pa_name.lower())
            if canonical:
                target = canonical
                break
            # Alias match
            alias_target = ALBUM_ALIASES.get(pa_name)
            if alias_target:
                canonical = our_lower.get(alias_target.lower())
                if canonical:
                    target = canonical
                    break

        if target is None and not skip_new:
            for pa_name in pa_albums:
                if pa_name not in SYSTEM_ALBUMS:
                    target = ALBUM_ALIASES.get(pa_name, pa_name)
                    is_new = target not in our_main
                    break

        if target is None:
            stats["no_match"] += 1
            continue

        current = existing_corrections.get(uuid, our_assignments.get(uuid, ""))
        current_main = (current.removesuffix("-unsure")
                        if current.endswith("-unsure") else current)
        if current_main.lower() == target.lower():
            stats["already_correct"] += 1
            continue

        stats["new_album" if is_new else "matched"] += 1
        to_write.append((uuid, target))

    return to_write, stats


# ── Alias resolution (𝕏 → Twitter) ───────────────────────────────────────────

class TestAliasResolution:
    def test_x_merged_into_existing_twitter(self):
        """𝕏 photos must be routed to the existing Twitter album."""
        uuid_albums = {
            "UUID-X": ["𝕏"],
            "UUID-TW": ["Twitter"],
        }
        our_assignments = {
            "UUID-EXISTING": "Twitter",  # Twitter exists in our system
        }
        to_write, stats = simulate_run_import(
            uuid_albums, our_assignments, {}, set()
        )
        targets = dict(to_write)
        assert targets.get("UUID-X") == "Twitter"
        assert targets.get("UUID-TW") == "Twitter"

    def test_x_becomes_twitter_when_twitter_is_new(self):
        """When Twitter doesn't yet exist, 𝕏 creates it as 'Twitter' (not '𝕏')."""
        uuid_albums = {"UUID-X": ["𝕏"]}
        our_assignments = {}  # no Twitter yet
        to_write, stats = simulate_run_import(
            uuid_albums, our_assignments, {}, set()
        )
        assert len(to_write) == 1
        assert to_write[0] == ("UUID-X", "Twitter")

    def test_x_and_twitter_create_single_album(self):
        """Both 𝕏 and Twitter photos end up in the same 'Twitter' album."""
        uuid_albums = {
            "UUID-X1": ["𝕏"],
            "UUID-X2": ["𝕏"],
            "UUID-TW1": ["Twitter"],
        }
        our_assignments = {}
        to_write, _ = simulate_run_import(uuid_albums, our_assignments, {}, set())
        targets = {target for _, target in to_write}
        assert targets == {"Twitter"}, "Both 𝕏 and Twitter should merge into 'Twitter'"


# ── Manual correction protection ─────────────────────────────────────────────

class TestManualProtection:
    def test_manual_correction_not_overwritten(self):
        """UUIDs with manual corrections must be skipped."""
        uuid_albums = {"UUID-MANUAL": ["Garten"]}
        our_assignments = {"UUID-MANUAL": "Kochen"}
        manual_uuids = {"UUID-MANUAL"}
        to_write, stats = simulate_run_import(
            uuid_albums, our_assignments, {}, manual_uuids
        )
        assert len(to_write) == 0
        assert stats["skipped_manual"] == 1

    def test_non_manual_uuid_is_not_protected(self):
        """UUIDs without manual corrections may be updated."""
        uuid_albums = {"UUID-FREE": ["Garten"]}
        our_assignments = {
            "UUID-FREE": "Diverses",
            "UUID-SEED": "Garten",  # so Garten exists in our_main
        }
        to_write, _ = simulate_run_import(uuid_albums, our_assignments, {}, set())
        assert len(to_write) == 1
        assert to_write[0][0] == "UUID-FREE"


# ── Already-correct skipping ─────────────────────────────────────────────────

class TestAlreadyCorrect:
    def test_already_in_correct_album_skipped(self):
        """Photos already in the right album must not produce a correction."""
        uuid_albums = {"UUID-OK": ["Garten"]}
        our_assignments = {
            "UUID-OK": "Garten",
            "UUID-SEED": "Garten",
        }
        to_write, stats = simulate_run_import(
            uuid_albums, our_assignments, {}, set()
        )
        assert len(to_write) == 0
        assert stats["already_correct"] == 1

    def test_unsure_variant_counts_as_correct(self):
        """A photo in 'Garten-unsure' is already in the right main album."""
        uuid_albums = {"UUID-UNSURE": ["Garten"]}
        our_assignments = {
            "UUID-UNSURE": "Garten-unsure",
            "UUID-SEED": "Garten",
        }
        to_write, stats = simulate_run_import(
            uuid_albums, our_assignments, {}, set()
        )
        assert len(to_write) == 0
        assert stats["already_correct"] == 1

    def test_wrong_album_generates_correction(self):
        """A photo in the wrong album must generate a correction."""
        uuid_albums = {"UUID-WRONG": ["Garten"]}
        our_assignments = {
            "UUID-WRONG": "Kochen",
            "UUID-SEED": "Garten",
        }
        to_write, stats = simulate_run_import(
            uuid_albums, our_assignments, {}, set()
        )
        assert len(to_write) == 1
        assert to_write[0] == ("UUID-WRONG", "Garten")


# ── skip-new flag ─────────────────────────────────────────────────────────────

class TestSkipNew:
    def test_skip_new_ignores_unknown_albums(self):
        """With skip_new=True, only photos in known albums are imported."""
        uuid_albums = {
            "UUID-KNOWN": ["Garten"],
            "UUID-NEW": ["BrandNewAlbum"],
        }
        our_assignments = {
            "UUID-SEED": "Garten",
        }
        to_write, stats = simulate_run_import(
            uuid_albums, our_assignments, {}, set(), skip_new=True
        )
        uuids_written = {u for u, _ in to_write}
        assert "UUID-NEW" not in uuids_written
        assert stats["no_match"] == 1

    def test_skip_new_false_creates_new_albums(self):
        """With skip_new=False (default), photos in new albums are imported."""
        uuid_albums = {"UUID-NEW": ["BrandNewAlbum"]}
        our_assignments = {}
        to_write, stats = simulate_run_import(
            uuid_albums, our_assignments, {}, set(), skip_new=False
        )
        assert len(to_write) == 1
        assert to_write[0] == ("UUID-NEW", "BrandNewAlbum")
        assert stats["new_album"] == 1


# ── System albums excluded ───────────────────────────────────────────────────

class TestSystemAlbumsExcluded:
    def test_system_album_only_photo_gets_no_match(self):
        """Photos only in system albums (e.g. 'Favorites') must be skipped."""
        uuid_albums = {"UUID-FAV": ["Favorites"]}
        to_write, stats = simulate_run_import(
            uuid_albums, {}, {}, set(), skip_new=False
        )
        assert len(to_write) == 0
        assert stats["no_match"] == 1
