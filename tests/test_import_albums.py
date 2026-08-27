"""Acceptance tests: 08_import_photos_albums.py run_import(), against
PRODUCTION code.

An earlier version of this file re-implemented the matching/alias/protection
logic in a local `simulate_run_import()` (plus its own copies of
SYSTEM_ALBUMS and ALBUM_ALIASES) and asserted against the replica --
mutating the alias lookup in production (`ALBUM_ALIASES.get(pa_name)` ->
`None`) left all 11 tests green. This was the third instance of the replica
pattern; see tests/test_retrain.py (#33) and tests/test_corrections.py (#34)
for the other two, fixed the same way.

These tests build a file sandbox (assignments.csv + corrections.csv), feed a
synthetic Photos.app dump into the real `run_import(dry_run=False)`, and
assert on the corrections.csv rows it writes plus the stats counters it
returns. 08_import_photos_albums.py is not a valid module identifier
(leading digit), so it is loaded via importlib from its file path -- safe to
import: the osxphotos code lives in a heredoc for a subprocess, and
run_import() only touches the PIXEL_ROOT-derived CSV paths monkeypatched
below.
"""
import csv
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "import_albums_08", REPO_ROOT / "08_import_photos_albums.py"
)
import_albums = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(import_albums)


def run_real_import(
    tmp_path,
    monkeypatch,
    uuid_albums: dict[str, list[str]],
    our_assignments: dict[str, str],
    existing_corrections: list[tuple[str, str, str]] = (),
    skip_new: bool = False,
    dry_run: bool = False,
) -> tuple[list[tuple[str, str]], dict]:
    """Run the REAL run_import() against a file sandbox.

    existing_corrections: (uuid, new_album, source) rows seeded into
    corrections.csv before the run -- source 'manual'/'confirm'/
    'similar-batch' drives the manual-protection logic exactly as in
    production.

    Returns (written, stats): `written` are the (uuid, new_album) rows
    run_import appended to corrections.csv (all verified to carry
    source='photos_app'), `stats` is the counter dict it returns.
    """
    assign_csv = tmp_path / "assignments.csv"
    corr_csv = tmp_path / "corrections.csv"

    with assign_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "album", "confidence", "source"])
        for uid, album in our_assignments.items():
            w.writerow([uid, album, "0.9000", "retrain"])

    with corr_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "new_album", "timestamp", "source"])
        for uid, album, source in existing_corrections:
            w.writerow([uid, album, "2024-01-01T00:00:00", source])

    monkeypatch.setattr(import_albums, "ASSIGN_CSV", assign_csv)
    monkeypatch.setattr(import_albums, "CORRECTIONS_CSV", corr_csv)

    album_counts: dict[str, int] = {}
    for albums in uuid_albums.values():
        for a in albums:
            album_counts[a] = album_counts.get(a, 0) + 1
    dump = {"uuid_albums": uuid_albums, "album_counts": album_counts}

    stats = import_albums.run_import(dump, dry_run=dry_run, skip_new=skip_new)

    with corr_csv.open() as f:
        rows = list(csv.DictReader(f))
    appended = rows[len(existing_corrections):]
    for r in appended:
        assert r["source"] == "photos_app", (
            f"run_import must write source='photos_app', got {r['source']!r}"
        )
    written = [(r["uuid"], r["new_album"]) for r in appended]
    return written, stats


# ── Alias resolution (𝕏 → Twitter) ───────────────────────────────────────────

class TestAliasResolution:
    def test_x_merged_into_existing_twitter(self, tmp_path, monkeypatch):
        """𝕏 photos must be routed to the existing Twitter album."""
        written, _ = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-X": ["𝕏"], "UUID-TW": ["Twitter"]},
            our_assignments={"UUID-EXISTING": "Twitter"},
        )
        targets = dict(written)
        assert targets.get("UUID-X") == "Twitter"
        assert targets.get("UUID-TW") == "Twitter"

    def test_x_becomes_twitter_when_twitter_is_new(self, tmp_path, monkeypatch):
        """When Twitter doesn't yet exist, 𝕏 creates it as 'Twitter' (not '𝕏')."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-X": ["𝕏"]},
            our_assignments={},
        )
        assert written == [("UUID-X", "Twitter")]
        assert stats["new_album"] == 1

    def test_alias_resolved_even_with_skip_new(self, tmp_path, monkeypatch):
        """--skip-new must still merge 𝕏 into the EXISTING Twitter album.

        This is the test that goes red under the documented mutation
        (`alias_target = ALBUM_ALIASES.get(pa_name)` -> `None` in
        08_import_photos_albums.py): with skip_new=True the new-album
        fallback (which applies the alias a second time and therefore masks
        the mutation in the other alias tests) never runs, so killing the
        primary alias lookup turns this photo into a no_match."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-X": ["𝕏"]},
            our_assignments={"UUID-EXISTING": "Twitter"},
            skip_new=True,
        )
        assert written == [("UUID-X", "Twitter")]
        assert stats["no_match"] == 0

    def test_x_and_twitter_create_single_album(self, tmp_path, monkeypatch):
        """Both 𝕏 and Twitter photos end up in the same 'Twitter' album."""
        written, _ = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={
                "UUID-X1": ["𝕏"],
                "UUID-X2": ["𝕏"],
                "UUID-TW1": ["Twitter"],
            },
            our_assignments={},
        )
        targets = {target for _, target in written}
        assert targets == {"Twitter"}, (
            "Both 𝕏 and Twitter should merge into 'Twitter'"
        )


# ── Manual correction protection ─────────────────────────────────────────────

class TestManualProtection:
    def test_manual_correction_not_overwritten(self, tmp_path, monkeypatch):
        """UUIDs with manual corrections must be skipped."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-MANUAL": ["Garten"]},
            our_assignments={"UUID-SEED": "Garten"},
            existing_corrections=[("UUID-MANUAL", "Kochen", "manual")],
        )
        assert written == []
        assert stats["skipped_manual"] == 1

    def test_tombstoned_manual_correction_is_not_protected(self, tmp_path, monkeypatch):
        """A manual correction later removed by a tombstone (empty new_album)
        no longer protects the uuid -- replay semantics, not row-by-row."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-RESTORED": ["Garten"]},
            our_assignments={"UUID-SEED": "Garten"},
            existing_corrections=[
                ("UUID-RESTORED", "Kochen", "manual"),
                ("UUID-RESTORED", "", "restore"),
            ],
        )
        assert written == [("UUID-RESTORED", "Garten")]
        assert stats["skipped_manual"] == 0

    def test_non_manual_uuid_is_not_protected(self, tmp_path, monkeypatch):
        """UUIDs without manual corrections may be updated."""
        written, _ = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-FREE": ["Garten"]},
            our_assignments={"UUID-FREE": "Diverses", "UUID-SEED": "Garten"},
        )
        assert written == [("UUID-FREE", "Garten")]


# ── Already-correct skipping ─────────────────────────────────────────────────

class TestAlreadyCorrect:
    def test_already_in_correct_album_skipped(self, tmp_path, monkeypatch):
        """Photos already in the right album must not produce a correction."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-OK": ["Garten"]},
            our_assignments={"UUID-OK": "Garten", "UUID-SEED": "Garten"},
        )
        assert written == []
        assert stats["already_correct"] == 1

    def test_unsure_variant_counts_as_correct(self, tmp_path, monkeypatch):
        """A photo in 'Garten-unsure' is already in the right main album."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-UNSURE": ["Garten"]},
            our_assignments={"UUID-UNSURE": "Garten-unsure", "UUID-SEED": "Garten"},
        )
        assert written == []
        assert stats["already_correct"] == 1

    def test_wrong_album_generates_correction(self, tmp_path, monkeypatch):
        """A photo in the wrong album must generate a correction."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-WRONG": ["Garten"]},
            our_assignments={"UUID-WRONG": "Kochen", "UUID-SEED": "Garten"},
        )
        assert written == [("UUID-WRONG", "Garten")]
        assert stats["matched"] == 1


# ── skip-new flag ─────────────────────────────────────────────────────────────

class TestSkipNew:
    def test_skip_new_ignores_unknown_albums(self, tmp_path, monkeypatch):
        """With skip_new=True, only photos in known albums are imported."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={
                "UUID-KNOWN": ["Garten"],
                "UUID-NEW": ["BrandNewAlbum"],
            },
            our_assignments={"UUID-SEED": "Garten"},
            skip_new=True,
        )
        uuids_written = {u for u, _ in written}
        assert "UUID-NEW" not in uuids_written
        assert stats["no_match"] == 1

    def test_skip_new_false_creates_new_albums(self, tmp_path, monkeypatch):
        """With skip_new=False (default), photos in new albums are imported."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-NEW": ["BrandNewAlbum"]},
            our_assignments={},
            skip_new=False,
        )
        assert written == [("UUID-NEW", "BrandNewAlbum")]
        assert stats["new_album"] == 1


# ── System albums excluded ───────────────────────────────────────────────────

class TestSystemAlbumsExcluded:
    def test_system_album_only_photo_gets_no_match(self, tmp_path, monkeypatch):
        """Photos only in system albums (e.g. 'Favorites') must be skipped."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-FAV": ["Favorites"]},
            our_assignments={},
            skip_new=False,
        )
        assert written == []
        assert stats["no_match"] == 1


# ── dry-run writes nothing ───────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_appends_no_rows(self, tmp_path, monkeypatch):
        """--dry-run must leave corrections.csv untouched while still
        reporting what it would have written."""
        written, stats = run_real_import(
            tmp_path, monkeypatch,
            uuid_albums={"UUID-WRONG": ["Garten"]},
            our_assignments={"UUID-WRONG": "Kochen", "UUID-SEED": "Garten"},
            dry_run=True,
        )
        assert written == []
        assert stats["matched"] == 1
