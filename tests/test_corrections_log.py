"""Tests for corrections_log.py — the ONE shared replay implementation —
plus one real-code-path test per consumer that had no pinning test yet
(04_contact_sheets.py as a subprocess, 07_incremental.run_process via
importlib).

The other consumers are pinned elsewhere: server.py boot by
tests/test_server_boot.py (real process), 06_retrain.load_corrections by
tests/test_corrections.py (real module), 08_import_photos_albums.run_import
by tests/test_import_albums.py (real module). 07_incremental's second call
site (the classifier training set) needs torch + scikit-learn and real new
photos, so it is not exercised in the fast suite — it is a one-line call
into replay_excluding_skip_albums(), whose semantics are pinned directly
below.
"""
import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import corrections_log
from .conftest import REPO_ROOT, TEST_UUIDS, make_pixel_root


def _write_corrections(path: Path, rows: list[tuple]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "new_album", "timestamp", "source"])
        for row in rows:
            w.writerow(row)


# ── replay(): pure replay semantics ──────────────────────────────────────────

class TestReplay:
    def test_missing_file_is_empty_log(self, tmp_path):
        assert corrections_log.replay(tmp_path / "nope.csv") == {}

    def test_last_entry_wins(self, tmp_path):
        corr = tmp_path / "c.csv"
        _write_corrections(corr, [
            ("U1", "Garten", "t", "manual"),
            ("U1", "Kochen", "t", "manual"),
            ("U2", "Astro", "t", "manual"),
        ])
        assert corrections_log.replay(corr) == {"U1": "Kochen", "U2": "Astro"}

    def test_tombstone_removes_and_can_be_reactivated(self, tmp_path):
        corr = tmp_path / "c.csv"
        _write_corrections(corr, [
            ("U1", "Garten", "t", "manual"),
            ("U1", "", "t", "restore"),
            ("U2", "Kochen", "t", "manual"),
            ("U2", "", "t", "restore"),
            ("U2", "Urlaub", "t", "manual"),
        ])
        assert corrections_log.replay(corr) == {"U2": "Urlaub"}

    def test_replay_keeps_skip_albums(self, tmp_path):
        """Plain replay must KEEP Thrash/_test_ entries — 04_contact_sheets
        needs them to remove those photos from the album sheets."""
        corr = tmp_path / "c.csv"
        _write_corrections(corr, [("U1", "Thrash", "t", "manual")])
        assert corrections_log.replay(corr) == {"U1": "Thrash"}


# ── replay_excluding_skip_albums(): the training view ────────────────────────

class TestReplayExcludingSkipAlbums:
    def test_final_state_filter_not_row_by_row(self, tmp_path):
        """The historic divergence bug (#34, and latent in 07's classifier
        copy): a later Thrash entry must REMOVE the earlier album, not be
        skipped row-by-row leaving the stale album in training."""
        corr = tmp_path / "c.csv"
        _write_corrections(corr, [
            ("U1", "Garten", "t", "manual"),
            ("U1", "Thrash", "t", "manual"),
            ("U2", "Kochen", "t", "manual"),
        ])
        assert corrections_log.replay_excluding_skip_albums(corr) == {"U2": "Kochen"}

    def test_correction_after_thrash_is_active(self, tmp_path):
        corr = tmp_path / "c.csv"
        _write_corrections(corr, [
            ("U1", "Thrash", "t", "manual"),
            ("U1", "Kochen", "t", "manual"),
        ])
        assert corrections_log.replay_excluding_skip_albums(corr) == {"U1": "Kochen"}

    def test_tombstone_removes_entry(self, tmp_path):
        corr = tmp_path / "c.csv"
        _write_corrections(corr, [
            ("U1", "Garten", "t", "manual"),
            ("U1", "", "t", "restore"),
        ])
        assert corrections_log.replay_excluding_skip_albums(corr) == {}


# ── replay_with_manual_protection(): the Photos.app-import view ──────────────

class TestReplayWithManualProtection:
    def test_manual_sources_are_protected(self, tmp_path):
        corr = tmp_path / "c.csv"
        _write_corrections(corr, [
            ("U1", "Garten", "t", "manual"),
            ("U2", "Kochen", "t", "confirm"),
            ("U3", "Astro", "t", "similar-batch"),
            ("U4", "Urlaub", "t", "photos_app"),
        ])
        active, protected = corrections_log.replay_with_manual_protection(corr)
        assert active == {"U1": "Garten", "U2": "Kochen",
                          "U3": "Astro", "U4": "Urlaub"}
        assert protected == {"U1", "U2", "U3"}

    def test_tombstone_lifts_protection(self, tmp_path):
        corr = tmp_path / "c.csv"
        _write_corrections(corr, [
            ("U1", "Garten", "t", "manual"),
            ("U1", "", "t", "restore"),
        ])
        active, protected = corrections_log.replay_with_manual_protection(corr)
        assert active == {}
        assert protected == set()

    def test_non_manual_entry_after_manual_keeps_protection(self, tmp_path):
        """Asymmetry kept from the original 08 implementation: a manual
        correction followed by a photos_app one still protects the uuid —
        only a tombstone lifts protection."""
        corr = tmp_path / "c.csv"
        _write_corrections(corr, [
            ("U1", "Garten", "t", "manual"),
            ("U1", "Kochen", "t", "photos_app"),
        ])
        active, protected = corrections_log.replay_with_manual_protection(corr)
        assert active == {"U1": "Kochen"}
        assert protected == {"U1"}

    def test_missing_file(self, tmp_path):
        active, protected = corrections_log.replay_with_manual_protection(
            tmp_path / "nope.csv")
        assert active == {}
        assert protected == set()


# ── consumer: 04_contact_sheets.py (real subprocess) ─────────────────────────

class TestContactSheetsConsumer:
    def test_sheets_apply_replayed_corrections(self, tmp_path):
        """Run the real 04_contact_sheets.py against a sandbox whose
        corrections.csv needs actual REPLAY (last-entry-wins + tombstone +
        Thrash removal) and assert on the generated album sheets."""
        root = make_pixel_root(tmp_path / "root")
        corr = root / "metadata" / "corrections.csv"
        _write_corrections(corr, [
            # UUID[0] (Garten): moved to Kochen, then to Urlaub → last wins
            (TEST_UUIDS[0], "Kochen", "t", "manual"),
            (TEST_UUIDS[0], "Urlaub", "t", "manual"),
            # UUID[1] (Kochen): corrected away, then restored → stays Kochen
            (TEST_UUIDS[1], "Garten", "t", "manual"),
            (TEST_UUIDS[1], "", "t", "restore"),
            # UUID[2] (Diverses): trashed → removed from its sheet
            (TEST_UUIDS[2], "Thrash", "t", "manual"),
        ])

        env = {"PIXEL_ROOT": str(root), "PATH": "/usr/bin:/bin"}
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "04_contact_sheets.py")],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"04_contact_sheets.py exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        review = root / "review"
        urlaub = (review / "Urlaub.html").read_text()
        kochen = (review / "Kochen.html").read_text()
        diverses_path = review / "Diverses.html"
        diverses = diverses_path.read_text() if diverses_path.exists() else ""

        assert TEST_UUIDS[0] in urlaub, "last-entry-wins correction not applied"
        garten_path = review / "Garten.html"
        garten = garten_path.read_text() if garten_path.exists() else ""
        assert TEST_UUIDS[0] not in garten, "photo left in its old album sheet"
        assert TEST_UUIDS[1] in kochen, (
            "tombstoned correction must leave the photo in its original album"
        )
        assert TEST_UUIDS[2] not in diverses, (
            "trashed photo must be removed from its album sheet"
        )


# ── consumer: 07_incremental.run_process (real module) ───────────────────────

_spec = importlib.util.spec_from_file_location(
    "incremental_07", REPO_ROOT / "07_incremental.py"
)
incremental = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(incremental)


class TestIncrementalProcessConsumer:
    def test_manual_inbox_commit_uses_replayed_corrections(self, tmp_path, monkeypatch):
        """Call the real run_process() with a manual-only inbox (every photo
        has an active correction, so the torch/sklearn ML path is never
        entered) and assert the committed assignments come from the REPLAYED
        log: last entry wins, Thrash photos stay in the inbox."""
        root = make_pixel_root(tmp_path / "root")
        meta = root / "metadata"

        # Inbox: three photos — A manual, B re-corrected (last wins), C trashed
        uid_a, uid_b, uid_c = "INBOX-A", "INBOX-B", "INBOX-C"
        with (meta / "inbox.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["uuid", "original_filename", "date",
                        "derivative_path", "has_local_derivative"])
            for uid in (uid_a, uid_b, uid_c):
                w.writerow([uid, f"{uid}.jpg", "2024-06-01",
                            f"/fake/path/{uid}.jpg", "True"])

        _write_corrections(meta / "corrections.csv", [
            (uid_a, "Garten", "t", "manual"),
            (uid_b, "Kochen", "t", "manual"),
            (uid_b, "Urlaub", "t", "manual"),   # last entry wins
            (uid_c, "Thrash", "t", "manual"),   # waits for Empty Trash
        ])

        for attr, value in {
            "ROOT": root,
            "INV_CSV": meta / "inventory.csv",
            "INBOX_CSV": meta / "inbox.csv",
            "ASSIGN_CSV": meta / "assignments.csv",
            "CORRECTIONS_CSV": meta / "corrections.csv",
            "EMB_NPY": root / "embeddings" / "clip_vitb32.npy",
            "EMB_IDX": root / "embeddings" / "clip_vitb32_uuids.json",
        }.items():
            monkeypatch.setattr(incremental, attr, value)

        incremental.run_process(regen=False)

        with (meta / "assignments.csv").open() as f:
            assigned = {r["uuid"]: (r["album"], r["source"])
                        for r in csv.DictReader(f)}
        assert assigned[uid_a] == ("Garten", "manual_inbox")
        assert assigned[uid_b] == ("Urlaub", "manual_inbox"), (
            "run_process must commit the REPLAYED (last) correction, "
            "not the first entry"
        )
        assert uid_c not in assigned, "trashed inbox photo must not be committed"

        with (meta / "inbox.csv").open() as f:
            remaining = [r["uuid"] for r in csv.DictReader(f)]
        assert remaining == [uid_c], (
            "inbox must keep only the trashed photo (waiting for Empty Trash)"
        )
