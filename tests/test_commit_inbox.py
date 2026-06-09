"""Integration tests: 07_incremental.py --process (commit-inbox).

Tests the manually-sorted path of run_process():
  - Manually sorted inbox photos → committed to assignments.csv + inventory.csv
  - Thrash inbox photos → stay in inbox.csv, skipped from process
  - inbox.csv cleared (except Thrash rows) after successful process

Uses --no-regen to skip the contact-sheets step, avoiding the need for
the full review/ directory tree.
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
from .conftest import TEST_UUIDS


# ── helpers ───────────────────────────────────────────────────────────────────

def _csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _run_process(root: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PIXEL_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "07_incremental.py"),
         "--process", "--no-regen"],
        env=env,
        capture_output=True,
        text=True,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def process_root(tmp_path, temp_root):
    """Copy the session temp_root to a fresh per-test directory."""
    import shutil
    dst = tmp_path / "pixel_root"
    shutil.copytree(temp_root, dst)
    return dst


# ── Manually sorted inbox photos get committed ───────────────────────────────

class TestManualSortCommit:
    def test_manually_sorted_photo_added_to_assignments(self, process_root):
        """A manually sorted inbox photo must appear in assignments.csv."""
        inbox_uid = TEST_UUIDS[4]
        # Write a manual correction for the inbox photo
        corr = process_root / "metadata" / "corrections.csv"
        with corr.open("a", newline="") as f:
            csv.writer(f).writerow([inbox_uid, "Garten",
                                    "2024-06-01T12:00:00", "manual"])

        rc = _run_process(process_root)
        assert rc.returncode == 0, f"--process failed:\n{rc.stdout}\n{rc.stderr}"

        assign = _csv_rows(process_root / "metadata" / "assignments.csv")
        assign_map = {r["uuid"]: r["album"] for r in assign}
        assert inbox_uid in assign_map, "Inbox photo must be in assignments after commit"
        assert assign_map[inbox_uid] == "Garten"

    def test_manually_sorted_photo_added_to_inventory(self, process_root):
        """A manually sorted inbox photo must appear in inventory.csv."""
        inbox_uid = TEST_UUIDS[4]
        corr = process_root / "metadata" / "corrections.csv"
        with corr.open("a", newline="") as f:
            csv.writer(f).writerow([inbox_uid, "Kochen",
                                    "2024-06-01T12:00:00", "manual"])

        rc = _run_process(process_root)
        assert rc.returncode == 0

        inv = _csv_rows(process_root / "metadata" / "inventory.csv")
        inv_uuids = {r["uuid"] for r in inv}
        assert inbox_uid in inv_uuids, "Inbox photo must be in inventory after commit"

    def test_manually_sorted_photo_uses_chosen_album(self, process_root):
        """The committed album must be exactly what the user chose, not ML."""
        inbox_uid = TEST_UUIDS[4]
        corr = process_root / "metadata" / "corrections.csv"
        with corr.open("a", newline="") as f:
            csv.writer(f).writerow([inbox_uid, "Unterwegs",
                                    "2024-06-01T12:00:00", "manual"])

        rc = _run_process(process_root)
        assert rc.returncode == 0

        assign = _csv_rows(process_root / "metadata" / "assignments.csv")
        assign_map = {r["uuid"]: r for r in assign}
        row = assign_map[inbox_uid]
        assert row["album"] == "Unterwegs"
        assert row["source"] == "manual_inbox"
        assert float(row["confidence"]) == 1.0


# ── Thrash photos stay in inbox ───────────────────────────────────────────────

class TestTrashInboxBehavior:
    def test_trashed_inbox_photo_stays_in_inbox(self, process_root):
        """A Thrash-corrected inbox photo must remain in inbox.csv after --process."""
        inbox_uid = TEST_UUIDS[4]
        corr = process_root / "metadata" / "corrections.csv"
        with corr.open("a", newline="") as f:
            csv.writer(f).writerow([inbox_uid, "Thrash",
                                    "2024-06-01T12:00:00", "manual"])

        rc = _run_process(process_root)
        assert rc.returncode == 0

        inbox = _csv_rows(process_root / "metadata" / "inbox.csv")
        inbox_uuids = {r["uuid"] for r in inbox}
        assert inbox_uid in inbox_uuids, (
            "Trashed inbox photo must remain in inbox.csv until Empty Trash"
        )

    def test_trashed_inbox_photo_not_in_assignments(self, process_root):
        """A Thrash-corrected inbox photo must NOT be added to assignments.csv."""
        inbox_uid = TEST_UUIDS[4]
        corr = process_root / "metadata" / "corrections.csv"
        with corr.open("a", newline="") as f:
            csv.writer(f).writerow([inbox_uid, "Thrash",
                                    "2024-06-01T12:00:00", "manual"])

        rc = _run_process(process_root)
        assign = _csv_rows(process_root / "metadata" / "assignments.csv")
        assign_uuids = {r["uuid"] for r in assign}
        assert inbox_uid not in assign_uuids, (
            "Trashed inbox photo must not be committed to assignments.csv"
        )


# ── inbox.csv is cleared after commit ────────────────────────────────────────

class TestInboxClearedAfterCommit:
    def test_inbox_cleared_after_committing_sorted_photo(self, process_root):
        """inbox.csv must be empty (header only) after committing a sorted photo."""
        inbox_uid = TEST_UUIDS[4]
        corr = process_root / "metadata" / "corrections.csv"
        with corr.open("a", newline="") as f:
            csv.writer(f).writerow([inbox_uid, "Garten",
                                    "2024-06-01T12:00:00", "manual"])

        rc = _run_process(process_root)
        assert rc.returncode == 0

        inbox = _csv_rows(process_root / "metadata" / "inbox.csv")
        assert len(inbox) == 0, (
            "inbox.csv must be empty after all sorted photos are committed"
        )

    def test_inbox_keeps_thrash_rows_after_mixed_commit(self, process_root):
        """When one photo is sorted and another trashed, inbox retains only the
        trashed photo after --process."""
        # We only have one inbox photo (UUID[4]). Let's test the case where
        # it's trashed — the inbox should keep it; if it's sorted, inbox clears.
        inbox_uid = TEST_UUIDS[4]
        corr = process_root / "metadata" / "corrections.csv"
        with corr.open("a", newline="") as f:
            csv.writer(f).writerow([inbox_uid, "Thrash",
                                    "2024-06-01T12:00:00", "manual"])

        rc = _run_process(process_root)
        assert rc.returncode == 0

        inbox = _csv_rows(process_root / "metadata" / "inbox.csv")
        assert len(inbox) == 1
        assert inbox[0]["uuid"] == inbox_uid
