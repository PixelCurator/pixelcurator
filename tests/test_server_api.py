"""Integration tests: PixelCurator server HTTP API.

All tests run against an ephemeral test server on port 8766, backed by
a temp directory with synthetic data (see conftest.py). The live server
on port 8765 is never touched.

Scenarios covered:
  1.  /api/albums — lists albums from assignments
  2.  /api/current — shows current album for a photo
  3.  /api/corrections — initially empty
  4.  Reassign a photo → correction written, undo available
  5.  Reassign to Thrash → NOT counted in retrain-status
  6.  /api/undo-status → can_undo after reassign
  7.  Undo a reassign → photo reverts to original album
  8.  Redo after undo → photo returns to new album
  9.  Undo for inbox photo (no prior album) → tombstone written
  10. /api/retrain-status → 0 new after all corrections were retrained
  11. /api/retrain-status → new_since_retrain > 0 after new correction
  12. /api/inbox-count → unsorted count excludes Thrash
  13. /api/restore → removes Thrash correction for a photo
  14. Retrain-status new_since_retrain excludes Thrash corrections
"""
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from .conftest import (
    TEST_UUIDS,
    reset_corrections,
    reset_server_state,
    reset_inbox,
)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _post(url: str, body: dict) -> dict:
    """POST JSON; returns parsed body for both 2xx and 4xx/5xx responses."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        # Parse JSON error body if available
        try:
            return json.loads(exc.read())
        except Exception:
            return {"_http_error": exc.code, "msg": str(exc)}


# ── Basic read endpoints ──────────────────────────────────────────────────────

class TestReadEndpoints:
    def test_albums_endpoint(self, server_url):
        albums = _get(f"{server_url}/api/albums")
        assert isinstance(albums, list)
        assert "Garten" in albums
        assert "Kochen" in albums

    def test_current_endpoint_known_photo(self, server_url):
        result = _get(f"{server_url}/api/current?uuid={TEST_UUIDS[0]}")
        assert result["uuid"] == TEST_UUIDS[0]
        assert result["album"] == "Garten"
        assert result["corrected"] is False

    def test_corrections_initially_empty(self, server_url, temp_root):
        reset_corrections(temp_root)
        reset_server_state(server_url)
        result = _get(f"{server_url}/api/corrections")
        for uid in TEST_UUIDS:
            assert uid not in result

    def test_undo_status_initially_no_undo(self, server_url, temp_root):
        reset_corrections(temp_root)
        reset_server_state(server_url)
        result = _get(f"{server_url}/api/undo-status")
        assert result["can_undo"] is False
        assert result["can_redo"] is False


# ── Reassign flow ─────────────────────────────────────────────────────────────

class TestReassign:
    def test_reassign_writes_correction(self, server_url, temp_root):
        """After reassign, /api/current reflects the new album."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        uid = TEST_UUIDS[0]  # was Garten

        result = _post(f"{server_url}/api/reassign",
                       {"uuid": uid, "new_album": "Kochen"})
        assert result["ok"] is True
        assert result["count"] == 1

        current = _get(f"{server_url}/api/current?uuid={uid}")
        assert current["album"] == "Kochen"
        assert current["corrected"] is True

    def test_reassign_makes_undo_available(self, server_url, temp_root):
        """After reassign, can_undo must be True."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        _post(f"{server_url}/api/reassign",
              {"uuid": TEST_UUIDS[1], "new_album": "Garten"})
        status = _get(f"{server_url}/api/undo-status")
        assert status["can_undo"] is True

    def test_reassign_clears_redo_stack(self, server_url, temp_root):
        """A new reassign must clear the redo stack."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        uid = TEST_UUIDS[0]
        _post(f"{server_url}/api/reassign", {"uuid": uid, "new_album": "Kochen"})
        _post(f"{server_url}/api/undo", {})
        _post(f"{server_url}/api/reassign", {"uuid": uid, "new_album": "Garten"})
        status = _get(f"{server_url}/api/undo-status")
        assert status["can_redo"] is False

    def test_reassign_batch(self, server_url, temp_root):
        """Batch reassign (uuids list) must update all photos."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        uids = TEST_UUIDS[:2]
        result = _post(f"{server_url}/api/reassign",
                       {"uuids": uids, "new_album": "Unterwegs"})
        assert result["ok"] is True
        assert result["count"] == len(uids)
        for uid in uids:
            c = _get(f"{server_url}/api/current?uuid={uid}")
            assert c["album"] == "Unterwegs"


# ── Thrash flow ───────────────────────────────────────────────────────────────

class TestTrash:
    def test_trash_photo_shows_in_corrections(self, server_url, temp_root):
        reset_corrections(temp_root)
        reset_server_state(server_url)
        uid = TEST_UUIDS[2]
        _post(f"{server_url}/api/reassign",
              {"uuid": uid, "new_album": "Thrash"})
        current = _get(f"{server_url}/api/current?uuid={uid}")
        assert current["album"] == "Thrash"

    def test_trash_not_counted_in_retrain_status_new(self, server_url, temp_root):
        """Thrash corrections must not increment new_since_retrain."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        (temp_root / "metadata" / "last_retrain.json").write_text(
            '{"timestamp":"2026-01-01T00:00:00","corrections_count":0}'
        )
        _post(f"{server_url}/api/reassign",
              {"uuid": TEST_UUIDS[2], "new_album": "Thrash"})
        status = _get(f"{server_url}/api/retrain-status")
        assert status["new_since_retrain"] == 0, (
            "Thrash correction must not appear in new_since_retrain"
        )

    def test_inbox_count_excludes_trashed_photo(self, server_url, temp_root):
        """Trashed inbox photos must not appear in the unsorted count."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        reset_inbox(temp_root)
        inbox_uid = TEST_UUIDS[4]

        # Before trashing: 1 unsorted inbox photo
        count_before = _get(f"{server_url}/api/inbox-count")
        assert count_before["count"] == 1

        # Trash the inbox photo
        _post(f"{server_url}/api/reassign",
              {"uuid": inbox_uid, "new_album": "Thrash"})

        count_after = _get(f"{server_url}/api/inbox-count")
        assert count_after["count"] == 0, (
            "Trashed inbox photo must not count as 'unsorted'"
        )


# ── Undo / Redo ───────────────────────────────────────────────────────────────

class TestUndoRedo:
    def test_undo_reverts_to_previous_album(self, server_url, temp_root):
        """After undo, the photo returns to its original album."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        uid = TEST_UUIDS[0]  # assigned to Garten (base assignment, no correction)

        _post(f"{server_url}/api/reassign", {"uuid": uid, "new_album": "Yves"})
        assert _get(f"{server_url}/api/current?uuid={uid}")["album"] == "Yves"

        undo_result = _post(f"{server_url}/api/undo", {})
        assert undo_result["ok"] is True
        assert undo_result["reverted"] == 1

        # After undo: photo is back to "Garten".
        # The server may write "Garten" as a correction (corrected=True) because
        # prev="Garten" was the value stored in assignments.csv — both outcomes
        # are acceptable; what matters is the album value.
        result = _get(f"{server_url}/api/current?uuid={uid}")
        assert result["album"] == "Garten"

    def test_undo_makes_redo_available(self, server_url, temp_root):
        """After undo, can_redo must be True."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        _post(f"{server_url}/api/reassign",
              {"uuid": TEST_UUIDS[0], "new_album": "Yves"})
        _post(f"{server_url}/api/undo", {})
        status = _get(f"{server_url}/api/undo-status")
        assert status["can_redo"] is True

    def test_redo_reapplies_reassign(self, server_url, temp_root):
        """After redo, the photo returns to the reassigned album."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        uid = TEST_UUIDS[0]
        _post(f"{server_url}/api/reassign", {"uuid": uid, "new_album": "Yves"})
        _post(f"{server_url}/api/undo", {})
        _post(f"{server_url}/api/redo", {})
        assert _get(f"{server_url}/api/current?uuid={uid}")["album"] == "Yves"

    def test_undo_nothing_to_undo_returns_error(self, server_url, temp_root):
        """Undo when stack is empty must return ok=False."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        # Stack is now empty — immediate undo must fail
        result = _post(f"{server_url}/api/undo", {})
        assert result["ok"] is False

    def test_undo_inbox_photo_writes_tombstone(self, server_url, temp_root):
        """Undo of an inbox-photo sort (no prior album) must write a tombstone,
        reverting the photo to 'unsorted' state (not a specific album)."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        reset_inbox(temp_root)
        inbox_uid = TEST_UUIDS[4]  # was unsorted inbox photo

        # Sort the inbox photo to an album
        _post(f"{server_url}/api/reassign",
              {"uuid": inbox_uid, "new_album": "Garten"})
        assert _get(f"{server_url}/api/current?uuid={inbox_uid}")["album"] == "Garten"

        # Undo: should revert to unsorted (no album), not crash
        undo_result = _post(f"{server_url}/api/undo", {})
        assert undo_result["ok"] is True
        assert undo_result["reverted"] == 1

        # After undo, photo should be back to unsorted (empty / not corrected)
        current = _get(f"{server_url}/api/current?uuid={inbox_uid}")
        assert current["corrected"] is False
        assert current["album"] in ("", None, []), (
            f"Expected empty album after undo of inbox sort, got {current['album']!r}"
        )

        # Inbox count should be back to 1 unsorted photo
        count = _get(f"{server_url}/api/inbox-count")
        assert count["count"] == 1


# ── Restore from Thrash ───────────────────────────────────────────────────────

class TestRestore:
    def test_restore_removes_thrash_correction(self, server_url, temp_root):
        """After restore, the photo is no longer in Thrash."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        uid = TEST_UUIDS[3]

        _post(f"{server_url}/api/reassign",
              {"uuid": uid, "new_album": "Thrash"})
        assert _get(f"{server_url}/api/current?uuid={uid}")["album"] == "Thrash"

        restore_result = _post(f"{server_url}/api/restore", {"uuid": uid})
        assert restore_result["ok"] is True

        current = _get(f"{server_url}/api/current?uuid={uid}")
        assert current["album"] != "Thrash"
        assert current["corrected"] is False

    def test_restore_missing_uuid_returns_error(self, server_url):
        """Restore with no UUID body must return an error (not crash)."""
        result = _post(f"{server_url}/api/restore", {})
        assert "error" in result


# ── Retrain status ────────────────────────────────────────────────────────────

class TestRetrainStatus:
    def test_new_since_retrain_zero_after_all_used(self, server_url, temp_root):
        """After resetting state and setting corrections_count=0, new=0."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        (temp_root / "metadata" / "last_retrain.json").write_text(
            '{"timestamp":"2026-01-01T00:00:00","corrections_count":0}'
        )
        result = _get(f"{server_url}/api/retrain-status")
        assert result["new_since_retrain"] == 0

    def test_new_since_retrain_increments_on_correction(self, server_url, temp_root):
        """After a non-Thrash correction, new_since_retrain must be > 0."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        (temp_root / "metadata" / "last_retrain.json").write_text(
            '{"timestamp":"2026-01-01T00:00:00","corrections_count":0}'
        )
        _post(f"{server_url}/api/reassign",
              {"uuid": TEST_UUIDS[0], "new_album": "Garten"})
        result = _get(f"{server_url}/api/retrain-status")
        assert result["new_since_retrain"] > 0

    def test_thrash_correction_does_not_increment_new(self, server_url, temp_root):
        """Thrash corrections must be excluded from new_since_retrain count."""
        reset_corrections(temp_root)
        reset_server_state(server_url)
        (temp_root / "metadata" / "last_retrain.json").write_text(
            '{"timestamp":"2026-01-01T00:00:00","corrections_count":0}'
        )
        _post(f"{server_url}/api/reassign",
              {"uuid": TEST_UUIDS[0], "new_album": "Thrash"})
        result = _get(f"{server_url}/api/retrain-status")
        assert result["new_since_retrain"] == 0
