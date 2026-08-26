"""Unit tests: corrections.csv replay semantics, against PRODUCTION code.

corrections.csv is an append-only event log: one row per user decision,
where the last row per uuid wins and an empty new_album is a tombstone
(written by empty-thrash / restore) that removes the correction.

An earlier version of this file tested two local replicas of the loading
logic ("Replicate the load-corrections logic from server.py /
06_retrain.py") -- deleting the tombstone branch from server.py left every
test green, and 06_retrain.py's real load_corrections() had in fact
diverged from the replica: it filtered row-by-row instead of replaying, so
tombstones and later Thrash corrections did not remove earlier entries and
retrain resurrected albums the user had explicitly un-corrected. These
tests import 06_retrain.py itself (via importlib -- the leading digit
makes it unimportable by name) and pin the replay semantics there.
The server boot path is pinned separately by tests/test_server_boot.py,
which exercises server.py's loader through a real process.
"""
import csv
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "retrain_06", REPO_ROOT / "06_retrain.py"
)
retrain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrain)


def _write_corrections(path: Path, rows: list[tuple]) -> None:
    """Write a corrections.csv with given (uuid, new_album, timestamp, source) rows."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "new_album", "timestamp", "source"])
        for row in rows:
            w.writerow(row)


def _load(monkeypatch, path: Path) -> dict:
    """Run the real retrain.load_corrections() against `path`."""
    monkeypatch.setattr(retrain, "CORRECTIONS_CSV", path)
    return retrain.load_corrections()


# ── Core replay semantics ─────────────────────────────────────────────────────

class TestLastEntryWins:
    def test_second_correction_overwrites_first(self, tmp_path, monkeypatch):
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-1", "Garten", "2024-01-01T10:00:00", "manual"),
            ("UUID-1", "Kochen", "2024-01-02T10:00:00", "manual"),
        ])
        assert _load(monkeypatch, corr)["UUID-1"] == "Kochen"

    def test_multiple_uuids_independent(self, tmp_path, monkeypatch):
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-A", "Garten",   "2024-01-01T00:00:00", "manual"),
            ("UUID-B", "Kochen",   "2024-01-01T00:00:00", "manual"),
            ("UUID-A", "Diverses", "2024-01-02T00:00:00", "manual"),
        ])
        result = _load(monkeypatch, corr)
        assert result["UUID-A"] == "Diverses"
        assert result["UUID-B"] == "Kochen"

    def test_missing_file_yields_empty(self, tmp_path, monkeypatch):
        assert _load(monkeypatch, tmp_path / "does-not-exist.csv") == {}


class TestTombstone:
    def test_empty_new_album_removes_uuid(self, tmp_path, monkeypatch):
        """Regression for the retrain divergence: a tombstone must remove the
        earlier correction, otherwise retrain pins a restored photo back to
        its stale album with conf=1.0 on every retrain."""
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-1", "Garten", "2024-01-01T00:00:00", "manual"),
            ("UUID-1", "",       "2024-01-02T00:00:00", "restore"),
        ])
        assert "UUID-1" not in _load(monkeypatch, corr)

    def test_correction_after_tombstone_is_active_again(self, tmp_path, monkeypatch):
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-1", "Garten", "2024-01-01T00:00:00", "manual"),
            ("UUID-1", "",       "2024-01-02T00:00:00", "restore"),
            ("UUID-1", "Kochen", "2024-01-03T00:00:00", "manual"),
        ])
        assert _load(monkeypatch, corr)["UUID-1"] == "Kochen"

    def test_tombstone_for_unknown_uuid_is_harmless(self, tmp_path, monkeypatch):
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-NEVER-CORRECTED", "", "2024-01-01T00:00:00", "restore"),
        ])
        assert _load(monkeypatch, corr) == {}


# ── SKIP_ALBUMS exclusion (final state, not row-by-row) ───────────────────────

class TestSkipAlbums:
    def test_thrash_excluded_from_retrain_corrections(self, tmp_path, monkeypatch):
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-KEPT", "Garten", "2024-01-01T00:00:00", "manual"),
            ("UUID-SKIP", "Thrash", "2024-01-01T00:00:00", "manual"),
        ])
        result = _load(monkeypatch, corr)
        assert "UUID-KEPT" in result
        assert "UUID-SKIP" not in result

    def test_test_album_excluded_from_retrain_corrections(self, tmp_path, monkeypatch):
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-T", "_test_", "2024-01-01T00:00:00", "manual"),
        ])
        assert _load(monkeypatch, corr) == {}

    def test_later_thrash_correction_removes_earlier_album(self, tmp_path, monkeypatch):
        """Regression for the retrain divergence: correcting Garten -> Thrash
        must not leave the photo training/pinned as Garten. Row-by-row
        filtering skipped the Thrash row without overwriting, so the stale
        Garten entry survived."""
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-1", "Garten", "2024-01-01T00:00:00", "manual"),
            ("UUID-1", "Thrash", "2024-01-02T00:00:00", "manual"),
        ])
        assert "UUID-1" not in _load(monkeypatch, corr)

    def test_correction_after_thrash_is_active(self, tmp_path, monkeypatch):
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-1", "Thrash",  "2024-01-01T00:00:00", "manual"),
            ("UUID-1", "Kochen",  "2024-01-02T00:00:00", "manual"),
        ])
        assert _load(monkeypatch, corr)["UUID-1"] == "Kochen"
