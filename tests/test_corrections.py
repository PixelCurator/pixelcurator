"""Unit tests: corrections.csv business logic.

These tests exercise the rules that govern corrections.csv without
starting a server — pure CSV + Python logic.
"""
import csv
from pathlib import Path


def _write_corrections(path: Path, rows: list[tuple]) -> None:
    """Write a corrections.csv with given (uuid, new_album, timestamp, source) rows."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "new_album", "timestamp", "source"])
        for row in rows:
            w.writerow(row)


def _load_corrections(path: Path) -> dict[str, str]:
    """Replicate the load-corrections logic from server.py / 06_retrain.py."""
    result: dict[str, str] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["new_album"]:
                result[r["uuid"]] = r["new_album"]
            else:
                result.pop(r["uuid"], None)
    return result


# ── Core corrections logic ────────────────────────────────────────────────────

class TestLastEntryWins:
    def test_second_correction_overwrites_first(self, tmp_path):
        """Most recent correction for a UUID takes precedence."""
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-1", "Garten",   "2024-01-01T10:00:00", "manual"),
            ("UUID-1", "Kochen",   "2024-01-02T10:00:00", "manual"),
        ])
        result = _load_corrections(corr)
        assert result["UUID-1"] == "Kochen"

    def test_multiple_uuids_independent(self, tmp_path):
        """Each UUID is tracked independently."""
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-A", "Garten",  "2024-01-01T00:00:00", "manual"),
            ("UUID-B", "Kochen",  "2024-01-01T00:00:00", "manual"),
            ("UUID-A", "Diverses","2024-01-02T00:00:00", "manual"),
        ])
        result = _load_corrections(corr)
        assert result["UUID-A"] == "Diverses"
        assert result["UUID-B"] == "Kochen"


class TestTombstone:
    def test_empty_new_album_removes_uuid(self, tmp_path):
        """An empty new_album entry is a tombstone: removes the UUID from results."""
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-1", "Garten", "2024-01-01T00:00:00", "manual"),
            ("UUID-1", "",       "2024-01-02T00:00:00", "undo"),
        ])
        result = _load_corrections(corr)
        assert "UUID-1" not in result

    def test_tombstone_then_new_correction_restores(self, tmp_path):
        """A tombstone can be superseded by a later correction."""
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-1", "Garten", "2024-01-01T00:00:00", "manual"),
            ("UUID-1", "",       "2024-01-02T00:00:00", "undo"),
            ("UUID-1", "Yves",   "2024-01-03T00:00:00", "redo"),
        ])
        result = _load_corrections(corr)
        assert result["UUID-1"] == "Yves"

    def test_multiple_tombstones_ok(self, tmp_path):
        """Multiple tombstones for same UUID — last tombstone still removes it."""
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-1", "Garten", "2024-01-01T00:00:00", "manual"),
            ("UUID-1", "Kochen", "2024-01-02T00:00:00", "manual"),
            ("UUID-1", "",       "2024-01-03T00:00:00", "emptied"),
            ("UUID-1", "",       "2024-01-04T00:00:00", "emptied"),
        ])
        result = _load_corrections(corr)
        assert "UUID-1" not in result


class TestSkipAlbums:
    """Thrash and _test_ corrections are loaded by server but excluded from
    retrain-status 'new_since_retrain' count (via server /api/retrain-status).
    The load_corrections() in 06_retrain.py already filters them out."""

    SKIP_ALBUMS = {"Thrash", "_test_"}

    def _load_retrain_corrections(self, path: Path) -> dict[str, str]:
        """Replicate 06_retrain.py load_corrections() which excludes SKIP_ALBUMS."""
        result: dict[str, str] = {}
        with path.open() as f:
            for r in csv.DictReader(f):
                if r["new_album"] and r["new_album"] not in self.SKIP_ALBUMS:
                    result[r["uuid"]] = r["new_album"]
        return result

    def test_thrash_excluded_from_retrain_corrections(self, tmp_path):
        """Thrash corrections must not appear in retrain training set."""
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-KEPT", "Garten", "2024-01-01T00:00:00", "manual"),
            ("UUID-SKIP", "Thrash", "2024-01-01T00:00:00", "manual"),
        ])
        result = self._load_retrain_corrections(corr)
        assert "UUID-KEPT" in result
        assert "UUID-SKIP" not in result

    def test_test_album_excluded_from_retrain_corrections(self, tmp_path):
        """_test_ album must not appear in retrain training set."""
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-SKIP", "_test_", "2024-01-01T00:00:00", "manual"),
        ])
        result = self._load_retrain_corrections(corr)
        assert "UUID-SKIP" not in result

    def test_retrain_count_excludes_trash_and_test(self, tmp_path):
        """new_since_retrain must not count Thrash or _test_ corrections."""
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [
            ("UUID-A", "Garten",  "2024-01-01T00:00:00", "manual"),
            ("UUID-B", "Thrash",  "2024-01-01T00:00:00", "manual"),
            ("UUID-C", "_test_",  "2024-01-01T00:00:00", "manual"),
            ("UUID-D", "Kochen",  "2024-01-01T00:00:00", "manual"),
        ])
        # Simulated retrain-status logic from server.py
        corrections_map = {}
        with corr.open() as f:
            for r in csv.DictReader(f):
                if r["new_album"]:
                    corrections_map[r["uuid"]] = r["new_album"]
                else:
                    corrections_map.pop(r["uuid"], None)
        _skip = {"Thrash", "_test_"}
        total = sum(1 for v in corrections_map.values() if v not in _skip)
        assert total == 2  # only UUID-A and UUID-D count


class TestEmptyFile:
    def test_empty_corrections_file_returns_empty_dict(self, tmp_path):
        corr = tmp_path / "corrections.csv"
        _write_corrections(corr, [])
        assert _load_corrections(corr) == {}
