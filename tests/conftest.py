"""Shared test fixtures for PixelCurator acceptance tests.

Server integration tests run against a fresh test server on port 8766
using a temp directory with minimal synthetic data — the live server
on port 8765 is never touched.
"""
import csv
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

# Repo root (parent of tests/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Five synthetic UUIDs used across all tests
TEST_UUIDS = [
    "AAAAAAAA-0000-0000-0000-000000000001",  # assigned to Garten
    "BBBBBBBB-0000-0000-0000-000000000002",  # assigned to Kochen
    "CCCCCCCC-0000-0000-0000-000000000003",  # assigned to Diverses
    "DDDDDDDD-0000-0000-0000-000000000004",  # assigned to Yves
    "EEEEEEEE-0000-0000-0000-000000000005",  # inbox photo (not in inventory)
]

TEST_PORT = 8766


# ── temp root (session-scoped: created once, reused across tests) ─────────────

@pytest.fixture(scope="session")
def temp_root(tmp_path_factory):
    """Create a minimal PixelCurator root directory with synthetic data."""
    root = tmp_path_factory.mktemp("pixel_root")
    meta = root / "metadata"
    emb_dir = root / "embeddings"
    review = root / "review"
    logs = root / "logs"
    for d in (meta, emb_dir, review, logs):
        d.mkdir()

    # inventory.csv — first 4 UUIDs are "known" photos
    with (meta / "inventory.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "original_filename", "date", "ismissing",
                    "has_local_original", "has_local_derivative", "derivative_path",
                    "derivative_size_bytes", "uti_original", "isphoto", "ismovie"])
        for i, uid in enumerate(TEST_UUIDS[:4]):
            w.writerow([uid, f"photo_{i}.jpg", "2024-01-01", "False",
                        "False", "True", f"/fake/path/{uid}.jpg",
                        "1000000", "", "True", "False"])

    # assignments.csv
    with (meta / "assignments.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "album", "confidence", "source"])
        w.writerow([TEST_UUIDS[0], "Garten",   "0.9000", "retrain"])
        w.writerow([TEST_UUIDS[1], "Kochen",   "0.8500", "retrain"])
        w.writerow([TEST_UUIDS[2], "Diverses", "0.6000", "retrain"])
        w.writerow([TEST_UUIDS[3], "Yves",     "0.9500", "retrain"])

    # corrections.csv — empty header
    with (meta / "corrections.csv").open("w", newline="") as f:
        csv.writer(f).writerow(["uuid", "new_album", "timestamp", "source"])

    # inbox.csv — UUID[4] is an unsorted inbox photo
    with (meta / "inbox.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "original_filename", "date",
                    "derivative_path", "has_local_derivative"])
        w.writerow([TEST_UUIDS[4], "inbox_photo.jpg", "2024-06-01",
                    f"/fake/path/{TEST_UUIDS[4]}.jpg", "True"])

    # Small random embeddings for all 5 UUIDs
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((5, 512)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(emb_dir / "clip_vitb32.npy", emb)
    (emb_dir / "clip_vitb32_uuids.json").write_text(json.dumps(TEST_UUIDS))

    # Minimal index.html so server can serve "/"
    (review / "index.html").write_text("<html><body>PixelCurator test</body></html>")

    # last_retrain.json — 0 corrections used → 0 new
    (meta / "last_retrain.json").write_text(
        json.dumps({"timestamp": "2026-01-01T00:00:00", "corrections_count": 0})
    )

    return root


# ── test server (session-scoped: started once, shared across tests) ───────────

@pytest.fixture(scope="session")
def server_url(temp_root):
    """Start a test server against temp_root on port 8766; yield base URL."""
    env = os.environ.copy()
    env["PIXEL_ROOT"] = str(temp_root)
    env["PIXEL_PORT"] = str(TEST_PORT)
    env["PIXEL_TEST_MODE"] = "1"  # enables /api/test-reset endpoint

    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "server.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait up to 10 s for the server to be ready
    base = f"http://127.0.0.1:{TEST_PORT}"
    for _ in range(20):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(f"{base}/api/albums", timeout=1)
            break
        except (urllib.error.URLError, ConnectionRefusedError):
            pass
    else:
        proc.kill()
        out, _ = proc.communicate()
        raise RuntimeError(
            f"Test server on port {TEST_PORT} failed to start.\n"
            f"Output:\n{(out or b'').decode()[:1000]}"
        )

    yield base

    proc.kill()
    proc.wait()


# ── helper: reset corrections.csv to empty ────────────────────────────────────

def reset_corrections(temp_root: Path) -> None:
    """Wipe corrections.csv back to header-only (file only — call
    reset_server_state() afterward to sync the running server's memory)."""
    corr = temp_root / "metadata" / "corrections.csv"
    with corr.open("w", newline="") as f:
        csv.writer(f).writerow(["uuid", "new_album", "timestamp", "source"])


def reset_server_state(server_url: str) -> None:
    """POST /api/test-reset — reloads corrections from CSV + clears undo/redo
    stacks in the running server without a restart. Requires PIXEL_TEST_MODE=1."""
    import json as _json
    data = _json.dumps({}).encode()
    req = urllib.request.Request(
        f"{server_url}/api/test-reset",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return _json.loads(r.read())


def reset_inbox(temp_root: Path) -> None:
    """Restore inbox.csv to initial state (UUID[4] unsorted)."""
    inbox = temp_root / "metadata" / "inbox.csv"
    with inbox.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "original_filename", "date",
                    "derivative_path", "has_local_derivative"])
        w.writerow([TEST_UUIDS[4], "inbox_photo.jpg", "2024-06-01",
                    f"/fake/path/{TEST_UUIDS[4]}.jpg", "True"])
