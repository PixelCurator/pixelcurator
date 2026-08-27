"""Shared test fixtures for PixelCurator acceptance tests.

Server integration tests run against fresh test servers on EPHEMERAL ports
(PIXEL_PORT=0: the kernel picks a free port, the server announces it on
stdout) using a temp directory with minimal synthetic data — the live
server on port 8765 is never touched, and two parallel pytest runs on one
box no longer collide on a fixed port.
"""
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
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

# ── temp root (session-scoped: created once, reused across tests) ─────────────

def make_pixel_root(root: Path) -> Path:
    """Populate `root` as a minimal PixelCurator sandbox with synthetic data.

    Plain function (not a fixture) so tests that need their own isolated
    root -- e.g. test_server_boot.py, which must seed corrections.csv
    *before* the server process starts -- can build one without touching
    the shared session-scoped `temp_root`."""
    root.mkdir(parents=True, exist_ok=True)
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


@pytest.fixture(scope="session")
def temp_root(tmp_path_factory):
    """Session-scoped minimal PixelCurator root shared by the server tests."""
    return make_pixel_root(tmp_path_factory.mktemp("pixel_root"))


# ── test server (session-scoped: started once, shared across tests) ───────────

_PORT_LINE = re.compile(r"Serving review on http://127\.0\.0\.1:(\d+)/")


def start_server_process(root: Path, *, test_mode: bool,
                         timeout: float = 20.0):
    """Start server.py against `root` on an EPHEMERAL port and return
    (proc, base_url).

    PIXEL_PORT=0 makes the kernel pick a free port; the server announces the
    actual bound port on stdout, which is parsed here. This replaces the old
    fixed ports 8766/8767, where two parallel pytest runs on one box
    collided. A pump thread keeps draining stdout afterwards so the child
    can never block on a full pipe.

    test_mode=True sets PIXEL_TEST_MODE=1 (enables /api/test-reset);
    boot-path tests pass False so they cannot cheat via test-reset."""
    env = os.environ.copy()
    env["PIXEL_ROOT"] = str(root)
    env["PIXEL_PORT"] = "0"
    env.pop("PIXEL_TEST_MODE", None)
    if test_mode:
        env["PIXEL_TEST_MODE"] = "1"

    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "server.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    lines_q: queue.Queue = queue.Queue()

    def _pump():
        for line in proc.stdout:
            lines_q.put(line)
        lines_q.put(None)  # EOF sentinel

    threading.Thread(target=_pump, daemon=True).start()

    seen: list[str] = []
    port = None
    deadline = time.monotonic() + timeout
    while port is None and time.monotonic() < deadline:
        try:
            line = lines_q.get(timeout=min(1.0, max(0.05, deadline - time.monotonic())))
        except queue.Empty:
            continue
        if line is None:  # process exited before announcing its port
            break
        seen.append(line)
        m = _PORT_LINE.search(line)
        if m:
            port = int(m.group(1))
    if port is None:
        proc.kill()
        proc.wait()
        raise RuntimeError(
            "Test server did not announce its port.\n"
            "Output:\n" + "".join(seen)[:1000]
        )

    # The socket is bound before the announcement, so this is a formality —
    # but poll readiness anyway to keep the fixture robust.
    base = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            urllib.request.urlopen(f"{base}/api/albums", timeout=1)
            return proc, base
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.25)
    proc.kill()
    proc.wait()
    raise RuntimeError(
        f"Test server on {base} never became ready.\n"
        "Output:\n" + "".join(seen)[:1000]
    )


@pytest.fixture(scope="session")
def server_url(temp_root):
    """Start a test server against temp_root on an ephemeral port; yield base URL."""
    proc, base = start_server_process(temp_root, test_mode=True)
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
