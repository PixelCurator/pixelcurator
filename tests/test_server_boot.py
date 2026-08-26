"""Boot-path regression tests: server.py must load corrections.csv correctly
at process start, before any /api/test-reset is ever called.

Why this file exists: every other server test talks to the shared
session-scoped server and calls /api/test-reset for isolation. test-reset
used to contain its own hand-written copy of the corrections-loading loop,
so the whole suite exercised only that copy -- mutating the *boot* loader
(dropping the tombstone branch) left all 56 tests green. Boot and reset now
share load_corrections_from_csv(); this test pins the boot path by seeding
corrections.csv first and only then starting a fresh server process on its
own port, asserting on /api/corrections without touching test-reset.
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

import pytest

from .conftest import REPO_ROOT, TEST_UUIDS, make_pixel_root

BOOT_TEST_PORT = 8767  # own port -- never the shared session server's 8766


@pytest.fixture()
def boot_server(tmp_path):
    """Build an isolated sandbox, let the test seed corrections.csv, then
    start a fresh server.py process against it. Yields (root, start) where
    start() boots the server and returns its base URL."""
    root = make_pixel_root(tmp_path / "pixel_root_boot")
    procs = []

    def start() -> str:
        env = os.environ.copy()
        env["PIXEL_ROOT"] = str(root)
        env["PIXEL_PORT"] = str(BOOT_TEST_PORT)
        # Deliberately NOT setting PIXEL_TEST_MODE: this test must observe
        # the plain production boot path, and must not be able to "cheat"
        # via /api/test-reset even by accident.
        proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "server.py")],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(proc)
        base = f"http://127.0.0.1:{BOOT_TEST_PORT}"
        for _ in range(20):
            time.sleep(0.5)
            try:
                urllib.request.urlopen(f"{base}/api/albums", timeout=1)
                return base
            except (urllib.error.URLError, ConnectionRefusedError):
                pass
        proc.kill()
        out, _ = proc.communicate()
        raise RuntimeError(
            f"Boot-test server on port {BOOT_TEST_PORT} failed to start.\n"
            f"Output:\n{(out or b'').decode()[:1000]}"
        )

    yield root, start

    for proc in procs:
        proc.kill()
        proc.wait()


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def test_boot_replays_tombstones_and_last_entry_wins(boot_server):
    """A corrections.csv containing active corrections, a tombstone, and a
    superseded entry must be replayed correctly by the boot loader."""
    root, start = boot_server
    corr = root / "metadata" / "corrections.csv"
    with corr.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "new_album", "timestamp", "source"])
        # UUID[0]: corrected, then tombstoned -> must NOT survive boot
        w.writerow([TEST_UUIDS[0], "Garten", "2026-01-01T00:00:00", "manual"])
        w.writerow([TEST_UUIDS[0], "", "2026-01-01T00:01:00", "restore"])
        # UUID[1]: corrected twice -> last entry wins
        w.writerow([TEST_UUIDS[1], "Kochen", "2026-01-01T00:02:00", "manual"])
        w.writerow([TEST_UUIDS[1], "Urlaub", "2026-01-01T00:03:00", "manual"])
        # UUID[2]: plain single correction -> survives as-is
        w.writerow([TEST_UUIDS[2], "Astro", "2026-01-01T00:04:00", "manual"])

    base = start()
    loaded = _get_json(f"{base}/api/corrections")

    assert TEST_UUIDS[0] not in loaded, (
        "tombstone (empty new_album) was ignored by the boot loader -- "
        "a restored/emptied photo would resurrect its stale correction "
        "after a server restart"
    )
    assert loaded.get(TEST_UUIDS[1]) == "Urlaub", "last entry per uuid must win"
    assert loaded.get(TEST_UUIDS[2]) == "Astro"
    assert set(loaded) == {TEST_UUIDS[1], TEST_UUIDS[2]}


def test_boot_tombstone_for_never_corrected_uuid_is_harmless(boot_server):
    """A tombstone for a uuid that never had a correction (possible via
    restore on an inbox photo) must not crash boot or leak an entry."""
    root, start = boot_server
    corr = root / "metadata" / "corrections.csv"
    with corr.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "new_album", "timestamp", "source"])
        w.writerow([TEST_UUIDS[3], "", "2026-01-01T00:00:00", "restore"])

    base = start()
    loaded = _get_json(f"{base}/api/corrections")
    assert loaded == {}
