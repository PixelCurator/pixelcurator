"""Same-origin regression tests for the review UI.

The review stack is a single origin: server.py serves the generated
contact sheets AND the /api + /img endpoints. Two things used to violate
that model and are pinned here:

1. server.py sent `Access-Control-Allow-Origin: *` on every response,
   which invited any web page in the user's browser to read personal
   photo bytes / album data from 127.0.0.1 and POST state-changing
   requests (reassign, retrain, commit-inbox). Same-origin requests need
   no CORS headers at all, so none must be sent.

2. 04_contact_sheets.py hardcoded `http://127.0.0.1:8765` into all
   generated HTML/JS (31 occurrences), which (a) required the CORS
   wildcard to begin with and (b) silently broke every API call and
   image load whenever PIXEL_PORT != 8765 -- including the test servers,
   which run on ephemeral ports. All URLs are now relative.
"""
import csv
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_api_responses_carry_no_cors_headers(server_url):
    with urllib.request.urlopen(f"{server_url}/api/albums", timeout=5) as r:
        headers = {k.lower() for k in r.headers.keys()}
    assert "access-control-allow-origin" not in headers, (
        "server.py must not send CORS headers: the review UI is same-origin, "
        "and a wildcard would re-open the API to arbitrary web pages"
    )


def test_generated_sheets_use_relative_urls(tmp_path):
    """Run the real 04_contact_sheets.py against a minimal sandbox and assert
    no generated file hardcodes a scheme://host origin."""
    root = tmp_path / "root"
    meta = root / "metadata"
    for d in (meta, root / "review", root / "logs"):
        d.mkdir(parents=True)

    uuids = [f"{c*8}-0000-0000-0000-00000000000{i}"
             for i, c in enumerate("AB", start=1)]
    with (meta / "inventory.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "original_filename", "date", "ismissing",
                    "has_local_original", "has_local_derivative",
                    "derivative_path", "derivative_size_bytes",
                    "uti_original", "isphoto", "ismovie"])
        for i, uid in enumerate(uuids):
            w.writerow([uid, f"photo_{i}.jpg", "2024-01-01", "False", "False",
                        "True", f"/fake/path/{uid}.jpg", "1000000", "",
                        "True", "False"])
    with (meta / "assignments.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "album", "confidence", "source"])
        w.writerow([uuids[0], "Garten", "0.9000", "retrain"])
        w.writerow([uuids[1], "Kochen", "0.8500", "retrain"])

    import os
    env = os.environ.copy()
    env["PIXEL_ROOT"] = str(root)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "04_contact_sheets.py")],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"04_contact_sheets.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    generated = sorted((root / "review").glob("*.html"))
    assert generated, "no contact sheets were generated"
    offenders = {
        p.name: line.strip()
        for p in generated
        for line in p.read_text().splitlines()
        if "http://127.0.0.1" in line or "http://localhost" in line
    }
    assert not offenders, (
        f"generated sheets hardcode an absolute origin (breaks any "
        f"PIXEL_PORT other than 8765 and forces CORS wildcards): {offenders}"
    )
