#!/usr/bin/env python3
"""Localhost review server: serves contact sheets + images + reassign API.

Endpoints:
  GET  /                          — index.html
  GET  /<album>.html              — album sheet
  GET  /img/<uuid>                — JPEG bytes
  GET  /api/albums                — JSON list of current album names
  GET  /api/current?uuid=X        — current album for one photo
  GET  /api/similar?uuid=X&n=20   — top-N similar uuids by CLIP cosine
  POST /api/reassign              — body {uuid, new_album} or {uuids:[...], new_album}
  GET  /api/corrections           — JSON map uuid -> corrected album
"""
import csv
import datetime as dt
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path.home() / "photo-sort"
REVIEW = ROOT / "review"
INV = ROOT / "metadata" / "inventory.csv"
ASSIGN = ROOT / "metadata" / "assignments.csv"
MULTI = ROOT / "metadata" / "multi_assignments.csv"
EMPTY_TXT = ROOT / "metadata" / "empty_albums.txt"
CORRECTIONS = ROOT / "metadata" / "corrections.csv"
EMB = ROOT / "embeddings" / "clip_vitb32.npy"
IDX = ROOT / "embeddings" / "clip_vitb32_uuids.json"

PORT = 8765
BIND = "127.0.0.1"

# uuid -> derivative path
path_by_uuid = {}
with INV.open() as f:
    for r in csv.DictReader(f):
        if r["has_local_derivative"] == "True":
            path_by_uuid[r["uuid"]] = r["derivative_path"]

# uuid -> primary assigned album
assigned = {}
with ASSIGN.open() as f:
    for r in csv.DictReader(f):
        assigned[r["uuid"]] = r["album"]

# uuid -> list of all album memberships (multi)
multi_memberships = {}
if MULTI.exists():
    from collections import defaultdict
    _mm = defaultdict(list)
    with MULTI.open() as f:
        for r in csv.DictReader(f):
            _mm[r["uuid"]].append(r["album"])
    multi_memberships = dict(_mm)

# Empty seed albums (created always, populated by user via UI)
empty_seeds = []
if EMPTY_TXT.exists():
    empty_seeds = [l.strip() for l in EMPTY_TXT.read_text().splitlines() if l.strip()]

# Initialize corrections file
if not CORRECTIONS.exists():
    with CORRECTIONS.open("w", newline="") as f:
        csv.writer(f).writerow(["uuid", "new_album", "timestamp", "source"])

corrections = {}
with CORRECTIONS.open() as f:
    for r in csv.DictReader(f):
        corrections[r["uuid"]] = r["new_album"]

# Embeddings for similarity
print(f"Loading embeddings...", flush=True)
emb = np.load(EMB)
uuids_emb = json.loads(IDX.read_text())
uuid_to_idx = {u: i for i, u in enumerate(uuids_emb)}
print(f"  {emb.shape} loaded", flush=True)

_lock = threading.Lock()


def current_album(uid: str) -> str:
    return corrections.get(uid, assigned.get(uid, ""))


def all_albums() -> list:
    s = set(assigned.values()) | set(corrections.values()) | set(empty_seeds)
    for mems in multi_memberships.values():
        s.update(mems)
    # Sort: seeds first, clusters next, Unsortiert last
    SEED = ["Pornografie", "Pornografie-unsure", "Nacktheit", "Nacktheit-unsure",
            "Motorrad-Werkstatt", "Motorrad-Werkstatt-unsure",
            "Aquarium", "Aquarium-unsure", "Garten", "Garten-unsure"]

    def key(a):
        if a in SEED:
            return (0, SEED.index(a))
        if a == "Unsortiert":
            return (3, 0)
        if a.startswith("Cluster-"):
            return (1, a)
        return (2, a)
    return sorted(s, key=key)


def find_similar(uid: str, n: int = 20) -> list:
    if uid not in uuid_to_idx:
        return []
    target = emb[uuid_to_idx[uid]]
    sims = emb @ target
    top = np.argsort(-sims)[:n + 1]
    return [(uuids_emb[int(j)], float(sims[int(j)])) for j in top if uuids_emb[int(j)] != uid][:n]


def apply_correction(uid: str, album: str, source: str = "manual"):
    with _lock:
        with CORRECTIONS.open("a", newline="") as f:
            csv.writer(f).writerow([uid, album, dt.datetime.utcnow().isoformat(), source])
        corrections[uid] = album


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _json(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_bytes(self, data: bytes, ctype: str, cache: bool = False):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", "max-age=86400")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw.decode())
        except Exception:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0]
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        query = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)

        # API
        if path == "/api/albums":
            self._json(all_albums()); return
        if path == "/api/current":
            uid = query.get("uuid", "")
            self._json({
                "uuid": uid,
                "album": current_album(uid),
                "corrected": uid in corrections,
                "all_memberships": multi_memberships.get(uid, []),
            }); return
        if path == "/api/similar":
            uid = query.get("uuid", "")
            n = int(query.get("n", "20"))
            res = find_similar(uid, n)
            self._json({"uuid": uid, "similar": [{"uuid": u, "score": s, "album": current_album(u)} for u, s in res]}); return
        if path == "/api/corrections":
            self._json(corrections); return

        # Images
        if path.startswith("/img/"):
            uid = path[5:]
            p = path_by_uuid.get(uid)
            if not p:
                self.send_error(404); return
            try:
                self._serve_bytes(Path(p).read_bytes(), "image/jpeg", cache=True)
            except Exception:
                self.send_error(404)
            return

        # Static review files
        if path == "/" or path == "":
            target = REVIEW / "index.html"
        else:
            rel = path.lstrip("/")
            if ".." in rel:
                self.send_error(403); return
            target = REVIEW / rel
            if not target.exists() and target.suffix == "":
                target = target.with_suffix(".html")

        if target.exists():
            ctype = "text/html; charset=utf-8" if target.suffix == ".html" else (
                "text/javascript; charset=utf-8" if target.suffix == ".js" else
                "text/css; charset=utf-8" if target.suffix == ".css" else
                "application/octet-stream"
            )
            self._serve_bytes(target.read_bytes(), ctype)
        else:
            self.send_error(404, f"no such file: {target.name}")

    def do_POST(self):
        if self.path == "/api/reassign":
            body = self._read_body()
            album = body.get("new_album", "").strip()
            if not album:
                self._json({"error": "new_album required"}, 400); return
            uids = body.get("uuids") or ([body["uuid"]] if "uuid" in body else [])
            src = body.get("source", "manual")
            for u in uids:
                if u in path_by_uuid:
                    apply_correction(u, album, src)
            self._json({"ok": True, "count": len(uids), "album": album}); return
        self.send_error(404)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")
        sys.stderr.flush()


if __name__ == "__main__":
    print(f"Serving review on http://{BIND}:{PORT}/", flush=True)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
