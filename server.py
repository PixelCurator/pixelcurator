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
import subprocess
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
        if r["new_album"]:            # non-empty = active correction
            corrections[r["uuid"]] = r["new_album"]
        else:
            corrections.pop(r["uuid"], None)   # empty = tombstone from empty-thrash

# Embeddings for similarity
print(f"Loading embeddings...", flush=True)
emb = np.load(EMB)
uuids_emb = json.loads(IDX.read_text())
uuid_to_idx = {u: i for i, u in enumerate(uuids_emb)}
print(f"  {emb.shape} loaded", flush=True)

_lock = threading.Lock()
_retrain_running = False


def current_album(uid: str) -> str:
    return corrections.get(uid, assigned.get(uid, ""))


def all_albums() -> list:
    s = set(assigned.values()) | set(corrections.values()) | set(empty_seeds)
    for mems in multi_memberships.values():
        s.update(mems)
    return sorted(s, key=lambda a: a.lower())


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


def reload_state():
    """Reload assigned dict from assignments.csv after a retrain."""
    global assigned
    new_assigned = {}
    with ASSIGN.open() as f:
        for r in csv.DictReader(f):
            new_assigned[r["uuid"]] = r["album"]
    with _lock:
        assigned.clear()
        assigned.update(new_assigned)
    return len(new_assigned)


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
        if path == "/api/thrash-count":
            n = sum(1 for v in corrections.values() if v == "Thrash")
            self._json({"count": n}); return
        if path == "/api/retrain-status":
            total = len(corrections)
            last_file = ROOT / "metadata" / "last_retrain.json"
            if last_file.exists():
                import json as _json
                info = _json.loads(last_file.read_text())
                used = info.get("corrections_count", 0)
                ts = info.get("timestamp")
            else:
                used, ts = 0, None
            self._json({"total": total, "new_since_retrain": max(0, total - used), "last_retrain": ts}); return

        if path == "/api/retrain-stream":
            global _retrain_running
            if _retrain_running:
                self.send_response(409)
                self.send_header("Content-Type", "text/plain")
                self._cors()
                self.end_headers()
                self.wfile.write(b"Retrain already in progress")
                return
            regen = query.get("regen", "1") != "0"
            cmd = [sys.executable, str(ROOT / "06_retrain.py")]
            if not regen:
                cmd.append("--no-regen")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()
            _retrain_running = True
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        payload = json.dumps({"type": "log", "msg": line})
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                proc.wait()
                if proc.returncode == 0:
                    n = reload_state()
                    payload = json.dumps({"type": "done", "loaded": n})
                else:
                    payload = json.dumps({"type": "error", "msg": f"exit code {proc.returncode}"})
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
            except BrokenPipeError:
                pass
            except Exception as e:
                try:
                    payload = json.dumps({"type": "error", "msg": str(e)})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    pass
            finally:
                _retrain_running = False
            return

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

        if self.path == "/api/retrain":
            body = self._read_body()
            regen = body.get("regenerate", True)
            cmd = [sys.executable, str(ROOT / "06_retrain.py")]
            if not regen:
                cmd.append("--no-regen")
            rc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if rc.returncode != 0:
                self._json({"ok": False, "error": rc.stderr[-1000:]}, 500); return
            n = reload_state()
            lines = [l for l in rc.stdout.splitlines() if l.strip()]
            self._json({"ok": True, "loaded": n, "log": lines[-30:]}); return

        if self.path == "/api/empty-thrash":
            body = self._read_body()
            if body.get("confirm") != "DELETE":
                self._json({"error": "confirm=DELETE required"}, 400); return
            thrash_uids = [u for u, v in corrections.items() if v == "Thrash"]
            if not thrash_uids:
                self._json({"ok": True, "deleted": 0, "message": "Thrash is already empty"}); return
            # Write helper script and call osxphotos venv python to delete
            helper = ROOT / "_delete_helper.py"
            helper.write_text(_DELETE_HELPER_SRC)
            tmp = ROOT / "logs" / "_thrash_uuids.txt"
            tmp.write_text("\n".join(thrash_uids))
            osx_python = Path.home() / ".local" / "pipx" / "venvs" / "osxphotos" / "bin" / "python"
            rc = subprocess.run(
                [str(osx_python), str(helper), str(tmp)],
                capture_output=True, text=True, timeout=120
            )
            if rc.returncode != 0:
                self._json({"ok": False, "error": rc.stderr[:500]}); return
            # Write tombstone entries so Thrash doesn't reappear after server restart,
            # then remove from in-memory corrections dict
            with _lock:
                ts = dt.datetime.utcnow().isoformat()
                with CORRECTIONS.open("a", newline="") as f:
                    w = csv.writer(f)
                    for u in thrash_uids:
                        w.writerow([u, "", ts, "emptied"])  # empty new_album = tombstone
                for u in thrash_uids:
                    corrections.pop(u, None)
            self._json({"ok": True, "deleted": len(thrash_uids), "message": rc.stdout.strip()}); return

        self.send_error(404)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")
        sys.stderr.flush()


_DELETE_HELPER_SRC = '''#!/usr/bin/env python3
"""Move Photos.app photos by UUID to the Recently Deleted album (trash)."""
import sys
from pathlib import Path
try:
    import photoscript
except ImportError:
    print("ERROR: photoscript not installed. Run: pipx inject osxphotos photoscript")
    sys.exit(2)

uuid_file = Path(sys.argv[1])
uuids = [u.strip() for u in uuid_file.read_text().splitlines() if u.strip()]
lib = photoscript.PhotosLibrary()
deleted = 0
errors = 0
for u in uuids:
    try:
        photo = photoscript.Photo(u)
        lib.delete([photo])
        deleted += 1
    except Exception as e:
        errors += 1
        print(f"  skip {u}: {e}", file=sys.stderr)
print(f"Moved {deleted}/{len(uuids)} to Recently Deleted (errors: {errors})")
'''

if __name__ == "__main__":
    print(f"Serving review on http://{BIND}:{PORT}/", flush=True)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
