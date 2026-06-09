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
from collections import deque
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

# Inbox: load derivative paths for photos not yet in inventory
INBOX = ROOT / "metadata" / "inbox.csv"
if INBOX.exists():
    with INBOX.open() as f:
        for r in csv.DictReader(f):
            if r.get("has_local_derivative") == "True" and r.get("derivative_path"):
                path_by_uuid[r["uuid"]] = r["derivative_path"]

# Embeddings for similarity
print(f"Loading embeddings...", flush=True)
emb = np.load(EMB)
uuids_emb = json.loads(IDX.read_text())
uuid_to_idx = {u: i for i, u in enumerate(uuids_emb)}
print(f"  {emb.shape} loaded", flush=True)

_lock = threading.Lock()
_retrain_running = False
_undo_stack = deque(maxlen=10)
_redo_stack = deque(maxlen=10)


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


def apply_correction(uid: str, album: str, source: str = "manual") -> str:
    prev = current_album(uid)
    with _lock:
        with CORRECTIONS.open("a", newline="") as f:
            csv.writer(f).writerow([uid, album, dt.datetime.utcnow().isoformat(), source])
        corrections[uid] = album
    return prev


def _safe_fn(name: str) -> str:
    """Same logic as safe_album_filename in 04_contact_sheets.py."""
    return (name.replace("/", "_").replace(" ", "_")
            .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"))


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
        if path == "/api/undo-status":
            self._json({
                "can_undo": bool(_undo_stack),
                "undo_desc": _undo_stack[-1]["desc"] if _undo_stack else None,
                "can_redo": bool(_redo_stack),
                "redo_desc": _redo_stack[-1]["desc"] if _redo_stack else None,
                "undo_count": len(_undo_stack),
                "redo_count": len(_redo_stack),
            }); return
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

        if path == "/api/inbox-count":
            inbox = ROOT / "metadata" / "inbox.csv"
            count = 0
            if inbox.exists():
                with inbox.open() as f:
                    count = sum(1 for r in csv.DictReader(f) if r.get("has_local_derivative") == "True")
            self._json({"count": count}); return

        if path in ("/api/inbox-detect-stream", "/api/inbox-process-stream"):
            mode = "--detect" if path == "/api/inbox-detect-stream" else "--process"
            cmd = [sys.executable, str(ROOT / "07_incremental.py"), mode]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
                count = 0
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        if line.startswith("RESULT:"):
                            count = int(line.split(":")[1])
                        payload = json.dumps({"type": "log", "msg": line})
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                proc.wait()
                if proc.returncode == 0:
                    # Reload paths from inbox.csv so thumbnails are immediately servable
                    inbox = ROOT / "metadata" / "inbox.csv"
                    if inbox.exists():
                        with inbox.open() as f:
                            for r in csv.DictReader(f):
                                if r.get("has_local_derivative") == "True" and r.get("derivative_path"):
                                    path_by_uuid[r["uuid"]] = r["derivative_path"]
                    if mode == "--process":
                        reload_state()
                    payload = json.dumps({"type": "done", "count": count})
                else:
                    payload = json.dumps({"type": "error", "msg": f"exit {proc.returncode}"})
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
            except BrokenPipeError:
                pass
            except Exception as e:
                try:
                    self.wfile.write(f"data: {json.dumps({'type':'error','msg':str(e)})}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    pass
            return

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
        if self.path in ("/api/undo", "/api/redo"):
            is_undo = self.path == "/api/undo"
            src_stack = _undo_stack if is_undo else _redo_stack
            dst_stack = _redo_stack if is_undo else _undo_stack
            if not src_stack:
                self._json({"ok": False, "error": "Nothing to " + ("undo" if is_undo else "redo")}); return
            action = src_stack.pop()
            tag = "undo" if is_undo else "redo"

            if action["type"] == "reassign":
                rev_changes = []
                for ch in action["changes"]:
                    target_album = ch["prev"] if is_undo else ch["new"]
                    if target_album:
                        apply_correction(ch["uuid"], target_album, tag)
                        rev_changes.append({"uuid": ch["uuid"], "prev": ch["new"] if is_undo else ch["prev"], "new": target_album})
                dst_stack.append({
                    "type": "reassign",
                    "desc": ("↩ " if is_undo else "↪ ") + action["desc"],
                    "changes": rev_changes,
                })
                self._json({"ok": True, "type": "reassign", "reverted": len(rev_changes)}); return

            elif action["type"] == "rename":
                rev_old = action["data"]["new_name"] if is_undo else action["data"]["old_name"]
                rev_new = action["data"]["old_name"] if is_undo else action["data"]["new_name"]
                name_map = {rev_old: rev_new}
                if not rev_old.endswith("-unsure"):
                    name_map[rev_old + "-unsure"] = rev_new + "-unsure"
                with ASSIGN.open() as f:
                    reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
                for r in rows:
                    if r["album"] in name_map: r["album"] = name_map[r["album"]]
                with ASSIGN.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
                with CORRECTIONS.open() as f:
                    cr = csv.DictReader(f); cf = cr.fieldnames; crows = list(cr)
                for r in crows:
                    if r["new_album"] in name_map: r["new_album"] = name_map[r["new_album"]]
                with CORRECTIONS.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cf); w.writeheader(); w.writerows(crows)
                with _lock:
                    for uid in list(assigned):
                        if assigned[uid] in name_map: assigned[uid] = name_map[assigned[uid]]
                    for uid in list(corrections):
                        if corrections[uid] in name_map: corrections[uid] = name_map[corrections[uid]]
                for old in name_map:
                    stale = REVIEW / (_safe_fn(old) + ".html")
                    if stale.exists(): stale.unlink()
                subprocess.run([sys.executable, str(ROOT / "04_contact_sheets.py")], capture_output=True, timeout=120)
                dst_stack.append({
                    "type": "rename",
                    "desc": ("↩ " if is_undo else "↪ ") + action["desc"],
                    "data": {"old_name": rev_new, "new_name": rev_old},
                })
                self._json({"ok": True, "type": "rename", "map": name_map}); return

            elif action["type"] == "merge":
                if is_undo:
                    source = action["data"]["source"]
                    target = action["data"]["target"]
                    affected = set(action["data"]["affected_uuids"])
                    name_map_rev = {target: source, target + "-unsure": source + "-unsure"}
                    with ASSIGN.open() as f:
                        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
                    for r in rows:
                        if r["uuid"] in affected and r["album"] in name_map_rev:
                            r["album"] = name_map_rev[r["album"]]
                    with ASSIGN.open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
                    with CORRECTIONS.open() as f:
                        cr = csv.DictReader(f); cf = cr.fieldnames; crows = list(cr)
                    for r in crows:
                        if r["uuid"] in affected and r["new_album"] in name_map_rev:
                            r["new_album"] = name_map_rev[r["new_album"]]
                    with CORRECTIONS.open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=cf); w.writeheader(); w.writerows(crows)
                    with _lock:
                        for uid in list(assigned):
                            if uid in affected and assigned[uid] in name_map_rev:
                                assigned[uid] = name_map_rev[assigned[uid]]
                        for uid in list(corrections):
                            if uid in affected and corrections[uid] in name_map_rev:
                                corrections[uid] = name_map_rev[corrections[uid]]
                    subprocess.run([sys.executable, str(ROOT / "04_contact_sheets.py")], capture_output=True, timeout=120)
                    dst_stack.append({
                        "type": "merge",
                        "desc": "↪ " + action["desc"],
                        "data": action["data"],
                    })
                    self._json({"ok": True, "type": "merge_undo", "restored": len(affected)}); return
                else:
                    source = action["data"]["source"]
                    target = action["data"]["target"]
                    name_map = {source: target, source + "-unsure": target + "-unsure"}
                    with ASSIGN.open() as f:
                        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
                    for r in rows:
                        if r["album"] in name_map: r["album"] = name_map[r["album"]]
                    with ASSIGN.open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
                    subprocess.run([sys.executable, str(ROOT / "04_contact_sheets.py")], capture_output=True, timeout=120)
                    dst_stack.append({
                        "type": "merge",
                        "desc": "↩ " + action["desc"],
                        "data": action["data"],
                    })
                    self._json({"ok": True, "type": "merge_redo"}); return

            self._json({"ok": False, "error": f"Unknown action type: {action['type']}"}); return

        if self.path == "/api/reassign":
            body = self._read_body()
            album = body.get("new_album", "").strip()
            if not album:
                self._json({"error": "new_album required"}, 400); return
            uids = body.get("uuids") or ([body["uuid"]] if "uuid" in body else [])
            src = body.get("source", "manual")
            changes = []
            for u in uids:
                if u in path_by_uuid:
                    prev = apply_correction(u, album, src)
                    changes.append({"uuid": u, "prev": prev, "new": album})
            if changes and src not in ("undo", "redo"):
                _undo_stack.append({
                    "type": "reassign",
                    "desc": f"Moved {len(changes)} photo(s) → {album}",
                    "changes": changes,
                })
                _redo_stack.clear()
            self._json({"ok": True, "count": len(changes), "album": album}); return

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

        if self.path == "/api/merge-album":
            body = self._read_body()
            source = body.get("source", "").strip()
            target = body.get("target", "").strip()
            if not source or not target or source == target:
                self._json({"error": "source and target required and must differ"}, 400); return
            # Map source → target, source-unsure → target-unsure
            name_map = {source: target, source + "-unsure": target + "-unsure"}
            # Rewrite assignments.csv
            with ASSIGN.open() as f:
                reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
            merged = 0
            # Identify affected UUIDs BEFORE replacement
            affected_uuids = []
            for r in rows:
                if r["album"] in name_map:
                    affected_uuids.append(r["uuid"])
                    r["album"] = name_map[r["album"]]; merged += 1
            # Record undo BEFORE writing CSV
            _undo_stack.append({
                "type": "merge",
                "desc": f"Merged '{source}' → '{target}' ({len(affected_uuids)} photos)",
                "data": {"source": source, "target": target, "affected_uuids": affected_uuids},
            })
            _redo_stack.clear()
            with ASSIGN.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            # Rewrite corrections.csv
            with CORRECTIONS.open() as f:
                cr = csv.DictReader(f); cfields = cr.fieldnames; crows = list(cr)
            for r in crows:
                if r["new_album"] in name_map:
                    r["new_album"] = name_map[r["new_album"]]
            with CORRECTIONS.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cfields); w.writeheader(); w.writerows(crows)
            # Update in-memory state
            with _lock:
                for uid in list(assigned):
                    if assigned[uid] in name_map: assigned[uid] = name_map[assigned[uid]]
                for uid in list(corrections):
                    if corrections[uid] in name_map: corrections[uid] = name_map[corrections[uid]]
            # Remove source HTML files
            for src in name_map:
                stale = REVIEW / (_safe_fn(src) + ".html")
                if stale.exists(): stale.unlink()
            subprocess.run([sys.executable, str(ROOT / "04_contact_sheets.py")],
                           capture_output=True, timeout=120)
            self._json({"ok": True, "merged": merged, "map": name_map}); return

        if self.path == "/api/rename-album":
            body = self._read_body()
            old_name = body.get("old_name", "").strip()
            new_name = body.get("new_name", "").strip()
            if not old_name or not new_name:
                self._json({"error": "old_name and new_name required"}, 400); return
            if old_name.strip().lower() == new_name.strip().lower():
                self._json({"ok": True, "renamed": 0}); return
            # If new_name matches an existing album (case-insensitive), redirect to merge
            existing_lower = {a.lower(): a for a in all_albums()}
            canonical = existing_lower.get(new_name.lower())
            if canonical and canonical.lower() != old_name.lower():
                # Silently redirect: rename becomes a merge into the canonical album
                body2 = {"source": old_name, "target": canonical}
                # Inline merge logic (reuse same block via adjusted body)
                source, target = old_name, canonical
                name_map2 = {source: target, source + "-unsure": target + "-unsure"}
                with ASSIGN.open() as f:
                    reader = csv.DictReader(f); fields2 = reader.fieldnames; rows2 = list(reader)
                merged2 = 0
                for r in rows2:
                    if r["album"] in name_map2: r["album"] = name_map2[r["album"]]; merged2 += 1
                with ASSIGN.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields2); w.writeheader(); w.writerows(rows2)
                with CORRECTIONS.open() as f:
                    cr2 = csv.DictReader(f); cfields2 = cr2.fieldnames; crows2 = list(cr2)
                for r in crows2:
                    if r["new_album"] in name_map2: r["new_album"] = name_map2[r["new_album"]]
                with CORRECTIONS.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cfields2); w.writeheader(); w.writerows(crows2)
                with _lock:
                    for uid in list(assigned):
                        if assigned[uid] in name_map2: assigned[uid] = name_map2[assigned[uid]]
                    for uid in list(corrections):
                        if corrections[uid] in name_map2: corrections[uid] = name_map2[corrections[uid]]
                for src in name_map2:
                    stale = REVIEW / (_safe_fn(src) + ".html")
                    if stale.exists(): stale.unlink()
                subprocess.run([sys.executable, str(ROOT / "04_contact_sheets.py")],
                               capture_output=True, timeout=120)
                self._json({"ok": True, "action": "merged", "target": canonical, "merged": merged2}); return
            # Record undo BEFORE applying the rename
            _undo_stack.append({
                "type": "rename",
                "desc": f"Renamed '{old_name}' → '{new_name}'",
                "data": {"old_name": old_name, "new_name": new_name},
            })
            _redo_stack.clear()
            # Build mapping: also rename the -unsure variant
            name_map = {old_name: new_name}
            if not old_name.endswith("-unsure"):
                name_map[old_name + "-unsure"] = new_name + "-unsure"
            # Rewrite assignments.csv
            with ASSIGN.open() as f:
                reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
            renamed = 0
            for r in rows:
                if r["album"] in name_map:
                    r["album"] = name_map[r["album"]]; renamed += 1
            with ASSIGN.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            # Rewrite corrections.csv
            with CORRECTIONS.open() as f:
                cr = csv.DictReader(f); cfields = cr.fieldnames; crows = list(cr)
            for r in crows:
                if r["new_album"] in name_map:
                    r["new_album"] = name_map[r["new_album"]]
            with CORRECTIONS.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cfields); w.writeheader(); w.writerows(crows)
            # Update in-memory state
            with _lock:
                for uid in list(assigned):
                    if assigned[uid] in name_map: assigned[uid] = name_map[assigned[uid]]
                for uid in list(corrections):
                    if corrections[uid] in name_map: corrections[uid] = name_map[corrections[uid]]
            # Remove stale HTML files, regenerate all sheets
            for old in name_map:
                stale = REVIEW / (_safe_fn(old) + ".html")
                if stale.exists(): stale.unlink()
            subprocess.run([sys.executable, str(ROOT / "04_contact_sheets.py")],
                           capture_output=True, timeout=120)
            self._json({"ok": True, "renamed": renamed, "map": name_map}); return

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
