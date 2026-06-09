#!/usr/bin/env python3
"""Stage 4: Generate HTML contact sheets per album + STATUS.md.

Output:
  ~/photo-sort/review/index.html — list of all albums with counts + jumps
  ~/photo-sort/review/<album>.html — thumbnails grid per album
  ~/photo-sort/STATUS.md — overview for Yves

Cluster albums show CLIP-derived label suggestions so Yves can rename them.
"""
import csv
import html
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

ROOT = Path.home() / "photo-sort"
INV_CSV = ROOT / "metadata" / "inventory.csv"
ASSIGN_CSV = ROOT / "metadata" / "assignments.csv"
CLUSTER_META = ROOT / "metadata" / "cluster_meta.json"
REVIEW_DIR = ROOT / "review"
STATUS_MD = ROOT / "STATUS.md"
LOG_PATH = ROOT / "logs" / "04_sheets.log"

MAX_PHOTOS_PER_SHEET = 500
RANDOM_SEED = 42

REVIEW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sheets")

# Load inventory: uuid -> derivative_path
log.info("Loading inventory...")
uid_to_path = {}
uid_to_filename = {}
uid_to_date = {}
with INV_CSV.open() as f:
    for r in csv.DictReader(f):
        uid_to_path[r["uuid"]] = r["derivative_path"]
        uid_to_filename[r["uuid"]] = r["original_filename"]
        uid_to_date[r["uuid"]] = r["date"]

# Load assignments
log.info("Loading assignments...")
album_to_uids = defaultdict(list)
uid_to_assign = {}
with ASSIGN_CSV.open() as f:
    for r in csv.DictReader(f):
        album_to_uids[r["album"]].append(r["uuid"])
        uid_to_assign[r["uuid"]] = (r["album"], float(r["confidence"]), r["source"])

log.info(f"  {len(album_to_uids)} albums, {len(uid_to_assign)} photos assigned")

# Apply corrections on top of assignments (same logic as server.py / current_album())
CORRECTIONS_CSV = ROOT / "metadata" / "corrections.csv"
SKIP_ALBUMS_SET = {"Thrash", "_test_"}
if CORRECTIONS_CSV.exists():
    _corr: dict[str, str] = {}
    with CORRECTIONS_CSV.open() as f:
        for r in csv.DictReader(f):
            if r.get("new_album"):
                _corr[r["uuid"]] = r["new_album"]
            else:
                _corr.pop(r["uuid"], None)
    _moved = 0
    for _uid, _new_album in _corr.items():
        if _new_album in SKIP_ALBUMS_SET:
            # Remove from current album but don't add to Thrash/test in sheets
            _old = uid_to_assign.get(_uid, (None,))[0]
            if _old and _old in album_to_uids:
                try:
                    album_to_uids[_old].remove(_uid)
                except ValueError:
                    pass
            continue
        _old_album = uid_to_assign.get(_uid, (None,))[0]
        if _old_album == _new_album:
            continue
        if _old_album and _old_album in album_to_uids:
            try:
                album_to_uids[_old_album].remove(_uid)
            except ValueError:
                pass
        album_to_uids[_new_album].append(_uid)
        uid_to_assign[_uid] = (_new_album, 1.0, "correction")
        _moved += 1
    # Remove empty albums created by corrections
    album_to_uids = defaultdict(list, {k: v for k, v in album_to_uids.items() if v})
    if _moved:
        log.info(f"  Applied {len(_corr)} corrections ({_moved} photos moved between albums)")

# Cluster suggestions
cluster_meta = {}
if CLUSTER_META.exists():
    cluster_meta = json.loads(CLUSTER_META.read_text())


def safe_album_filename(album: str) -> str:
    return album.replace("/", "_").replace(" ", "_").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")


SERVER_BASE = "http://127.0.0.1:8765/img"


def img_url(uuid: str) -> str:
    return f"{SERVER_BASE}/{uuid}"


HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body{{font-family:-apple-system,sans-serif;background:#111;color:#eee;margin:0;padding:20px}}
  h1{{font-size:18px;margin:0 0 8px}}
  .meta{{color:#999;font-size:12px;margin-bottom:16px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}}
  .cell{{background:#222;border-radius:4px;overflow:hidden;position:relative;aspect-ratio:1;cursor:pointer;border:2px solid transparent}}
  .cell.corrected{{border-color:#4a8;display:none}}
  body.show-corrected .cell.corrected{{display:block}}
  body.show-corrected .cell.corrected{{opacity:0.5}}
  #toggleBar{{margin:10px 0;padding:6px 10px;background:#1a1a1a;border-radius:4px;font-size:12px;color:#999;display:flex;gap:12px;align-items:center}}
  #toggleBar button{{background:#333;color:#eee;border:0;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:12px}}
  #toggleBar button:hover{{background:#444}}
  .cell img{{width:100%;height:100%;object-fit:cover;display:block;pointer-events:none}}
  .cell .info{{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.7);font-size:9px;padding:2px 4px;font-family:monospace;color:#ccc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .cell .conf{{position:absolute;top:2px;right:2px;background:rgba(255,255,255,0.85);color:#000;font-size:9px;padding:1px 4px;border-radius:2px;font-weight:600}}
  .cell .moved{{position:absolute;top:2px;left:2px;background:#4a8;color:#000;font-size:9px;padding:1px 4px;border-radius:2px;font-weight:600}}
  .cell .trash-btn{{position:absolute;bottom:4px;right:4px;background:#c44;color:#fff;border:0;border-radius:3px;font-size:13px;width:22px;height:22px;cursor:pointer;display:none;line-height:22px;text-align:center;padding:0;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,0.6)}}
  .cell:hover .trash-btn{{display:block}}
  .cell.trashed{{border-color:#c55 !important}}
  .cell.trashed .moved{{background:#c55}}
  .nav{{margin:16px 0;padding:8px;background:#1a1a1a;border-radius:4px}}
  .nav a{{color:#7ac;margin-right:12px;text-decoration:none}}
  .suggestions{{background:#1c2530;padding:8px 12px;border-left:3px solid #4a8;margin:12px 0;font-size:13px}}
  .legend{{font-size:11px;color:#888;margin-bottom:12px}}

  /* Modal */
  #modal{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:100;overflow-y:auto;padding:20px}}
  #modal.open{{display:block}}
  #modal .panel{{max-width:1200px;margin:0 auto;background:#1a1a1a;border-radius:8px;padding:20px}}
  #modal h2{{margin:0 0 10px;font-size:16px}}
  #modal .row{{display:flex;gap:16px;margin-bottom:16px;align-items:flex-start}}
  #modal .preview{{flex:0 0 320px}}
  #modal .preview img{{width:100%;border-radius:4px}}
  #modal .controls{{flex:1}}
  #modal .controls label{{display:block;font-size:11px;color:#888;margin-bottom:4px}}
  #modal select, #modal input{{background:#222;color:#eee;border:1px solid #444;padding:6px 8px;border-radius:4px;font-size:14px;width:100%;box-sizing:border-box;font-family:inherit}}
  #modal button{{background:#4a8;color:#000;border:0;padding:8px 14px;border-radius:4px;cursor:pointer;font-weight:600;margin-right:8px;font-size:13px}}
  #modal button.secondary{{background:#444;color:#eee}}
  #modal button.danger{{background:#c55;color:#fff}}
  #modal .similar-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:6px;margin-top:12px;max-height:60vh;overflow-y:auto}}
  #modal .similar-grid .cell{{aspect-ratio:1;position:relative}}
  #modal .similar-grid .cell.selected{{outline:3px solid #4a8}}
  #modal .similar-grid .cell .score{{position:absolute;top:2px;left:2px;background:rgba(0,0,0,0.7);color:#fff;font-size:9px;padding:1px 3px;border-radius:2px}}
  #modal .closebtn{{position:absolute;top:12px;right:16px;background:none;color:#888;font-size:24px;border:0;cursor:pointer}}
  #modal .album-badge{{display:inline-block;background:#333;color:#aaa;padding:2px 8px;border-radius:3px;font-size:11px;margin-left:6px}}
  details>summary{{cursor:pointer;padding:8px 10px;background:#1a1a1a;border-radius:4px;font-size:13px;color:#888;user-select:none;margin-top:16px}}
  details>summary::marker{{color:#4a8;font-size:14px}}
  details>summary::-webkit-details-marker{{color:#4a8;font-size:14px}}
</style></head><body>
<div class="nav">{nav}</div>
<h1>{title}</h1>
<div class="meta">{meta}</div>
"""

HTML_TAIL = """
<div id="modal" onclick="if(event.target.id==='modal')closeModal()">
  <div class="panel" style="position:relative">
    <button class="closebtn" onclick="closeModal()">×</button>
    <h2 id="m-title">Reassign photo</h2>
    <div class="row">
      <div class="preview"><img id="m-img" src=""></div>
      <div class="controls">
        <div style="margin-bottom:10px">Current: <span id="m-current" class="album-badge"></span></div>
        <label>Target album</label>
        <select id="m-album"></select>
        <div style="margin-top:6px"><input id="m-new-album" type="text" placeholder="…or type a new album name"></div>
        <div style="margin-top:12px">
          <button id="m-confirm-btn" onclick="confirmOne()" style="background:#27ae60;color:#fff;display:none">✓ Confirm</button>
          <button onclick="saveOne()">Save this photo</button>
          <button class="secondary" onclick="loadSimilar()">Find similar →</button>
          <button class="danger" onclick="trashModal()" style="margin-top:6px">🗑 Thrash</button>
        </div>
        <div id="m-status" style="margin-top:10px;color:#888;font-size:12px"></div>
      </div>
    </div>
    <div id="m-similar-wrap" style="display:none">
      <div style="margin-bottom:6px;font-size:13px">Click thumbnails to toggle selection. Then:</div>
      <div style="margin-bottom:8px">
        <button onclick="selectAll(true)" class="secondary">Select all</button>
        <button onclick="selectAll(false)" class="secondary">Clear</button>
        <button onclick="saveSelected()">Reassign selected to target album</button>
      </div>
      <div id="m-similar" class="similar-grid"></div>
    </div>
  </div>
</div>
<script>
let albums = [];
let activeUuid = null;
let currentAlbum = null;
let similarItems = [];

async function loadAlbums() {
  if (albums.length) return;
  const r = await fetch('http://127.0.0.1:8765/api/albums');
  albums = await r.json();
  const sel = document.getElementById('m-album');
  sel.innerHTML = '';
  for (const a of albums) {
    const opt = document.createElement('option');
    opt.value = a; opt.textContent = a;
    sel.appendChild(opt);
  }
}

function setStatus(msg, color='#888') {
  const el = document.getElementById('m-status');
  el.textContent = msg; el.style.color = color;
}

// Returns the next non-corrected uuid after activeUuid in the grid
function getNextUuid() {
  const cells = [...document.querySelectorAll('.grid .cell[data-uuid]')];
  const idx = cells.findIndex(c => c.dataset.uuid === activeUuid);
  for (let i = idx + 1; i < cells.length; i++) {
    if (!cells[i].classList.contains('corrected')) return cells[i].dataset.uuid;
  }
  return null;
}

async function advanceModal() {
  const next = getNextUuid();
  if (next) {
    await openModal(next);
  } else {
    closeModal();
  }
}

async function openModal(uuid) {
  await loadAlbums();
  activeUuid = uuid;
  document.getElementById('m-img').src = `http://127.0.0.1:8765/img/${uuid}`;
  document.getElementById('m-title').textContent = `Reassign ${uuid.slice(0,8)}…`;
  const cur = await (await fetch(`http://127.0.0.1:8765/api/current?uuid=${uuid}`)).json();
  currentAlbum = cur.album;
  document.getElementById('m-current').textContent = cur.album + (cur.corrected ? ' (corrected)' : '');
  // Show/hide Confirm button
  const confirmBtn = document.getElementById('m-confirm-btn');
  if (cur.album.endsWith('-unsure')) {
    const mainAlbum = cur.album.slice(0, -'-unsure'.length);
    confirmBtn.textContent = `✓ Confirm → ${mainAlbum}`;
    confirmBtn.style.display = '';
  } else {
    confirmBtn.style.display = 'none';
  }
  const sel = document.getElementById('m-album');
  let target = cur.album;
  if (target.endsWith('-unsure')) {
    const main = target.slice(0, -'-unsure'.length);
    if (albums.includes(main)) target = main;
  }
  if (albums.includes(target)) sel.value = target;
  document.getElementById('m-new-album').value = '';
  document.getElementById('m-similar-wrap').style.display = 'none';
  document.getElementById('m-similar').innerHTML = '';
  similarItems = [];
  setStatus('');
  document.getElementById('modal').classList.add('open');
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
  activeUuid = null;
}

function targetAlbum() {
  const newName = document.getElementById('m-new-album').value.trim();
  return newName || document.getElementById('m-album').value;
}

async function saveOne() {
  const album = targetAlbum();
  if (!album) { setStatus('no album', '#c55'); return; }
  const r = await fetch('http://127.0.0.1:8765/api/reassign', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({uuid: activeUuid, new_album: album, source: 'manual'})
  });
  const j = await r.json();
  if (j.ok) {
    setStatus(`saved → ${album}`, '#4a8');
    markCellCorrected(activeUuid, album);
    await advanceModal();
  } else setStatus('error: ' + (j.error || 'unknown'), '#c55');
}

async function confirmOne() {
  if (!currentAlbum || !currentAlbum.endsWith('-unsure')) return;
  const album = currentAlbum.slice(0, -'-unsure'.length);
  const r = await fetch('http://127.0.0.1:8765/api/reassign', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({uuid: activeUuid, new_album: album, source: 'confirm'})
  });
  const j = await r.json();
  if (j.ok) {
    markCellCorrected(activeUuid, album);
    await advanceModal();
  } else setStatus('error: ' + (j.error || 'unknown'), '#c55');
}

// Trash a single uuid — callable from grid (event) or modal (no event)
async function trashOne(uuid, event) {
  if (event) event.stopPropagation();
  const r = await fetch('http://127.0.0.1:8765/api/reassign', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({uuid, new_album: 'Thrash', source: 'trash'})
  });
  const j = await r.json();
  if (j.ok) markCellCorrected(uuid, 'Thrash', true);
}

// Trash from modal then advance
async function trashModal() {
  await trashOne(activeUuid, null);
  await advanceModal();
}

async function loadSimilar() {
  setStatus('searching similar…');
  const target = targetAlbum();
  // Fetch extra candidates so we still have 24 after filtering out the target album
  const r = await fetch(`http://127.0.0.1:8765/api/similar?uuid=${activeUuid}&n=96`);
  const j = await r.json();
  // Only show photos NOT already in the target album
  similarItems = j.similar.filter(s => s.album !== target).slice(0, 24);
  if (similarItems.length === 0) {
    setStatus('No similar photos found in other albums.', '#888');
    return;
  }
  const grid = document.getElementById('m-similar');
  grid.innerHTML = '';
  for (const s of similarItems) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.dataset.uuid = s.uuid;
    cell.innerHTML = `<img src="http://127.0.0.1:8765/img/${s.uuid}" loading="lazy"><span class="score">${s.score.toFixed(2)}</span><span class="info">${s.album}</span>`;
    cell.onclick = () => cell.classList.toggle('selected');
    grid.appendChild(cell);
  }
  document.getElementById('m-similar-wrap').style.display = '';
  const skipped = j.similar.length - j.similar.filter(s => s.album !== target).slice(0, 24).length;
  setStatus(`${similarItems.length} similar from other albums (filtered ${j.similar.filter(s=>s.album===target).length} already in "${target}").`);
}

function selectAll(yes) {
  for (const c of document.querySelectorAll('#m-similar .cell')) {
    c.classList.toggle('selected', yes);
  }
}

async function saveSelected() {
  const album = targetAlbum();
  if (!album) { setStatus('no album', '#c55'); return; }
  const sel = [...document.querySelectorAll('#m-similar .cell.selected')].map(c => c.dataset.uuid);
  if (!sel.length) { setStatus('nothing selected', '#c55'); return; }
  const allUuids = [activeUuid, ...sel.filter(u => u !== activeUuid)];
  const r = await fetch('http://127.0.0.1:8765/api/reassign', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({uuids: allUuids, new_album: album, source: 'similar-batch'})
  });
  const j = await r.json();
  if (j.ok) {
    setStatus(`saved ${allUuids.length} → ${album}`, '#4a8');
    for (const u of allUuids) markCellCorrected(u, album);
    await advanceModal();
  } else setStatus('error: ' + (j.error || 'unknown'), '#c55');
}

function markCellCorrected(uuid, album, isTrashed=false) {
  for (const cell of document.querySelectorAll(`.cell[data-uuid="${uuid}"]`)) {
    if (cell.closest('#modal')) continue;
    cell.classList.add('corrected');
    if (isTrashed) cell.classList.add('trashed');
    let moved = cell.querySelector('.moved');
    if (!moved) {
      moved = document.createElement('span'); moved.className = 'moved';
      cell.appendChild(moved);
    }
    moved.textContent = isTrashed ? '🗑' : ('→ ' + album.slice(0,12));
  }
  updateHiddenCount();
  // Refresh inbox grid pagination if on index page
  if (typeof _renderInbox === 'function') _renderInbox();
}

function updateHiddenCount() {
  const n = document.querySelectorAll('.grid .cell.corrected').length;
  const el = document.getElementById('hiddenCount');
  if (el) el.textContent = `${n} corrected hidden`;
}

function toggleCorrected() {
  document.body.classList.toggle('show-corrected');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
  if ((e.key === 's' || e.key === 'S') && (e.metaKey || e.ctrlKey)) { e.preventDefault(); saveOne(); }
  if (e.key === 'ArrowRight' && document.getElementById('modal').classList.contains('open')) { e.preventDefault(); advanceModal(); }
  if (e.key === 'Delete' && document.getElementById('modal').classList.contains('open')) { e.preventDefault(); trashModal(); }
});

document.addEventListener('click', e => {
  // Trash button in grid — stops propagation itself via onclick
  const cell = e.target.closest('.cell[data-uuid]');
  if (cell && !cell.closest('#modal') && !e.target.closest('.trash-btn')) {
    openModal(cell.dataset.uuid);
  }
});

// Mark already-corrected cells on load
(async () => {
  const r = await fetch('http://127.0.0.1:8765/api/corrections');
  const c = await r.json();
  for (const [uuid, album] of Object.entries(c)) {
    markCellCorrected(uuid, album, album === 'Thrash');
  }
})();
</script>
</body></html>
"""


def render_sheet(album: str, uids: list) -> str:
    n_total = len(uids)
    # Sort by confidence descending (most certain first)
    def conf_of(u):
        return uid_to_assign[u][1] if u in uid_to_assign else 0.0
    uids_sorted = sorted(uids, key=conf_of, reverse=False)

    if n_total > MAX_PHOTOS_PER_SHEET:
        shown = uids_sorted[:MAX_PHOTOS_PER_SHEET]
        sample_note = f" (showing {MAX_PHOTOS_PER_SHEET} least-confident of {n_total})"
    else:
        shown = uids_sorted
        sample_note = ""

    # cluster suggestions block
    sugg_html = ""
    if album.startswith("Cluster-"):
        cid = album.split("-")[1]
        try:
            ci = int(cid)
            meta = cluster_meta.get(str(ci))
            if meta:
                sugg_lines = "<br>".join(
                    f"<code>{html.escape(s[0])}</code> ({s[1]:.3f})"
                    for s in meta.get("suggestions", [])
                )
                sugg_html = f'<div class="suggestions"><b>CLIP label suggestions (for rename):</b><br>{sugg_lines}</div>'
        except Exception:
            pass

    parts = [HTML_HEAD.format(
        title=html.escape(album),
        meta=f"{n_total} photos{sample_note}",
        nav='<a href="/">← Index</a> <span style="color:#888;font-size:12px">| Click photo to reassign · Modal: Esc closes · Cmd/Ctrl+S saves</span>',
    )]
    parts.append(sugg_html)
    parts.append('<div class="legend">Confidence in top-right. Click image to open modal · corrected photos auto-hidden.</div>')
    parts.append('<div id="toggleBar"><span id="hiddenCount">0 corrected hidden</span> <button onclick="toggleCorrected()">Toggle visibility</button></div>')
    parts.append('<div class="grid">')
    for uid in shown:
        path = uid_to_path.get(uid, "")
        if not path:
            continue
        _, conf, src = uid_to_assign[uid]
        conf_label = f"{conf:.2f}" if src != "cluster" else src
        fn = uid_to_filename.get(uid, "")[:20]
        date = uid_to_date.get(uid, "")[:10]
        url = img_url(uid)
        parts.append(
            f'<div class="cell" data-uuid="{uid}" data-conf="{conf:.4f}">'
            f'<img src="{url}" loading="lazy">'
            f'<span class="conf">{conf_label}</span>'
            f'<span class="info">{html.escape(fn)} {html.escape(date)} {html.escape(uid[:8])}</span>'
            f'<button class="trash-btn" onclick="trashOne(\'{uid}\',event)" title="Move to Thrash">🗑</button>'
            f'</div>'
        )
    parts.append('</div>')
    parts.append(HTML_TAIL)
    return "".join(parts)


# Sort albums: seed albums first, then -unsure, then clusters (largest first), Unsortiert last
def sort_key(album_uids):
    album, uids = album_uids
    n = len(uids)
    seed_order = {"Pornografie": 0, "Pornografie-unsure": 1, "Nacktheit": 2, "Nacktheit-unsure": 3,
                  "Motorrad-Werkstatt": 4, "Motorrad-Werkstatt-unsure": 5,
                  "Aquarium": 6, "Aquarium-unsure": 7,
                  "Garten": 8, "Garten-unsure": 9}
    if album in seed_order:
        return (0, seed_order[album], 0)
    if album == "Unsortiert":
        return (2, 0, 0)
    if album.startswith("Cluster-"):
        return (1, 0, -n)
    return (1, 1, -n)


albums_sorted = sorted(album_to_uids.items(), key=lambda x: x[0].lower())

# Load inbox (new unprocessed photos)
INBOX_CSV = ROOT / "metadata" / "inbox.csv"
inbox_rows = []
if INBOX_CSV.exists():
    inbox_rows = [r for r in csv.DictReader(INBOX_CSV.open())
                  if r.get("has_local_derivative") == "True" and r.get("derivative_path")]

# Generate Inbox.html if there are unprocessed photos
if inbox_rows:
    log.info(f"Writing Inbox.html ({len(inbox_rows)} photos)...")
    inbox_parts = [HTML_HEAD.format(
        title="Inbox — New Photos",
        meta=f"{len(inbox_rows)} new photos not yet classified. Assign them manually or use Auto-classify.",
        nav='<a href="/">← Index</a> <span style="color:#888;font-size:12px">| New photos — assign manually or auto-classify from the index</span>',
    )]
    inbox_parts.append('<div class="legend">These photos have not been classified yet. Click to assign an album.</div>')
    inbox_parts.append('<div id="toggleBar"><span id="hiddenCount">0 corrected hidden</span> <button onclick="toggleCorrected()">Toggle visibility</button></div>')
    inbox_parts.append('<div class="grid">')
    for r in inbox_rows:
        uid = r["uuid"]
        fn = r.get("original_filename", "")[:20]
        date = r.get("date", "")[:10]
        url = f"http://127.0.0.1:8765/img/{uid}"
        inbox_parts.append(
            f'<div class="cell" data-uuid="{uid}" data-conf="0">'
            f'<img src="{url}" loading="lazy">'
            f'<span class="conf" style="background:#e8a;color:#000">new</span>'
            f'<span class="info">{html.escape(fn)} {html.escape(date)} {html.escape(uid[:8])}</span>'
            f'<button class="trash-btn" onclick="trashOne(\'{uid}\',event)" title="Move to Thrash">🗑</button>'
            f'</div>'
        )
    inbox_parts.append('</div>')
    inbox_parts.append(HTML_TAIL)
    (REVIEW_DIR / "Inbox.html").write_text("".join(inbox_parts))

# Index page
log.info("Writing index.html...")
parts = [HTML_HEAD.format(
    title="Photo Sort — Review Index",
    meta=f"Total {len(uid_to_assign):,} photos across {len(albums_sorted)} albums.&nbsp;<span id='inbox-meta-extra' style='color:#7ac'></span>",
    nav='<span style="color:#aaa;font-weight:600;font-size:14px">Photo Sort — Review</span>',
)]
# Undo/Redo bar — very top, only visible when steps exist
parts.append("""
<div id="undo-bar" style="display:none;margin:0 0 8px;padding:7px 12px;background:#1a1a1a;border-radius:4px;align-items:center;gap:10px;font-size:12px">
  <button id="undo-btn" onclick="doUndoRedo('undo')" disabled
    style="background:#333;color:#aaa;border:0;padding:3px 10px;border-radius:3px;cursor:pointer;font-size:12px">↩ Undo</button>
  <span id="undo-desc" style="color:#555;flex:1">—</span>
  <button id="redo-btn" onclick="doUndoRedo('redo')" disabled
    style="background:#333;color:#aaa;border:0;padding:3px 10px;border-radius:3px;cursor:pointer;font-size:12px">Redo ↪</button>
</div>
<script>
(async () => {
  try {
    const r = await fetch('http://127.0.0.1:8765/api/undo-status');
    const j = await r.json();
    if (!j.can_undo && !j.can_redo) return;
    const bar = document.getElementById('undo-bar');
    bar.style.display = 'flex';
    const ub = document.getElementById('undo-btn');
    const rb = document.getElementById('redo-btn');
    const ud = document.getElementById('undo-desc');
    ub.disabled = !j.can_undo;
    rb.disabled = !j.can_redo;
    ub.style.color = j.can_undo ? '#eee' : '#555';
    rb.style.color = j.can_redo ? '#eee' : '#555';
    if (j.undo_desc) ud.textContent = j.undo_desc + (j.undo_count > 1 ? ` (+${j.undo_count - 1} more)` : '');
  } catch(e) {}
})();
async function doUndoRedo(action) {
  const r = await fetch('http://127.0.0.1:8765/api/' + action, {method:'POST'});
  const j = await r.json();
  if (j.ok) location.reload();
  else alert((action === 'undo' ? 'Undo' : 'Redo') + ' failed: ' + (j.error || 'unknown'));
}
</script>
""")
# Retrain bar — top of page
parts.append("""
<div id="retrain-bar" style="margin:8px 0 16px;padding:10px 14px;background:#1a2a1a;border:1px solid #4a8;border-radius:6px">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
    <span style="font-size:13px">&#x1F9E0; <b>Retrain:</b> <span id="corr-new">…</span> new corrections <span id="corr-total" style="color:#555;font-size:11px"></span></span>
    <button id="retrain-btn" onclick="doRetrain()" style="background:#27ae60;color:#fff;border:0;padding:6px 14px;border-radius:4px;cursor:pointer;font-weight:600;font-size:13px">Retrain &amp; Refresh</button>
    <span id="retrain-status" style="font-size:12px;color:#888"></span>
  </div>
  <div id="retrain-progress-wrap" style="display:none;margin-top:10px">
    <progress id="retrain-progress" max="100" value="0" style="width:100%;height:10px;accent-color:#27ae60"></progress>
    <div id="retrain-log" style="font-size:11px;color:#666;margin-top:4px;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"></div>
  </div>
</div>
<script>
(async () => {
  try {
    const r = await fetch('http://127.0.0.1:8765/api/retrain-status');
    const j = await r.json();
    document.getElementById('corr-new').textContent = j.new_since_retrain;
    document.getElementById('corr-total').textContent = j.last_retrain
      ? `(${j.total} total, last retrain ${j.last_retrain.slice(0,10)})`
      : `(${j.total} total, no retrain yet)`;
    const btn = document.getElementById('retrain-btn');
    if (j.new_since_retrain < 50) {
      btn.title = `Only ${j.new_since_retrain} new corrections — threshold for a meaningful retrain is 200. Keep sorting!`;
      btn.style.opacity = '0.45';
    } else if (j.new_since_retrain < 200) {
      btn.title = `${j.new_since_retrain}/200 new corrections — retrain works now but improves significantly at 200+.`;
      btn.style.opacity = '0.75';
    } else {
      btn.title = `${j.new_since_retrain} new corrections — good time to retrain!`;
      btn.style.boxShadow = '0 0 0 2px #27ae60';
    }
  } catch(e) { document.getElementById('corr-new').textContent = '?'; }
})();
function doRetrain() {
  const btn = document.getElementById('retrain-btn');
  const st = document.getElementById('retrain-status');
  const wrap = document.getElementById('retrain-progress-wrap');
  const bar = document.getElementById('retrain-progress');
  const logEl = document.getElementById('retrain-log');
  btn.disabled = true;
  btn.textContent = 'Retraining…';
  st.textContent = 'Do not close this tab.';
  st.style.color = '#888';
  wrap.style.display = 'block';
  bar.value = 2;
  const phases = [
    ['Loading corrections', 5],
    ['Loading original', 10],
    ['Loading CLIP', 15],
    ['Building training', 22],
    ['Training Logistic', 32],
    ['Predicting all', 55],
    ['Backup', 60],
    ['Wrote', 62],
    ['Regenerating', 65],
    ['Writing index', 95],
    ['Retrain complete', 98],
  ];
  const es = new EventSource('http://127.0.0.1:8765/api/retrain-stream?regen=1');
  es.onmessage = function(e) {
    const ev = JSON.parse(e.data);
    if (ev.type === 'done') {
      bar.value = 100;
      st.textContent = '✓ Done — ' + ev.loaded + ' photos reloaded. Refreshing…';
      st.style.color = '#4a8';
      logEl.textContent = '';
      es.close();
      setTimeout(() => location.reload(), 1800);
    } else if (ev.type === 'error') {
      st.textContent = 'Error — check server log';
      st.style.color = '#c55';
      logEl.textContent = ev.msg || '';
      es.close();
      btn.disabled = false;
      btn.textContent = 'Retrain & Refresh';
    } else {
      const clean = ev.msg.replace(/^\\d{{4}}-\\d{{2}}-\\d{{2}} \\d{{2}}:\\d{{2}}:\\d{{2}},\\d{{3}} \\w+ /, '');
      logEl.textContent = clean;
      for (const [kw, pct] of phases) {
        if (ev.msg.includes(kw) && bar.value < pct) {{ bar.value = pct; break; }}
      }
    }
  };
  es.onerror = function() {
    if (bar.value < 100) {
      st.textContent = 'Connection lost — retrain may still be running in background.';
      st.style.color = '#c55';
      es.close();
    }
  };
}
</script>
""")
parts.append(f"""
<div id="inbox-bar" style="margin:6px 0 8px;padding:10px 14px;background:#1a1f2a;border:1px solid #7ac;border-radius:6px;display:flex;align-items:center;gap:14px;flex-wrap:wrap">
  <span id="inbox-label" style="font-size:13px">📥 <b>Inbox:</b> <span id="inbox-count">{"⏳" if not inbox_rows else len(inbox_rows)}</span> new photos not yet classified</span>
  <button id="inbox-scan-btn" onclick="runInboxPhase('detect')"
    style="background:#2a4a6a;color:#7ac;border:1px solid #7ac;padding:5px 12px;border-radius:4px;cursor:pointer;font-size:12px;font-weight:600">
    🔍 Scan for new
  </button>
  <button id="inbox-process-btn" onclick="runInboxPhase('process')"
    style="background:#27ae60;color:#fff;border:0;padding:5px 12px;border-radius:4px;cursor:pointer;font-size:12px;font-weight:600;{'opacity:0.45;pointer-events:none' if not inbox_rows else ''}">
    ⚡ Auto-classify {"(" + str(len(inbox_rows)) + " photos)" if inbox_rows else ""}
  </button>
  <span id="inbox-status" style="font-size:12px;color:#888"></span>
</div>
<div id="inbox-scan-progress" style="display:none;margin:0 0 6px;padding:8px 12px;background:#111;border-radius:4px">
  <progress id="inbox-scan-bar" max="100" value="0" style="width:100%;height:8px;accent-color:#7ac"></progress>
  <div id="inbox-scan-text" style="font-size:11px;color:#666;margin-top:3px;font-family:monospace"></div>
</div>
<div id="inbox-log-wrap" style="display:none;margin:0 0 8px;padding:8px 12px;background:#111;border-radius:4px;font-size:11px;font-family:monospace;color:#666;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"></div>
<script>
async function runInboxPhase(phase) {{
  const btn = document.getElementById(phase === 'detect' ? 'inbox-scan-btn' : 'inbox-process-btn');
  const st = document.getElementById('inbox-status');
  const logWrap = document.getElementById('inbox-log-wrap');
  const progressWrap = document.getElementById('inbox-scan-progress');
  const progressBar = document.getElementById('inbox-scan-bar');
  const progressText = document.getElementById('inbox-scan-text');
  btn.disabled = true;
  st.textContent = phase === 'detect' ? 'Scanning Photos.app…' : 'Embedding + classifying…';
  st.style.color = '#888';
  if (phase === 'detect') {{
    progressWrap.style.display = 'block';
    progressBar.value = 0; progressBar.max = 100;
    progressText.textContent = 'Starting scan…';
  }} else {{
    logWrap.style.display = 'block';
  }}
  const es = new EventSource('http://127.0.0.1:8765/api/inbox-' + phase + '-stream');
  es.onmessage = function(e) {{
    const ev = JSON.parse(e.data);
    if (ev.type === 'done') {{
      if (phase === 'detect') {{ progressBar.value = progressBar.max; progressText.textContent = 'Scan complete!'; }}
      st.textContent = '✓ Done' + (ev.count !== undefined ? ' — ' + ev.count + ' new photos' : '') + '. Refreshing…';
      st.style.color = '#4a8';
      es.close();
      setTimeout(() => location.reload(), 1200);
    }} else if (ev.type === 'error') {{
      st.textContent = 'Error: ' + ev.msg;
      st.style.color = '#c55';
      es.close();
      btn.disabled = false;
    }} else if (ev.type === 'progress') {{
      if (ev.total > 0) {{
        progressBar.max = ev.total; progressBar.value = ev.current;
        const pct = Math.round(ev.current / ev.total * 100);
        progressText.textContent = 'Scanning… ' + ev.current.toLocaleString() + ' / ' + ev.total.toLocaleString() + ' (' + pct + '%)';
        const mx = document.getElementById('inbox-meta-extra');
        if (mx) mx.textContent = ' · scanning ' + pct + '%…';
      }}
    }} else {{
      logWrap.textContent = ev.msg.replace(/^.*? (INFO|WARNING) /, '');
    }}
  }};
  es.onerror = function() {{
    if (!st.textContent.startsWith('✓')) {{
      st.textContent = 'Connection lost';
      st.style.color = '#c55';
      es.close();
    }}
  }};
}}
(async () => {{
  try {{
    const r = await fetch('http://127.0.0.1:8765/api/inbox-count');
    const j = await r.json();
    // Counts: j.count=unsorted, j.total=all inbox, j.sorted=manually assigned
    if (j.total === 0) {{
      // Empty inbox: muted bar, scan button stays visible for manual refresh
      const bar = document.getElementById('inbox-bar');
      bar.style.borderColor = '#2a2a2a';
      bar.style.background = '#141414';
      const lbl = document.getElementById('inbox-label');
      if (lbl) lbl.innerHTML = '<span style="color:#555;font-size:12px">📥 Inbox: no new photos</span>';
      document.getElementById('inbox-process-btn').style.display = 'none';
      return;
    }}
    document.getElementById('inbox-count').textContent = j.count;
    const mx = document.getElementById('inbox-meta-extra');
    if (mx) {{
      if (j.count > 0) mx.textContent = ' · ' + j.count + ' unsorted in Inbox' + (j.sorted > 0 ? ' · ✓ ' + j.sorted + ' sorted' : '');
      else if (j.total > 0) mx.textContent = ' · ✓ Inbox all sorted — click ⚡ to commit';
    }}
    const pb = document.getElementById('inbox-process-btn');
    if (j.total > 0) {{
      pb.style.opacity = '1'; pb.style.pointerEvents = '';
      pb.textContent = '⚡ ' + (j.sorted > 0 && j.count === 0 ? 'Commit to albums (' + j.total + ')' : 'Auto-classify (' + j.total + ' photos)');
      pb.title = j.sorted > 0
        ? 'Processes all ' + j.total + ' inbox photos. Manually sorted photos keep your album assignment. The ' + j.count + ' remaining get ML classification.'
        : 'Embed + classify all ' + j.total + ' inbox photos with CLIP ML model.';
    }}
  }} catch(e) {{}}
}})();
</script>
""")
# Inline inbox grid — directly below inbox-bar, visually connected
if inbox_rows:
    _ibox_ps = 48
    parts.append(f"""
<style>
  #inbox-grid .cell.inbox-selected {{ outline:3px solid #7ac; opacity:.9; }}
  #inbox-grid .cell {{ cursor:pointer; }}
</style>
<div id="inbox-section" style="margin:-1px 0 12px;padding:10px 16px 14px;background:#111d2a;border:1px solid #7ac;border-top:none;border-radius:0 0 6px 6px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap">
    <button id="inbox-sel-all-btn" onclick="inboxToggleSelectAll()"
      style="background:#1a2a3a;color:#7ac;border:1px solid #7ac;padding:3px 10px;border-radius:3px;cursor:pointer;font-size:12px">☐ Select all</button>
    <div id="inbox-batch-bar" style="display:none;align-items:center;gap:8px;flex-wrap:wrap">
      <span id="inbox-sel-count" style="font-size:12px;color:#7ac;font-weight:600"></span>
      <select id="inbox-batch-album"
        style="background:#222;color:#eee;border:1px solid #444;padding:3px 8px;border-radius:3px;font-size:12px;max-width:180px"></select>
      <button onclick="inboxBatchAssign()"
        style="background:#27ae60;color:#fff;border:0;padding:3px 10px;border-radius:3px;cursor:pointer;font-size:12px;font-weight:600">Assign →</button>
      <button onclick="inboxBatchTrash()"
        style="background:#c44;color:#fff;border:0;padding:3px 10px;border-radius:3px;cursor:pointer;font-size:12px">🗑 Trash</button>
      <button onclick="inboxDeselectAll()"
        style="background:none;color:#555;border:0;padding:3px 6px;cursor:pointer;font-size:14px">✕</button>
    </div>
  </div>
  <div id="inbox-grid" class="grid" style="margin-top:4px">""")
    for _r in inbox_rows:
        _uid = _r["uuid"]
        _fn = html.escape(_r.get("original_filename", "")[:20])
        _date = html.escape(_r.get("date", "")[:10])
        _url = f"http://127.0.0.1:8765/img/{_uid}"
        parts.append(
            f'<div class="cell" data-uuid="{_uid}" data-conf="0" style="display:none">'
            f'<img src="{_url}" loading="lazy">'
            f'<span class="conf" style="background:#e8a;color:#000">new</span>'
            f'<span class="info">{_fn} {_date} {html.escape(_uid[:8])}</span>'
            f'<button class="trash-btn" onclick="trashOne(\'{_uid}\',event)" title="Move to Thrash">🗑</button>'
            f'</div>'
        )
    parts.append(f"""  </div>
  <div id="inbox-pagination" style="margin-top:10px;display:flex;align-items:center;gap:10px">
    <button id="inbox-prev-btn" onclick="inboxPrevPage()"
      style="background:#333;color:#eee;border:0;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:12px">← Prev</button>
    <span id="inbox-page-info" style="color:#888;font-size:12px"></span>
    <button id="inbox-next-btn" onclick="inboxNextPage()"
      style="background:#333;color:#eee;border:0;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:12px">Next →</button>
  </div>
</div>
<script>
let _iboxPage = 0;
const _iboxCells = [...document.querySelectorAll('#inbox-grid .cell')];
function _renderInbox() {{
  const uncorrected = _iboxCells.filter(c => !c.classList.contains('corrected'));
  const total = _iboxCells.length;
  const sortedN = total - uncorrected.length;
  const section = document.getElementById('inbox-section');

  if (uncorrected.length === 0) {{
    if (section) section.style.display = 'none';
    const countEl = document.getElementById('inbox-count');
    if (countEl) countEl.textContent = '0';
    const mx = document.getElementById('inbox-meta-extra');
    if (mx) mx.textContent = total > 0 ? ' · ✓ Inbox all sorted — click ⚡ to commit' : '';
    return;
  }}

  if (section) section.style.display = '';
  const totalPages = Math.max(1, Math.ceil(uncorrected.length / {_ibox_ps}));
  if (_iboxPage >= totalPages) _iboxPage = Math.max(0, totalPages - 1);
  _iboxCells.forEach(c => {{ c.style.display = 'none'; }});
  uncorrected.slice(_iboxPage * {_ibox_ps}, (_iboxPage + 1) * {_ibox_ps}).forEach(c => {{ c.style.display = ''; }});

  const sortedNote = sortedN > 0 ? ' · ✓ ' + sortedN + ' sorted' : '';
  document.getElementById('inbox-page-info').textContent =
    'Page ' + (_iboxPage + 1) + ' of ' + totalPages + ' · ' + uncorrected.length + ' unsorted' + sortedNote;
  const pag = document.getElementById('inbox-pagination');
  if (pag) pag.style.display = totalPages <= 1 ? 'none' : '';
  document.getElementById('inbox-prev-btn').disabled = _iboxPage === 0;
  document.getElementById('inbox-next-btn').disabled = _iboxPage >= totalPages - 1;

  const countEl = document.getElementById('inbox-count');
  if (countEl) countEl.textContent = uncorrected.length;
  const mx = document.getElementById('inbox-meta-extra');
  if (mx) mx.textContent = ' · ' + uncorrected.length + ' unsorted in Inbox' + sortedNote;
}}
function inboxPrevPage() {{ if (_iboxPage > 0) {{ _iboxPage--; _renderInbox(); }} }}
function inboxNextPage() {{
  const uncorrected = _iboxCells.filter(c => !c.classList.contains('corrected'));
  if (_iboxPage < Math.ceil(uncorrected.length / {_ibox_ps}) - 1) {{ _iboxPage++; _renderInbox(); }}
}}

// ── Inbox selection & batch actions ─────────────────────────────────────────
let _inboxAlbumsLoaded = false;
async function _loadInboxBatchAlbums() {{
  if (_inboxAlbumsLoaded) return;
  const r = await fetch('http://127.0.0.1:8765/api/albums');
  const albs = await r.json();
  const sel = document.getElementById('inbox-batch-album');
  sel.innerHTML = albs.map(a => `<option value="${{a}}">${{a}}</option>`).join('');
  _inboxAlbumsLoaded = true;
}}
function _inboxVisible() {{
  return _iboxCells.filter(c => c.style.display !== 'none' && !c.classList.contains('corrected'));
}}
function _inboxSelected() {{
  return _iboxCells.filter(c => c.classList.contains('inbox-selected'));
}}
function _updateBatchBar() {{
  const sel = _inboxSelected();
  const bar = document.getElementById('inbox-batch-bar');
  const cnt = document.getElementById('inbox-sel-count');
  if (bar) bar.style.display = sel.length > 0 ? 'flex' : 'none';
  if (cnt) cnt.textContent = sel.length + ' selected';
  if (sel.length > 0) _loadInboxBatchAlbums();
  const btn = document.getElementById('inbox-sel-all-btn');
  if (!btn) return;
  const vis = _inboxVisible();
  btn.textContent = vis.length > 0 && vis.every(c => c.classList.contains('inbox-selected'))
    ? '☑ Deselect all' : '☐ Select all';
}}
function inboxToggleSelectAll() {{
  const vis = _inboxVisible();
  const allSel = vis.every(c => c.classList.contains('inbox-selected'));
  vis.forEach(c => c.classList.toggle('inbox-selected', !allSel));
  _updateBatchBar();
}}
function inboxDeselectAll() {{
  _iboxCells.forEach(c => c.classList.remove('inbox-selected'));
  _updateBatchBar();
}}
async function inboxBatchAssign() {{
  const uuids = _inboxSelected().map(c => c.dataset.uuid);
  if (!uuids.length) return;
  const album = document.getElementById('inbox-batch-album').value;
  if (!album) return;
  const r = await fetch('http://127.0.0.1:8765/api/reassign', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{uuids, new_album: album, source:'manual'}})
  }});
  const j = await r.json();
  if (j.ok) {{ for (const u of uuids) markCellCorrected(u, album); inboxDeselectAll(); }}
}}
async function inboxBatchTrash() {{
  const uuids = _inboxSelected().map(c => c.dataset.uuid);
  if (!uuids.length) return;
  const r = await fetch('http://127.0.0.1:8765/api/reassign', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{uuids, new_album:'Thrash', source:'trash'}})
  }});
  const j = await r.json();
  if (j.ok) {{
    for (const u of uuids) markCellCorrected(u, 'Thrash', true);
    inboxDeselectAll();
    // Refresh thrash counter
    const t = await fetch('http://127.0.0.1:8765/api/thrash-count');
    const tj = await t.json();
    const el = document.getElementById('thrash-count');
    if (el) el.textContent = tj.count;
    const btn = document.getElementById('empty-thrash-btn');
    if (btn) btn.textContent = 'Empty Thrash (' + tj.count + ')';
    const bar = document.getElementById('thrash-bar');
    if (bar) bar.style.display = tj.count === 0 ? 'none' : '';
  }}
}}
// Click in inbox grid = toggle selection (not open modal)
document.getElementById('inbox-grid').addEventListener('click', function(e) {{
  if (e.target.closest('.trash-btn')) return;
  const cell = e.target.closest('.cell[data-uuid]');
  if (!cell) return;
  e.stopPropagation();
  cell.classList.toggle('inbox-selected');
  _updateBatchBar();
}});
_renderInbox();
</script>
""")
# undo bar inserted above (before inbox bar)
parts.append("""
<div id="thrash-bar" style="margin:12px 0;padding:10px 14px;background:#2a1a1a;border:1px solid #c55;border-radius:6px;display:flex;align-items:center;gap:14px">
  <span style="font-size:13px">🗑 <b>Thrash:</b> <span id="thrash-count">…</span> photos marked for deletion</span>
  <button id="empty-thrash-btn" onclick="openEmptyThrash()" style="background:#c55;color:#fff;border:0;padding:6px 14px;border-radius:4px;cursor:pointer;font-weight:600;font-size:13px">Empty Thrash</button>
  <a id="view-thrash-link" href="/Thrash.html" style="color:#c55;font-size:13px;text-decoration:none">View Thrash →</a>
</div>
<div id="empty-thrash-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:200;align-items:center;justify-content:center">
  <div style="background:#1a1a1a;border:1px solid #c55;border-radius:8px;padding:28px 32px;max-width:420px;width:100%">
    <h2 style="margin:0 0 10px;color:#c55">⚠️ Empty Thrash</h2>
    <p id="empty-thrash-msg" style="color:#ccc;font-size:14px;margin:0 0 16px"></p>
    <p style="color:#888;font-size:12px;margin:0 0 12px">This moves photos to <b>Recently Deleted</b> in Photos.app. iCloud will sync this to all devices. You have 30 days to recover them.</p>
    <p style="color:#888;font-size:12px;margin:0 0 16px">Type <b>DELETE</b> to confirm:</p>
    <input id="empty-thrash-confirm" type="text" placeholder="DELETE" style="background:#222;color:#eee;border:1px solid #555;padding:8px;border-radius:4px;font-size:14px;width:100%;box-sizing:border-box;margin-bottom:14px">
    <div>
      <button onclick="doEmptyThrash()" style="background:#c55;color:#fff;border:0;padding:8px 18px;border-radius:4px;cursor:pointer;font-weight:600;margin-right:8px">Delete from Photos.app</button>
      <button onclick="closeEmptyThrash()" style="background:#444;color:#eee;border:0;padding:8px 14px;border-radius:4px;cursor:pointer">Cancel</button>
    </div>
    <div id="empty-thrash-status" style="margin-top:10px;font-size:13px;color:#888"></div>
  </div>
</div>
<script>
(async () => {
  const r = await fetch('http://127.0.0.1:8765/api/thrash-count');
  const j = await r.json();
  document.getElementById('thrash-count').textContent = j.count;
  if (j.count === 0) {
    document.getElementById('thrash-bar').style.display = 'none';
  } else {
    document.getElementById('empty-thrash-btn').textContent = `Empty Thrash (${j.count})`;
  }
})();
function openEmptyThrash() {
  const count = document.getElementById('thrash-count').textContent;
  document.getElementById('empty-thrash-msg').textContent = `${count} photos will be moved to Recently Deleted in Photos.app.`;
  document.getElementById('empty-thrash-confirm').value = '';
  document.getElementById('empty-thrash-status').textContent = '';
  document.getElementById('empty-thrash-modal').style.display = 'flex';
}
function closeEmptyThrash() {
  document.getElementById('empty-thrash-modal').style.display = 'none';
}
async function doEmptyThrash() {
  const confirm = document.getElementById('empty-thrash-confirm').value.trim();
  if (confirm !== 'DELETE') {
    document.getElementById('empty-thrash-status').textContent = 'Type DELETE to confirm.';
    document.getElementById('empty-thrash-status').style.color = '#c55';
    return;
  }
  document.getElementById('empty-thrash-status').textContent = 'Deleting…';
  document.getElementById('empty-thrash-status').style.color = '#888';
  const r = await fetch('http://127.0.0.1:8765/api/empty-thrash', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({confirm: 'DELETE'})
  });
  const j = await r.json();
  if (j.ok) {
    document.getElementById('empty-thrash-status').textContent = `✓ ${j.message}`;
    document.getElementById('empty-thrash-status').style.color = '#4a8';
    document.getElementById('thrash-count').textContent = '0';
    document.getElementById('empty-thrash-btn').textContent = 'Empty Thrash (0)';
    document.getElementById('empty-thrash-btn').disabled = true;
    setTimeout(closeEmptyThrash, 2000);
  } else {
    document.getElementById('empty-thrash-status').textContent = 'Error: ' + (j.error || 'unknown');
    document.getElementById('empty-thrash-status').style.color = '#c55';
  }
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeEmptyThrash(); });
</script>
""")

TABLE_HEADER = ('<tr style="background:#1a1a1a">'
                '<th style="text-align:left;padding:6px 8px">Album</th>'
                '<th style="text-align:right;padding:6px 8px">Count</th>'
                '<th style="width:28px"></th></tr>')

main_albums = [(a, u) for a, u in albums_sorted if not a.endswith('-unsure')]
unsure_albums = [(a, u) for a, u in albums_sorted if a.endswith('-unsure')]
unsure_by_main = {a[:-len('-unsure')]: (a, u) for a, u in unsure_albums}


unsure_total = sum(len(u) for _, u in unsure_albums)
parts.append(
    '<div style="margin:24px 0 10px;display:flex;align-items:center;gap:10px">'
    '<span style="font-size:12px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:.06em">Albums</span>'
    '<div style="flex:1;height:1px;background:#252525"></div>'
    '</div>'
)
parts.append(f'<p style="color:#555;font-size:11px;margin:0 0 6px">'
             f'{len(main_albums)} albums · {len(unsure_albums)} with unsure photos ({unsure_total:,} total)</p>')
parts.append('<table style="width:100%;border-collapse:collapse;margin-bottom:32px">')
parts.append(TABLE_HEADER)
for album, uids in main_albums:
    fn = safe_album_filename(album) + ".html"
    row_id = f"ur-{safe_album_filename(album)}"
    unsure_row_html = ""

    if album in unsure_by_main:
        u_album, u_uids = unsure_by_main[album]
        u_fn = safe_album_filename(u_album) + ".html"
        toggle = (f'<button onclick="toggleUnsure(\'{row_id}\',this)" '
                  f'title="{len(u_uids):,} unsure photos — click to expand" '
                  f'style="background:none;border:none;color:#555;cursor:pointer;'
                  f'padding:0 5px 0 0;font-size:11px;line-height:1;vertical-align:middle">▶</button>')
        unsure_row_html = (
            f'<tr id="{row_id}" style="display:none">'
            f'<td style="padding:3px 8px 3px 24px;background:#161616">'
            f'<a href="{u_fn}" style="color:#8ab;font-size:12px">↳ {html.escape(u_album)}</a></td>'
            f'<td style="text-align:right;padding:3px 8px;background:#161616;'
            f'font-size:12px;color:#666">{len(u_uids):,}</td>'
            f'<td style="background:#161616"></td></tr>'
        )
    else:
        toggle = '<span style="display:inline-block;width:20px"></span>'

    esc = html.escape(album).replace("'", "\\'")
    edit_btn = (f'<button onclick="startRename(\'{esc}\',this)" title="Rename album" '
                f'style="background:none;border:none;color:#555;cursor:pointer;'
                f'padding:2px 3px;font-size:12px;line-height:1">✏️</button>')
    merge_btn = (f'<button onclick="startMerge(\'{esc}\',this)" title="Merge into another album" '
                 f'style="background:none;border:none;color:#555;cursor:pointer;'
                 f'padding:2px 3px;font-size:12px;line-height:1">⊕</button>')

    parts.append(
        f'<tr style="border-bottom:1px solid #222">'
        f'<td style="padding:6px 8px">{toggle}'
        f'<a href="{fn}" style="color:#7ac">{html.escape(album)}</a></td>'
        f'<td style="text-align:right;padding:6px 8px">{len(uids):,}</td>'
        f'<td style="width:52px;padding:2px 4px;text-align:right;white-space:nowrap">'
        f'{edit_btn}{merge_btn}</td></tr>'
    )
    if unsure_row_html:
        parts.append(unsure_row_html)

parts.append('</table>')
parts.append("""
<script>
function toggleUnsure(id, btn) {
  const row = document.getElementById(id);
  const hidden = row.style.display === 'none';
  row.style.display = hidden ? '' : 'none';
  btn.textContent = hidden ? '▼' : '▶';
  btn.style.color = hidden ? '#4a8' : '#555';
}
function startRename(albumName, btn) {
  const td = btn.closest('tr').querySelector('td:first-child');
  const link = td.querySelector('a');
  if (td.querySelector('input')) return;
  const input = document.createElement('input');
  input.value = albumName;
  input.style.cssText = 'background:#222;color:#eee;border:1px solid #4a8;padding:2px 6px;border-radius:3px;font-size:13px;width:180px;margin-left:2px';
  input.onkeydown = (e) => {
    if (e.key === 'Enter') doRename(albumName, input.value.trim(), input, btn, link);
    if (e.key === 'Escape') { input.remove(); link.style.display=''; btn.style.display=''; }
  };
  link.style.display = 'none';
  btn.style.display = 'none';
  td.insertBefore(input, link);
  input.focus(); input.select();
}
async function doRename(oldName, newName, input, btn, link) {
  if (!newName || newName === oldName) {
    input.remove(); link.style.display=''; btn.style.display=''; return;
  }
  input.disabled = true;
  input.style.opacity = '0.5';
  const r = await fetch('http://127.0.0.1:8765/api/rename-album', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({old_name: oldName, new_name: newName})
  });
  const j = await r.json();
  if (j.ok) { location.reload(); }
  else { alert('Rename failed: ' + (j.error || 'unknown')); input.disabled=false; input.style.opacity='1'; }
}
function startMerge(albumName, btn) {
  const actionCell = btn.closest('td');
  if (actionCell.querySelector('select')) return;
  // Collect all main album names from the table (exclude self and -unsure rows)
  const allAlbums = [...document.querySelectorAll('table tr td:first-child > a')]
    .map(a => a.textContent.trim())
    .filter(n => !n.endsWith('-unsure') && n !== albumName);
  const sel = document.createElement('select');
  sel.style.cssText = 'position:absolute;right:0;top:100%;background:#1a1a1a;color:#eee;border:1px solid #e8a;padding:4px 6px;border-radius:4px;font-size:12px;z-index:50;min-width:180px;max-height:300px;overflow-y:auto';
  sel.innerHTML = '<option value="">— merge into… —</option>' +
    allAlbums.map(n => `<option value="${n}">${n}</option>`).join('');
  sel.onchange = () => {
    const target = sel.value;
    if (!target) return;
    const hasUnsure = document.getElementById('ur-' + albumName.replace(/ /g,'_').replace(/[^a-zA-Z0-9_()]/g,'_'));
    const unsureNote = hasUnsure ? `\\n${albumName}-unsure → ${target}-unsure` : '';
    if (confirm(`Merge "${albumName}" into "${target}"?${unsureNote}\\n\\nAll photos will be moved. "${albumName}" will be removed. This cannot be undone easily.`)) {
      doMerge(albumName, target, sel);
    } else { sel.remove(); }
  };
  sel.onkeydown = (e) => { if (e.key === 'Escape') sel.remove(); };
  sel.onblur = () => setTimeout(() => sel.remove(), 200);
  // Position relative to action cell
  actionCell.style.position = 'relative';
  actionCell.appendChild(sel);
  sel.focus();
}
async function doMerge(source, target, sel) {
  sel.disabled = true;
  const r = await fetch('http://127.0.0.1:8765/api/merge-album', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({source, target})
  });
  const j = await r.json();
  if (j.ok) { location.reload(); }
  else { alert('Merge failed: ' + (j.error || 'unknown')); sel.remove(); }
}
</script>
""")
parts.append(HTML_TAIL)
(REVIEW_DIR / "index.html").write_text("".join(parts))

# Per-album pages
log.info("Writing per-album pages...")
for album, uids in albums_sorted:
    fn = safe_album_filename(album) + ".html"
    (REVIEW_DIR / fn).write_text(render_sheet(album, uids))

# STATUS.md
log.info("Writing STATUS.md...")
total = sum(len(u) for _, u in albums_sorted)
status_lines = [
    "# Photo Sort — Status",
    "",
    f"**Total photos processed:** {total:,}",
    f"**Albums generated:** {len(albums_sorted)}",
    "",
    "## Review",
    "",
    f"Open `~/photo-sort/review/index.html` in a browser. Each album is a contact sheet — confidence badge top-right, filename + date overlay.",
    "",
    f"Cluster albums (`Cluster-XX`) include CLIP-derived label suggestions to help you rename them.",
    "",
    "## Album sizes",
    "",
    "| Album | Count | Notes |",
    "| --- | ---: | --- |",
]
for album, uids in albums_sorted:
    note = ""
    if album.endswith("-unsure"):
        note = "below 80% confidence — manual recheck"
    elif album == "Unsortiert":
        note = "cluster noise + below 50% confidence"
    elif album.startswith("Cluster-"):
        try:
            cid = int(album.split("-")[1])
            m = cluster_meta.get(str(cid), {})
            if m.get("suggestions"):
                note = "Suggested: " + ", ".join(s[0] for s in m["suggestions"][:2])
        except Exception:
            pass
    status_lines.append(f"| {album} | {len(uids):,} | {note} |")

status_lines.extend([
    "",
    "## Next steps",
    "",
    "1. Open `review/index.html` and click through albums.",
    "2. For each `Cluster-XX`: decide on a final album name (or merge with another).",
    "3. For each `-unsure` album: spot-check; tell me which ones look fine to merge into the main album.",
    "4. Once names are decided, I write the albums to Photos.app via `osxphotos`. **iCloud will sync** the new albums to all your devices.",
    "",
    "## Caveats",
    "",
    "- NSFW detection (NudeNet) has false positives on artistic nudes, babies, beach photos. Spot-check `Pornografie` and `Nacktheit` carefully.",
    "- CLIP zero-shot on the small thumbnails (60% of photos are <100KB) is less precise than on full-res. Expect 70-85% accuracy on the seed categories.",
    "- Cluster suggestions are CLIP nearest-text-prompts to the cluster centroid — they're hints, not labels.",
    "",
])
STATUS_MD.write_text("\n".join(status_lines))

log.info(f"DONE. {len(albums_sorted)} sheets in {REVIEW_DIR}")
log.info(f"Status: {STATUS_MD}")
