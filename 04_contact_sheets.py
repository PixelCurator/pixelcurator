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
</style></head><body>
<div class="nav"><a href="/">← Index</a> <span style="color:#888;font-size:12px">| Click photo to reassign · Modal: Esc closes · Cmd/Ctrl+S saves</span></div>
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
          <button onclick="saveOne()">Save this photo</button>
          <button class="secondary" onclick="loadSimilar()">Find similar →</button>
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

async function openModal(uuid) {
  await loadAlbums();
  activeUuid = uuid;
  document.getElementById('m-img').src = `http://127.0.0.1:8765/img/${uuid}`;
  document.getElementById('m-title').textContent = `Reassign ${uuid.slice(0,8)}…`;
  const cur = await (await fetch(`http://127.0.0.1:8765/api/current?uuid=${uuid}`)).json();
  document.getElementById('m-current').textContent = cur.album + (cur.corrected ? ' (corrected)' : '');
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
  } else setStatus('error: ' + (j.error || 'unknown'), '#c55');
}

async function loadSimilar() {
  setStatus('searching similar…');
  const r = await fetch(`http://127.0.0.1:8765/api/similar?uuid=${activeUuid}&n=24`);
  const j = await r.json();
  similarItems = j.similar;
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
  setStatus(`${similarItems.length} similar found. Click to select then reassign.`);
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
  const r = await fetch('http://127.0.0.1:8765/api/reassign', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({uuids: sel, new_album: album, source: 'similar-batch'})
  });
  const j = await r.json();
  if (j.ok) {
    setStatus(`saved ${sel.length} → ${album}`, '#4a8');
    for (const u of sel) markCellCorrected(u, album);
  } else setStatus('error: ' + (j.error || 'unknown'), '#c55');
}

function markCellCorrected(uuid, album) {
  for (const cell of document.querySelectorAll(`.cell[data-uuid="${uuid}"]`)) {
    // only hide in main grid (not inside the modal's similar grid)
    if (cell.closest('#modal')) continue;
    cell.classList.add('corrected');
    let moved = cell.querySelector('.moved');
    if (!moved) {
      moved = document.createElement('span'); moved.className = 'moved';
      cell.appendChild(moved);
    }
    moved.textContent = '→ ' + album.slice(0,12);
  }
  updateHiddenCount();
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
});

document.addEventListener('click', e => {
  const cell = e.target.closest('.cell[data-uuid]');
  if (cell && !cell.closest('#modal')) {
    openModal(cell.dataset.uuid);
  }
});

// Mark already-corrected cells on load
(async () => {
  const r = await fetch('http://127.0.0.1:8765/api/corrections');
  const c = await r.json();
  for (const [uuid, album] of Object.entries(c)) markCellCorrected(uuid, album);
})();
</script>
</body></html>
"""


def render_sheet(album: str, uids: list) -> str:
    n_total = len(uids)
    if n_total > MAX_PHOTOS_PER_SHEET:
        random.seed(RANDOM_SEED)
        shown = random.sample(uids, MAX_PHOTOS_PER_SHEET)
        sample_note = f" (showing {MAX_PHOTOS_PER_SHEET} random of {n_total})"
    else:
        shown = uids
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
            f'<div class="cell" data-uuid="{uid}">'
            f'<img src="{url}" loading="lazy">'
            f'<span class="conf">{conf_label}</span>'
            f'<span class="info">{html.escape(fn)} {html.escape(date)} {html.escape(uid[:8])}</span>'
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


albums_sorted = sorted(album_to_uids.items(), key=sort_key)

# Index page
log.info("Writing index.html...")
parts = [HTML_HEAD.format(title="Photo Sort — Review Index", meta=f"Total {len(uid_to_assign)} photos across {len(albums_sorted)} albums.")]
parts.append('<table style="width:100%;border-collapse:collapse">')
parts.append('<tr style="background:#1a1a1a"><th style="text-align:left;padding:6px">Album</th><th style="text-align:right;padding:6px">Count</th><th style="text-align:left;padding:6px">Suggestion</th></tr>')
for album, uids in albums_sorted:
    fn = safe_album_filename(album) + ".html"
    sugg = ""
    if album.startswith("Cluster-"):
        try:
            cid = int(album.split("-")[1])
            m = cluster_meta.get(str(cid), {})
            if m.get("suggestions"):
                sugg = ", ".join(s[0] for s in m["suggestions"][:3])
        except Exception:
            pass
    parts.append(
        f'<tr style="border-bottom:1px solid #333"><td style="padding:6px"><a href="{fn}" style="color:#7ac">{html.escape(album)}</a></td>'
        f'<td style="text-align:right;padding:6px">{len(uids)}</td>'
        f'<td style="padding:6px;color:#999;font-size:12px">{html.escape(sugg)}</td></tr>'
    )
parts.append('</table>')
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
