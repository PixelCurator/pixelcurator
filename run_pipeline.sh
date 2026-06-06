#!/usr/bin/env bash
# Orchestrator: wait for embedding done, then run NSFW, classify+cluster, contact sheets.
# Writes ~/photo-sort/PIPELINE_STATUS each transition.
set -u

ROOT="$HOME/photo-sort"
LOG="$ROOT/logs/pipeline.log"
STATUS="$ROOT/PIPELINE_STATUS"
ACT_PY="$ROOT/venv/bin/python"

mkdir -p "$ROOT/logs"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
set_status() { echo "$1" > "$STATUS"; log "STATUS → $1"; }

set_status "WAITING_FOR_EMBED"
log "=== Pipeline orchestrator started ==="

# 1) wait for embed to complete
while ! grep -q "DONE\." "$ROOT/logs/01_embed.log" 2>/dev/null; do
  # also bail on a fatal traceback
  if grep -q "Traceback" "$ROOT/logs/01_embed.log" 2>/dev/null; then
    if ! grep -q "DONE\." "$ROOT/logs/01_embed.log" 2>/dev/null; then
      set_status "FAILED_EMBED"
      log "Embedding traceback detected without DONE — aborting."
      exit 1
    fi
  fi
  sleep 30
done
log "Embedding complete."

# 2) NSFW
set_status "RUNNING_NSFW"
log "Starting Stage 2 (NSFW)..."
"$ACT_PY" "$ROOT/02_nsfw.py" >> "$ROOT/logs/02_nsfw.stdout" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  set_status "FAILED_NSFW"
  log "NSFW failed (rc=$rc) — aborting."
  exit 1
fi
log "NSFW done."

# 3) Classify + cluster
set_status "RUNNING_CLASSIFY"
log "Starting Stage 3 (classify + cluster)..."
"$ACT_PY" "$ROOT/03_classify_cluster.py" >> "$ROOT/logs/03_classify.stdout" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  set_status "FAILED_CLASSIFY"
  log "Classify failed (rc=$rc) — aborting."
  exit 1
fi
log "Classify+cluster done."

# 4) Contact sheets + STATUS.md
set_status "RUNNING_SHEETS"
log "Starting Stage 4 (contact sheets)..."
"$ACT_PY" "$ROOT/04_contact_sheets.py" >> "$ROOT/logs/04_sheets.stdout" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  set_status "FAILED_SHEETS"
  log "Sheets failed (rc=$rc) — aborting."
  exit 1
fi

set_status "DONE"
log "=== Pipeline complete ==="
