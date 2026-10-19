#!/usr/bin/env bash
# G1.3: back up each run's checkpoint to the author's Drive every ~3 h, md5-verified.
#
# A hard link is taken first, so the trainer's atomic os.replace cannot change the bytes
# under an upload in progress. The step-14,751 partial checkpoint that the previous
# instance uploaded to TiJepa/checkpoints_v1/ is only overwritten after a verified copy
# exists in TiJepa/checkpoints_v1_partial/ (G0.5).
#
# Never prints rclone config or tokens; only file names, sizes and md5s are logged.
# Usage: 98_backup_loop_v2.sh [interval_seconds] [--once]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="gdrive:TiJepa"
SNAP="$REPO/checkpoints/drive_snapshot"
LOG="$REPO/logs/v2_backup.log"
INTERVAL="${1:-10800}"
ONCE="${2:-}"
mkdir -p "$SNAP" "$REPO/logs"
say() { echo "[$(date -u +%FT%TZ)] $*" >>"$LOG"; }

verify() {  # local_file remote_dir
  local l r name
  name="$(basename "$1")"
  l="$(md5sum "$1" | cut -d' ' -f1)"
  r="$(rclone md5sum "$2" --include "$name" 2>>"$LOG" | cut -d' ' -f1)"
  if [ -n "$r" ] && [ "$l" = "$r" ]; then say "VERIFIED $2/$name md5 $l"; return 0; fi
  say "MISMATCH or missing $2/$name local $l remote ${r:-none}"; return 1
}

partial_ok() {
  local p="$REPO/checkpoints/v1_partial"
  if verify "$p/tijepa1_T1_seed0.pt" "$REMOTE/checkpoints_v1_partial" 2>/dev/null; then
    return 0
  fi
  rclone copy "$p" "$REMOTE/checkpoints_v1_partial/" --include "tijepa1_T1_seed0.pt" \
    --include "pretrain_log_tijepa1_seed0.csv" >>"$LOG" 2>&1 || return 1
  verify "$p/tijepa1_T1_seed0.pt" "$REMOTE/checkpoints_v1_partial" &&
    verify "$p/pretrain_log_tijepa1_seed0.csv" "$REMOTE/checkpoints_v1_partial"
}

while true; do
  say "=== backup pass"
  if ! rclone lsd "$REMOTE" >/dev/null 2>&1; then
    say "rclone remote unusable (not authorised on this instance); nothing copied"
  elif ! partial_ok; then
    say "partial checkpoint not yet verified on Drive; run checkpoints left untouched"
  else
    for run in tijepa1 tijepa1_spatial; do
      ck="$REPO/checkpoints/${run}_T1_seed0.pt"
      [ -f "$ck" ] || continue
      ln -f "$ck" "$SNAP/${run}_T1_seed0.pt"
      cp -f "$REPO/checkpoints/pretrain_log_${run}_seed0.csv" "$SNAP/" 2>/dev/null
      rclone copy "$SNAP" "$REMOTE/checkpoints_v1/" --include "${run}_T1_seed0.pt" \
        --include "pretrain_log_${run}_seed0.csv" >>"$LOG" 2>&1
      verify "$SNAP/${run}_T1_seed0.pt" "$REMOTE/checkpoints_v1"
    done
  fi
  [ "$ONCE" = "--once" ] && break
  sleep "$INTERVAL"
done
