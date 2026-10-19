#!/usr/bin/env bash
# F0.1: back up the v0 checkpoint to the author's Google Drive BEFORE anything else.
#
# Google Drive is the author's private storage and is authorised for backups
# (PLAN_v1.md Section 1 rule 6). It is not publication. Nothing here prints, logs or
# commits rclone config, tokens or credentials: the only thing read from rclone is the
# remote's NAME and TYPE, via `rclone listremotes --long`.
set -uo pipefail
cd "$(dirname "$0")/.."

if ! command -v rclone >/dev/null; then
  echo "rclone not installed; see reports/BACKUP.md for manual commands"; exit 1
fi
REMOTE="$(rclone listremotes --long 2>/dev/null | awk '$2=="drive"{print $1; exit}')"
REMOTE="${REMOTE%:}"
if [ -z "$REMOTE" ]; then
  echo "no rclone remote of type 'drive'; see reports/BACKUP.md"; exit 1
fi
echo "using remote of type drive: ${REMOTE}"

if [ ! -f checkpoints/tijepa0_T1_seed0.pt ]; then
  echo "v0 checkpoint not on this instance; nothing to back up"; exit 0
fi
rclone copy checkpoints/ "${REMOTE}:TiJepa/checkpoints_v0/" \
  --include "*.pt" --include "*.csv" --include "*.json" --stats-one-line
rclone check checkpoints/ "${REMOTE}:TiJepa/checkpoints_v0/" --one-way --include "*.pt"
echo "local  md5: $(md5sum checkpoints/tijepa0_T1_seed0.pt | cut -d' ' -f1)"
echo "remote md5: $(rclone md5sum "${REMOTE}:TiJepa/checkpoints_v0/tijepa0_T1_seed0.pt" | cut -d' ' -f1)"
