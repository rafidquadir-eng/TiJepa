#!/usr/bin/env bash
# F9.1: back up all v1 checkpoints and outputs to the author's Google Drive and verify.
#
# As with the v0 backup, only the remote's name and type are ever read from rclone; no
# config, token or credential is printed, logged or committed.
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

copy_and_check () {  # <local dir> <remote subdir> [extra rclone args...]
  local src="$1" dst="$2"; shift 2
  [ -e "$src" ] || { echo "  skip ${src}: absent"; return 0; }
  echo "== ${src} -> ${REMOTE}:TiJepa/${dst}/"
  rclone copy "$src" "${REMOTE}:TiJepa/${dst}/" --stats-one-line "$@"
  rclone check "$src" "${REMOTE}:TiJepa/${dst}/" --one-way "$@"
}

copy_and_check checkpoints          checkpoints_v1 --include "tijepa1*"
copy_and_check results              results_v1
copy_and_check figures              figures_v1
copy_and_check paper                paper_v1
copy_and_check reports              reports_v1
copy_and_check requirements-lock.txt lockfile_v1

# the corpus index is the one large artefact worth keeping off the instance
if [ -f data/processed/v1/pretrain_index.csv ]; then
  copy_and_check data/processed/v1/pretrain_index.csv corpus_index_v1
fi

echo "== md5s of the v1 checkpoints"
for f in checkpoints/tijepa1*.pt; do
  [ -e "$f" ] || continue
  echo "local  $(md5sum "$f")"
  echo "remote $(rclone md5sum "${REMOTE}:TiJepa/checkpoints_v1/$(basename "$f")")"
done
