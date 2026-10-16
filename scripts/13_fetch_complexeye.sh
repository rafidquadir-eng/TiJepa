#!/usr/bin/env bash
# F1.4.1: fetch the ComplexEye archives (Zenodo 10082674, CC BY 4.0) ONE AT A TIME,
# with resume and MD5 verification against the Zenodo record.
#
# zenodo.org is reachable from this instance only intermittently (roughly one
# connection in three succeeds; see results/v1/preflight.json and the v1 run report),
# so every transfer is retried indefinitely with backoff rather than failed. aria2c
# resumes a partial file instead of restarting it.
#
# An archive is downloaded only if it is absent or its MD5 does not match the record.
# R08.zip failed verification in v0 (its transfer was interrupted, which is what
# produced the 14 corrupt movie members) and is therefore re-fetched.
#
# Order is "verify what is here, then fetch what is missing" so that a restart of this
# script costs nothing. Extraction and preprocessing are NOT done here: the corpus
# builder deletes each zip after preprocessing it, which is what bounds disk use.
#
# Usage: 13_fetch_complexeye.sh [KEY ...]
#   with no arguments, every archive in the record is considered (smallest first);
#   with arguments, only those keys are fetched. The corpus orchestrator calls it one
#   key at a time so that a zip is deleted after preprocessing and the set never has to
#   fit on disk all at once.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/data/raw/complexeye"
REC="$DEST/.zenodo_record_10082674.json"
LOG="$REPO/logs/v1_complexeye_fetch.log"
MAX_GB=200           # PLAN_v1.md F1.4.1 hard cap on bytes downloaded
mkdir -p "$DEST" "$REPO/logs"

if [ ! -f "$REC" ]; then
  echo "FATAL: $REC absent; fetch the Zenodo record first" | tee -a "$LOG"; exit 1
fi

# key<TAB>md5<TAB>size, smallest first so that cheap wins land early
mapfile -t ROWS < <(python3 - "$REC" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for f in sorted(d["files"], key=lambda f: f["size"]):
    print(f"{f['key']}\t{f['checksum'].split(':')[1]}\t{f['size']}")
PY
)

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
WANT=("$@")
if [ ${#WANT[@]} -gt 0 ]; then
  keep=()
  for row in "${ROWS[@]}"; do
    k="${row%%$'\t'*}"
    for w in "${WANT[@]}"; do [ "$k" = "$w" ] && keep+=("$row"); done
  done
  ROWS=("${keep[@]}")
fi
say "=== ComplexEye fetch: ${#ROWS[@]} archive(s) selected, cap ${MAX_GB} GB"

downloaded_bytes=0
for row in "${ROWS[@]}"; do
  key="${row%%$'\t'*}"; rest="${row#*$'\t'}"
  md5="${rest%%$'\t'*}"; size="${rest#*$'\t'}"
  f="$DEST/$key"

  if [ -f "$f" ]; then
    if echo "$md5  $f" | md5sum -c - >/dev/null 2>&1; then
      say "OK (already verified) $key"; continue
    fi
    say "MD5 MISMATCH, refetching $key"
  fi

  gb=$(( size / 1000000000 ))
  if [ $(( downloaded_bytes / 1000000000 + gb )) -gt $MAX_GB ]; then
    say "STOP: ${MAX_GB} GB download cap would be exceeded by $key"; break
  fi

  # Zenodo stalls often; retry forever with backoff. aria2c resumes partials.
  attempt=0
  while true; do
    attempt=$(( attempt + 1 ))
    say "fetch $key attempt $attempt ($(( size / 1000000 )) MB)"
    aria2c --continue=true --max-tries=5 --retry-wait=20 --timeout=60 \
           --connect-timeout=30 --max-connection-per-server=4 --split=4 \
           --min-split-size=200M --summary-interval=0 --console-log-level=warn \
           --auto-file-renaming=false --allow-overwrite=true \
           -d "$DEST" -o "$key" \
           "https://zenodo.org/records/10082674/files/${key}?download=1" \
           >>"$LOG" 2>&1
    if echo "$md5  $f" | md5sum -c - >/dev/null 2>&1; then
      say "VERIFIED $key"
      downloaded_bytes=$(( downloaded_bytes + size ))
      break
    fi
    if [ -f "$f" ]; then
      say "  incomplete or corrupt after attempt $attempt ($(stat -c%s "$f") / $size bytes)"
    fi
    if [ $attempt -ge 40 ]; then
      say "GIVING UP on $key after $attempt attempts"; break
    fi
    sleep $(( attempt < 10 ? 30 : 120 ))
  done
done

if [ ${#WANT[@]} -gt 0 ]; then
  say "=== fetch pass complete for ${WANT[*]}"
  exit 0
fi

say "=== fetch pass complete; verifying the whole set"
python3 - "$REC" "$DEST" <<'PY' | tee -a "$LOG"
import hashlib, json, pathlib, sys
rec, dest = sys.argv[1], pathlib.Path(sys.argv[2])
d = json.load(open(rec))
ok = miss = bad = 0
for f in d["files"]:
    p = dest / f["key"]
    if not p.exists():
        print(f"ABSENT   {f['key']}"); miss += 1; continue
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() == f["checksum"].split(":")[1]:
        print(f"VERIFIED {f['key']}"); ok += 1
    else:
        print(f"CORRUPT  {f['key']}"); bad += 1
print(f"SUMMARY verified={ok} absent={miss} corrupt={bad}")
PY
