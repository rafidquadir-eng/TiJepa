"""F1.4: build the v1 pretraining corpus, one archive at a time.

The v0 corpus used 7 of 21 ComplexEye archives, one source, no QC, no leakage check and
two temporal strides (review findings D4a-e, D5a, C5). This script addresses each of
those, and it is structured around the fact that the full archive set is 187 GB while
this instance has roughly 220 GB free: an archive is fetched, preprocessed, and its zip
is then **deleted** before the next is fetched, so peak disk use is one archive plus the
shards.

Per archive:
  1. fetch if absent or if its MD5 does not match the Zenodo record (13_fetch_complexeye.sh);
  2. for every movie member, in parallel across cores:
     a. fingerprint it and drop it if it matches an S-BIAD2196 evaluation movie
        (F1.3; the fingerprints are loaded from the leakage check, so exclusion still
        works after the zip is gone),
     b. measure frame QC (F1.4.4),
     c. resample to the common 0.65 um/px using a **verified** pixel size,
     d. detect and track cells, and cut cell-centred clips at strides {1, 3, 9, 27},
     e. make a fraction of clips static (F1.4.7);
  3. append to the shard writer and the index;
  4. delete the zip.

State is written after every archive, so the script is resumable: re-running it skips
archives already recorded as done.

Pixel sizes are verified, not assumed (F1.4.2). From Cibir et al., Nat Commun 2023
(DOI 10.1038/s41467-023-43765-3) Methods: the ComplexEye imager is an OmniVision OV9712
with 1280 x 800 native resolution over a 3888 x 2430 um active area (3.0375 um per
sensor pixel) at 4.69x magnification, giving 0.648 um/px in the sample plane, which
agrees within 1% with the 0.641 um/px stated in the S-BIAD2196 record for the same
instrument. The reference microscope is a Leica DMI6000 B whose reported field of view
is 662.9 x 495.2 um; the Leica movies here are 1392 x 1040, so the pixel size is
662.9/1392 = 0.4762 um/px and 495.2/1040 = 0.4762 um/px - consistent to four decimal
places in both axes, which is why the Leica movies are **included** in v1 rather than
dropped for an unverifiable scale.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.preprocess import detect as D  # noqa: E402
from tijepa.preprocess import fingerprint as FP  # noqa: E402
from tijepa.preprocess import qc as QC  # noqa: E402

RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed" / "v1"
STAGE = REPO / "data" / "interim" / "stage"
OUT = REPO / "results" / "v1"
CE = RAW / "complexeye"
RECORD = CE / ".zenodo_record_10082674.json"
FETCH = REPO / "scripts" / "13_fetch_complexeye.sh"

CLIP, NF = 96, 16
CLIP_BYTES = NF * CLIP * CLIP
SHARD_CLIPS = 7000           # ~1.03 GB per shard, satisfying the >= 1 GB shard rule
STRIDES = (1, 3, 9, 27)      # F1.4.5
STATIC_P = 0.1               # F1.4.7

# Verified pixel sizes; see the module docstring for the derivation of each.
UM_PER_PX_COMPLEXEYE = 0.641
UM_PER_PX_LEICA = 0.4762


def source_of(zip_name: str) -> str:
    return "complexeye_leica" if "leica" in zip_name.lower() else "complexeye"


def um_per_px_for(zip_name: str) -> float:
    return (UM_PER_PX_LEICA if source_of(zip_name) == "complexeye_leica"
            else UM_PER_PX_COMPLEXEYE)


def members_of(z: Path) -> list[str]:
    try:
        with zipfile.ZipFile(z) as zf:
            return sorted(m for m in zf.namelist()
                          if m.lower().endswith((".tif", ".tiff"))
                          and not m.startswith("__MACOSX"))
    except zipfile.BadZipFile:
        return []


def movie_worker(args: tuple) -> dict:
    """Fingerprint, QC, resample, track and clip one movie member."""
    (zip_path, member, um_per_px, out_npy, max_clips, seed,
     eval_vectors, eval_shapes, eval_sha1s) = args
    t_start = time.time()
    tmpdir = tempfile.mkdtemp(prefix="tijepa_", dir=str(STAGE))
    key = f"{Path(zip_path).name}:{member}"
    try:
        import tifffile

        with zipfile.ZipFile(zip_path) as zf:
            local = Path(tmpdir) / Path(member).name
            with zf.open(member) as src, local.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
        with tifffile.TiffFile(local) as tf:
            n_pages = len(tf.pages)
            if n_pages < NF:
                return {"key": key, "status": "skipped_too_short", "n_frames": n_pages}
            frames = np.stack([tf.pages[i].asarray() for i in range(n_pages)])
        if frames.ndim != 3:
            return {"key": key, "status": "skipped_not_3d",
                    "shape": str(frames.shape)}

        # ---- (a) leakage: is this an S-BIAD2196 evaluation movie?
        fp = FP.fingerprint(frames)
        for name, vec in eval_vectors.items():
            r = FP.correlate(fp["vector"], vec)
            identical = fp["frame0_sha1"] == eval_sha1s[name]
            same_shape = list(frames.shape[1:]) == list(eval_shapes[name][1:])
            if identical or (same_shape and r >= FP.MATCH_R):
                return {"key": key, "status": "excluded_leakage",
                        "matches_eval_movie": name, "pearson_r": round(r, 5),
                        "frame0_sha1_identical": bool(identical)}

        # ---- (b) QC metrics (thresholds are applied by the parent, pooled)
        qc_metrics = QC.frame_metrics(frames)

        # ---- (c) resample to the common pixel size
        frames = D.maybe_resample(frames, um_per_px)

        # ---- (d) detect, track, clip at several strides
        det = D.detect_movie(frames, stride=1)
        linked = D.link_tracks(det, min_len=NF)
        rng = np.random.default_rng(seed)
        clips, meta = D.extract_clips(frames, linked, crop=CLIP, n_frames=NF,
                                      strides=STRIDES, max_clips=max_clips, rng=rng)

        # ---- (e) static-clip augmentation
        clips, static_mask = QC.apply_static_augmentation(clips, rng, p=STATIC_P)
        for m, s in zip(meta, static_mask.tolist(), strict=False):
            m["static"] = bool(s)

        if len(clips):
            np.save(out_npy, clips)
        return {
            "key": key, "status": "ok", "n_frames": int(n_pages),
            "n_detections": int(len(det)),
            "n_tracks": int(linked["particle"].nunique()) if len(linked) else 0,
            "n_clips": int(len(clips)), "n_static": int(static_mask.sum()),
            "meta": meta, "qc": qc_metrics,
            "npy": str(out_npy) if len(clips) else None,
            "um_per_px": um_per_px, "seconds": round(time.time() - t_start, 1),
        }
    except Exception as exc:
        return {"key": key, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class ShardWriter:
    """Accumulates clips and flushes >= 1 GB uint8 shards, resumably."""

    def __init__(self, out_dir: Path, start_id: int = 0):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_id = start_id
        self.buf: list[np.ndarray] = []
        self.buf_meta: list[dict] = []
        self.index_rows: list[dict] = []

    @property
    def buffered(self) -> int:
        return sum(len(b) for b in self.buf)

    def add(self, arr: np.ndarray, metas: list[dict]) -> None:
        self.buf.append(arr)
        self.buf_meta.extend(metas)
        while self.buffered >= SHARD_CLIPS:
            self.flush(full_only=True)

    def flush(self, full_only: bool = False) -> None:
        if not self.buf:
            return
        arr = np.concatenate(self.buf)
        take = SHARD_CLIPS if (full_only and len(arr) > SHARD_CLIPS) else len(arr)
        out, rest = arr[:take], arr[take:]
        metas, rest_meta = self.buf_meta[:take], self.buf_meta[take:]
        path = self.dir / f"pretrain_shard_{self.shard_id:04d}.npy"
        np.save(path, out)
        for off, m in enumerate(metas):
            self.index_rows.append({**m, "shard": path.name, "offset": off})
        print(f"    wrote {path.name}: {len(out)} clips, "
              f"{path.stat().st_size/1e9:.2f} GB")
        self.shard_id += 1
        self.buf = [rest] if len(rest) else []
        self.buf_meta = rest_meta


def fetch_archive(key: str, log: Path) -> bool:
    """Run the fetch script for one archive; return True if it verifies."""
    print(f"  fetching {key} (this can take a long while: zenodo.org is flaky here)")
    r = subprocess.run(["bash", str(FETCH), key], capture_output=True, text=True)
    log.write_text(log.read_text() + r.stdout[-4000:] if log.exists() else r.stdout)
    return "VERIFIED" in r.stdout


def md5_ok(path: Path, want: str) -> bool:
    import hashlib
    if not path.exists():
        return False
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest() == want


def main() -> int:
    ap = argparse.ArgumentParser()
    # 700 clips/movie over ~900 movies is ~630k clips = ~93 GB of uint8 shards, which
    # fits this instance's free disk alongside one in-flight archive. v0 had 288k.
    ap.add_argument("--max-clips-per-movie", type=int, default=700)
    ap.add_argument("--max-clips-total", type=int, default=630_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--archives", default="all",
                    help="comma-separated archive keys, or 'all'")
    ap.add_argument("--keep-zips", action="store_true",
                    help="do not delete each zip after preprocessing (needs ~190 GB)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="only process archives already on disk")
    args = ap.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "pretrain_corpus.json"
    fetch_log = REPO / "logs" / "v1_complexeye_fetch.log"

    record = json.loads(RECORD.read_text())
    want_md5 = {f["key"]: f["checksum"].split(":")[1] for f in record["files"]}
    sizes = {f["key"]: f["size"] for f in record["files"]}
    keys = ([k.strip() for k in args.archives.split(",")]
            if args.archives != "all"
            else sorted(want_md5, key=lambda k: sizes[k]))

    # ---- evaluation fingerprints, for leakage exclusion after the zips are gone
    leak_path = OUT / "leakage_check.json"
    eval_vectors: dict[str, np.ndarray] = {}
    eval_shapes: dict[str, list] = {}
    eval_sha1s: dict[str, str] = {}
    fp_cache = REPO / "data" / "interim" / "eval_fingerprints.npz"
    if fp_cache.exists():
        z = np.load(fp_cache, allow_pickle=True)
        for name in z["names"]:
            eval_vectors[str(name)] = z[f"vec_{name}"]
            eval_shapes[str(name)] = z[f"shape_{name}"].tolist()
            eval_sha1s[str(name)] = str(z[f"sha_{name}"])
    else:
        from tijepa.io import sbiad
        names = []
        store: dict[str, np.ndarray] = {}
        for m in sorted((RAW / "s-biad2196" / "Celltrack_data").glob("*.zip")):
            raw, _ = sbiad.load_movie(m)
            fp = FP.fingerprint(raw)
            eval_vectors[m.name] = fp["vector"]
            eval_shapes[m.name] = fp["shape"]
            eval_sha1s[m.name] = fp["frame0_sha1"]
            names.append(m.name)
            store[f"vec_{m.name}"] = fp["vector"]
            store[f"shape_{m.name}"] = np.asarray(fp["shape"])
            store[f"sha_{m.name}"] = np.asarray(fp["frame0_sha1"])
        fp_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(fp_cache, names=np.asarray(names), **store)
        print(f"fingerprinted {len(names)} evaluation movies -> {fp_cache.name}")
    print(f"leakage reference: {len(eval_vectors)} S-BIAD2196 movies "
          f"({'from ' + leak_path.name if leak_path.exists() else 'freshly computed'})")

    # ---- resume
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    done = set(state.get("archives_done", []))
    writer = ShardWriter(PROC, start_id=state.get("next_shard_id", 0))
    index_rows: list[dict] = []
    idx_csv = PROC / "pretrain_index.csv"
    if idx_csv.exists():
        with idx_csv.open() as fh:
            index_rows = list(csv.DictReader(fh))
    movie_records: list[dict] = state.get("movies", [])
    qc_all: list[dict] = []
    total_clips = len(index_rows)
    excluded_leak: list[dict] = state.get("leakage_exclusions", [])
    bytes_downloaded = state.get("bytes_downloaded", 0)

    print(f"\n{len(keys)} archive(s) to consider; {len(done)} already done; "
          f"{total_clips} clips already indexed; {args.workers} workers")

    for ai, key in enumerate(keys, 1):
        if key in done:
            print(f"[{ai}/{len(keys)}] {key}: already processed, skipping")
            continue
        z = CE / key
        print(f"\n[{ai}/{len(keys)}] === {key} ({sizes[key]/1e9:.1f} GB)")
        if not md5_ok(z, want_md5[key]):
            if args.no_fetch:
                print("  absent or unverified and --no-fetch given; skipping")
                continue
            if not fetch_archive(key, fetch_log):
                print(f"  COULD NOT VERIFY {key}; recording and moving on")
                movie_records.append({"archive": key, "status": "fetch_failed"})
                continue
            bytes_downloaded += sizes[key]
        members = members_of(z)
        if not members:
            print("  no readable TIFF members (bad zip); recording and moving on")
            movie_records.append({"archive": key, "status": "bad_zip"})
            done.add(key)
            continue
        print(f"  {len(members)} movie members")

        umpp = um_per_px_for(key)
        jobs = [(str(z), m, umpp, str(STAGE / f"v1__{Path(key).stem}__{i:04d}.npy"),
                 args.max_clips_per_movie, 7000 + i,
                 eval_vectors, eval_shapes, eval_sha1s)
                for i, m in enumerate(members)]

        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(movie_worker, j) for j in jobs]
            for k, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                if r["status"] == "excluded_leakage":
                    excluded_leak.append(r)
                    print(f"    LEAKAGE: {r['key']} matches {r['matches_eval_movie']} "
                          f"(r={r['pearson_r']}) -> excluded from pretraining")
                elif r["status"] not in ("ok",):
                    print(f"    [{k}/{len(jobs)}] {r['status']} {r['key']}"
                          f"{' ' + r.get('error', '') if r.get('error') else ''}")
                elif k % 20 == 0:
                    so_far = sum(x.get("n_clips", 0) for x in results)
                    print(f"    [{k}/{len(jobs)}] ok, {so_far} clips so far")

        # pooled QC thresholds, then accept movies and write shards
        qc_all.extend([r["qc"] for r in results if r.get("qc")])
        thresholds = QC.fit_thresholds(qc_all)
        n_qc_dropped = 0
        for r in results:
            if r.get("status") != "ok" or not r.get("npy"):
                continue
            verdict = QC.judge(r["qc"], thresholds)
            r["qc_verdict"] = verdict
            if not verdict["pass"]:
                n_qc_dropped += 1
                r["status"] = "excluded_qc"
                Path(r["npy"]).unlink(missing_ok=True)
                continue
            arr = np.load(r["npy"])
            src = source_of(key)
            metas = [{
                "source": src, "archive": key, "movie": r["key"].split(":", 1)[1],
                "track_id": m["particle"], "t0": m["t0"], "stride": m["stride"],
                "static": m.get("static", False),
                "um_per_px": umpp,
                "frame_interval_s": 8.0,
                "effective_interval_s": 8.0 * m["stride"],
            } for m in r["meta"][: len(arr)]]
            writer.add(arr, metas)
            Path(r["npy"]).unlink(missing_ok=True)
            total_clips += len(arr)

        movie_records.extend([{k2: v for k2, v in r.items()
                               if k2 not in ("meta", "qc", "npy")} for r in results])
        # Flush before marking the archive done. Clips still sitting in the buffer
        # would otherwise be lost on a restart, because the archive is recorded as
        # processed and its zip has been deleted. A few short tail shards are a much
        # smaller cost than silently dropping an archive's clips.
        writer.flush()
        done.add(key)
        if not args.keep_zips:
            z.unlink(missing_ok=True)
            print(f"  deleted {key} to bound disk use")

        # ---- persist state after every archive so this is resumable
        writer_rows = index_rows + writer.index_rows
        if writer_rows:
            with idx_csv.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(writer_rows[0].keys()))
                w.writeheader()
                w.writerows(writer_rows)
        state = {
            "status": "in_progress",
            "archives_done": sorted(done),
            "archives_in_record": sorted(want_md5),
            "next_shard_id": writer.shard_id,
            "bytes_downloaded": bytes_downloaded,
            "n_clips": len(writer_rows),
            "qc_thresholds": thresholds,
            "n_movies_qc_dropped_this_archive": n_qc_dropped,
            "leakage_exclusions": excluded_leak,
            "movies": movie_records,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        state_path.write_text(json.dumps(state, indent=2, default=str) + "\n")
        print(f"  archive done: {len(writer_rows)} clips total, "
              f"{n_qc_dropped} movies dropped by QC, "
              f"{len(excluded_leak)} leakage exclusions so far")
        if total_clips >= args.max_clips_total:
            print("clip cap reached; stopping")
            break

    writer.flush()
    all_rows = index_rows + writer.index_rows
    if all_rows:
        with idx_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    ok = [r for r in movie_records if r.get("status") == "ok"]
    by_source: dict[str, int] = {}
    for row in all_rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    strides = {}
    for row in all_rows:
        s = str(row["stride"])
        strides[s] = strides.get(s, 0) + 1
    n_static = sum(1 for row in all_rows if str(row.get("static")).lower() == "true")

    state.update({
        "status": "complete" if len(done) == len(want_md5) else "partial",
        "n_archives_done": len(done),
        "n_archives_in_record": len(want_md5),
        "n_movies_ok": len(ok),
        "n_movies_excluded_leakage": len(excluded_leak),
        "n_movies_excluded_qc": sum(1 for r in movie_records
                                    if r.get("status") == "excluded_qc"),
        "n_movies_failed": sum(1 for r in movie_records
                               if r.get("status") == "failed"),
        "n_clips": len(all_rows),
        "n_shards": writer.shard_id,
        "total_gb": round(len(all_rows) * CLIP_BYTES / 1e9, 2),
        "clips_by_source": by_source,
        "clips_by_stride": strides,
        "n_static_clips": n_static,
        "static_fraction": round(n_static / max(1, len(all_rows)), 4),
        "strides_requested": list(STRIDES),
        "target_um_per_px": D.TARGET_UM_PER_PX,
        "pixel_size_provenance": {
            "complexeye": {
                "um_per_px": UM_PER_PX_COMPLEXEYE,
                "source": "S-BIAD2196 BioStudies per-file metadata (same instrument), "
                          "cross-checked against Cibir et al. Nat Commun 2023 Methods: "
                          "OV9712 sensor, 1280x800 over 3888x2430 um active area = "
                          "3.0375 um/sensor px at 4.69x = 0.648 um/px, agreeing to 1%",
                "verified": True,
            },
            "complexeye_leica": {
                "um_per_px": UM_PER_PX_LEICA,
                "source": "Cibir et al. Nat Commun 2023 Methods report the Leica "
                          "DMI6000 B field of view as 662.9 x 495.2 um; these movies "
                          "are 1392 x 1040 px, so 662.9/1392 = 0.4762 and "
                          "495.2/1040 = 0.4762 um/px, consistent in both axes",
                "verified": True,
            },
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    state_path.write_text(json.dumps(state, indent=2, default=str) + "\n")
    print(f"\n{len(all_rows)} clips in {writer.shard_id} shards "
          f"({state['total_gb']} GB) from {len(ok)} movies; "
          f"{len(excluded_leak)} excluded for leakage, "
          f"{state['n_movies_excluded_qc']} by QC")
    print(f"by source: {by_source}")
    print(f"by stride: {strides}; static clips: {n_static}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
