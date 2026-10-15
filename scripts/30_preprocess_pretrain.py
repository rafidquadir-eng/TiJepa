"""M4: build the pretraining clip corpus as large sequential shards.

Reads the downloaded CC BY archives, detects and tracks cells, extracts cell-centred
96x96x16 clips, and writes >=1 GB uint8 .npy shards plus an index CSV
(source, movie, track_id, t0, stride, um_per_px) as PLAN.md A1.5 requires.

Movies are processed in parallel across CPU cores; each worker stages its clips to a
small .npy, and the parent concatenates them into big shards.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from tijepa.preprocess import detect as D  # noqa: E402

RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"
STAGE = REPO / "data" / "interim" / "stage"
CLIP, NF = 96, 16
CLIP_BYTES = NF * CLIP * CLIP
SHARD_CLIPS = 7000  # ~1.03 GB per shard

# Pixel sizes: verified where a record states one, else flagged.
SOURCES = {
    "complexeye": {
        "zips": sorted((RAW / "complexeye").glob("*.zip")),
        "um_per_px": 0.641,
        "um_per_px_source": "S-BIAD2196 BioStudies file metadata (same group/instrument "
                            "family, 0.641 x 0.641 um); the ComplexEye OME headers carry "
                            "no PhysicalSize. For the *_leica.zip movies this is an "
                            "UNVERIFIED assumption.",
        "frame_interval_s": 8.0,
        "licence": "CC BY 4.0",
    },
}


def movie_worker(args: tuple) -> dict:
    zip_path, member, um_per_px, out_npy, max_clips, seed = args
    t0 = time.time()
    tmpdir = tempfile.mkdtemp(prefix="tijepa_", dir=str(STAGE))
    try:
        import tifffile

        with zipfile.ZipFile(zip_path) as zf:
            local = Path(tmpdir) / Path(member).name
            with zf.open(member) as src, local.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
        with tifffile.TiffFile(local) as tf:
            n_pages = len(tf.pages)
            frames = np.stack([tf.pages[i].asarray() for i in range(n_pages)])
        if frames.ndim != 3:
            return {"member": member, "status": "skipped_not_3d", "shape": str(frames.shape)}
        frames = D.maybe_resample(frames, um_per_px)
        det = D.detect_movie(frames, stride=1)
        linked = D.link_tracks(det, min_len=NF)
        rng = np.random.default_rng(seed)
        clips, meta = D.extract_clips(
            frames, linked, crop=CLIP, n_frames=NF, strides=(1, 3), max_clips=max_clips,
            rng=rng,
        )
        if len(clips):
            np.save(out_npy, clips)
        return {
            "member": member, "status": "ok", "n_frames": int(n_pages),
            "n_detections": int(len(det)),
            "n_tracks": int(linked["particle"].nunique()) if len(linked) else 0,
            "n_clips": int(len(clips)), "meta": meta,
            "npy": str(out_npy) if len(clips) else None,
            "seconds": round(time.time() - t0, 1),
        }
    except Exception as exc:
        return {"member": member, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips-total", type=int, default=400_000)
    ap.add_argument("--max-clips-per-movie", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    jobs, source_info = [], {}
    for name, spec in SOURCES.items():
        source_info[name] = {k: v for k, v in spec.items() if k != "zips"}
        source_info[name]["zips"] = [p.name for p in spec["zips"]]
        for z in spec["zips"]:
            try:
                with zipfile.ZipFile(z) as zf:
                    members = [m for m in zf.namelist()
                               if m.lower().endswith((".tif", ".tiff"))
                               and not m.startswith("__MACOSX")]
            except zipfile.BadZipFile:
                print(f"  BAD ZIP (incomplete download?): {z.name}")
                continue
            for i, m in enumerate(members):
                out = STAGE / f"{name}__{z.stem}__{i:04d}.npy"
                jobs.append((str(z), m, spec["um_per_px"], str(out),
                             args.max_clips_per_movie, 1000 + i))
    print(f"{len(jobs)} movies queued across {len(SOURCES)} sources; "
          f"{args.workers} workers")

    results, total_clips = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(movie_worker, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            total_clips += r.get("n_clips", 0)
            if k % 10 == 0 or r["status"] != "ok":
                print(f"  [{k}/{len(jobs)}] {r['status']} {Path(r['member']).name} "
                      f"clips={r.get('n_clips', 0)} total={total_clips}")
            if total_clips >= args.max_clips_total:
                print("  clip cap reached; cancelling remaining movies")
                for f in futs:
                    f.cancel()
                break

    # assemble shards
    index_rows, shard_id, buf, buf_meta = [], 0, [], []

    def flush() -> None:
        nonlocal shard_id, buf, buf_meta
        if not buf:
            return
        arr = np.concatenate(buf)[:SHARD_CLIPS]
        path = PROC / f"pretrain_shard_{shard_id:04d}.npy"
        np.save(path, arr)
        for off, m in enumerate(buf_meta[: len(arr)]):
            index_rows.append({**m, "shard": path.name, "offset": off})
        print(f"  wrote {path.name}: {len(arr)} clips, {path.stat().st_size/1e9:.2f} GB")
        shard_id += 1
        buf, buf_meta = [], []

    for r in results:
        if r.get("status") != "ok" or not r.get("npy"):
            continue
        a = np.load(r["npy"], mmap_mode="r")
        src = Path(r["npy"]).name.split("__")[0]
        for i, m in enumerate(r["meta"][: len(a)]):
            buf_meta.append({
                "source": src, "movie": Path(r["member"]).name,
                "track_id": m["particle"], "t0": m["t0"], "stride": m["stride"],
                "um_per_px": SOURCES[src]["um_per_px"],
                "frame_interval_s": SOURCES[src]["frame_interval_s"],
                "effective_interval_s": SOURCES[src]["frame_interval_s"] * m["stride"],
            })
        buf.append(np.asarray(a))
        if sum(len(b) for b in buf) >= SHARD_CLIPS:
            flush()
    flush()

    with (PROC / "pretrain_index.csv").open("w", newline="") as fh:
        if index_rows:
            w = csv.DictWriter(fh, fieldnames=list(index_rows[0].keys()))
            w.writeheader()
            w.writerows(index_rows)

    ok = [r for r in results if r.get("status") == "ok"]
    eff = [r["effective_interval_s"] for r in index_rows] if index_rows else []
    summary = {
        "n_movies_attempted": len(results),
        "n_movies_ok": len(ok),
        "n_movies_failed": sum(1 for r in results if r.get("status") == "failed"),
        "failures": [r for r in results if r.get("status") == "failed"][:20],
        "n_clips": len(index_rows),
        "n_shards": shard_id,
        "clip_shape": [NF, CLIP, CLIP],
        "bytes_per_clip": CLIP_BYTES,
        "total_gb": round(len(index_rows) * CLIP_BYTES / 1e9, 2),
        "sources": source_info,
        "target_um_per_px": D.TARGET_UM_PER_PX,
        "effective_interval_s_distribution": {
            "unique": sorted(set(eff)),
            "counts": {str(v): int(sum(1 for e in eff if e == v)) for v in sorted(set(eff))},
        },
        "detector": "temporal-median background + MAD threshold + area filter; "
                    "trackpy linking (search_range 12 px, memory 2, min_len 16)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (REPO / "results" / "partA").mkdir(parents=True, exist_ok=True)
    (REPO / "results" / "partA" / "pretrain_corpus.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n{len(index_rows)} clips in {shard_id} shards "
          f"({summary['total_gb']} GB) from {len(ok)}/{len(results)} movies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
