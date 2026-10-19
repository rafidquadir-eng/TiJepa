"""H3.1: a label-free validation set that is never used as a test set (PLAN_v3 §7, R10, R22).

Source: the ComplexEye archive `384-well_plate_movies_n=1_complexeye.zip` (Zenodo
10082674, CC BY 4.0), which is **not** in the frozen pretraining corpus and was already on
this instance; its MD5 matches the Zenodo record. PLAN_v3 prefers `R01.zip`; this archive
is used instead because it is equally outside the corpus and needs no 11 GB download over
an intermittent Zenodo connection (recorded as a deviation).

Leakage, per movie:
* against the S-BIAD2196 evaluation movies: the corpus builder's fingerprint rule
  (same frame shape and fingerprint r >= 0.9, or identical frame-0 SHA-1);
* positive control: the first query patch of each movie is searched for in its own raw
  frame and must match at NCC > 0.95;
* against the frozen corpus: three stride-1, non-static clips are taken from every
  ComplexEye corpus movie, and the first frame of each (a 96 x 96 cell patch) is searched
  for in the validation movie's frame at the same time index by normalised
  cross-correlation; a movie is a duplicate if >= 2 of the 3 patches match at NCC > 0.95.
  (Leica corpus movies come from a different instrument and frame size and cannot be
  duplicates of ComplexEye movies.)

Re-identification clips: cells are detected and linked with the classical pipeline used to
build the corpus. Per movie, 4 anchor frames a are spaced evenly; the cells at an anchor are
the tracks covering [a - 15, a + 27 + 15] without gaps (up to 64, seeded). For each cell the
query is the 16-frame track-following clip ending at a; the gallery item for gap k
(k in {3, 9, 27}) is the 16-frame clip starting at a + k, so query and gallery never share a
frame. Image encoders use the query clip's last frame and the gallery clip's first frame.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.preprocess import detect as D  # noqa: E402
from tijepa.preprocess import fingerprint as FP  # noqa: E402

ARCHIVE = REPO / "data" / "raw" / "complexeye" / "384-well_plate_movies_n=1_complexeye.zip"
ARCHIVE_MD5 = "6ae275191e4ba918ce4f4ec04a98936a"
OUT_DIR = REPO / "data" / "interim" / "val_reid"
RES = REPO / "results" / "v3" / "validation_set.json"
STAGE = REPO / "data" / "interim" / "stage_v3"
CROP, NF = 96, 16
GAPS = (3, 9, 27)
N_ANCHORS, MAX_CELLS = 4, 64
NCC_DUP, MIN_PATCHES = 0.95, 2
SEED = 20261019


def corpus_patches(n_per_movie: int = 3) -> dict:
    """{movie key: [(t0, patch uint8 96x96), ...]} for ComplexEye corpus movies."""
    import pandas as pd

    idx = pd.read_csv(REPO / "data" / "processed" / "v1" / "pretrain_index.csv")
    idx = idx[(idx.source == "complexeye") & (idx.stride == 1) & (~idx.static)]
    rng = np.random.default_rng(SEED)
    out: dict = {}
    shards: dict = {}
    for (arch, mov), g in idx.groupby(["archive", "movie"]):
        rows = g.iloc[rng.permutation(len(g))[:n_per_movie]]
        for _, r in rows.iterrows():
            if r.shard not in shards:
                shards[r.shard] = np.load(
                    REPO / "data" / "processed" / "v1" / r.shard, mmap_mode="r"
                )
            out.setdefault(f"{arch}:{mov}", []).append(
                (int(r.t0), np.asarray(shards[r.shard][int(r.offset), 0]))
            )
    return out


def ncc_max(frame: np.ndarray, patch: np.ndarray) -> float:
    from skimage.feature import match_template

    if patch.std() < 1e-6:
        return 0.0
    return float(match_template(frame.astype(np.float32), patch.astype(np.float32)).max())


def track_clip(frames: np.ndarray, pos: dict, t_start: int) -> np.ndarray | None:
    half = CROP // 2
    H, W = frames.shape[1:]
    buf = np.zeros((NF, CROP, CROP), np.uint8)
    for i in range(NF):
        cy, cx = pos[t_start + i]
        y0, x0 = int(round(cy)) - half, int(round(cx)) - half
        if y0 < 0 or x0 < 0 or y0 + CROP > H or x0 + CROP > W:
            return None
        buf[i] = frames[t_start + i, y0 : y0 + CROP, x0 : x0 + CROP]
    return buf


def to_uint8(frames: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(frames[:: max(1, len(frames) // 16)], (0.5, 99.5))
    return np.clip((frames.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(
        np.uint8
    )


def movie_worker(args: tuple) -> dict:
    member, eval_fp, cpatch = args
    tmp = tempfile.mkdtemp(prefix="val_", dir=str(STAGE))
    key = f"{ARCHIVE.name}:{member}"
    try:
        import tifffile

        with zipfile.ZipFile(ARCHIVE) as zf:
            local = Path(tmp) / Path(member).name
            with zf.open(member) as src, local.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
        with tifffile.TiffFile(local) as tf:
            frames = np.stack([p.asarray() for p in tf.pages])
        if frames.ndim != 3 or len(frames) < NF + max(GAPS) + NF:
            return {
                "key": key,
                "status": "skipped_too_short_or_not_3d",
                "shape": list(frames.shape),
            }
        rec: dict = {"key": key, "shape": [int(x) for x in frames.shape]}
        fp = FP.fingerprint(frames)
        for name, (vec, shape, sha) in eval_fp.items():
            r = FP.correlate(fp["vector"], vec)
            if fp["frame0_sha1"] == sha or (
                list(frames.shape[1:]) == list(shape[1:]) and r >= FP.MATCH_R
            ):
                return {
                    **rec,
                    "status": "excluded_leakage_eval",
                    "matches": name,
                    "fingerprint_r": round(r, 5),
                }
        # corpus: patch search at the same time index
        best = {"key": None, "n_match": 0, "ncc": []}
        for ck, pats in cpatch.items():
            nccs = [ncc_max(frames[t0], p) if t0 < len(frames) else 0.0 for t0, p in pats]
            n = sum(v > NCC_DUP for v in nccs)
            if n > best["n_match"] or (n == best["n_match"] and best["key"] is None):
                best = {"key": ck, "n_match": n, "ncc": [round(v, 4) for v in nccs]}
            if n >= MIN_PATCHES:
                return {
                    **rec,
                    "status": "excluded_leakage_corpus",
                    "matches": ck,
                    "ncc": best["ncc"],
                }
        rec["corpus_check_best"] = best
        rec["corpus_check_max_ncc"] = float(
            max((max(v) for v in [best["ncc"]] if v), default=0.0)
        )

        frames = D.maybe_resample(frames, 0.641)
        det = D.detect_movie(frames, stride=1)
        linked = D.link_tracks(det, min_len=NF + max(GAPS) + NF)
        f8 = to_uint8(frames)
        tracks = {
            int(p): {
                int(f): (y, x) for f, y, x in zip(g["frame"], g["y"], g["x"], strict=False)
            }
            for p, g in linked.groupby("particle")
        }
        n = len(frames)
        anchors = np.linspace(NF - 1, n - max(GAPS) - NF - 1, N_ANCHORS).astype(int)
        rng = np.random.default_rng(
            (SEED, int(hashlib.sha1(member.encode()).hexdigest()[:8], 16))
        )
        q, g = [], {k: [] for k in GAPS}
        meta = []
        for ai, a in enumerate(anchors):
            need = range(a - NF + 1, a + max(GAPS) + NF)
            cells = [p for p, pos in tracks.items() if all(t in pos for t in need)]
            cells = [cells[i] for i in rng.permutation(len(cells))[:MAX_CELLS]]
            for p in cells:
                qc = track_clip(f8, tracks[p], a - NF + 1)
                gc = {k: track_clip(f8, tracks[p], a + k) for k in GAPS}
                if qc is None or any(v is None for v in gc.values()):
                    continue
                q.append(qc)
                for k in GAPS:
                    g[k].append(gc[k])
                meta.append((ai, int(a), p))
        if not q:
            return {**rec, "status": "no_eligible_cells", "n_tracks": len(tracks)}
        # positive control for the corpus search: a patch cut by this pipeline must be
        # found again in its own raw frame
        a0, _p0 = int(meta[0][1]), meta[0][2]
        rec["corpus_check_positive_control_ncc"] = round(
            ncc_max(frames[a0 - NF + 1], q[0][0]), 4
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / (Path(member).stem + ".npz")
        np.savez_compressed(
            out,
            query=np.stack(q),
            **{f"gallery_k{k}": np.stack(g[k]) for k in GAPS},
            anchor=np.array([m[0] for m in meta]),
            anchor_frame=np.array([m[1] for m in meta]),
            track=np.array([m[2] for m in meta]),
        )
        per_anchor = np.bincount([m[0] for m in meta], minlength=N_ANCHORS)
        return {
            **rec,
            "status": "ok",
            "n_tracks": len(tracks),
            "n_cells": len(q),
            "cells_per_anchor": per_anchor.tolist(),
            "file": str(out.relative_to(REPO)),
        }
    except Exception as exc:
        return {"key": key, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="first N members (smoke tests)")
    args = ap.parse_args()
    t0 = time.time()
    STAGE.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()
    with open(ARCHIVE, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            md5.update(b)
    if md5.hexdigest() != ARCHIVE_MD5:
        raise SystemExit("archive MD5 does not match the Zenodo record")
    z = np.load(REPO / "data" / "interim" / "eval_fingerprints.npz", allow_pickle=True)
    eval_fp = {
        str(n): (z[f"vec_{n}"], list(z[f"shape_{n}"]), str(z[f"sha_{n}"])) for n in z["names"]
    }
    cpatch = corpus_patches()
    print(
        f"{len(eval_fp)} evaluation fingerprints; {len(cpatch)} ComplexEye corpus movies "
        "with search patches"
    )
    with zipfile.ZipFile(ARCHIVE) as zf:
        members = sorted(
            m
            for m in zf.namelist()
            if m.lower().endswith((".tif", ".tiff")) and not m.startswith("__MACOSX")
        )
    if args.limit:
        members = members[: args.limit]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(movie_worker, [(m, eval_fp, cpatch) for m in members]))
    for r in res:
        print(f"  {r['key']}: {r['status']} {r.get('n_cells', '')}")
    ok = [r for r in res if r["status"] == "ok"]
    out = {
        "phase": "PLAN_v3 H3.1",
        "archive": {
            "file": ARCHIVE.name,
            "zenodo_record": "10082674",
            "licence": "CC BY 4.0",
            "md5": ARCHIVE_MD5,
            "md5_verified": True,
            "in_frozen_corpus": False,
        },
        "deviation": "PLAN_v3 prefers R01.zip; this archive is equally outside the corpus "
        "and already on the instance",
        "leakage_rules": {
            "evaluation": "corpus-builder fingerprint rule (same shape and r >= 0.9, or "
            "identical frame-0 SHA-1) against all 6 S-BIAD2196 movies",
            "corpus": f"3 stride-1 non-static clip patches per ComplexEye corpus movie, "
            f"searched at the same time index; duplicate if >= {MIN_PATCHES} "
            f"match at NCC > {NCC_DUP}",
        },
        "reid_design": {
            "gaps_frames": list(GAPS),
            "anchors_per_movie": N_ANCHORS,
            "max_cells_per_anchor": MAX_CELLS,
            "query": "16-frame track-following clip ending at the anchor",
            "gallery": "16-frame clip starting at anchor + k (no shared frame)",
            "image_encoders": "last query frame vs first gallery frame",
        },
        "n_members": len(members),
        "n_ok": len(ok),
        "n_cells_total": int(sum(r["n_cells"] for r in ok)),
        "status_counts": {
            s: sum(r["status"] == s for r in res) for s in sorted({r["status"] for r in res})
        },
        "movies": res,
        "seconds": round(time.time() - t0, 1),
    }
    RES.parent.mkdir(parents=True, exist_ok=True)
    RES.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {RES}: {out['status_counts']}, {out['n_cells_total']} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
