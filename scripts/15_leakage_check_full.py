"""G3 (PLAN_v2.md §6; finding N5): leakage check over EVERY archive in the frozen corpus.

The v1 check fingerprinted the 8 archives still on disk; R05 and R12 had already been
deleted and were never checked. Here each of the 10 archives in
`results/v1/pretrain_corpus.json` -> `archives_done` is fetched again (MD5-verified
against the Zenodo record), checked, and deleted before the next, so disk use stays at
one archive.

Two tests per (evaluation movie, corpus movie) pair:

  1. the v1 fingerprint (frames {0, n/2, n-1}, 40x25 block means, z-scored) - cheap,
     run on every pair, but weakly discriminative (226 pairs above r = 0.9 in v1);
  2. for every pair with fingerprint r > 0.9: **full-resolution normalised
     cross-correlation** of frames {0, n/4, n/2, 3n/4, n-1} (each movie indexed by its
     own length), plus frame count and frame shape. A pair is a **duplicate** when the
     full-resolution r > 0.95 on at least 3 of the 5 frames.

If a corpus movie with a different frame shape correlates, its frames are resized to
the evaluation movie's shape before the full-resolution comparison and that is recorded.

A new duplicate in the frozen corpus does **not** trigger a rebuild: the affected
evaluation movie is excluded from tracking/forecasting for every model trained on this
corpus (G3.3), via `excluded_eval_movies` in the output.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
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

from tijepa.io import sbiad  # noqa: E402
from tijepa.preprocess import fingerprint as FP  # noqa: E402

RAW = REPO / "data" / "raw"
DEST = RAW / "complexeye"
REC = DEST / ".zenodo_record_10082674.json"
OUT = REPO / "results" / "v2"
STAGE = REPO / "data" / "interim" / "stage_v2"
FETCH = REPO / "scripts" / "13_fetch_complexeye.sh"

FULL_R = 0.95
FULL_MIN_FRAMES = 3
CANDIDATE_R = 0.90
EXPLICIT_PAIRS = [("03_F2", "R16.zip:E2.tiff")]  # PLAN_v2 G3.2: re-examine explicitly


def frame_idx(n: int) -> list[int]:
    return [0, n // 4, n // 2, (3 * n) // 4, n - 1]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.0


def resize_to(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if img.shape == tuple(shape):
        return img
    from skimage.transform import resize
    return resize(img.astype(np.float32), shape, order=1, preserve_range=True,
                  anti_aliasing=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def eval_movies() -> dict[str, dict]:
    out = {}
    for m in sorted((RAW / "s-biad2196" / "Celltrack_data").glob("*.zip")):
        raw, _ = sbiad.load_movie(m, max_frames=None)
        n = raw.shape[0]
        out[m.name] = {"fp": FP.fingerprint(raw), "n_frames": n,
                       "shape": [int(x) for x in raw.shape],
                       "frames5": np.stack([raw[i] for i in frame_idx(n)]).astype(np.float32)}
        print(f"  eval {m.name}: {raw.shape}")
    return out


def member_worker(args: tuple) -> dict:
    """Fingerprint one corpus movie; also return its 5 full-res frames (small)."""
    zip_path, member, stage = args
    key = f"{Path(zip_path).name}:{member}"
    tmpdir = tempfile.mkdtemp(prefix="fp2_", dir=stage)
    try:
        import tifffile

        with zipfile.ZipFile(zip_path) as zf:
            local = Path(tmpdir) / Path(member).name
            with zf.open(member) as src, local.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
        with tifffile.TiffFile(local) as tf:
            n = len(tf.pages)
            if n < 1:
                return {"key": key, "status": "no_pages"}
            need = sorted(set(frame_idx(n)) | {0, n // 2, n - 1})
            frames = {i: tf.pages[i].asarray() for i in need}
        fp = FP.fingerprint(np.stack([frames[i] for i in sorted({0, n // 2, n - 1})]))
        h, w = frames[0].shape[-2:]
        return {"key": key, "status": "ok", "vector": fp["vector"],
                "frame0_sha1": fp["frame0_sha1"], "n_frames": n, "shape": [n, h, w],
                "frames5": np.stack([frames[i] for i in frame_idx(n)]).astype(np.float32)}
    except Exception as exc:
        return {"key": key, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:200]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def compare(ev: dict, tr: dict) -> dict:
    r = FP.correlate(ev["fp"]["vector"], tr["vector"])
    rec = {"fingerprint_r": round(r, 5), "eval_shape": ev["shape"],
           "train_shape": tr["shape"],
           "same_frame_shape": ev["shape"][1:] == tr["shape"][1:],
           "same_frame_count": ev["n_frames"] == tr["n_frames"],
           "frame0_sha1_identical": ev["fp"]["frame0_sha1"] == tr["frame0_sha1"]}
    return rec


def full_res(ev: dict, tr: dict) -> dict:
    shp = tuple(ev["frames5"].shape[1:])
    rs = [ncc(ev["frames5"][k], resize_to(tr["frames5"][k], shp)) for k in range(5)]
    n_hi = sum(r > FULL_R for r in rs)
    return {"full_res_r": [round(x, 5) for x in rs], "n_frames_above": n_hi,
            "resized": tuple(tr["frames5"].shape[1:]) != shp,
            "duplicate": n_hi >= FULL_MIN_FRAMES}


def fetch(key: str) -> bool:
    r = subprocess.run(["bash", str(FETCH), key], capture_output=True, text=True)
    return "VERIFIED" in r.stdout or "already verified" in r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((REPO / "results" / "v1" / "pretrain_corpus.json").read_text())
    archives = corpus["archives_done"]
    status_by_movie = {m["key"]: m["status"] for m in corpus["movies"] if "key" in m}
    n_ok_by_archive = {a: sum(1 for k, st in status_by_movie.items()
                              if k.startswith(a + ":") and st == "ok") for a in archives}
    record = {f["key"]: f["checksum"].split(":")[1] for f in
              json.loads(REC.read_text())["files"]}
    part_path = OUT / "leakage_check_full.partial.json"
    state = json.loads(part_path.read_text()) if part_path.exists() else {
        "archives": {}, "pairs": [], "explicit": [], "failures": []}

    print("evaluation movies:")
    ev = eval_movies()

    for key in archives:
        if key in state["archives"] and state["archives"][key].get("checked"):
            print(f"{key}: already checked")
            continue
        z = DEST / key
        was_present = z.exists()
        t0 = time.time()
        print(f"{key}: fetching (present before: {was_present})")
        if not fetch(key) or md5(z) != record[key]:
            state["archives"][key] = {"checked": False, "reason": "fetch/MD5 failed"}
            part_path.write_text(json.dumps(state, indent=1, default=str))
            print(f"  {key}: FETCH FAILED; recorded and continuing")
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                members = [m for m in zf.namelist()
                           if m.lower().endswith((".tif", ".tiff"))
                           and not m.startswith("__MACOSX")]
        except zipfile.BadZipFile:
            state["archives"][key] = {
                "checked": True, "md5_verified": True, "readable": False,
                "n_corpus_movies_ok": n_ok_by_archive[key],
                "note": "the published archive has no end-of-central-directory record "
                        "(results/v1/complexeye_archive_defect.json); the corpus took "
                        f"{n_ok_by_archive[key]} movies from it, so there is nothing "
                        "to check"}
            if not was_present:
                z.unlink()
            part_path.write_text(json.dumps(state, indent=1, default=str))
            print(f"  {key}: unreadable zip, {n_ok_by_archive[key]} corpus movies")
            continue
        n_ok = n_cand = 0
        jobs = [(str(z), m, str(STAGE)) for m in members]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(member_worker, j) for j in jobs]):
                tr = fut.result()
                if tr["status"] != "ok":
                    state["failures"].append({k: tr[k] for k in ("key", "status")
                                              if k in tr} | {"error": tr.get("error")})
                    continue
                n_ok += 1
                for ek, e in ev.items():
                    rec = compare(e, tr)
                    explicit = any(a in ek and b == tr["key"] for a, b in EXPLICIT_PAIRS)
                    if (rec["fingerprint_r"] > CANDIDATE_R or rec["frame0_sha1_identical"]
                            or explicit):
                        rec.update(full_res(e, tr))
                        rec.update({"eval_movie": ek, "train_movie": tr["key"],
                                    "in_corpus_status": status_by_movie.get(tr["key"],
                                                                            "not in corpus"),
                                    "explicitly_requested": explicit})
                        state["pairs"].append(rec)
                        n_cand += 1
        md5_ok = md5(z) == record[key]
        state["archives"][key] = {"checked": True, "md5_verified": md5_ok,
                                  "n_members": len(members), "n_fingerprinted": n_ok,
                                  "n_corpus_movies_ok": n_ok_by_archive[key],
                                  "n_candidate_pairs": n_cand,
                                  "seconds": round(time.time() - t0, 1)}
        if not args.keep_archives and not was_present:
            z.unlink()
        part_path.write_text(json.dumps(state, indent=1, default=str))
        print(f"  {key}: {n_ok}/{len(members)} fingerprinted, {n_cand} candidate pairs")

    pairs = sorted(state["pairs"], key=lambda p: -p["fingerprint_r"])
    dups = [p for p in pairs if p["duplicate"]]
    in_corpus_dups = [p for p in dups if p["in_corpus_status"] == "ok"]
    res = {
        "phase": "G3",
        "corpus": "data/processed/v1 (pretrain_index md5 43b19b36849e88d29cefa484066e35a1)",
        "archives_in_corpus": archives,
        "archives": state["archives"],
        "all_archives_checked": all(state["archives"].get(a, {}).get("checked")
                                    for a in archives),
        "n_eval_movies": len(ev),
        "n_corpus_movies_fingerprinted": sum(a.get("n_fingerprinted", 0)
                                             for a in state["archives"].values()),
        "rule": f"duplicate = full-resolution NCC > {FULL_R} on >= {FULL_MIN_FRAMES} of the "
                f"5 frames {{0, n/4, n/2, 3n/4, n-1}}, tested for every pair with "
                f"fingerprint r > {CANDIDATE_R} (or identical frame-0 SHA-1, or "
                "explicitly requested)",
        "n_candidate_pairs": len(pairs),
        "n_duplicates": len(dups),
        "duplicates": dups,
        "duplicates_in_frozen_corpus": in_corpus_dups,
        "excluded_eval_movies_for_corpus_v1_models": sorted(
            {p["eval_movie"] for p in in_corpus_dups}),
        "explicit_checks": [p for p in pairs if p["explicitly_requested"]],
        "pairs": pairs,
        "fingerprint_failures": state["failures"],
        "policy": "corpus is frozen and not rebuilt; an evaluation movie with a twin in the "
                  "corpus is excluded from tracking/forecasting for every model trained on "
                  "it (PLAN_v2 G3.3). Excluded twins (status excluded_leakage) were never "
                  "in the corpus and do not trigger exclusion.",
    }
    (OUT / "leakage_check_full.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(f"{len(pairs)} candidate pairs, {len(dups)} duplicates "
          f"({len(in_corpus_dups)} in the frozen corpus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
