"""F1.3: check whether any S-BIAD2196 evaluation movie also appears in ComplexEye.

Both are human neutrophil brightfield time-lapses, so a held-out claim about the
tracking evaluation is worth nothing without this check (review finding D3).

Every S-BIAD2196 movie and every ComplexEye movie member is fingerprinted (see
`tijepa.preprocess.fingerprint`). Matching ComplexEye movies are written to
`results/v1/leakage_check.json` and the corpus builder excludes them from pretraining.
An S-BIAD2196 movie whose twin has been excluded is still usable for evaluation.

Fingerprinting a ComplexEye member means extracting a TIFF from a multi-GB zip, so the
work is spread over CPU cores and only the three frames needed are decoded.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
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
OUT = REPO / "results" / "v1"
STAGE = REPO / "data" / "interim" / "stage"


def ce_members(z: Path) -> list[str]:
    try:
        with zipfile.ZipFile(z) as zf:
            return [m for m in zf.namelist()
                    if m.lower().endswith((".tif", ".tiff"))
                    and not m.startswith("__MACOSX")]
    except zipfile.BadZipFile:
        return []


def ce_worker(args: tuple) -> dict:
    """Fingerprint one ComplexEye movie, decoding only the frames needed."""
    zip_path, member = args
    tmpdir = tempfile.mkdtemp(prefix="fp_", dir=str(STAGE))
    try:
        import tifffile

        with zipfile.ZipFile(zip_path) as zf:
            local = Path(tmpdir) / Path(member).name
            with zf.open(member) as src, local.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
        with tifffile.TiffFile(local) as tf:
            n = len(tf.pages)
            if n < 1:
                return {"key": f"{Path(zip_path).name}:{member}", "status": "no_pages"}
            idx = sorted({0, n // 2, n - 1})
            frames = np.stack([tf.pages[i].asarray() for i in idx])
            shape = [n, int(frames.shape[1]), int(frames.shape[2])]
        fp = FP.fingerprint(frames)
        fp["shape"] = shape  # report the movie's true length, not the 3 sampled frames
        return {"key": f"{Path(zip_path).name}:{member}", "status": "ok",
                "vector": fp["vector"].tolist(), "shape": shape,
                "frame0_sha1": fp["frame0_sha1"], "n_frames": n}
    except Exception as exc:
        return {"key": f"{Path(zip_path).name}:{member}", "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:200]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-frames-eval", type=int, default=0,
                    help="0 = use the whole movie (only 3 frames are fingerprinted)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------- evaluation side: S-BIAD2196
    eval_fps: dict[str, dict] = {}
    movies = sorted((RAW / "s-biad2196" / "Celltrack_data").glob("*.zip"))
    print(f"S-BIAD2196: {len(movies)} movies on disk")
    for m in movies:
        try:
            raw, _ = sbiad.load_movie(m, max_frames=args.max_frames_eval or None)
            fp = FP.fingerprint(raw)
            eval_fps[m.name] = fp
            print(f"  {m.name}: {raw.shape}")
        except Exception as exc:
            print(f"  {m.name}: FAILED {type(exc).__name__}: {exc}")

    # ------------------------------------------- pretraining side: ComplexEye
    zips = sorted((RAW / "complexeye").glob("*.zip"))
    jobs = [(str(z), m) for z in zips for m in ce_members(z)]
    print(f"\nComplexEye: {len(jobs)} movie members across {len(zips)} archives on disk")
    train_fps: dict[str, dict] = {}
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(ce_worker, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["status"] == "ok":
                train_fps[r["key"]] = {
                    "vector": np.asarray(r["vector"], dtype=np.float32),
                    "shape": r["shape"], "frame0_sha1": r["frame0_sha1"],
                    "n_frames": r["n_frames"]}
            else:
                failures.append(r)
            if k % 25 == 0:
                print(f"  [{k}/{len(jobs)}] fingerprinted {len(train_fps)} ok, "
                      f"{len(failures)} failed")

    # ------------------------------------------------------------ compare
    res = FP.compare_sets(eval_fps, train_fps)
    res["status"] = "ok"
    res["eval_source"] = "S-BIAD2196 CellTrack (tracking evaluation)"
    res["train_source"] = "ComplexEye Zenodo 10082674 (pretraining candidates)"
    res["archives_on_disk"] = [z.name for z in zips]
    res["n_fingerprint_failures"] = len(failures)
    res["fingerprint_failures"] = failures[:20]
    res["interpretation"] = (
        "No pair above the reporting threshold means no evidence of overlap between "
        "the pretraining corpus and the tracking evaluation set, which is what the "
        "held-out claim requires."
        if res["n_pairs_matched"] == 0 else
        f"{res['n_pairs_matched']} matching pair(s): the listed ComplexEye movies are "
        "excluded from pretraining, and the affected S-BIAD2196 movies remain usable "
        "for evaluation once their twin is excluded.")
    res["elapsed_s"] = round(time.time() - t_start, 1)

    (OUT / "leakage_check.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n{res['n_pairs_reported']} pair(s) above r={FP.REPORT_R}, "
          f"{res['n_pairs_matched']} judged leakage")
    for p in res["pairs"][:10]:
        print(f"  r={p['pearson_r']:.4f} {p['eval_movie']} ~ {p['train_movie']} "
              f"({p['decision']})")
    print(f"excluded from pretraining: {len(res['excluded_train_movies'])} movies")
    print(f"wrote {OUT/'leakage_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
