"""M6 / PLAN.md A4.3: point tracking on held-out S-BIAD2196 movies.

Given a query point at t0 (a tracked cell's centroid), track it for the rest of the
sequence. Methods:
  (i)  nearest-detection linking baseline (classical detector, no learned features)
  (ii) patch-feature correlation tracking using each encoder's patch tokens
       (identical tracker for every encoder)
Metric: TAP-Vid-style <delta> = mean fraction of predictions within
{1, 2, 4, 8, 16} um of ground truth, plus median track survival length.

This is the one evaluation in the run with real temporal structure, so it carries the
frame-shuffle control (E5) that the static BBBC054 probe cannot.
CoTracker was not run: dl.fbaipublicfiles.com was unreachable in preflight.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.io import sbiad  # noqa: E402
from tijepa.models.video_jepa import VideoJEPA  # noqa: E402
from tijepa.preprocess import detect as D  # noqa: E402

OUT = REPO / "results" / "partA"
THRESH_UM = (1.0, 2.0, 4.0, 8.0, 16.0)
CROP, NF = 96, 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def precompute_detections(frames: np.ndarray) -> list[np.ndarray]:
    """Detect once per frame for the whole movie; reused by every query track."""
    bg = D.temporal_background(frames)
    return [D.detect_frame(frames[t], bg) for t in range(frames.shape[0])]


def nearest_detection_tracker(dets: list[np.ndarray], start: tuple[float, float],
                              t0: int, n_steps: int, n_frames: int,
                              search_px: float = 20.0) -> np.ndarray:
    """Greedy nearest-detection linking from a query point."""
    pos = np.asarray(start, dtype=float)
    out = [pos.copy()]
    for t in range(t0 + 1, min(t0 + 1 + n_steps, n_frames)):
        c = dets[t]
        if len(c) == 0:
            out.append(pos.copy())
            continue
        d = np.hypot(c[:, 0] - pos[0], c[:, 1] - pos[1])
        j = int(np.argmin(d))
        if d[j] <= search_px:
            pos = c[j].astype(float)
        out.append(pos.copy())
    return np.asarray(out)


@torch.no_grad()
def feature_correlation_tracker(model, frames: np.ndarray, start: tuple[float, float],
                                t0: int, n_steps: int, grid, search_px: float = 24.0,
                                shuffle_frames: bool = False, seed: int = 0
                                ) -> np.ndarray:
    """Track by matching the query's patch token to candidate patch tokens.

    A window around the current position is embedded at each step; the candidate patch
    whose token is most similar (cosine) to the query token becomes the new position.
    """
    gt, gy, gx = grid
    stride_y, stride_x = CROP // gy, CROP // gx
    pos = np.asarray(start, dtype=float)
    out = [pos.copy()]
    rng = np.random.default_rng(seed)

    def window(t: int, p: np.ndarray) -> torch.Tensor | None:
        half = CROP // 2
        y0, x0 = int(round(p[0])) - half, int(round(p[1])) - half
        H, W = frames.shape[1], frames.shape[2]
        if y0 < 0 or x0 < 0 or y0 + CROP > H or x0 + CROP > W:
            return None
        idx = [min(frames.shape[0] - 1, t + k) for k in range(NF)]
        if shuffle_frames:
            idx = list(rng.permutation(idx))
        clip = frames[idx, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32)
        clip = (clip - clip.mean()) / (clip.std() + 1e-6)
        return torch.from_numpy(clip)[None, None].to(DEV)

    w = window(t0, pos)
    if w is None:
        return np.asarray(out)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        tok = model.target(w).float()
    tok = tok.reshape(gt, gy, gx, -1)
    qy, qx = gy // 2, gx // 2
    query = tok[:, qy, qx, :].mean(0)
    query = query / (query.norm() + 1e-6)

    for t in range(t0 + 1, min(t0 + 1 + n_steps, frames.shape[0])):
        w = window(t, pos)
        if w is None:
            out.append(pos.copy())
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            tk = model.target(w).float()
        tk = tk.reshape(gt, gy, gx, -1).mean(0)
        tk = tk / (tk.norm(dim=-1, keepdim=True) + 1e-6)
        sim = (tk @ query).cpu().numpy()
        by, bx = np.unravel_index(int(np.argmax(sim)), sim.shape)
        dy = (by - qy) * stride_y
        dx = (bx - qx) * stride_x
        step = np.array([dy, dx], dtype=float)
        if np.hypot(*step) <= search_px:
            pos = pos + step
        out.append(pos.copy())
    return np.asarray(out)


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    n = min(len(pred), len(gt))
    if n == 0:
        return {"deltas": {}, "delta_avg": None, "survival": 0}
    err_um = np.hypot(pred[:n, 0] - gt[:n, 0], pred[:n, 1] - gt[:n, 1]) * sbiad.UM_PER_PX
    deltas = {f"within_{t:g}um": float((err_um <= t).mean()) for t in THRESH_UM}
    within8 = err_um <= 8.0
    surv = int(np.argmin(within8)) if (~within8).any() else n
    return {"deltas": deltas,
            "delta_avg": float(np.mean(list(deltas.values()))),
            "median_error_um": float(np.median(err_um)),
            "survival_frames": surv, "n_steps": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--n-tracks", type=int, default=25)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    movies = sorted((REPO / "data" / "raw" / "s-biad2196" / "Celltrack_data").glob("*.zip"))
    if not movies:
        (OUT / "tracking.json").write_text(json.dumps(
            {"status": "skipped", "reason": "no S-BIAD2196 movies on disk"}, indent=2))
        return 0
    ckpt = Path(args.ckpt) if args.ckpt else next(
        iter(sorted((REPO / "checkpoints").glob("tijepa0_T1_seed*.pt"))), None)

    encoders: dict[str, VideoJEPA | None] = {}
    if ckpt and ckpt.exists():
        m = VideoJEPA(clip=(NF, CROP, CROP))
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(sd["model"])
        encoders["E4_tijepa0"] = m.eval().to(DEV)
        print(f"loaded checkpoint step {sd.get('step')}")
    else:
        encoders["E4_tijepa0"] = None
        print("no checkpoint: E4/E5 will be skipped")
    mr = VideoJEPA(clip=(NF, CROP, CROP))
    encoders["E6_tijepa0_random_init"] = mr.eval().to(DEV)

    per_movie, agg = [], {}
    for mv in movies:
        raw, lab = sbiad.load_movie(mv, max_frames=args.max_frames)
        tracks = sbiad.tracks_from_labels(lab, min_len=args.n_steps + 1)
        ids = sorted(tracks)[: args.n_tracks]
        print(f"{mv.name}: {raw.shape} {len(tracks)} tracks >= {args.n_steps+1} frames; "
              f"using {len(ids)}")
        dets = precompute_detections(raw)
        rows = []
        for tid in ids:
            tr = tracks[tid]
            t0 = int(tr[0, 0])
            gt = tr[:, 1:3]
            start = (gt[0, 0], gt[0, 1])
            res = {"movie": mv.name, "track_id": tid, "t0": t0,
                   "gt_len": int(len(gt))}
            pred = nearest_detection_tracker(dets, start, t0, args.n_steps,
                                             raw.shape[0])
            res["baseline_nearest_detection"] = evaluate(pred, gt)
            for key, model in encoders.items():
                if model is None:
                    res[key] = {"status": "skipped_no_checkpoint"}
                    continue
                p = feature_correlation_tracker(model, raw, start, t0, args.n_steps,
                                                model.grid)
                res[key] = evaluate(p, gt)
            if encoders.get("E4_tijepa0") is not None:
                p = feature_correlation_tracker(
                    encoders["E4_tijepa0"], raw, start, t0, args.n_steps,
                    encoders["E4_tijepa0"].grid, shuffle_frames=True)
                res["E5_tijepa0_frame_shuffled"] = evaluate(p, gt)
            rows.append(res)
        per_movie.extend(rows)

    methods = ["baseline_nearest_detection", "E4_tijepa0",
               "E5_tijepa0_frame_shuffled", "E6_tijepa0_random_init"]
    for meth in methods:
        vals = [r[meth]["delta_avg"] for r in per_movie
                if meth in r and r[meth].get("delta_avg") is not None]
        surv = [r[meth]["survival_frames"] for r in per_movie
                if meth in r and r[meth].get("survival_frames") is not None]
        if not vals:
            agg[meth] = {"status": "unavailable"}
            continue
        # bootstrap over movies (the grouping unit)
        rng = np.random.default_rng(0)
        mv_names = sorted({r["movie"] for r in per_movie})
        boots = []
        for _ in range(2000):
            drawn = rng.choice(mv_names, len(mv_names), replace=True)
            v = [r[meth]["delta_avg"] for d in drawn for r in per_movie
                 if r["movie"] == d and meth in r
                 and r[meth].get("delta_avg") is not None]
            if v:
                boots.append(np.mean(v))
        agg[meth] = {
            "delta_avg_mean": float(np.mean(vals)),
            "delta_avg_ci": [float(np.quantile(boots, 0.025)),
                             float(np.quantile(boots, 0.975))] if boots else None,
            "median_survival_frames": float(np.median(surv)),
            "n_tracks": len(vals),
        }
        print(f"  {meth:32s} <delta> {agg[meth]['delta_avg_mean']:.3f} "
              f"survival {agg[meth]['median_survival_frames']:.0f} frames")

    (OUT / "tracking.json").write_text(json.dumps({
        "status": "ok",
        "dataset": "S-BIAD2196 CellTrack (held out from pretraining entirely)",
        "licence": "CC BY 4.0",
        "um_per_px": sbiad.UM_PER_PX, "frame_interval_s": sbiad.FRAME_INTERVAL_S,
        "n_movies": len(movies), "movies": [m.name for m in movies],
        "max_frames_per_movie": args.max_frames, "n_steps": args.n_steps,
        "thresholds_um": list(THRESH_UM),
        "cotracker": "NOT RUN: dl.fbaipublicfiles.com unreachable in preflight",
        "ltdb": "NOT RUN: ndownloader.figshare.com unreachable in preflight; see "
                "reports/MANUAL_DOWNLOADS.md",
        "immunemap": "NOT RUN: deferred as future work per PLAN.md A4.3",
        "aggregate": agg, "per_track": per_movie,
    }, indent=2, default=float) + "\n")
    print(f"wrote {OUT/'tracking.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
