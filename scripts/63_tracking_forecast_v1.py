"""F3.3 tracking and F3.4 latent forecasting on held-out movies.

### F3.3 Tracking (fixes D6)

v0 tracked on 5 S-BIAD2196 movies with no leakage check. v1 uses **all 6** movies the
deposit actually contains (the BioStudies API exposes only 5; the sixth was found in the
FIRE tree), and the leakage check has established which ComplexEye movies are their
twins so those are excluded from pretraining. The affected evaluation movie stays usable,
which is the point of excluding the twin rather than the evaluation movie.

LTDB is used if the author provided it, and CoTracker if it is reachable; both statuses
are recorded either way rather than left implicit.

Six movies is still below the 10-group threshold of PLAN_v1.md rule 3, so **no bootstrap
CI is reported over movies**; per-movie values and the exact permutation over movie
labels are given instead. That is a change from v0, which bootstrapped over 5 movies.

### F3.4 Latent forecasting (new; fixes C4, and tests the "world model" framing)

Each model is given the first 12 frames of a held-out clip and asked to predict its own
target-encoder latents for the last 4, through its own predictor with the future mask it
was (or was not) trained on. The score is

    ratio = L1(predictor) / L1(copy-last-observed-latent)

so a value below 1 means the model forecasts better than assuming nothing changes.
**Only within-model ratios are comparable**, because each model has its own latent space
and the absolute L1 values live on different scales; comparing raw L1 across models would
be meaningless and is not done. A `mean-latent` baseline is reported as well.

TiJepa-1-spatial is queried with the same future mask although it never saw one in
training, so its ratio is an out-of-distribution query and is labelled as such.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.io import sbiad  # noqa: E402
from tijepa.models.video_jepa import VideoJEPA, future_mask  # noqa: E402
from tijepa.preprocess import detect as D  # noqa: E402

OUT = REPO / "results" / "v1" / "partA"
RAW = REPO / "data" / "raw"
THRESH_UM = (1.0, 2.0, 4.0, 8.0, 16.0)
CROP, NF = 96, 16
N_FUTURE_SLOTS = 2          # last 2 tubelet slots = last 4 frames
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------- tracking
def nearest_detection_tracker(dets, start, t0, n_steps, n_frames,
                              search_px: float = 20.0) -> np.ndarray:
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
def feature_correlation_tracker(model, frames, start, t0, n_steps, grid,
                                search_px: float = 24.0, shuffle_frames: bool = False,
                                seed: int = 0) -> np.ndarray:
    gt, gy, gx = grid
    stride_y, stride_x = CROP // gy, CROP // gx
    pos = np.asarray(start, dtype=float)
    out = [pos.copy()]
    rng = np.random.default_rng(seed)

    def window(t: int, p: np.ndarray):
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
        step = np.array([(by - qy) * stride_y, (bx - qx) * stride_x], dtype=float)
        if np.hypot(*step) <= search_px:
            pos = pos + step
        out.append(pos.copy())
    return np.asarray(out)


def evaluate_track(pred: np.ndarray, gt: np.ndarray) -> dict:
    n = min(len(pred), len(gt))
    if n == 0:
        return {"deltas": {}, "delta_avg": None, "survival_frames": 0}
    err = np.hypot(pred[:n, 0] - gt[:n, 0],
                   pred[:n, 1] - gt[:n, 1]) * sbiad.UM_PER_PX
    deltas = {f"within_{t:g}um": float((err <= t).mean()) for t in THRESH_UM}
    within8 = err <= 8.0
    return {"deltas": deltas, "delta_avg": float(np.mean(list(deltas.values()))),
            "median_error_um": float(np.median(err)),
            "survival_frames": int(np.argmin(within8)) if (~within8).any() else n,
            "n_steps": n}


# ------------------------------------------------------------------ forecasting
@torch.no_grad()
def forecast_ratio(model: VideoJEPA, clips: torch.Tensor, batch: int = 32) -> dict:
    """L1 of the predictor's future latents against copy-last and mean baselines."""
    grid = model.grid
    mask = future_mask(grid, n_future_slots=N_FUTURE_SLOTS).to(DEV)
    t, h, w = grid
    n_ctx = t - N_FUTURE_SLOTS
    l1_pred, l1_copy, l1_mean, n = 0.0, 0.0, 0.0, 0
    for i in range(0, len(clips), batch):
        c = clips[i:i + batch].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            res = model(c, mask)
            tgt_all = model.target(c).float()
        tgt_all = torch.nn.functional.layer_norm(tgt_all, (tgt_all.shape[-1],))
        pred = res["pred"].float()
        tgt = res["target"].float()

        # copy-last-observed-latent: the last CONTEXT slot, tiled over the future slots
        toks = tgt_all.reshape(tgt_all.shape[0], t, h * w, -1)
        last_ctx = toks[:, n_ctx - 1, :, :]
        copy = last_ctx[:, None, :, :].repeat(1, N_FUTURE_SLOTS, 1, 1)
        copy = copy.reshape(tgt_all.shape[0], -1, tgt_all.shape[-1])
        # mean over the context slots and positions -> (B, 1, dim), then tiled over
        # every future target position
        mean_lat = toks[:, :n_ctx].mean(dim=(1, 2))[:, None, :]
        mean_b = mean_lat.expand(-1, N_FUTURE_SLOTS * h * w, -1).contiguous()
        # the future-mask targets are exactly the last slots, in the same order
        k = copy.shape[1]
        l1_pred += float(torch.nn.functional.l1_loss(
            pred[:, -k:], tgt[:, -k:], reduction="sum"))
        l1_copy += float(torch.nn.functional.l1_loss(copy, tgt[:, -k:],
                                                     reduction="sum"))
        l1_mean += float(torch.nn.functional.l1_loss(mean_b, tgt[:, -k:],
                                                     reduction="sum"))
        n += int(pred[:, -k:].numel())
    if n == 0:
        return {"status": "no_clips"}
    lp, lc, lm = l1_pred / n, l1_copy / n, l1_mean / n
    return {
        "status": "ok",
        "l1_predictor": lp, "l1_copy_last_latent": lc, "l1_mean_latent": lm,
        "ratio_vs_copy_last": lp / lc if lc > 0 else None,
        "ratio_vs_mean_latent": lp / lm if lm > 0 else None,
        "n_elements": n,
        "interpretation": "ratio < 1 means the predictor beats assuming the latent does "
                          "not change. Only WITHIN-model ratios are comparable: each "
                          "model has its own latent space, so raw L1 across models is "
                          "meaningless.",
    }


def exact_movie_permutation(per_movie: dict[str, float], ref: dict[str, float]) -> dict:
    """Exact permutation over movie labels for a paired method-vs-baseline contrast."""
    movies = sorted(set(per_movie) & set(ref))
    d = np.array([per_movie[m] - ref[m] for m in movies])
    obs = float(d.mean())
    flips = [float(np.mean(d * np.array(sg)))
             for sg in itertools.product([1, -1], repeat=len(d))]
    return {"n_movies": len(movies), "per_movie_difference": d.tolist(),
            "mean_difference": obs,
            "exact_sign_flip_p_two_sided": float(
                np.mean(np.abs(np.asarray(flips)) >= abs(obs))),
            "minimum_achievable_p": float(2.0 / len(flips)) if len(d) else None,
            "confidence_interval": None,
            "confidence_interval_reason":
                f"{len(movies)} movies is below the 10-group minimum (rule 3); v0 "
                "bootstrapped over 5"}


def load_encoders(args) -> dict:
    enc: dict[str, dict] = {}
    main = (Path(args.ckpt) if args.ckpt
            else REPO / "checkpoints" / "tijepa1_T1_seed0.pt")
    specs = [
        ("E4_tijepa1", main, "the v1 model, mixed spatial + future masking"),
        ("E4s_tijepa1_spatial", REPO / "checkpoints" / "tijepa1_spatial_T1_seed0.pt",
         "matched-step spatial-only ablation; queried with a future mask it never saw "
         "in training, so its forecasting ratio is OUT OF DISTRIBUTION"),
        ("E0_tijepa0_reference", REPO / "checkpoints" / "tijepa0_T1_seed0.pt",
         "historical reference: different corpus, not a valid ablation"),
    ]
    for name, path, note in specs:
        if not path.exists():
            continue
        m = VideoJEPA(clip=(NF, CROP, CROP))
        sd = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(sd["model"])
        enc[name] = {"model": m.eval().to(DEV), "note": note,
                     "checkpoint": path.name, "step": sd.get("step")}
    for s in (0, 1, 2):
        torch.manual_seed(9000 + s)
        enc[f"E6_random_init_draw{s}"] = {
            "model": VideoJEPA(clip=(NF, CROP, CROP)).eval().to(DEV),
            "note": f"independent random-init draw {s}", "checkpoint": None,
            "step": None}
    return enc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--n-tracks", type=int, default=25)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--forecast-clips", type=int, default=256)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    movies = sorted((RAW / "s-biad2196" / "Celltrack_data").glob("*.zip"))
    ltdb = sorted((RAW / "ltdb").glob("*")) if (RAW / "ltdb").exists() else []
    leak_path = REPO / "results" / "v1" / "leakage_check.json"
    leak = json.loads(leak_path.read_text()) if leak_path.exists() else {}

    enc = load_encoders(args)
    print(f"{len(movies)} S-BIAD2196 movies, {len(enc)} encoders: {sorted(enc)}")

    result: dict = {
        "phase": "F3.3 + F3.4",
        "dataset": "S-BIAD2196 CellTrack, all 6 movies the deposit contains",
        "n_movies": len(movies),
        "movies": [m.name for m in movies],
        "leakage": {
            "checked": bool(leak),
            "excluded_pretraining_movies": leak.get("excluded_train_movies", []),
            "eval_movies_with_a_twin": leak.get("eval_movies_with_a_twin", []),
            "policy": "the matching ComplexEye movie is excluded from pretraining, so "
                      "the evaluation movie stays usable (F1.3.4)",
        },
        "ltdb": ("provided: " + str(len(ltdb)) + " files" if ltdb else
                 "NOT PROVIDED by the author; see results/v1/manual_intake.json"),
        "cotracker": "not run: dl.fbaipublicfiles.com was unreachable in preflight; "
                     "the Hugging Face route was not attempted in this run",
        "encoders": {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                     for k, v in enc.items()},
        "statistics": "6 movies is below the 10-group minimum, so no bootstrap CI is "
                      "reported (rule 3); per-movie values and an exact sign-flip "
                      "permutation are given instead.",
    }

    per_track: list[dict] = []
    forecast: dict[str, dict] = {k: {} for k in enc}
    for mv in movies:
        raw, lab = sbiad.load_movie(mv, max_frames=args.max_frames)
        tracks = sbiad.tracks_from_labels(lab, min_len=args.n_steps + 1)
        ids = sorted(tracks)[: args.n_tracks]
        print(f"{mv.name}: {raw.shape}, {len(tracks)} tracks >= {args.n_steps+1}, "
              f"using {len(ids)}")
        dets = [D.detect_frame(raw[t], D.temporal_background(raw))
                for t in range(raw.shape[0])]

        clips_for_forecast = []
        for tid in ids:
            tr = tracks[tid]
            t0 = int(tr[0, 0])
            gt = tr[:, 1:3]
            start = (gt[0, 0], gt[0, 1])
            row = {"movie": mv.name, "track_id": int(tid), "t0": t0,
                   "gt_len": int(len(gt))}
            row["baseline_nearest_detection"] = evaluate_track(
                nearest_detection_tracker(dets, start, t0, args.n_steps,
                                          raw.shape[0]), gt)
            for name, e in enc.items():
                row[name] = evaluate_track(feature_correlation_tracker(
                    e["model"], raw, start, t0, args.n_steps, e["model"].grid), gt)
            if "E4_tijepa1" in enc:
                row["E5_frame_shuffled"] = evaluate_track(
                    feature_correlation_tracker(
                        enc["E4_tijepa1"]["model"], raw, start, t0, args.n_steps,
                        enc["E4_tijepa1"]["model"].grid, shuffle_frames=True), gt)
            per_track.append(row)

            # a cell-centred clip for the forecasting evaluation
            half = CROP // 2
            y0, x0 = int(round(gt[0, 0])) - half, int(round(gt[0, 1])) - half
            if (y0 >= 0 and x0 >= 0 and y0 + CROP <= raw.shape[1]
                    and x0 + CROP <= raw.shape[2] and t0 + NF <= raw.shape[0]):
                c = raw[t0:t0 + NF, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32)
                c = (c - c.mean()) / (c.std() + 1e-6)
                clips_for_forecast.append(torch.from_numpy(c)[None])

        if clips_for_forecast:
            cl = torch.stack(clips_for_forecast)[: args.forecast_clips]
            for name, e in enc.items():
                forecast[name][mv.name] = forecast_ratio(e["model"], cl)
            print(f"   forecasting on {len(cl)} clips: " + ", ".join(
                f"{n}={forecast[n][mv.name].get('ratio_vs_copy_last'):.3f}"
                for n in enc if forecast[n][mv.name].get("ratio_vs_copy_last")))

    # ---- aggregate tracking
    methods = ["baseline_nearest_detection", *sorted(enc), "E5_frame_shuffled"]
    agg: dict[str, dict] = {}
    per_movie_delta: dict[str, dict[str, float]] = {}
    for meth in methods:
        vals = [r[meth]["delta_avg"] for r in per_track
                if meth in r and r[meth].get("delta_avg") is not None]
        if not vals:
            agg[meth] = {"status": "unavailable"}
            continue
        surv = [r[meth]["survival_frames"] for r in per_track if meth in r]
        pm = {}
        for mvn in {r["movie"] for r in per_track}:
            v = [r[meth]["delta_avg"] for r in per_track
                 if r["movie"] == mvn and meth in r
                 and r[meth].get("delta_avg") is not None]
            if v:
                pm[mvn] = float(np.mean(v))
        per_movie_delta[meth] = pm
        agg[meth] = {"delta_avg_mean": float(np.mean(vals)),
                     "median_survival_frames": float(np.median(surv)),
                     "n_tracks": len(vals), "per_movie_delta_avg": pm,
                     "confidence_interval": None}
        print(f"  {meth:28s} <delta> {np.mean(vals):.3f} "
              f"survival {np.median(surv):.0f} frames")

    base = per_movie_delta.get("baseline_nearest_detection", {})
    for meth in methods:
        if meth in per_movie_delta and meth != "baseline_nearest_detection" and base:
            agg[meth]["vs_baseline_exact_permutation"] = exact_movie_permutation(
                per_movie_delta[meth], base)

    result["tracking"] = {"aggregate": agg, "per_track": per_track}
    result["forecasting"] = {
        "n_future_slots": N_FUTURE_SLOTS,
        "frames_context_then_predicted": [NF - N_FUTURE_SLOTS * 2,
                                          N_FUTURE_SLOTS * 2],
        "per_movie": forecast,
        "summary": {
            name: {
                "mean_ratio_vs_copy_last": float(np.mean(
                    [v["ratio_vs_copy_last"] for v in forecast[name].values()
                     if v.get("ratio_vs_copy_last")])) if forecast[name] else None,
                "mean_ratio_vs_mean_latent": float(np.mean(
                    [v["ratio_vs_mean_latent"] for v in forecast[name].values()
                     if v.get("ratio_vs_mean_latent")])) if forecast[name] else None,
                "note": enc[name]["note"],
            } for name in enc},
    }
    (OUT / "tracking_forecast_v1.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n")
    print(f"wrote {OUT/'tracking_forecast_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
