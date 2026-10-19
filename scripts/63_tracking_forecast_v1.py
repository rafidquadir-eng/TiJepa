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

### v2 (PLAN_v2.md G2.3, G3.3; findings N5, N7, N1)

* Up to 50 ground-truth tracks per movie (all if fewer), chosen by a seeded rule that is
  recorded, over the whole movie rather than its first 160 frames; `n_tracks` per movie.
* Per-threshold success curves (1, 2, 4, 8, 16 um). Survival = number of frames before
  the first frame whose error exceeds 8 um. When <delta> and survival rank two methods
  differently, the output says so.
* Leakage: TiJepa-0's corpus contained `R08:H4`, the twin of `08_H4`, so every E0
  comparison is reported with and without `08_H4`. Evaluation movies with a twin in the
  frozen v1 corpus (`results/v2/leakage_check_full.json`) are excluded for every model
  trained on that corpus.
* Forecasting on every movie, from clips sampled along all of its tracks; any movie that
  cannot be evaluated gets a stated reason. Primary comparison: TiJepa-1 against
  TiJepa-1-spatial, per-movie ratio, exact sign-flip test over movies. E0 is a reference
  row; random-init ratios are supplementary (their predictor is untrained).
* Each learned encoder records its checkpoint (file, run, step, SHA-256).
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

from tijepa.eval import provenance as PV  # noqa: E402
from tijepa.io import sbiad  # noqa: E402
from tijepa.models.video_jepa import VideoJEPA, future_mask  # noqa: E402
from tijepa.preprocess import detect as D  # noqa: E402

OUT = REPO / "results" / "v2" / "partA"
TRACK_SEED = 20261016
E0_TWIN_MOVIE = "08_H4.ome.zarr.zip"  # twin of R08:H4, which was in TiJepa-0's corpus
CORPUS_V1_RUNS = ("tijepa1", "tijepa1_spatial")
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
    specs = [
        ("tijepa1", Path(args.ckpt) if args.ckpt else
         REPO / "checkpoints" / "tijepa1_T1_seed0.pt",
         "mixed spatial + future masking"),
        ("tijepa1_spatial", REPO / "checkpoints" / "tijepa1_spatial_T1_seed0.pt",
         "matched-step spatial-only ablation; queried with a future mask it never saw "
         "in training, so its forecasting ratio is out of distribution"),
        ("tijepa0", REPO / "checkpoints" / "tijepa0_T1_seed0.pt",
         "historical reference: different corpus, not a valid ablation"),
    ]
    for run, path, note in specs:
        if not path.exists():
            print(f"  {run}: checkpoint absent, not evaluated")
            continue
        sd = torch.load(path, map_location="cpu", weights_only=False)
        rec = PV.checkpoint_record(path, sd)
        if run != "tijepa0" and not rec["final"] and not args.allow_unfinished:
            raise SystemExit(f"{path.name} is at step {rec['step']}, not its planned final "
                             f"step {rec['planned_final_step']}; evaluate final checkpoints "
                             "only (or pass --allow-unfinished for a smoke test)")
        m = VideoJEPA(clip=(NF, CROP, CROP))
        m.load_state_dict(sd["model"])
        enc[PV.key_for(rec)] = {"model": m.eval().to(DEV), "note": note, "checkpoint": rec}
    for s in (0, 1, 2):
        torch.manual_seed(9000 + s)
        enc[f"E6_random_init_draw{s}"] = {
            "model": VideoJEPA(clip=(NF, CROP, CROP)).eval().to(DEV),
            "note": f"independent random-init draw {s}",
            "checkpoint": PV.random_init_record(9000 + s)}
    for k, e in enc.items():
        PV.label_for_key(k, e["checkpoint"])
    return enc


def select_tracks(tracks: dict, movie: str, n: int) -> list[int]:
    """Up to `n` track ids, a seeded draw from all eligible ids (recorded)."""
    ids = sorted(int(t) for t in tracks)
    if len(ids) <= n:
        return ids
    seed = (TRACK_SEED, int.from_bytes(movie.encode()[:8].ljust(8, b"_"), "little"))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(ids, size=n, replace=False))


def excluded_movies(enc_key: str, enc: dict, corpus_twins: set[str]) -> set[str]:
    run = (enc.get("checkpoint") or {}).get("run_name")
    out = set()
    if run in CORPUS_V1_RUNS or enc_key == "E5_frame_shuffled":
        out |= corpus_twins
    if run == "tijepa0":
        out.add(E0_TWIN_MOVIE)
    return out


def rank_note(agg: dict, a: str, b: str) -> str | None:
    """A sentence when <delta> and survival order two methods differently."""
    if a not in agg or b not in agg or "delta_avg_mean" not in agg[a] \
            or "delta_avg_mean" not in agg[b]:
        return None
    d = np.sign(agg[a]["delta_avg_mean"] - agg[b]["delta_avg_mean"])
    sv = np.sign(agg[a]["median_survival_frames"] - agg[b]["median_survival_frames"])
    if d != 0 and sv != 0 and d != sv:
        return (f"{a} and {b} are ordered differently by <delta> and by median survival: "
                "<delta> averages success over the whole track at five thresholds, while "
                "survival counts only frames before the first error above 8 um, so a "
                "tracker that drifts slowly but rarely jumps can win one and lose the other")
    return None


def aggregate(per_track: list[dict], meth: str, skip: set[str]) -> dict | None:
    rows = [r for r in per_track if meth in r and r["movie"] not in skip
            and r[meth].get("delta_avg") is not None]
    if not rows:
        return None
    pm: dict[str, float] = {}
    for mvn in sorted({r["movie"] for r in rows}):
        pm[mvn] = float(np.mean([r[meth]["delta_avg"] for r in rows if r["movie"] == mvn]))
    curve = {f"within_{t:g}um": float(np.mean([r[meth]["deltas"][f"within_{t:g}um"]
                                               for r in rows])) for t in THRESH_UM}
    return {"delta_avg_mean": float(np.mean([r[meth]["delta_avg"] for r in rows])),
            "median_survival_frames": float(np.median([r[meth]["survival_frames"]
                                                       for r in rows])),
            "per_threshold_success": curve,
            "n_tracks": len(rows), "per_movie_delta_avg": pm,
            "movies_excluded": sorted(skip), "confidence_interval": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole movie")
    ap.add_argument("--n-tracks", type=int, default=50, help="per movie")
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--forecast-clips", type=int, default=256, help="per movie")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--allow-unfinished", action="store_true")
    ap.add_argument("--movies", default="", help="comma-separated subset (smoke tests)")
    ap.add_argument("--out-name", default="tracking_forecast.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    movies = sorted((RAW / "s-biad2196" / "Celltrack_data").glob("*.zip"))
    if args.movies:
        movies = [m for m in movies if any(k in m.name for k in args.movies.split(","))]
    ltdb = sorted((RAW / "ltdb").glob("*")) if (RAW / "ltdb").exists() else []
    leak_path = REPO / "results" / "v2" / "leakage_check_full.json"
    leak = json.loads(leak_path.read_text()) if leak_path.exists() else {}
    corpus_twins = set(leak.get("excluded_eval_movies_for_corpus_v1_models", []))

    enc = load_encoders(args)
    print(f"{len(movies)} S-BIAD2196 movies, {len(enc)} encoders: {sorted(enc)}")

    result: dict = {
        "phase": "G2.3 (F3.3 + F3.4)",
        "dataset": "S-BIAD2196 CellTrack, all movies the deposit contains",
        "n_movies": len(movies),
        "movies": [m.name for m in movies],
        "leakage": {
            "full_check": str(leak_path.relative_to(REPO)) if leak else None,
            "full_check_complete": bool(leak.get("all_archives_checked")),
            "excluded_for_corpus_v1_models": sorted(corpus_twins),
            "excluded_for_tijepa0": [E0_TWIN_MOVIE],
            "policy": "TiJepa-0's corpus contained R08:H4, the twin of 08_H4, so E0 "
                      "comparisons are given with and without 08_H4; an evaluation movie "
                      "with a twin in the frozen v1 corpus is excluded for every model "
                      "trained on it",
        },
        "track_selection": {
            "rule": f"tracks of >= {args.n_steps + 1} frames over the whole movie; up to "
                    f"{args.n_tracks} per movie drawn without replacement with "
                    f"numpy default_rng(({TRACK_SEED}, movie-name bytes)); all if fewer",
            "per_movie": {}},
        "survival_definition": "number of frames tracked before the first frame whose "
                               "position error exceeds 8 um (the whole track if never)",
        "ltdb": ("provided: " + str(len(ltdb)) + " files" if ltdb else
                 "not provided by the author; see results/v2/manual_intake.json"),
        "cotracker": "not run: dl.fbaipublicfiles.com was unreachable in v1 preflight; "
                     "the Hugging Face route was not attempted",
        "encoders": {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                     for k, v in enc.items()},
        "statistics": f"{len(movies)} movies is below the 10-group minimum, so no "
                      "bootstrap CI is reported (rule 3); per-movie values and an exact "
                      "sign-flip permutation are given instead.",
    }

    per_track: list[dict] = []
    forecast: dict[str, dict] = {k: {} for k in enc}
    forecast_status: dict[str, str] = {}
    for mv in movies:
        raw, lab = sbiad.load_movie(mv, max_frames=args.max_frames or None)
        tracks = sbiad.tracks_from_labels(lab, min_len=args.n_steps + 1)
        ids = select_tracks(tracks, mv.name, args.n_tracks)
        result["track_selection"]["per_movie"][mv.name] = {
            "n_frames": int(raw.shape[0]), "n_eligible": len(tracks),
            "n_tracks": len(ids), "track_ids": ids}
        print(f"{mv.name}: {raw.shape}, {len(tracks)} tracks >= {args.n_steps+1}, "
              f"using {len(ids)}")
        bg = D.temporal_background(raw)
        dets = [D.detect_frame(raw[t], bg) for t in range(raw.shape[0])]

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

        # forecasting clips: cell-centred windows along EVERY track, every NF frames
        half = CROP // 2
        cands = []
        for tid in sorted(tracks):
            tr = tracks[tid]
            for k in range(0, len(tr) - NF + 1, NF):
                t0 = int(tr[k, 0])
                y0 = int(round(tr[k, 1])) - half
                x0 = int(round(tr[k, 2])) - half
                if (y0 >= 0 and x0 >= 0 and y0 + CROP <= raw.shape[1]
                        and x0 + CROP <= raw.shape[2] and t0 + NF <= raw.shape[0]):
                    cands.append((t0, y0, x0))
        if not cands:
            forecast_status[mv.name] = ("no 16-frame cell-centred window fits inside the "
                                        "frame along any track")
            continue
        rng = np.random.default_rng((TRACK_SEED, 1, len(cands)))
        pick = rng.permutation(len(cands))[: args.forecast_clips]
        clips = []
        for i in sorted(pick):
            t0, y0, x0 = cands[i]
            c = raw[t0:t0 + NF, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32)
            clips.append(torch.from_numpy((c - c.mean()) / (c.std() + 1e-6))[None])
        cl = torch.stack(clips)
        forecast_status[mv.name] = f"evaluated on {len(cl)} clips of {len(cands)} candidates"
        for name, e in enc.items():
            forecast[name][mv.name] = forecast_ratio(e["model"], cl)
        print(f"   forecasting on {len(cl)} clips: " + ", ".join(
            f"{n}={forecast[n][mv.name].get('ratio_vs_copy_last'):.3f}"
            for n in enc if forecast[n][mv.name].get("ratio_vs_copy_last")))

    # ---- aggregate tracking
    methods = ["baseline_nearest_detection", *sorted(enc), "E5_frame_shuffled"]
    agg: dict[str, dict] = {}
    for meth in methods:
        skip = excluded_movies(meth, enc.get(meth, {}), corpus_twins)
        a = aggregate(per_track, meth, skip)
        if a is None:
            agg[meth] = {"status": "unavailable"}
            continue
        if meth == "E0_tijepa0":
            a["with_08_H4_reference_only"] = aggregate(per_track, meth, set())
        agg[meth] = a
        print(f"  {meth:28s} <delta> {a['delta_avg_mean']:.3f} "
              f"survival {a['median_survival_frames']:.0f} frames ({a['n_tracks']} tracks)")
    base_rows = {mvn: v for mvn, v in agg["baseline_nearest_detection"].get(
        "per_movie_delta_avg", {}).items()}
    for meth in methods:
        if meth == "baseline_nearest_detection" or "per_movie_delta_avg" not in agg[meth]:
            continue
        pm = agg[meth]["per_movie_delta_avg"]
        agg[meth]["vs_baseline_exact_permutation"] = exact_movie_permutation(
            pm, {m: base_rows[m] for m in pm if m in base_rows})
        if meth == "E0_tijepa0":
            w = agg[meth]["with_08_H4_reference_only"]["per_movie_delta_avg"]
            agg[meth]["with_08_H4_reference_only"]["vs_baseline_exact_permutation"] = \
                exact_movie_permutation(w, base_rows)
    notes = [n for a, b in itertools.combinations(
        [m for m in methods if "delta_avg_mean" in agg[m]], 2)
        if (n := rank_note(agg, a, b))]
    result["tracking"] = {"aggregate": agg, "rank_consistency_notes": notes,
                          "per_track": per_track}

    # ---- forecasting
    def summary(name: str) -> dict:
        skip = excluded_movies(name, enc[name], corpus_twins)
        vals = {m: v["ratio_vs_copy_last"] for m, v in forecast[name].items()
                if v.get("ratio_vs_copy_last") and m not in skip}
        mv = {m: v["ratio_vs_mean_latent"] for m, v in forecast[name].items()
              if v.get("ratio_vs_mean_latent") and m not in skip}
        return {"per_movie_ratio_vs_copy_last": vals,
                "mean_ratio_vs_copy_last": float(np.mean(list(vals.values()))) if vals
                else None,
                "mean_ratio_vs_mean_latent": float(np.mean(list(mv.values()))) if mv
                else None,
                "movies_excluded": sorted(skip), "note": enc[name]["note"],
                "checkpoint": enc[name]["checkpoint"]}

    main_rows = {k: summary(k) for k in enc if not k.startswith("E6_")}
    supp_rows = {k: summary(k) for k in enc if k.startswith("E6_")}
    primary = None
    if "E4_tijepa1" in main_rows and "E4s_tijepa1_spatial" in main_rows:
        a = main_rows["E4_tijepa1"]["per_movie_ratio_vs_copy_last"]
        b = main_rows["E4s_tijepa1_spatial"]["per_movie_ratio_vs_copy_last"]
        primary = {"comparison": "TiJepa-1 ratio minus TiJepa-1-spatial ratio, per movie "
                                 "(negative = TiJepa-1 forecasts better relative to its "
                                 "own copy-last baseline)",
                   **exact_movie_permutation(a, b)}
    result["forecasting"] = {
        "n_future_slots": N_FUTURE_SLOTS,
        "frames_context_then_predicted": [NF - N_FUTURE_SLOTS * 2, N_FUTURE_SLOTS * 2],
        "per_movie_status": forecast_status,
        "per_movie": forecast,
        "primary_tijepa1_vs_spatial": primary
        or {"status": "not run", "reason": "both TiJepa-1 checkpoints are required"},
        "summary": main_rows,
        "supplementary_random_init": {
            "rows": supp_rows,
            "why_not_comparable": "a random-init model's predictor is untrained, so its "
                                  "ratio measures an arbitrary map, not forecasting; it is "
                                  "listed for completeness only"},
    }
    problems = PV.check_results({k: v for k, v in result["encoders"].items()})
    if problems:
        raise SystemExit("provenance check failed: " + "; ".join(problems))
    (OUT / args.out_name).write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"wrote {OUT / args.out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
