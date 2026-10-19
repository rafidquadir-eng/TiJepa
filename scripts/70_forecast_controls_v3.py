"""H2: verify or retract the latent-forecasting result (PLAN_v3 §6, finding R6, R11).

The v2 metric compared each model's predictor with "copy the last context latent".
Target latents at future slots also carry temporal-position structure that the predictor
receives through its positional embeddings, so beating copy-last does not by itself show
that cell dynamics are modelled. For TiJepa-1, the spatial-only ablation and TiJepa-0,
on every S-BIAD2196 movie, this script reports:

1. slot-conditional mean baselines (mean target latent per future slot, and per slot and
   spatial position), fitted on the other half of the movies (2-fold over movies);
2. a context-swap control: the predictor gets the context of a different clip from the
   same movie and is scored against the true clip's targets;
3. a static control: one frame (the last context frame) repeated 16 times;
4. per-stride results, stride 1, 3, 9 and 27 (frame interval 8 s to 3.6 min, so the
   4-frame forecast horizon spans 32 s to 14.4 min);
5. a pixel-space sanity check: a ridge decoder from target latents to the future tubelet's
   4 x 4-downsampled patch pixels, fitted on the other half of the movies, applied to the
   predictor's and to copy-last's latents, and compared with copying the last context
   pixels.

G-Forecast (src/tijepa/eval/forecast_controls.gate) is applied to TiJepa-1 at stride 1,
the stride the v2 claim was made at. Six movies is below the 10-group minimum, so
statistics are per-movie values with exact sign-flip tests and no confidence intervals.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.eval import forecast_controls as FC  # noqa: E402
from tijepa.eval import provenance as PV  # noqa: E402
from tijepa.io import sbiad  # noqa: E402
from tijepa.models.video_jepa import VideoJEPA, future_mask  # noqa: E402

OUT = REPO / "results" / "v3" / "partA"
RAW = REPO / "data" / "raw"
CROP, NF, N_FUT = 96, 16, 2
STRIDES = (1, 3, 9, 27)
FRAME_INTERVAL_S = 8.0
SEED = 20261019
E0_TWIN_MOVIE = "08_H4.ome.zarr.zip"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = (("tijepa1", "tijepa1_T1_seed0.pt"),
        ("tijepa1_spatial", "tijepa1_spatial_T1_seed0.pt"),
        ("tijepa0", "tijepa0_T1_seed0.pt"))


def load_models() -> dict:
    out = {}
    for run, fn in RUNS:
        path = REPO / "checkpoints" / fn
        sd = torch.load(path, map_location="cpu", weights_only=False)
        rec = PV.checkpoint_record(path, sd)
        if run != "tijepa0" and not rec["final"]:
            raise SystemExit(f"{fn} is not at its planned final step")
        m = VideoJEPA(clip=(NF, CROP, CROP))
        m.load_state_dict(sd["model"])
        key = PV.key_for(rec)
        PV.label_for_key(key, rec)
        out[key] = {"model": m.eval().to(DEV), "checkpoint": rec}
    return out


def clip_windows(raw: np.ndarray, tracks: dict, stride: int, n: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Up to n clips of 16 frames at the given stride, each a fixed 96 x 96 window
    centred on a tracked cell at the clip's first frame (candidates every 16 frames
    along every track)."""
    half, span = CROP // 2, (NF - 1) * stride
    cands = []
    for tid in sorted(tracks):
        tr = tracks[tid]
        for k in range(0, len(tr), NF):
            t0 = int(tr[k, 0])
            y0, x0 = int(round(tr[k, 1])) - half, int(round(tr[k, 2])) - half
            if (y0 >= 0 and x0 >= 0 and y0 + CROP <= raw.shape[1]
                    and x0 + CROP <= raw.shape[2] and t0 + span < raw.shape[0]):
                cands.append((t0, y0, x0))
    if not cands:
        return np.zeros((0, 1, NF, CROP, CROP), np.float32), 0
    pick = sorted(rng.permutation(len(cands))[:n])
    clips = np.empty((len(pick), 1, NF, CROP, CROP), np.float32)
    for j, i in enumerate(pick):
        t0, y0, x0 = cands[i]
        c = raw[t0:t0 + span + 1:stride, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32)
        clips[j, 0] = (c - c.mean()) / (c.std() + 1e-6)
    return clips, len(cands)


def static_of(clips: np.ndarray) -> np.ndarray:
    last_ctx = NF - 2 * N_FUT - 1          # frame 11, the last context frame
    return np.repeat(clips[:, :, last_ctx:last_ctx + 1], NF, axis=2)


def patch_pixels(clips: torch.Tensor, frames: slice) -> torch.Tensor:
    """(B,1,T,96,96) -> per future-slot token, 4x4-pooled mean of the tubelet frames:
    (B, n_slots, 144, 16)."""
    x = clips[:, 0, frames]                                     # (B, 2*S, 96, 96)
    b, f = x.shape[:2]
    x = x.reshape(b, f // 2, 2, CROP, CROP).mean(2)             # (B, S, 96, 96)
    x = torch.nn.functional.avg_pool2d(x, 2)                    # (B, S, 48, 48)
    x = x.reshape(b, f // 2, 12, 4, 12, 4).permute(0, 1, 2, 4, 3, 5)
    return x.reshape(b, f // 2, 144, 16)


@torch.no_grad()
def run_model(model: VideoJEPA, clips: np.ndarray, perm: np.ndarray,
              batch: int = 32) -> dict:
    t, h, w = model.grid
    n_ctx, p = t - N_FUT, h * w
    mask = future_mask(model.grid, n_future_slots=N_FUT).to(DEV)
    keys = ("pred", "tgt", "copy", "pred_swap", "pix", "pix_copy")
    acc = {k: [] for k in keys}
    st = {"l1_pred": 0.0, "l1_copy": 0.0, "n": 0}
    x_all = torch.from_numpy(clips)
    xs_all = torch.from_numpy(static_of(clips))
    for i in range(0, len(clips), batch):
        idx = np.arange(i, min(i + batch, len(clips)))
        c = x_all[idx].to(DEV)
        cs = x_all[perm[idx]].to(DEV)
        cst = xs_all[idx].to(DEV)
        out = {}
        for name, inp in (("real", c), ("swap", cs), ("static", cst)):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                res = model(inp, mask)
                ta = model.target(inp).float()
            ta = torch.nn.functional.layer_norm(ta, (ta.shape[-1],))
            toks = ta.reshape(len(idx), t, p, -1)
            out[name] = (res["pred"].float().reshape(len(idx), N_FUT, p, -1),
                         toks[:, n_ctx:], toks[:, n_ctx - 1:n_ctx].expand(-1, N_FUT, -1, -1))
        pred, tgt, copy = out["real"]
        acc["pred"].append(pred.half().cpu())
        acc["tgt"].append(tgt.half().cpu())
        acc["copy"].append(copy.half().cpu())
        acc["pred_swap"].append(out["swap"][0].half().cpu())
        acc["pix"].append(patch_pixels(c.float(), slice(2 * n_ctx, 2 * t)).cpu())
        acc["pix_copy"].append(patch_pixels(c.float(), slice(2 * n_ctx - 2, 2 * n_ctx))
                               .expand(-1, N_FUT, -1, -1).cpu())
        sp, stg, scp = out["static"]
        st["l1_pred"] += float((sp - stg).abs().sum())
        st["l1_copy"] += float((scp - stg).abs().sum())
        st["n"] += int(sp.numel())
    arr = {k: torch.cat(v).numpy() for k, v in acc.items()}
    arr["static_ratio_vs_copy_last"] = st["l1_pred"] / st["l1_copy"]
    return arr


def pixel_check(store: dict, folds: dict, rng: np.random.Generator,
                n_fit: int = 40000) -> dict:
    """Ridge decoder latent -> future patch pixels, 2-fold over movies."""
    res = {}
    for m, fit_movies in folds.items():
        xs, ys = [], []
        for fm in fit_movies:
            t = store[fm]["tgt"].reshape(-1, store[fm]["tgt"].shape[-1])
            y = store[fm]["pix"].reshape(-1, 16)
            k = rng.choice(len(t), size=min(n_fit // len(fit_movies), len(t)),
                           replace=False)
            xs.append(t[k]), ys.append(y[k])
        w = FC.ridge_fit(np.concatenate(xs), np.concatenate(ys), lam=10.0)
        s = store[m]
        d = s["tgt"].shape[-1]
        y = s["pix"].reshape(-1, 16).astype(np.float64)
        dec = {k: FC.ridge_apply(w, s[k].reshape(-1, d)) for k in ("pred", "copy", "tgt")}
        mse = {f"mse_decoded_{k}": float(np.mean((v - y) ** 2)) for k, v in dec.items()}
        mse["mse_pixel_copy_last"] = float(np.mean(
            (s["pix_copy"].reshape(-1, 16) - y) ** 2))
        mse["mse_pixel_zero"] = float(np.mean(y ** 2))
        mse["ratio_decoded_pred_vs_decoded_copy"] = (mse["mse_decoded_pred"]
                                                     / mse["mse_decoded_copy"])
        mse["ratio_decoded_pred_vs_pixel_copy_last"] = (mse["mse_decoded_pred"]
                                                        / mse["mse_pixel_copy_last"])
        res[m] = mse
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=256, help="per movie and stride")
    ap.add_argument("--movies", default="", help="comma-separated subset (smoke tests)")
    ap.add_argument("--strides", default=",".join(map(str, STRIDES)))
    ap.add_argument("--out", default=str(OUT / "forecast_controls.json"))
    args = ap.parse_args()
    strides = [int(s) for s in args.strides.split(",")]
    t_start = time.time()

    movies = sorted((RAW / "s-biad2196" / "Celltrack_data").glob("*.zip"))
    if args.movies:
        movies = [m for m in movies if any(k in m.name for k in args.movies.split(","))]
    models = load_models()
    print(f"{len(movies)} movies; models {sorted(models)}; strides {strides}")

    clips: dict[int, dict[str, np.ndarray]] = {s: {} for s in strides}
    n_cands: dict[int, dict[str, int]] = {s: {} for s in strides}
    for mv in movies:
        raw, lab = sbiad.load_movie(mv)
        tracks = sbiad.tracks_from_labels(lab, min_len=1)
        for s in strides:
            rng = np.random.default_rng((SEED, s, int.from_bytes(mv.name.encode()[:8],
                                                                  "little")))
            c, nc = clip_windows(raw, tracks, s, args.clips, rng)
            clips[s][mv.name], n_cands[s][mv.name] = c, nc
        print(f"  {mv.name}: {raw.shape[0]} frames; clips per stride "
              + ", ".join(f"{s}:{len(clips[s][mv.name])}" for s in strides))
        del raw, lab

    result: dict = {
        "phase": "PLAN_v3 H2 (finding R6, R11)",
        "dataset": "S-BIAD2196 CellTrack, all 6 movies",
        "frame_interval_s": FRAME_INTERVAL_S,
        "forecast_horizon": f"last {N_FUT} tubelet slots = last 4 frames of a 16-frame clip",
        "strides": {s: {"frame_interval_s": s * FRAME_INTERVAL_S,
                        "horizon_s": 4 * s * FRAME_INTERVAL_S,
                        "n_clips": {m: int(len(v)) for m, v in clips[s].items()},
                        "n_candidates": n_cands[s]} for s in strides},
        "clip_rule": f"fixed {CROP}x{CROP} window centred on a tracked cell at the clip's "
                     f"first frame; candidates every {NF} frames along every track; up to "
                     f"{args.clips} per movie and stride, seeded (seed {SEED})",
        "static_control": "frame 11 (the last context frame) repeated 16 times",
        "context_swap": "context of another clip from the same movie (seeded cyclic shift); "
                        "scored against the true clip's targets",
        "slot_mean_baselines": "mean target latent per future slot (and per slot x "
                               "spatial position), fitted on the other half of the movies "
                               "(2-fold); the better of the two is the reference",
        "pixel_check": "ridge (lambda 10) from LayerNorm-ed target latents to the 4x4 "
                       "pooled pixels of each future tubelet patch, fitted on 40,000 "
                       "tokens of the other half of the movies",
        "statistics": "6 movies < 10 groups: per-movie values and exact sign-flip tests "
                      "over movies, no confidence intervals",
        "leakage": f"TiJepa-0's corpus contained the twin of {E0_TWIN_MOVIE}; that movie is "
                   "excluded from every TiJepa-0 summary",
        "models": {k: {"checkpoint": v["checkpoint"]} for k, v in models.items()},
        "results": {},
    }

    for key, mv in models.items():
        result["results"][key] = {}
        for s in strides:
            store, static = {}, {}
            for name, c in clips[s].items():
                if key == "E0_tijepa0" and name == E0_TWIN_MOVIE:
                    continue
                if len(c) < 2:
                    continue
                rng = np.random.default_rng((SEED, 7, s))
                shift = int(rng.integers(1, len(c)))
                perm = (np.arange(len(c)) + shift) % len(c)
                store[name] = run_model(mv["model"], c, perm)
                static[name] = store[name].pop("static_ratio_vs_copy_last")
            if len(store) < 2:
                result["results"][key][str(s)] = {"status": "fewer than 2 movies"}
                continue
            folds = FC.two_fold(list(store))
            per_movie = {}
            for name in store:
                means = FC.slot_means([store[f]["tgt"] for f in folds[name]])
                sc = FC.movie_scores(store[name]["pred"], store[name]["tgt"],
                                     store[name]["copy"], store[name]["pred_swap"], means)
                sc["static_ratio_vs_copy_last"] = static[name]
                per_movie[name] = sc
            pix = pixel_check(store, folds, np.random.default_rng((SEED, 11, s)))
            for name in per_movie:
                per_movie[name]["pixel"] = pix[name]
            summ = {k: float(np.mean([v[k] for v in per_movie.values()]))
                    for k in ("ratio_vs_copy_last", "ratio_vs_slot_mean",
                              "swap_ratio_vs_copy_last", "swap_degradation",
                              "static_ratio_vs_copy_last")}
            summ["pixel_ratio_decoded_pred_vs_decoded_copy"] = float(np.mean(
                [v["ratio_decoded_pred_vs_decoded_copy"] for v in pix.values()]))
            summ["pixel_ratio_decoded_pred_vs_pixel_copy_last"] = float(np.mean(
                [v["ratio_decoded_pred_vs_pixel_copy_last"] for v in pix.values()]))
            summ["sign_flip_log_ratio_vs_copy_last"] = FC.sign_flip(
                [float(np.log(v["ratio_vs_copy_last"])) for v in per_movie.values()])
            summ["sign_flip_log_ratio_vs_slot_mean"] = FC.sign_flip(
                [float(np.log(v["ratio_vs_slot_mean"])) for v in per_movie.values()])
            entry = {"per_movie": per_movie, "summary": summ,
                     "gate": FC.gate(per_movie, static)}
            result["results"][key][str(s)] = entry
            print(f"{key} stride {s}: vs copy {summ['ratio_vs_copy_last']:.3f}  "
                  f"vs slot-mean {summ['ratio_vs_slot_mean']:.3f}  "
                  f"swap {summ['swap_ratio_vs_copy_last']:.3f}  "
                  f"static {summ['static_ratio_vs_copy_last']:.3f}  "
                  f"pix {summ['pixel_ratio_decoded_pred_vs_pixel_copy_last']:.3f}  "
                  f"gate={'pass' if entry['gate']['pass'] else 'fail'}")
            del store

    # E4 vs E4s, per stride: paired log-ratio difference over shared movies
    cmp = {}
    if {"E4_tijepa1", "E4s_tijepa1_spatial"} <= set(result["results"]):
        for s in map(str, strides):
            a = result["results"]["E4_tijepa1"][s]["per_movie"]
            b = result["results"]["E4s_tijepa1_spatial"][s]["per_movie"]
            ms = sorted(set(a) & set(b))
            cmp[s] = FC.sign_flip([float(np.log(a[m]["ratio_vs_slot_mean"])
                                         - np.log(b[m]["ratio_vs_slot_mean"])) for m in ms])
    result["tijepa1_vs_spatial_log_ratio_vs_slot_mean"] = cmp
    primary = result["results"].get("E4_tijepa1", {}).get("1", {}).get("gate")
    result["G_Forecast"] = {
        "applied_to": "TiJepa-1, stride 1 (the setting of the v2 claim)",
        "gate": primary,
        "decision": None if primary is None else (
            "pass: forecasting is reported as a finding, with the controls" if primary["pass"]
            else "fail: forecasting is reported as explained by temporal-position "
                 "structure and the forecasting claim is dropped"),
    }
    result["seconds"] = round(time.time() - t_start, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"G-Forecast: {result['G_Forecast']['decision']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
