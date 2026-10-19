"""H3.1 validation score (re-identification) and H3.2 collapse diagnosis (PLAN_v3 §7).

Validation clips: results/v3/validation_set.json (64 non-corpus ComplexEye movies). This
set is never used as a test set; its score and effective rank are the only quantities
used to compare training recipes (rule 1).

Re-ID: src/tijepa/eval/reid.py. Encoders: TiJepa-1, the spatial ablation, TiJepa-0,
three random-init draws, TiJepa-1 with frames shuffled (E5), ImageNet ViT-S, DINOv2 and
V-JEPA 2 (video clips; image models see the query's last and the gallery's first frame),
and a raw-pixel reference (48 x 48 downsampled, z-scored). Per-movie scores; the summary
is the mean over movies with a 95% bootstrap CI over movies (64 groups).

Collapse diagnosis (no training): per-dimension std, effective rank and normalised
singular-value spectra of the target encoder (pooled and per token) and the context
encoder (pooled) of each video JEPA, on the validation clips and on the 1,051 BBBC054
monitor crops; the training-log trajectories of rank against EMA momentum and LR; and
across-model Spearman correlations of rank with the BBBC054 and validation scores
(descriptive: few models).
"""

from __future__ import annotations

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

from tijepa.eval import encoders as ENC  # noqa: E402
from tijepa.eval import probe as P  # noqa: E402
from tijepa.eval import provenance as PV  # noqa: E402
from tijepa.eval import reid as R  # noqa: E402
from tijepa.models.video_jepa import VideoJEPA  # noqa: E402

VAL = REPO / "data" / "interim" / "val_reid"
OUT = REPO / "results" / "v3" / "partA"
DEV = ENC.DEV
SEED = 20261019
N_DIAG = 2000


def load_val() -> dict:
    meta = json.loads((REPO / "results" / "v3" / "validation_set.json").read_text())
    movies = {}
    for m in meta["movies"]:
        if m["status"] != "ok":
            continue
        z = np.load(REPO / m["file"])
        movies[Path(m["file"]).stem] = {k: z[k] for k in z.files}
    return movies


def zclip(c: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(c.astype(np.float32) / 255.0)[:, None]
    m = t.mean(dim=(1, 2, 3, 4), keepdim=True)
    s = t.std(dim=(1, 2, 3, 4), keepdim=True)
    return (t - m) / (s + 1e-6)


@torch.no_grad()
def jepa_embed(
    model: VideoJEPA, clips: np.ndarray, what: str = "target", batch: int = 128
) -> np.ndarray:
    out = []
    for i in range(0, len(clips), batch):
        c = zclip(clips[i : i + batch]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            if what == "target":
                z = model.target(c).float().mean(1)
            elif what == "context":
                z = model.encoder(c).float().mean(1)
            else:  # per-token target, a random 16 tokens per clip
                tok = model.target(c).float()
                g = torch.Generator(device="cpu").manual_seed(SEED + i)
                idx = torch.randperm(tok.shape[1], generator=g)[:16].to(DEV)
                z = tok[:, idx].reshape(-1, tok.shape[-1])
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def shuffle_frames(clips: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([c[rng.permutation(c.shape[0])] for c in clips])


def raw_pixels(frames: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(frames.astype(np.float32))[:, None]
    x = torch.nn.functional.avg_pool2d(t, 2)[:, 0].reshape(len(frames), -1).numpy()
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def spectrum(z: np.ndarray, k: int = 32) -> dict:
    z = np.asarray(z, np.float64)
    zc = z - z.mean(0)
    s = np.linalg.svd(zc, compute_uv=False) ** 2
    s = s / max(s.sum(), 1e-12)
    return {
        "effective_rank": P.effective_rank(z),
        "std_mean": float(zc.std(0).mean()),
        "n": int(len(z)),
        "dim": int(z.shape[1]),
        "top_variance_fractions": [float(v) for v in s[:k]],
        "variance_in_top1": float(s[0]),
        "variance_in_top8": float(s[:8].sum()),
    }


def reid_for(embed_q, embed_g, movies: dict) -> dict:
    per_movie = {}
    pooled_q = []
    for name, d in movies.items():
        zq = embed_q(d["query"])
        zg = {k: embed_g(d[f"gallery_k{k}"]) for k in R.GAPS}
        per_movie[name] = R.score(zq, zg, d["anchor"])
        pooled_q.append(zq)
    vals = np.array(
        [
            v["validation_score"]
            for v in per_movie.values()
            if v["validation_score"] is not None
        ]
    )
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(vals, len(vals)).mean() for _ in range(2000)]
    zq = np.concatenate(pooled_q)
    sel = np.random.default_rng(SEED).permutation(len(zq))[:N_DIAG]
    return {
        "validation_score_mean": float(vals.mean()),
        "ci95_over_movies": [
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        ],
        "per_gap_mean": {
            str(k): float(
                np.mean(
                    [
                        v["per_gap"][str(k)]
                        for v in per_movie.values()
                        if v["per_gap"][str(k)] is not None
                    ]
                )
            )
            for k in R.GAPS
        },
        "chance_mean": float(
            np.mean([v["chance"] for v in per_movie.values() if v["chance"] is not None])
        ),
        "effective_rank_query": P.effective_rank(zq[sel]),
        "n_movies": int(len(vals)),
        "per_movie": per_movie,
    }


def log_trajectory(csv: Path) -> dict:
    import pandas as pd

    if not csv.exists():
        return {"status": "absent"}
    df = pd.read_csv(csv)
    df = df.drop_duplicates("step", keep="last").sort_values("step")
    pick = df.iloc[np.linspace(0, len(df) - 1, 25).astype(int)]
    out = {
        "note": "training-batch diagnostic (64 clips, target encoder pooled), so rank "
        "is capped at 64 and noisy",
        "steps": pick["step"].astype(int).tolist(),
        "effective_rank": pick["effective_rank"].round(3).tolist(),
        "std_mean": pick["std_mean"].round(5).tolist(),
        "lr": pick["lr"].tolist(),
        "ema_m": pick["ema_m"].round(6).tolist(),
    }
    sm = df["effective_rank"].rolling(41, center=True, min_periods=5).mean()
    out["peak_rank_step"] = int(df["step"].iloc[int(np.nanargmax(sm.to_numpy()))])
    out["spearman_rank_vs_ema_m"] = float(
        df["effective_rank"].corr(df["ema_m"], method="spearman")
    )
    return out


def main() -> int:
    t0 = time.time()
    movies = load_val()
    print(
        f"{len(movies)} validation movies, {sum(len(d['query']) for d in movies.values())} "
        "cells"
    )
    mon = np.load(REPO / "data" / "interim" / "monitor_bbbc054_subset.npz", allow_pickle=True)[
        "X"
    ].astype(np.float32)
    res: dict = {
        "phase": "PLAN_v3 H3.1 + H3.2 (findings R4, R10, R22)",
        "validation_set": "results/v3/validation_set.json",
        "reid": {},
        "diagnosis": {},
        "encoders": {},
    }

    # raw-pixel reference
    res["reid"]["raw_pixels"] = reid_for(
        lambda c: raw_pixels(c[:, -1]), lambda c: raw_pixels(c[:, 0]), movies
    )
    print(f"  raw pixels: {res['reid']['raw_pixels']['validation_score_mean']:.3f}")

    # image models and V-JEPA 2
    for key in ("E2_imagenet_vit_s", "E3_dinov2_vit_s"):
        def fn(X, k=key):
            return ENC.embed_timm(ENC.TIMM[k], X)
        res["reid"][key] = reid_for(
            lambda c, f=fn: f(ENC.minmax(c[:, -1])),
            lambda c, f=fn: f(ENC.minmax(c[:, 0])),
            movies,
        )
        print(f"  {key}: {res['reid'][key]['validation_score_mean']:.3f}")
    from tijepa.eval import vjepa2 as VJ2

    def vf(c):
        return VJ2.embed_clips(c.astype(np.float32) / 255.0, DEV)
    res["reid"][VJ2.KEY] = reid_for(vf, vf, movies)
    res["encoders"][VJ2.KEY] = VJ2.record()
    print(f"  {VJ2.KEY}: {res['reid'][VJ2.KEY]['validation_score_mean']:.3f}")

    # video JEPAs
    jepas = []
    for run, fn in ENC.JEPA_RUNS:
        m, rec = ENC.load_jepa(run, fn)
        jepas.append((PV.key_for(rec), rec, m))
    for s in ENC.RANDOM_SEEDS:
        torch.manual_seed(s)
        jepas.append(
            (
                f"E6_random_init_draw{s - ENC.RANDOM_SEEDS[0]}",
                PV.random_init_record(s),
                VideoJEPA(clip=(ENC.NF, ENC.CROP, ENC.CROP)).eval().to(DEV),
            )
        )
    pool = np.concatenate([d["query"] for d in movies.values()])
    diag_clips = pool[np.random.default_rng(SEED).permutation(len(pool))[:N_DIAG]]
    for key, rec, m in jepas:
        PV.label_for_key(key, rec)
        res["encoders"][key] = {"checkpoint": rec}
        def f(c, m=m):
            return jepa_embed(m, c)
        res["reid"][key] = reid_for(f, f, movies)
        print(f"  {key}: {res['reid'][key]['validation_score_mean']:.3f}")
        if key == "E4_tijepa1":
            def fs(c, m=m):
                return jepa_embed(m, shuffle_frames(c, SEED))
            res["reid"]["E5_frame_shuffled"] = reid_for(fs, fs, movies)
            print(f"  E5: {res['reid']['E5_frame_shuffled']['validation_score_mean']:.3f}")
        mon_clips = (np.clip(mon, 0, 1) * 255).astype(np.uint8)[:, None].repeat(16, 1)
        res["diagnosis"][key] = {
            "validation_clips": {
                w: spectrum(jepa_embed(m, diag_clips, w))
                for w in ("target", "context", "token")
            },
            "bbbc054_monitor_crops": {
                w: spectrum(jepa_embed(m, mon_clips, w)) for w in ("target", "context")
            },
        }
        torch.cuda.empty_cache()

    logs = {
        "E0_tijepa0": "pretrain_log_seed0.csv",
        "E4_tijepa1": "pretrain_log_tijepa1_seed0.csv",
        "E4s_tijepa1_spatial": "pretrain_log_tijepa1_spatial_seed0.csv",
    }
    res["training_trajectories"] = {
        k: log_trajectory(REPO / "checkpoints" / v) for k, v in logs.items()
    }

    # across-model correlations (descriptive)
    from scipy.stats import spearmanr

    bb = json.loads((OUT / "bbbc054.json").read_text())["encoders"]
    rows = []
    for key in res["diagnosis"]:
        bk = "E6_random_init" if key.startswith("E6") else key
        if bk in bb and "three_class_mean" in bb[bk]:
            rows.append(
                (
                    key,
                    res["diagnosis"][key]["validation_clips"]["target"]["effective_rank"],
                    bb[bk]["three_class_mean"],
                    res["reid"][key]["validation_score_mean"],
                )
            )
    if len(rows) >= 4:
        rk, ba, vs = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
        res["rank_correlations"] = {
            "models": [r[0] for r in rows],
            "effective_rank": rk.tolist(),
            "bbbc054_three_class": ba.tolist(),
            "validation_score": vs.tolist(),
            "spearman_rank_vs_bbbc054": float(spearmanr(rk, ba).statistic),
            "spearman_rank_vs_validation": float(spearmanr(rk, vs).statistic),
            "note": f"{len(rows)} models (random-init draws share one BBBC054 value); "
            "descriptive only",
        }
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation_reid.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT / 'validation_reid.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
