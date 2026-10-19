"""H3.3: short recipe sweep and the G-Recipe gate (PLAN_v3 §7, findings R4, R5, R22).

Five 20,000-step runs from scratch on the frozen corpus, seed 0, immutable snapshots every
5,000 steps (rule 3). Each schedule (cosine LR, EMA) ends at 20,000 steps.

| ID | change vs the TiJepa-1 recipe (values fixed before any run) |
|----|----|
| S0 | none |
| S1 | EMA momentum 0.996 -> 0.9995 (never reaches 1.0) |
| S2 | VICReg terms on predictor outputs: variance 0.04, covariance 0.001 |
| S3 | effective batch 256 (2 x 128 accumulated), LR scaled linearly to 3e-4 |
| S4 | future-mask batches drawn only from stride-9/27 clips; predictor depth 8 |

Snapshots are scored only on the label-free validation set (re-ID score and effective
rank of the pooled target embedding; src/tijepa/eval/reid.py) — never on a test set.
G-Recipe (§2): a variant passes iff, at 20,000 steps, its validation score exceeds S0's by
>= 0.02 AND its effective rank is >= 2x S0's.

Usage: `--train` (sequential, resumable; skips finished variants), `--score` (after or
during training; scores every snapshot present and applies the gate when all 20k
snapshots exist).
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import torch

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from tijepa.train.pretrain_video import PretrainConfig, train  # noqa: E402

SHARDS = REPO / "data" / "processed" / "v1"
SWEEP = REPO / "checkpoints" / "sweep"
OUT = REPO / "results" / "v3" / "partA" / "recipe_sweep.json"
STEPS, SNAP = 20_000, 5_000
GATE_SCORE, GATE_RANK = 0.02, 2.0
VARIANTS = {
    "S0": ("baseline (TiJepa-1 recipe, 20k-step schedule)", {}),
    "S1": ("EMA 0.996 -> 0.9995", {"ema_end": 0.9995}),
    "S2": (
        "VICReg on predictor outputs (var 0.04, cov 0.001)",
        {"vicreg_var": 0.04, "vicreg_cov": 0.001},
    ),
    "S3": ("batch 256 (2 x 128 accumulated), LR 3e-4", {"grad_accum": 2, "lr": 3e-4}),
    "S4": (
        "future targets from stride-9/27 clips; predictor depth 8",
        {"future_strides": [9, 27], "pred_depth": 8},
    ),
}


def cfg_for(vid: str, workers: int) -> PretrainConfig:
    return PretrainConfig(
        shard_dir=str(SHARDS),
        out_dir=str(SWEEP),
        run_name=f"sweep_{vid}",
        seed=0,
        max_steps=STEPS,
        max_hours=8.0,
        num_workers=workers,
        snapshot_every=SNAP,
        **VARIANTS[vid][1],
    )


def do_train(workers: int, only: list[str]) -> None:
    SWEEP.mkdir(parents=True, exist_ok=True)
    for vid in only:
        final = SWEEP / f"sweep_{vid}" / f"step_{STEPS:06d}.pt"
        if final.exists():
            print(f"{vid}: finished, skipped")
            continue
        print(f"{vid}: training ({VARIANTS[vid][0]})")
        s = train(cfg_for(vid, workers))
        print(f"{vid}: step {s['steps']}, annealed {s['annealed']}, {s['gpu_hours']} GPU-h")
        if not s["annealed"]:
            print(f"{vid}: stopped early ({s['stop_reason']}); stopping the sweep")
            return


def do_score() -> dict:
    import importlib

    from tijepa.models.video_jepa import VideoJEPA

    rc = importlib.import_module("72_reid_collapse_v3")
    movies = rc.load_val()
    res: dict = {
        "phase": "PLAN_v3 H3.3",
        "variants": {},
        "steps": STEPS,
        "snapshot_every": SNAP,
        "selection_data": "label-free validation set only (results/v3/validation_set.json)",
        "gate_rule": f"validation score >= S0 + {GATE_SCORE} AND effective rank "
        f">= {GATE_RANK} x S0, both at {STEPS} steps",
    }
    for vid, (desc, over) in VARIANTS.items():
        d = SWEEP / f"sweep_{vid}"
        snaps = sorted(d.glob("step_*.pt")) if d.exists() else []
        summ = SWEEP / f"pretrain_summary_sweep_{vid}_seed0.json"
        entry = {
            "description": desc,
            "overrides": over,
            "snapshots": {},
            "summary": json.loads(summ.read_text()) if summ.exists() else None,
        }
        for sp in snaps:
            sd = torch.load(sp, map_location="cpu", weights_only=False)
            c = sd["config"]
            m = (
                VideoJEPA(
                    dim=c["model_dim"],
                    depth=c["model_depth"],
                    heads=c["model_heads"],
                    pred_dim=c["pred_dim"],
                    pred_depth=c["pred_depth"],
                    clip=(16, 96, 96),
                )
                .eval()
                .to(rc.DEV)
            )
            m.load_state_dict(sd["model"])
            def f(x, m=m):
                return rc.jepa_embed(m, x)
            r = rc.reid_for(f, f, movies)
            entry["snapshots"][str(sd["step"])] = {
                "validation_score": r["validation_score_mean"],
                "ci95_over_movies": r["ci95_over_movies"],
                "per_gap": r["per_gap_mean"],
                "effective_rank": r["effective_rank_query"],
                "sha256": __import__("tijepa.eval.provenance", fromlist=["x"]).sha256_file(sp),
            }
            print(
                f"  {vid} step {sd['step']}: score {r['validation_score_mean']:.3f} "
                f"rank {r['effective_rank_query']:.1f}"
            )
            del m
            torch.cuda.empty_cache()
        res["variants"][vid] = entry
    fin = {v: e["snapshots"].get(str(STEPS)) for v, e in res["variants"].items()}
    if all(fin.values()):
        base = fin["S0"]
        gate = {}
        for v in ("S1", "S2", "S3", "S4"):
            ds = fin[v]["validation_score"] - base["validation_score"]
            rr = fin[v]["effective_rank"] / base["effective_rank"]
            gate[v] = {
                "score_gain": ds,
                "rank_ratio": rr,
                "pass": bool(ds >= GATE_SCORE and rr >= GATE_RANK),
            }
        winners = [v for v, g in gate.items() if g["pass"]]
        res["G_Recipe"] = {
            "per_variant": gate,
            "passing": winners,
            "decision": (
                "pass: report " + ", ".join(winners) + " and its "
                "20k-step model as an extra row (no TiJepa-2 "
                "retrain, PLAN_v3 0A.2)"
            )
            if winners
            else "fail: stop recipe work; the collapse diagnosis is reported as a limitation",
        }
        print(f"G-Recipe: {res['G_Recipe']['decision']}")
    else:
        res["G_Recipe"] = {
            "decision": None,
            "reason": f"not every variant has a {STEPS}-step snapshot yet",
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--only", default=",".join(VARIANTS))
    a = ap.parse_args()
    if a.train:
        do_train(a.workers, a.only.split(","))
    if a.score:
        do_score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
