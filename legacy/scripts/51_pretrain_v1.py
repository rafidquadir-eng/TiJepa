"""F2 driver: TiJepa-1 pretraining, and the spatial-only ablation.

Two runs, in this order on one GPU (PLAN_v1.md Section 5.4, tier T1):

  1. `tijepa1` - mixed masking (future mask with p = 0.5), budget 12 GPU-hours;
  2. `tijepa1_spatial` - the ablation: identical corpus, spatial masking only, run for
     **the same number of steps** as run 1 rather than the same wall-clock, so the
     comparison isolates the objective and not the throughput.

Pretraining has a single seed on one GPU; Part A's seed variance comes from random-init
draws, cell subsamples and fold assignment (F3.5), and that must be stated in Methods.

The shard directory is read once at startup and the list is recorded in the run summary,
so a corpus build still running in the background cannot silently change the training
set underneath a run.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.train.pretrain_video import PretrainConfig, train  # noqa: E402

OUT = REPO / "results" / "v1" / "partA"


def run(name: str, seed: int, hours: float, steps: int, batch: int, workers: int,
        p_future: float) -> dict:
    cfg = PretrainConfig(
        shard_dir=str(REPO / "data" / "processed" / "v1"),
        out_dir=str(REPO / "checkpoints"),
        run_name=name, seed=seed, max_hours=hours, max_steps=steps,
        batch_size=batch, num_workers=workers, p_future=p_future,
    )
    print(f"\n=== {name}: p_future={p_future}, budget {hours} GPU-h, "
          f"max_steps {steps}")
    s = train(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pretrain_{name}_seed{seed}.json").write_text(
        json.dumps(s, indent=2) + "\n")
    print(json.dumps({k: s.get(k) for k in
                      ("run_name", "steps", "gpu_hours", "effective_epochs",
                       "stop_reason", "initial_loss_mean_first500",
                       "final_loss_mean_last500", "final_loss_spatial_last500",
                       "final_loss_future_last500")}, indent=2))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=12.0,
                    help="GPU-hour budget for the main run (Section 5.4, tier T1)")
    ap.add_argument("--max-steps", type=int, default=150_000,
                    help="bound for the cosine LR schedule; the budget usually stops "
                         "the run first")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=14)
    ap.add_argument("--which", default="both",
                    choices=["both", "main", "ablation"])
    ap.add_argument("--ablation-steps", type=int, default=0,
                    help="steps for the ablation; 0 = match the main run exactly")
    a = ap.parse_args()

    main_summary = None
    if a.which in ("both", "main"):
        main_summary = run("tijepa1", a.seed, a.hours, a.max_steps,
                           a.batch_size, a.num_workers, p_future=0.5)

    if a.which in ("both", "ablation"):
        steps = a.ablation_steps
        if steps <= 0:
            if main_summary is not None:
                steps = int(main_summary["steps"])
            else:
                prev = OUT / f"pretrain_tijepa1_seed{a.seed}.json"
                if not prev.exists():
                    raise SystemExit(
                        "the ablation must match the main run's step count, but "
                        f"{prev} is absent; pass --ablation-steps explicitly")
                steps = int(json.loads(prev.read_text())["steps"])
        print(f"\nablation will run for {steps} steps, matching the main run")
        # A generous hour cap so the step count, not the clock, is what stops it.
        run("tijepa1_spatial", a.seed, a.hours * 2.5, steps,
            a.batch_size, a.num_workers, p_future=0.0)
