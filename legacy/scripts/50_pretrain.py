"""M5 driver: TiJepa-0 pretraining."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.train.pretrain_video import PretrainConfig, train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=12)
    a = ap.parse_args()
    cfg = PretrainConfig(
        shard_dir=str(REPO / "data" / "processed"),
        out_dir=str(REPO / "checkpoints"),
        seed=a.seed, max_hours=a.max_hours, max_steps=a.max_steps,
        batch_size=a.batch_size, num_workers=a.num_workers,
    )
    s = train(cfg)
    (REPO / "results" / "partA").mkdir(parents=True, exist_ok=True)
    (REPO / "results" / "partA" / f"pretrain_seed{a.seed}.json").write_text(
        json.dumps(s, indent=2) + "\n")
    print(json.dumps({k: s[k] for k in
                      ("steps", "gpu_hours", "effective_epochs", "stop_reason",
                       "initial_loss_mean_first500", "final_loss_mean_last500")}, indent=2))
