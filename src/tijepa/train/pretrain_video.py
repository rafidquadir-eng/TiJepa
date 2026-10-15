"""M5: TiJepa-0 video-JEPA pretraining (tier T1: 1 GPU, ViT-S, bf16).

Reads the big sequential .npy shards built by scripts/30_preprocess_pretrain.py with a
multi-worker DataLoader, trains the latent-space masked predictor, logs curves and
collapse diagnostics to CSV, and checkpoints on a wall-clock interval so the run
survives an instance restart.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..models.video_jepa import VideoJEPA, collapse_diagnostics, multiblock_tube_mask


@dataclass
class PretrainConfig:
    shard_dir: str
    out_dir: str
    clip_frames: int = 16
    clip_size: int = 96
    batch_size: int = 128
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_steps: int = 60_000
    max_hours: float = 24.0
    ema_start: float = 0.996
    ema_end: float = 1.0
    mask_blocks: int = 8
    seed: int = 0
    num_workers: int = 12
    log_every: int = 50
    ckpt_minutes: float = 30.0
    amp_dtype: str = "bfloat16"


class ShardClipDataset(Dataset):
    """Memory-maps every shard once; __getitem__ returns a single uint8 clip."""

    def __init__(self, shard_dir: Path, flip: bool = True, seed: int = 0):
        self.paths = sorted(Path(shard_dir).glob("pretrain_shard_*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"no shards in {shard_dir}")
        self.maps: list[np.ndarray | None] = [None] * len(self.paths)
        self.lengths = []
        for p in self.paths:
            a = np.load(p, mmap_mode="r")
            self.lengths.append(a.shape[0])
            del a
        self.cum = np.cumsum([0] + self.lengths)
        self.flip = flip
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return int(self.cum[-1])

    def _get_map(self, i: int) -> np.ndarray:
        if self.maps[i] is None:
            self.maps[i] = np.load(self.paths[i], mmap_mode="r")
        return self.maps[i]

    def __getitem__(self, idx: int) -> torch.Tensor:
        s = int(np.searchsorted(self.cum, idx, side="right") - 1)
        clip = np.asarray(self._get_map(s)[idx - self.cum[s]], dtype=np.float32)
        if self.flip:
            if self._rng is None:
                info = torch.utils.data.get_worker_info()
                wid = 0 if info is None else info.id
                self._rng = np.random.default_rng(self.seed * 1_000_003 + wid)
            if self._rng.random() < 0.5:
                clip = clip[:, ::-1, :]
            if self._rng.random() < 0.5:
                clip = clip[:, :, ::-1]
        clip = np.ascontiguousarray(clip) / 255.0
        # per-clip standardisation: removes illumination offset between wells/plates
        clip = (clip - clip.mean()) / (clip.std() + 1e-6)
        return torch.from_numpy(clip)[None]  # (1, T, H, W)


def cosine_lr(step: int, cfg: PretrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    p = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def train(cfg: PretrainConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = getattr(torch, cfg.amp_dtype)

    ds = ShardClipDataset(Path(cfg.shard_dir), seed=cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=cfg.num_workers > 0,
                    prefetch_factor=4 if cfg.num_workers > 0 else None)

    model = VideoJEPA(clip=(cfg.clip_frames, cfg.clip_size, cfg.clip_size),
                      ema=(cfg.ema_start, cfg.ema_end)).to(dev)
    params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))

    ckpt_path = out / f"tijepa0_T1_seed{cfg.seed}.pt"
    log_path = out / f"pretrain_log_seed{cfg.seed}.csv"
    step = 0
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["optim"])
        step = int(sd.get("step", 0))
        print(f"resumed from {ckpt_path} at step {step}", flush=True)

    if not log_path.exists():
        with log_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(
                ["step", "loss", "lr", "ema_m", "clips_seen", "seconds",
                 "std_mean", "effective_rank"])

    gen = torch.Generator().manual_seed(cfg.seed)
    t_start = time.time()
    t_ckpt = t_start
    losses: list[float] = []
    stop_reason = "max_steps"
    n_epochs = 0

    while step < cfg.max_steps:
        n_epochs += 1
        for clips in dl:
            if step >= cfg.max_steps:
                break
            elapsed_h = (time.time() - t_start) / 3600.0
            if elapsed_h >= cfg.max_hours:
                stop_reason = "max_hours"
                break
            lr = cosine_lr(step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            clips = clips.to(dev, non_blocking=True)
            mask = multiblock_tube_mask(model.grid, n_blocks=cfg.mask_blocks,
                                        generator=gen).to(dev)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
                res = model(clips, mask)
            loss = res["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            m = model.update_target(step / max(1, cfg.max_steps))
            losses.append(float(loss.detach()))
            step += 1

            if step % cfg.log_every == 0:
                with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype,
                                                     enabled=dev == "cuda"):
                    diag = collapse_diagnostics(model.embed(clips[:64]).float())
                row = [step, float(np.mean(losses[-cfg.log_every:])), lr, m,
                       step * cfg.batch_size, round(time.time() - t_start, 1),
                       diag["std_mean"], diag["effective_rank"]]
                with log_path.open("a", newline="") as fh:
                    csv.writer(fh).writerow(row)
                print(f"step {step} loss {row[1]:.4f} lr {lr:.2e} ema {m:.4f} "
                      f"rank {diag['effective_rank']:.1f} "
                      f"{elapsed_h:.2f}h", flush=True)

            if (time.time() - t_ckpt) / 60.0 >= cfg.ckpt_minutes:
                torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                            "step": step, "config": asdict(cfg)}, ckpt_path)
                t_ckpt = time.time()
                print(f"  checkpoint at step {step}", flush=True)
        if stop_reason == "max_hours":
            break

    torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                "step": step, "config": asdict(cfg)}, ckpt_path)
    gpu_hours = (time.time() - t_start) / 3600.0
    summary = {
        "steps": step,
        "epochs_started": n_epochs,
        "clips_seen": step * cfg.batch_size,
        "n_clips_in_corpus": len(ds),
        "n_shards": len(ds.paths),
        "effective_epochs": round(step * cfg.batch_size / max(1, len(ds)), 2),
        "final_loss_mean_last500": float(np.mean(losses[-500:])) if losses else None,
        "initial_loss_mean_first500": float(np.mean(losses[:500])) if losses else None,
        "gpu_hours": round(gpu_hours, 2),
        "stop_reason": stop_reason,
        "checkpoint": str(ckpt_path),
        "log_csv": str(log_path),
        "config": asdict(cfg),
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    (out / f"pretrain_summary_seed{cfg.seed}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return summary
