"""M5 / F2 / G1: TiJepa video-JEPA pretraining (tier T1: 1 GPU, ViT-S, bf16).

v1 changes (PLAN_v1.md F2.1-F2.2):
  * the masking objective is **mixed**: per batch, either the v0 spatial tube mask or a
    **future mask** whose targets are the last temporal slots, making the predictor a
    latent forecaster (review finding C4). The two objectives are not on the same
    scale, so their losses are logged in separate columns rather than averaged into one
    uninterpretable curve;
  * the shard list is frozen at startup and recorded, so a corpus still being built in
    the background cannot change the training set mid-run.

v2 changes (PLAN_v2.md G1.1, review findings N3 and N13):
  * the budget is a **step count**: `max_steps` is the planned final step, and the cosine
    LR reaches `min_lr` and EMA momentum reaches `ema_end` exactly on the last update.
    `max_hours` is a safety cap only; hitting it ends the run *unannealed* and the summary
    says so;
  * SIGTERM/SIGINT finish the current step, save, write the summary, release the lock and
    return normally;
  * checkpoints are written atomically (tmp + fsync + `os.replace`) and the run holds an
    exclusive lock file `<run>.lock` (hostname, PID, start time) so two instances that
    share storage can never write the same checkpoint;
  * GPU and wall seconds are **cumulative** across resumes, with one `segments` entry per
    process (host, GPU, torch version, start/end step, seconds);
  * data order, flips and masks are a deterministic function of (seed, step), so a resume
    continues the stream instead of replaying it from the start.
"""
from __future__ import annotations

import csv
import json
import math
import os
import signal
import socket
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..models.video_jepa import VideoJEPA, collapse_diagnostics, sample_mask

LOG_HEADER = ["step", "loss", "loss_spatial", "loss_future", "n_spatial", "n_future", "lr",
              "ema_m", "clips_seen", "seconds", "std_mean", "effective_rank", "p_future"]


@dataclass
class PretrainConfig:
    shard_dir: str
    out_dir: str
    clip_frames: int = 16
    clip_size: int = 96
    batch_size: int = 128
    lr: float = 1.5e-4
    min_lr: float = 0.0
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_steps: int = 60_000
    max_hours: float = 24.0
    ema_start: float = 0.996
    ema_end: float = 1.0
    mask_blocks: int = 8
    p_future: float = 0.5
    n_future_slots: int = 2
    context_spatial_mask: float = 0.0
    run_name: str = "tijepa1"
    seed: int = 0
    num_workers: int = 12
    log_every: int = 50
    ckpt_minutes: float = 30.0
    amp_dtype: str = "bfloat16"
    # model size; the defaults are TiJepa's ViT-S. Tests use a tiny model.
    model_dim: int = 384
    model_depth: int = 12
    model_heads: int = 6
    pred_dim: int = 192
    pred_depth: int = 6
    # Segments that ran before this checkpoint recorded segments (G1.1.3). Used only if
    # the resumed checkpoint has no `segments` entry.
    prior_segments_if_missing: list = field(default_factory=list)
    # Benchmark mode: never write the checkpoint, log or summary (G1.2).
    dry_run: bool = False
    # PLAN_v3 rule 3 / H3.3 recipe variants (all off by default = the TiJepa-1 recipe)
    snapshot_every: int = 0          # immutable out_dir/<run>/step_XXXXXX.pt snapshots
    vicreg_var: float = 0.0          # variance hinge on predictor outputs (S2)
    vicreg_cov: float = 0.0          # off-diagonal covariance penalty (S2)
    grad_accum: int = 1              # micro-batches per optimiser step (S3)
    future_strides: list = field(default_factory=list)  # future-mask batches only from
    #                                  clips with these temporal strides (S4)


# --------------------------------------------------------------------------- schedule


def cosine_lr(step: int, cfg: PretrainConfig) -> float:
    """LR for the update with 0-based index `step`; equals `min_lr` on the last update."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    span = max(1, cfg.max_steps - 1 - cfg.warmup_steps)
    p = min(max((step - cfg.warmup_steps) / span, 0.0), 1.0)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * p))


def _cosine_lr_v1(step: int, cfg: PretrainConfig) -> float:
    """The v1 trainer's schedule (floor 0, denominator max_steps - warmup); for reporting
    the discontinuity at a resume from a v1 checkpoint only."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    p = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def ema_progress(step: int, cfg: PretrainConfig) -> float:
    """EMA progress for the update with 0-based index `step`; 1.0 on the last update."""
    return min((step + 1) / max(1, cfg.max_steps), 1.0)


def ema_momentum(step: int, cfg: PretrainConfig) -> float:
    return cfg.ema_start + (cfg.ema_end - cfg.ema_start) * ema_progress(step, cfg)


# ------------------------------------------------------------------- safe file writes


def atomic_write_bytes(path: Path, write_fn) -> None:
    """Write via `write_fn(fh)` to a tmp file in the same directory, fsync, then replace."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            write_fn(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def atomic_torch_save(obj, path: Path) -> None:
    atomic_write_bytes(path, lambda fh: torch.save(obj, fh))


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, lambda fh: fh.write(text.encode()))


class LockError(RuntimeError):
    pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    """Exclusive per-run lock file on shared storage (PLAN_v2 rule 3).

    A lock held by a live PID on this host, or any lock from another host (whose PID
    cannot be checked from here), refuses the start. Only a lock left by a dead PID on
    this same host is treated as stale and replaced.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.held = False

    def acquire(self) -> None:
        info = {"hostname": socket.gethostname(), "pid": os.getpid(),
                "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    other = json.loads(self.path.read_text())
                except (OSError, ValueError) as exc:
                    raise LockError(f"unreadable lock {self.path}; remove it by hand "
                                    "after checking no trainer runs") from exc
                same_host = other.get("hostname") == info["hostname"]
                if same_host and not _pid_alive(int(other.get("pid", -1))):
                    print(f"removing stale lock {self.path} (pid {other.get('pid')} dead)",
                          flush=True)
                    self.path.unlink()
                    continue
                raise LockError(f"{self.path} is held by {other}; refusing to start") from None
            with os.fdopen(fd, "w") as fh:
                json.dump(info, fh)
                fh.flush()
                os.fsync(fh.fileno())
            self.held = True
            return
        raise LockError(f"could not acquire {self.path}")

    def release(self) -> None:
        if self.held:
            try:
                other = json.loads(self.path.read_text())
                if other.get("pid") == os.getpid():
                    self.path.unlink()
            except (OSError, ValueError):
                pass
            self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class StopFlag:
    """SIGTERM/SIGINT set a flag; the loop finishes the step it is on and stops."""

    def __init__(self):
        self.signum: int | None = None
        self._old: dict = {}

    def install(self) -> None:
        for s in (signal.SIGTERM, signal.SIGINT):
            self._old[s] = signal.signal(s, self._handle)

    def _handle(self, signum, _frame) -> None:
        self.signum = signum
        print(f"signal {signal.Signals(signum).name} received; stopping after this step",
              flush=True)

    def restore(self) -> None:
        for s, h in self._old.items():
            signal.signal(s, h)

    @property
    def set(self) -> bool:
        return self.signum is not None


# ------------------------------------------------------------------------------- data


class ShardClipDataset(Dataset):
    """Memory-maps every shard once; __getitem__ returns a single uint8 clip.

    Flips are a deterministic function of (seed, index in the sample stream), so they do
    not depend on which worker serves a clip or on where a run was resumed.
    """

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

    def __len__(self) -> int:
        return int(self.cum[-1])

    def _get_map(self, i: int) -> np.ndarray:
        if self.maps[i] is None:
            self.maps[i] = np.load(self.paths[i], mmap_mode="r")
        return self.maps[i]

    def __getitem__(self, key) -> torch.Tensor:
        idx, pos = key if isinstance(key, tuple) else (key, key)
        s = int(np.searchsorted(self.cum, idx, side="right") - 1)
        clip = np.asarray(self._get_map(s)[idx - self.cum[s]], dtype=np.float32)
        if self.flip:
            r = np.random.default_rng((self.seed, int(pos))).random(2)
            if r[0] < 0.5:
                clip = clip[:, ::-1, :]
            if r[1] < 0.5:
                clip = clip[:, :, ::-1]
        clip = np.ascontiguousarray(clip) / 255.0
        # per-clip standardisation: removes illumination offset between wells/plates
        clip = (clip - clip.mean()) / (clip.std() + 1e-6)
        return torch.from_numpy(clip)[None]  # (1, T, H, W)


def vicreg_terms(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance hinge and covariance penalty on predictor outputs (B, N, D)."""
    z = pred.float().reshape(-1, pred.shape[-1])
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var = torch.relu(1.0 - std).mean()
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return var, (off ** 2).sum() / z.shape[1]


class SubsetStreamSampler(Sampler):
    """StreamSampler over a fixed subset of clip indices (same determinism contract)."""

    def __init__(self, subset: np.ndarray, seed: int, start: int = 0):
        self.subset = np.asarray(subset)
        self.inner = StreamSampler(len(self.subset), seed + 104_729, start)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for i, pos in self.inner:
            yield int(self.subset[i]), pos

    def __len__(self) -> int:  # pragma: no cover
        return 2**62


def stride_subset(shard_dir: Path, strides: list) -> np.ndarray:
    import pandas as pd
    idx = pd.read_csv(Path(shard_dir) / "pretrain_index.csv", usecols=["stride", "static"])
    return np.flatnonzero(idx["stride"].isin(list(strides)).to_numpy()
                          & ~idx["static"].astype(bool).to_numpy())


class StreamSampler(Sampler):
    """Endless stream of (clip index, stream position) pairs from position `start`.

    The stream is the concatenation of per-epoch permutations, epoch e seeded by
    (seed, e). Position p is therefore the same clip whatever process reaches it, which
    makes a resume continue the stream exactly.
    """

    def __init__(self, n: int, seed: int, start: int = 0):
        self.n, self.seed, self.start = n, seed, start

    def _perm(self, epoch: int) -> np.ndarray:
        return np.random.default_rng((self.seed, 7919, epoch)).permutation(self.n)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        epoch, off = divmod(self.start, self.n)
        pos = self.start
        while True:
            perm = self._perm(epoch)
            for i in perm[off:]:
                yield int(i), pos
                pos += 1
            epoch, off = epoch + 1, 0

    def __len__(self) -> int:  # pragma: no cover - endless; DataLoader does not need it
        return 2**62


def _worker_init(_wid: int) -> None:
    # A Ctrl-C to the process group must not kill workers before the main process saves.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ------------------------------------------------------------------------- accounting


def _gpu_name(dev: str) -> str:
    return torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"


def segment_totals(segments: list[dict]) -> dict:
    return {
        "gpu_seconds_total": float(sum(s.get("gpu_seconds", s["seconds"]) for s in segments)),
        "wall_seconds_total": float(sum(s["seconds"] for s in segments)),
        "steps_total": int(segments[-1]["end_step"]) if segments else 0,
    }


def _clean_log(log_path: Path, resume_step: int, p_future: float) -> int:
    """Drop rows logged after the checkpoint (they will be re-run) and upgrade the header.

    Returns the number of rows dropped. Old logs without a `p_future` column get the
    run's configured value, which did not change within a run.
    """
    with log_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    keep = [r for r in rows if int(r["step"]) <= resume_step]
    for r in keep:
        if not r.get("p_future"):
            r["p_future"] = p_future
    buf = [",".join(LOG_HEADER)]
    for r in keep:
        buf.append(",".join(str(r.get(k, "")) for k in LOG_HEADER))
    atomic_write_text(log_path, "\n".join(buf) + "\n")
    return len(rows) - len(keep)


# ------------------------------------------------------------------------------ train


def train(cfg: PretrainConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / f"{cfg.run_name}_T1_seed{cfg.seed}.pt"
    log_path = out / f"pretrain_log_{cfg.run_name}_seed{cfg.seed}.csv"
    summary_path = out / f"pretrain_summary_{cfg.run_name}_seed{cfg.seed}.json"
    lock = RunLock(out / f"{cfg.run_name}.lock")
    if not cfg.dry_run:
        lock.acquire()
    stop = StopFlag()
    stop.install()
    try:
        return _train(cfg, ckpt_path, log_path, summary_path, stop)
    finally:
        stop.restore()
        lock.release()


def _train(cfg: PretrainConfig, ckpt_path: Path, log_path: Path, summary_path: Path,
           stop: StopFlag) -> dict:
    t_proc = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = getattr(torch, cfg.amp_dtype)

    model = VideoJEPA(dim=cfg.model_dim, depth=cfg.model_depth, heads=cfg.model_heads,
                      pred_dim=cfg.pred_dim, pred_depth=cfg.pred_depth,
                      clip=(cfg.clip_frames, cfg.clip_size, cfg.clip_size),
                      ema=(cfg.ema_start, cfg.ema_end)).to(dev)
    params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))

    step = 0
    prior_segments: list[dict] = []
    resume_info: dict | None = None
    gen = torch.Generator().manual_seed(cfg.seed)
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["optim"])
        step = int(sd.get("step", 0))
        prior_segments = list(sd.get("segments") or cfg.prior_segments_if_missing)
        if prior_segments and int(prior_segments[-1]["end_step"]) != step:
            raise RuntimeError(f"segment history ends at {prior_segments[-1]['end_step']} "
                               f"but the checkpoint is at step {step}")
        if "mask_rng" in sd:
            gen.set_state(sd["mask_rng"].cpu())
        else:
            gen.manual_seed(cfg.seed * 1_000_003 + step)
        old = sd.get("config", {})
        resume_info = {"step": step, "old_max_steps": old.get("max_steps"),
                       "new_max_steps": cfg.max_steps}
        if old and old.get("max_steps") != cfg.max_steps:
            oc = PretrainConfig(**{k: v for k, v in old.items()
                                   if k in PretrainConfig.__dataclass_fields__})
            # the v1 trainer computed EMA progress as step / max_steps before the increment
            old_m = (oc.ema_start + (oc.ema_end - oc.ema_start) * (step - 1) / oc.max_steps
                     if "segments" not in sd else ema_momentum(step - 1, oc))
            old_lr = (cosine_lr(step, oc) if "segments" in sd else _cosine_lr_v1(step, oc))
            resume_info.update({
                "lr_old_schedule": old_lr, "lr_new_schedule": cosine_lr(step, cfg),
                "ema_old_schedule_last_update": old_m,
                "ema_new_schedule_next_update": ema_momentum(step, cfg),
            })
            lo = resume_info["lr_old_schedule"]
            resume_info["lr_relative_change"] = (
                resume_info["lr_new_schedule"] / lo - 1 if lo > 0 else None)
        print(f"resumed from {ckpt_path} at step {step}: {resume_info}", flush=True)
        del sd
        if log_path.exists() and not cfg.dry_run:
            dropped = _clean_log(log_path, step, cfg.p_future)
            resume_info["log_rows_dropped_after_checkpoint"] = dropped
    start_step = step

    ds = ShardClipDataset(Path(cfg.shard_dir), seed=cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size,
                    sampler=StreamSampler(len(ds), cfg.seed,
                                         start=step * cfg.batch_size * cfg.grad_accum),
                    num_workers=cfg.num_workers, pin_memory=dev == "cuda",
                    worker_init_fn=_worker_init if cfg.num_workers > 0 else None,
                    persistent_workers=False,
                    prefetch_factor=4 if cfg.num_workers > 0 else None)

    it_future = None
    if cfg.future_strides:
        sub = stride_subset(Path(cfg.shard_dir), cfg.future_strides)
        dl_f = DataLoader(ds, batch_size=cfg.batch_size,
                          sampler=SubsetStreamSampler(sub, cfg.seed,
                                                      start=step * cfg.batch_size),
                          num_workers=cfg.num_workers // 2,
                          pin_memory=dev == "cuda",
                          worker_init_fn=_worker_init if cfg.num_workers > 1 else None,
                          prefetch_factor=4 if cfg.num_workers > 1 else None)
        it_future = iter(dl_f)
    snap_dir = Path(cfg.out_dir) / cfg.run_name

    def snapshot() -> None:
        if cfg.dry_run or not cfg.snapshot_every:
            return
        snap_dir.mkdir(parents=True, exist_ok=True)
        p = snap_dir / f"step_{step:06d}.pt"
        if not p.exists():  # immutable: never overwritten
            atomic_torch_save({"model": model.state_dict(), "step": step,
                               "config": asdict(cfg), "snapshot": True}, p)

    if not log_path.exists() and not cfg.dry_run:
        atomic_write_text(log_path, ",".join(LOG_HEADER) + "\n")

    seg = {"host": socket.gethostname(), "gpu_name": _gpu_name(dev),
           "torch_version": torch.__version__, "start_step": start_step,
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t_loop = time.time()

    def segments_now() -> list[dict]:
        cur = dict(seg, end_step=step, seconds=round(time.time() - t_proc, 1),
                   gpu_seconds=round(time.time() - t_loop, 1),
                   end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return prior_segments + [cur]

    def save() -> None:
        if cfg.dry_run:
            return
        segs = segments_now()
        atomic_torch_save({"model": model.state_dict(), "optim": opt.state_dict(),
                           "step": step, "config": asdict(cfg), "segments": segs,
                           **segment_totals(segs), "mask_rng": gen.get_state()}, ckpt_path)

    t_ckpt = time.time()
    losses: list[float] = []
    losses_by_kind: dict[str, list[float]] = {"spatial": [], "future": []}
    kind_counts = {"spatial": 0, "future": 0}
    lr = cosine_lr(step, cfg)
    m = float("nan")
    stop_reason = "max_steps"

    it = iter(dl)
    while step < cfg.max_steps:
        if (time.time() - t_loop) / 3600.0 >= cfg.max_hours:
            stop_reason = "max_hours_safety_cap"
            break
        lr = cosine_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        mask, kind = sample_mask(
            model.grid, p_future=cfg.p_future, generator=gen,
            n_blocks=cfg.mask_blocks, n_future_slots=cfg.n_future_slots,
            context_spatial_mask=cfg.context_spatial_mask)
        while not mask.any():  # only possible on tiny test grids; never on 12x12
            mask, kind = sample_mask(
                model.grid, p_future=cfg.p_future, generator=gen,
                n_blocks=cfg.mask_blocks, n_future_slots=cfg.n_future_slots,
                context_spatial_mask=cfg.context_spatial_mask)
        mask = mask.to(dev)
        src = it_future if (it_future is not None and kind == "future") else it
        loss_acc = 0.0
        for _ in range(cfg.grad_accum):
            clips = next(src).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
                res = model(clips, mask)
            loss = res["loss"]
            total = loss
            if cfg.vicreg_var or cfg.vicreg_cov:
                v, c = vicreg_terms(res["pred"])
                total = loss + cfg.vicreg_var * v + cfg.vicreg_cov * c
            (total / cfg.grad_accum).backward()
            loss_acc += float(loss.detach()) / cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        m = model.update_target(ema_progress(step, cfg))
        lv = loss_acc
        losses.append(lv)
        losses_by_kind[kind].append(lv)
        kind_counts[kind] += 1
        step += 1
        if cfg.snapshot_every and step % cfg.snapshot_every == 0:
            snapshot()

        if step % cfg.log_every == 0 or step == cfg.max_steps:
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype,
                                                 enabled=dev == "cuda"):
                diag = collapse_diagnostics(model.embed(clips[:64]).float())
            n = cfg.log_every
            tail_s = losses_by_kind["spatial"][-n:]
            tail_f = losses_by_kind["future"][-n:]
            prior_s = sum(s.get("gpu_seconds", s["seconds"]) for s in prior_segments)
            row = [step, float(np.mean(losses[-n:])),
                   float(np.mean(tail_s)) if tail_s else "",
                   float(np.mean(tail_f)) if tail_f else "",
                   kind_counts["spatial"], kind_counts["future"], lr, m,
                   step * cfg.batch_size, round(prior_s + time.time() - t_loop, 1),
                   diag["std_mean"], diag["effective_rank"], cfg.p_future]
            if not cfg.dry_run:
                with log_path.open("a", newline="") as fh:
                    csv.writer(fh).writerow(row)
            print(f"step {step} loss {row[1]:.4f} lr {lr:.2e} ema {m:.6f} "
                  f"rank {diag['effective_rank']:.1f} "
                  f"seg {(time.time() - t_loop) / 3600:.2f}h", flush=True)

        if stop.set:
            stop_reason = f"signal_{signal.Signals(stop.signum).name}"
            break
        if (time.time() - t_ckpt) / 60.0 >= cfg.ckpt_minutes:
            save()
            t_ckpt = time.time()
            print(f"  checkpoint at step {step}", flush=True)

    del it, it_future
    save()
    segs = segments_now()
    totals = segment_totals(segs)
    annealed = step == cfg.max_steps
    summary = {
        "run_name": cfg.run_name,
        "objective": (f"mixed masking: future mask with probability {cfg.p_future} "
                      f"(last {cfg.n_future_slots} temporal slots as targets), "
                      f"spatial tube mask otherwise"
                      if cfg.p_future > 0 else
                      "spatial tube masking only (the ablation)"),
        "mask_kind_counts_this_segment": kind_counts,
        "final_loss_spatial_last500": (
            float(np.mean(losses_by_kind["spatial"][-500:]))
            if losses_by_kind["spatial"] else None),
        "final_loss_future_last500": (
            float(np.mean(losses_by_kind["future"][-500:]))
            if losses_by_kind["future"] else None),
        "shards_frozen_at_start": [p.name for p in ds.paths],
        "steps": step,
        "planned_final_step": cfg.max_steps,
        "annealed": annealed,
        "final_lr": lr,
        "final_ema_momentum": m,
        "clips_seen": step * cfg.batch_size * cfg.grad_accum,
        "n_clips_in_corpus": len(ds),
        "n_shards": len(ds.paths),
        "effective_epochs": round(step * cfg.batch_size / max(1, len(ds)), 2),
        "final_loss_mean_last500": float(np.mean(losses[-500:])) if losses else None,
        **totals,
        "gpu_hours": round(totals["gpu_seconds_total"] / 3600.0, 3),
        "segments": segs,
        "resume": resume_info,
        "stop_reason": stop_reason,
        "checkpoint": ckpt_path.name,
        "log_csv": log_path.name,
        "config": asdict(cfg),
        "device": _gpu_name(dev),
        "torch_version": torch.__version__,
    }
    if not annealed:
        summary["note"] = (f"stopped at step {step} of {cfg.max_steps} ({stop_reason}); "
                           "the LR and EMA schedules did not finish annealing")
    if not cfg.dry_run:
        atomic_write_text(summary_path, json.dumps(summary, indent=2) + "\n")
    return summary
