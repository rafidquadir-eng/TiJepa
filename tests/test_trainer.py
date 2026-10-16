"""Trainer safety and accounting (PLAN_v2.md G1.1 and G7.1).

Every test trains a tiny synthetic model on synthetic shards on CPU, so the suite runs
anywhere in a few seconds per test.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from tijepa.train.pretrain_video import (  # noqa: E402
    LockError,
    PretrainConfig,
    RunLock,
    StreamSampler,
    atomic_torch_save,
    cosine_lr,
    ema_momentum,
    train,
)


def synthetic_shards(d: Path, n: int = 40) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.save(d / "pretrain_shard_0000.npy",
            rng.integers(0, 255, (n, 16, 32, 32), dtype=np.uint8))
    return d


def tiny_cfg(tmp: Path, **kw) -> PretrainConfig:
    base = dict(shard_dir=str(synthetic_shards(tmp / "synthetic_shards")),
                out_dir=str(tmp / "ckpt"), clip_size=32, batch_size=4, max_steps=12,
                warmup_steps=2, num_workers=0, log_every=2, ckpt_minutes=1e9,
                model_dim=32, model_depth=1, model_heads=2, pred_dim=16, pred_depth=1,
                mask_blocks=2, run_name="synthetic_run")
    base.update(kw)
    return PretrainConfig(**base)


def test_schedule_anneals_exactly_on_the_planned_last_step():
    cfg = PretrainConfig(shard_dir="", out_dir="", max_steps=117_000, warmup_steps=500)
    assert cosine_lr(cfg.max_steps - 1, cfg) == pytest.approx(cfg.min_lr, abs=1e-15)
    assert ema_momentum(cfg.max_steps - 1, cfg) == pytest.approx(1.0)
    assert cosine_lr(cfg.max_steps // 2, cfg) > 0.4 * cfg.lr
    assert ema_momentum(0, cfg) < 0.9961


def test_tiny_run_finishes_annealed(tmp_path):
    s = train(tiny_cfg(tmp_path))
    assert s["annealed"] and s["steps"] == 12
    assert s["final_lr"] == pytest.approx(0.0, abs=1e-12)
    assert s["final_ema_momentum"] == pytest.approx(1.0)
    assert not (tmp_path / "ckpt" / "synthetic_run.lock").exists()


def test_segments_are_cumulative_across_a_resume(tmp_path):
    prior = [{"host": "old", "gpu_name": "x", "torch_version": "t", "start_step": 0,
              "end_step": 6, "seconds": 100.0, "gpu_seconds": 100.0}]
    cfg = tiny_cfg(tmp_path, max_steps=6)
    train(cfg)
    # simulate a checkpoint written by the old trainer (no segment record)
    ck = Path(cfg.out_dir) / "synthetic_run_T1_seed0.pt"
    sd = torch.load(ck, weights_only=False)
    for k in ("segments", "gpu_seconds_total", "wall_seconds_total", "steps_total",
              "mask_rng"):
        sd.pop(k)
    torch.save(sd, ck)
    s = train(tiny_cfg(tmp_path, max_steps=12, prior_segments_if_missing=prior))
    assert [seg["start_step"] for seg in s["segments"]] == [0, 6]
    assert s["segments"][-1]["end_step"] == 12
    assert s["gpu_seconds_total"] == pytest.approx(
        100.0 + s["segments"][-1]["gpu_seconds"])
    assert s["resume"]["old_max_steps"] == 6 and s["resume"]["new_max_steps"] == 12
    # a third segment keeps both earlier ones
    sd = torch.load(ck, weights_only=False)
    assert len(sd["segments"]) == 2 and sd["steps_total"] == 12
    # log rows are unique and monotonic
    steps = [int(r.split(",")[0]) for r in
             (Path(cfg.out_dir) / "pretrain_log_synthetic_run_seed0.csv")
             .read_text().splitlines()[1:]]
    assert steps == sorted(set(steps))


def test_resume_drops_log_rows_logged_after_the_checkpoint(tmp_path):
    cfg = tiny_cfg(tmp_path, max_steps=6)
    train(cfg)
    log = Path(cfg.out_dir) / "pretrain_log_synthetic_run_seed0.csv"
    rows = log.read_text().splitlines()
    log.write_text("\n".join(rows + [rows[-1].replace("6,", "8,", 1)]) + "\n")
    s = train(tiny_cfg(tmp_path, max_steps=10))
    assert s["resume"]["log_rows_dropped_after_checkpoint"] == 1


def test_lock_refuses_a_live_holder_and_replaces_a_dead_one(tmp_path):
    p = tmp_path / "run.lock"
    a = RunLock(p)
    a.acquire()
    info = json.loads(p.read_text())
    assert info["pid"] == os.getpid() and info["hostname"] == socket.gethostname()
    with pytest.raises(LockError):
        RunLock(p).acquire()
    a.release()
    assert not p.exists()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    p.write_text(json.dumps({"hostname": socket.gethostname(), "pid": dead.pid}))
    b = RunLock(p)
    b.acquire()  # stale: same host, dead pid
    b.release()
    p.write_text(json.dumps({"hostname": "another-instance", "pid": 1}))
    with pytest.raises(LockError):
        RunLock(p).acquire()


def test_trainer_refuses_to_start_while_locked(tmp_path):
    cfg = tiny_cfg(tmp_path)
    Path(cfg.out_dir).mkdir(parents=True)
    (Path(cfg.out_dir) / "synthetic_run.lock").write_text(
        json.dumps({"hostname": "another-instance", "pid": 1}))
    with pytest.raises(LockError):
        train(cfg)
    assert not (Path(cfg.out_dir) / "synthetic_run_T1_seed0.pt").exists()


def test_atomic_save_leaves_the_old_file_when_the_write_fails(tmp_path):
    p = tmp_path / "synthetic.pt"
    atomic_torch_save({"v": 1}, p)

    class Boom:
        def __reduce__(self):
            raise RuntimeError("synthetic failure mid-write")

    with pytest.raises(RuntimeError):
        atomic_torch_save({"v": 2, "x": Boom()}, p)
    assert torch.load(p)["v"] == 1
    assert [f.name for f in tmp_path.iterdir()] == ["synthetic.pt"]


def test_stream_sampler_resume_continues_the_stream():
    full = StreamSampler(10, seed=3, start=0)
    it = iter(full)
    first = [next(it) for _ in range(25)]
    it2 = iter(StreamSampler(10, seed=3, start=13))
    assert [next(it2) for _ in range(12)] == first[13:]
    assert sorted(i for i, _ in first[:10]) == list(range(10))


def test_sigterm_saves_and_exits_cleanly(tmp_path):
    cfg = tiny_cfg(tmp_path, max_steps=10**6, log_every=1)
    script = tmp_path / "synthetic_trainer.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(SRC)!r})
        from tijepa.train.pretrain_video import PretrainConfig, train
        cfg = PretrainConfig(**json.loads({json.dumps(json.dumps(cfg.__dict__))}))
        s = train(cfg)
        print("STOP", s["stop_reason"], s["steps"], flush=True)
    """))
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    log = Path(cfg.out_dir) / "pretrain_log_synthetic_run_seed0.csv"
    deadline = time.time() + 120
    while time.time() < deadline:
        if log.exists() and len(log.read_text().splitlines()) > 3:
            break
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0, out
    assert "STOP signal_SIGTERM" in out
    steps = int(out.split("STOP signal_SIGTERM")[1].split()[0])
    sd = torch.load(Path(cfg.out_dir) / "synthetic_run_T1_seed0.pt", weights_only=False)
    assert sd["step"] == steps and sd["segments"][-1]["end_step"] == steps
    summ = json.loads((Path(cfg.out_dir) /
                       "pretrain_summary_synthetic_run_seed0.json").read_text())
    assert summ["annealed"] is False and "did not finish annealing" in summ["note"]
    assert not (Path(cfg.out_dir) / "synthetic_run.lock").exists()
