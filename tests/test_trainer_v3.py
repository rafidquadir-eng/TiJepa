"""PLAN_v3 trainer additions: immutable snapshots, VICReg terms, gradient accumulation,
stride-filtered future batches (rule 3, H3.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_trainer import tiny_cfg  # noqa: E402

from tijepa.train.pretrain_video import stride_subset, train, vicreg_terms  # noqa: E402


def test_snapshots_are_kept_and_immutable(tmp_path):
    cfg = tiny_cfg(tmp_path, max_steps=8, snapshot_every=4)
    train(cfg)
    snap = Path(cfg.out_dir) / cfg.run_name
    files = sorted(p.name for p in snap.glob("step_*.pt"))
    assert files == ["step_000004.pt", "step_000008.pt"]
    before = (snap / "step_000004.pt").stat().st_mtime_ns
    cfg2 = tiny_cfg(tmp_path, max_steps=12, snapshot_every=4)
    train(cfg2)  # resumes; must not rewrite the existing snapshots
    assert (snap / "step_000004.pt").stat().st_mtime_ns == before
    assert (snap / "step_000012.pt").exists()
    sd = torch.load(snap / "step_000008.pt", weights_only=False)
    assert sd["step"] == 8 and sd["snapshot"]


def test_vicreg_terms_penalise_collapse():
    torch.manual_seed(0)
    healthy = torch.randn(4, 50, 16) * 2
    collapsed = torch.randn(4, 50, 1).expand(-1, -1, 16) * 0.01
    v_h, c_h = vicreg_terms(healthy)
    v_c, _ = vicreg_terms(collapsed)
    assert v_h < 0.05 and v_c > 0.9
    corr = torch.randn(4, 50, 1).expand(-1, -1, 16) * 3
    _, c_corr = vicreg_terms(corr)
    assert c_corr > 10 * c_h


def test_grad_accum_and_vicreg_run(tmp_path):
    s = train(tiny_cfg(tmp_path, max_steps=4, grad_accum=2, vicreg_var=0.04, vicreg_cov=0.001))
    assert s["annealed"] and s["clips_seen"] == 4 * 4 * 2
    assert np.isfinite(s["final_loss_mean_last500"])


def test_future_batches_come_from_the_requested_strides(tmp_path):
    cfg = tiny_cfg(tmp_path, max_steps=6, future_strides=[9, 27], p_future=1.0)
    d = Path(cfg.shard_dir)
    rows = [
        "source,archive,movie,track_id,t0,stride,static,um_per_px,frame_interval_s,"
        "effective_interval_s,shard,offset"
    ]
    for i in range(40):
        st = (1, 3, 9, 27)[i % 4]
        rows.append(f"s,a,m,{i},0,{st},{i % 8 == 0},0.65,8,8,pretrain_shard_0000.npy,{i}")
    (d / "pretrain_index.csv").write_text("\n".join(rows) + "\n")
    sub = stride_subset(d, [9, 27])
    assert all(i % 4 in (2, 3) and i % 8 != 0 for i in sub)
    assert train(cfg)["annealed"]
