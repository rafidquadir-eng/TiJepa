"""Unit tests on synthetic fixtures only (PLAN.md 1.1.2: synthetic_* lives here only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tijepa.eval.stats import benjamini_hochberg, cluster_bootstrap_ci  # noqa: E402
from tijepa.io import tay  # noqa: E402
from tijepa.models.trace_jepa import trace_features  # noqa: E402
from tijepa.models.video_jepa import multiblock_tube_mask  # noqa: E402
from tijepa.preprocess import detect as D  # noqa: E402


def synthetic_movie(n=24, h=128, w=128, n_cells=6, seed=0):
    rng = np.random.default_rng(seed)
    frames = np.full((n, h, w), 200, np.uint8)
    pos = rng.uniform(20, h - 20, size=(n_cells, 2))
    for t in range(n):
        for c in range(n_cells):
            y, x = (pos[c] + t * 0.8).astype(int) % (h - 20) + 10
            frames[t, y - 4:y + 4, x - 4:x + 4] = 60
    return frames


def test_tube_mask_is_constant_over_time():
    grid = (8, 12, 12)
    m = multiblock_tube_mask(grid).reshape(grid)
    assert m.shape == grid
    for t in range(1, grid[0]):
        assert bool((m[t] == m[0]).all()), "mask must be tubed over time"
    assert 0.0 < float(m.float().mean()) < 1.0


def test_detect_and_track_synthetic():
    frames = synthetic_movie()
    det = D.detect_movie(frames)
    assert len(det) > 0
    linked = D.link_tracks(det, min_len=8)
    assert linked["particle"].nunique() >= 1
    clips, meta = D.extract_clips(frames, linked, crop=32, n_frames=8)
    assert clips.dtype == np.uint8
    if len(clips):
        assert clips.shape[1:] == (8, 32, 32)
        assert len(meta) == len(clips)


def test_trace_features_shape_and_finiteness():
    rng = np.random.default_rng(0)
    win = np.sin(np.linspace(0, 6, 40))[None, :] + rng.normal(0, 0.1, (12, 40))
    f, names = trace_features(win)
    assert f.shape[0] == 12
    assert f.shape[1] == len(names)
    assert np.isfinite(f).all()


def test_cluster_bootstrap_respects_groups():
    y = np.array([0, 0, 1, 1] * 5)
    pred = y.copy()
    groups = np.repeat(np.arange(5), 4)
    out = cluster_bootstrap_ci(y, pred, groups, n_boot=100)
    assert out["point"] == pytest.approx(1.0)
    assert out["ci_low"] is not None


def test_benjamini_hochberg_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
    q = benjamini_hochberg(p)
    assert np.all(q >= p - 1e-12)
    assert np.all((q >= 0) & (q <= 1))
    assert np.all(np.diff(q) >= -1e-12)


def test_second_stim_index_matches_six_minute_frames():
    # 8 h after the first stimulus at frame 3, at 6 min per frame
    assert tay.second_stim_index(8.0) == 3 + 80
    assert tay.second_stim_index(4.0) == 3 + 40
