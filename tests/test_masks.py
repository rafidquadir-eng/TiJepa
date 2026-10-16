"""Unit tests for the v1 mask generators (PLAN_v1.md F2.1 and 11.4).

The future mask is the whole point of the v1 objective change, so its shape and
coverage are pinned here rather than trusted. All fixtures are synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tijepa.models.video_jepa import (  # noqa: E402
    future_mask,
    multiblock_tube_mask,
    sample_mask,
)

GRID = (8, 12, 12)  # 16 frames / tubelet depth 2 = 8 temporal slots, 12x12 spatial


def test_future_mask_targets_exactly_the_last_slots():
    m = future_mask(GRID, n_future_slots=2).reshape(*GRID)
    assert bool(m[-1].all()) and bool(m[-2].all())
    assert not bool(m[:-2].any())
    # 2 of 8 slots masked = a quarter of all tokens
    assert int(m.sum()) == 2 * GRID[1] * GRID[2]


def test_future_mask_context_is_contiguous_in_time():
    """The visible tokens must be the FIRST slots: a forecaster cannot see the future."""
    m = future_mask(GRID, n_future_slots=3).reshape(*GRID)
    visible_slots = [t for t in range(GRID[0]) if not bool(m[t].any())]
    assert visible_slots == list(range(GRID[0] - 3))


def test_future_mask_optional_context_spatial_masking():
    m = future_mask(GRID, n_future_slots=2, context_spatial_mask=0.15)
    m3 = m.reshape(*GRID)
    n_pos = GRID[1] * GRID[2]
    k = round(0.15 * n_pos)
    # each context slot loses the same k spatial positions
    for t in range(GRID[0] - 2):
        assert int(m3[t].sum()) == k
    # and the same positions in every context slot, since it is a tube
    first = m3[0].reshape(-1).nonzero().flatten()
    for t in range(1, GRID[0] - 2):
        assert torch.equal(m3[t].reshape(-1).nonzero().flatten(), first)
    assert int(m.sum()) == 2 * n_pos + (GRID[0] - 2) * k


def test_future_mask_rejects_degenerate_arguments():
    with pytest.raises(ValueError):
        future_mask(GRID, n_future_slots=0)
    with pytest.raises(ValueError):
        future_mask(GRID, n_future_slots=GRID[0])
    with pytest.raises(ValueError):
        # masking every slot's spatial positions would leave nothing visible
        future_mask(GRID, n_future_slots=GRID[0] - 1, context_spatial_mask=1.0)


def test_spatial_mask_is_tubed_and_leaves_context():
    g = torch.Generator().manual_seed(0)
    m = multiblock_tube_mask(GRID, generator=g).reshape(*GRID)
    # identical across time: that is what "tubed" means
    for t in range(1, GRID[0]):
        assert torch.equal(m[t], m[0])
    assert int((~m).sum()) >= 16
    assert int(m.sum()) > 0


def test_spatial_mask_does_not_target_whole_time_slots():
    """The distinguishing property against the future mask."""
    g = torch.Generator().manual_seed(1)
    m = multiblock_tube_mask(GRID, generator=g).reshape(*GRID)
    # no slot is fully masked and no slot is fully visible, for a typical draw
    assert not bool(m[-1].all())
    assert bool(m[-1].any())


def test_sample_mask_returns_both_kinds_with_the_expected_rate():
    g = torch.Generator().manual_seed(7)
    kinds = [sample_mask(GRID, p_future=0.5, generator=g)[1] for _ in range(400)]
    n_future = sum(1 for k in kinds if k == "future")
    assert 0.4 < n_future / len(kinds) < 0.6
    assert set(kinds) == {"future", "spatial"}


def test_sample_mask_extremes_are_deterministic():
    g = torch.Generator().manual_seed(3)
    assert all(sample_mask(GRID, p_future=1.0, generator=g)[1] == "future"
               for _ in range(20))
    assert all(sample_mask(GRID, p_future=0.0, generator=g)[1] == "spatial"
               for _ in range(20))


def test_masks_have_the_right_length_and_dtype():
    n = GRID[0] * GRID[1] * GRID[2]
    for m, _ in (sample_mask(GRID, p_future=1.0), sample_mask(GRID, p_future=0.0)):
        assert m.shape == (n,)
        assert m.dtype == torch.bool
