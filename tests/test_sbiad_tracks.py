"""The fast centroid tracker must equal the per-ID reference (synthetic labels only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tijepa.io import sbiad  # noqa: E402


def _reference(labels: np.ndarray, min_len: int) -> dict:
    out: dict = {}
    for t in range(labels.shape[0]):
        for i in np.unique(labels[t]):
            if i == 0:
                continue
            ys, xs = np.nonzero(labels[t] == i)
            out.setdefault(int(i), []).append((t, ys.mean(), xs.mean()))
    return {k: np.asarray(v) for k, v in out.items() if len(v) >= min_len}


def test_bincount_centroids_match_reference():
    rng = np.random.default_rng(0)
    synthetic_labels = rng.integers(0, 7, (6, 20, 30)).astype(np.int32)
    synthetic_labels[2][synthetic_labels[2] == 3] = 0  # an ID absent from one frame
    fast = sbiad.tracks_from_labels(synthetic_labels, 4)
    ref = _reference(synthetic_labels, 4)
    assert sorted(fast) == sorted(ref)
    for k in ref:
        np.testing.assert_allclose(fast[k], ref[k])
