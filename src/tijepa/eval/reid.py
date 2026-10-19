"""Label-free re-identification validation score (PLAN_v3 H3.1).

For each anchor frame, every eligible cell's query embedding (clip ending at the anchor)
is matched by cosine similarity against the gallery embeddings of the same set of cells
at gap k (clip starting at anchor + k). Top-1 accuracy is the fraction of queries whose
nearest gallery item is the same cell; chance is 1 / (cells at that anchor).

The validation score is the mean top-1 over k in {3, 9, 27}, averaged over anchors with
at least MIN_CELLS cells. It is the only quantity, with effective rank, used to compare
training recipes (PLAN_v3 rule 1).
"""
from __future__ import annotations

import numpy as np

GAPS = (3, 9, 27)
MIN_CELLS = 5


def top1(query: np.ndarray, gallery: np.ndarray) -> float:
    """query, gallery: (n, d), row i of each is the same cell."""
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    sim = q @ g.T
    return float(np.mean(sim.argmax(1) == np.arange(len(q))))


def score(zq: np.ndarray, zg: dict[int, np.ndarray], anchor_ids: np.ndarray) -> dict:
    """Per-gap top-1 averaged over anchors, plus chance, for one movie or a pool."""
    out: dict = {"per_gap": {}, "chance": None}
    chances = []
    for k in GAPS:
        accs = []
        for a in np.unique(anchor_ids):
            m = anchor_ids == a
            if m.sum() < MIN_CELLS:
                continue
            accs.append(top1(zq[m], zg[k][m]))
            if k == GAPS[0]:
                chances.append(1.0 / m.sum())
        out["per_gap"][str(k)] = float(np.mean(accs)) if accs else None
    vals = [v for v in out["per_gap"].values() if v is not None]
    out["validation_score"] = float(np.mean(vals)) if vals else None
    out["chance"] = float(np.mean(chances)) if chances else None
    return out
