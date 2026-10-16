"""Movie fingerprints for the pretraining/evaluation leakage check (PLAN_v1.md F1.3).

The v0 run pretrained on ComplexEye neutrophil movies and evaluated tracking on
S-BIAD2196 neutrophil movies. Both are human neutrophil brightfield time-lapses from
overlapping groups, so there was no evidence either way about whether any evaluation
movie also sat in the pretraining corpus. That is review finding D3, and a held-out
claim without such a check is not worth making.

A fingerprint is built from three frames (first, middle, last), each downsampled to
40x25 by block averaging and z-scored, then concatenated. Two movies match when their
shapes are identical and the Pearson correlation of their fingerprints exceeds 0.98;
the SHA-1 of the raw first frame is recorded as well, so byte-identical duplicates are
detectable without relying on a threshold at all.

Downsampling before z-scoring makes the comparison insensitive to intensity scaling and
to small differences in compression, which is what matters: the question is whether the
same field of view was imaged, not whether two files are byte-identical.
"""
from __future__ import annotations

import hashlib

import numpy as np

FP_H, FP_W = 25, 40
MATCH_R = 0.98
REPORT_R = 0.90


def _block_reduce(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Average-pool `img` to (h, w) by integer-ish blocks, cropping the remainder."""
    H, W = img.shape
    bh, bw = max(1, H // h), max(1, W // w)
    img = img[: bh * h, : bw * w].astype(np.float32)
    return img.reshape(h, bh, w, bw).mean(axis=(1, 3))


def fingerprint(frames: np.ndarray) -> dict:
    """Fingerprint a (T, Y, X) movie.

    Returns the vector, the shape it was computed from, and the SHA-1 of frame 0.
    """
    if frames.ndim != 3 or frames.shape[0] < 1:
        raise ValueError(f"expected (T, Y, X) with T >= 1, got {frames.shape}")
    t = frames.shape[0]
    idx = sorted({0, t // 2, t - 1})
    parts = []
    for i in idx:
        r = _block_reduce(frames[i], FP_H, FP_W)
        s = r.std()
        parts.append((r - r.mean()) / (s if s > 1e-8 else 1.0))
    vec = np.concatenate([p.ravel() for p in parts])
    # Pad to a fixed length so movies with 1 or 2 frames stay comparable.
    full = FP_H * FP_W * 3
    if vec.size < full:
        vec = np.concatenate([vec, np.zeros(full - vec.size, dtype=np.float32)])
    return {
        "vector": vec.astype(np.float32),
        "shape": [int(x) for x in frames.shape],
        "frame0_sha1": hashlib.sha1(
            np.ascontiguousarray(frames[0]).tobytes()).hexdigest(),
        "n_frames": int(t),
    }


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare_sets(eval_fps: dict[str, dict], train_fps: dict[str, dict],
                 *, match_r: float = MATCH_R, report_r: float = REPORT_R) -> dict:
    """Compare every evaluation movie against every candidate pretraining movie.

    Shape equality is required for a *match*, because two different fields of view at
    the same magnification can correlate moderately; but pairs above `report_r` are
    reported whatever their shapes, so the decision is auditable rather than hidden
    behind the shape test.
    """
    pairs, matched_train, matched_eval = [], set(), set()
    for ek, ev in eval_fps.items():
        for tk, tv in train_fps.items():
            r = correlate(ev["vector"], tv["vector"])
            same_shape = ev["shape"][1:] == tv["shape"][1:]
            identical = ev["frame0_sha1"] == tv["frame0_sha1"]
            if r < report_r and not identical:
                continue
            is_match = bool(identical or (same_shape and r >= match_r))
            pairs.append({
                "eval_movie": ek, "train_movie": tk, "pearson_r": round(r, 5),
                "eval_shape": ev["shape"], "train_shape": tv["shape"],
                "same_frame_shape": same_shape,
                "frame0_sha1_identical": identical,
                "decision": "LEAKAGE: exclude the pretraining movie" if is_match
                            else "reported only, not a match",
            })
            if is_match:
                matched_train.add(tk)
                matched_eval.add(ek)
    pairs.sort(key=lambda p: -p["pearson_r"])
    return {
        "n_eval_movies": len(eval_fps),
        "n_train_movies": len(train_fps),
        "match_threshold_r": match_r,
        "report_threshold_r": report_r,
        "n_pairs_reported": len(pairs),
        "n_pairs_matched": sum(1 for p in pairs if p["decision"].startswith("LEAKAGE")),
        "excluded_train_movies": sorted(matched_train),
        "eval_movies_with_a_twin": sorted(matched_eval),
        "pairs": pairs,
    }
