"""H4: detection-level association on S-BIAD2196 ground truth (PLAN_v3 §8, finding R9).

Point tracking by patch correlation cannot tell features apart (frame-shuffled = trained =
random), so the primary tracking metric becomes identity association between ground-
truth cells. For a frame pair (t, t + k), k in {1, 3, 9}, every ground-truth cell in each
frame is embedded from a 16-frame clip ending at that frame (a fixed 96 x 96 window centred
on the cell, so the clip carries the cell's recent motion; image encoders see the last
frame). Identities are matched with the Hungarian algorithm on

  (a) embedding cosine distance only, (b) centroid distance only (baseline), and
  (c) both, each cost matrix z-scored and the two summed with equal weight (fixed in
      advance, not tuned).

Association accuracy = fraction of cells present in both frames that are assigned to
themselves; cells present in only one frame remain as distractors. Anchors are every 45th
frame from frame 15. IDF1 over the movie is computed for the chain of k = 9 associations
through frames 15, 24, 33, ...: with ground-truth detections IDF1 = IDTP / N, IDTP from the
optimal one-to-one mapping between true and predicted identities.

Statistics: per-movie values and exact sign-flip tests over the 6 movies (no CI, < 10
groups). TiJepa-0 summaries exclude 08_H4 (its corpus held that movie's twin).
"""

from __future__ import annotations

import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.eval import encoders as ENC  # noqa: E402
from tijepa.eval import forecast_controls as FC  # noqa: E402
from tijepa.eval import provenance as PV  # noqa: E402
from tijepa.io import sbiad  # noqa: E402

OUT = REPO / "results" / "v3" / "partA" / "association.json"
GAPS = (1, 3, 9)
CHAIN_K, ANCHOR_EVERY, T0 = 9, 45, 15
CROP, NF = 96, 16
E0_TWIN = "08_H4.ome.zarr.zip"
SEED = 20261019


def cells_by_frame(lab: np.ndarray) -> dict[int, dict[int, tuple[float, float]]]:
    tr = sbiad.tracks_from_labels(lab, min_len=1)
    out: dict[int, dict] = {}
    for cid, a in tr.items():
        for t, y, x in a:
            out.setdefault(int(t), {})[cid] = (float(y), float(x))
    return out


def clip_at(raw_p: np.ndarray, t: int, y: float, x: float) -> np.ndarray:
    h = CROP // 2
    yy, xx = int(round(y)) + h, int(round(x)) + h  # raw_p is padded by h
    return raw_p[t - NF + 1 : t + 1, yy - h : yy + h, xx - h : xx + h]


def hungarian(cost: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    r, c = linear_sum_assignment(cost)
    out = np.full(cost.shape[0], -1)
    out[r] = c
    return out


def z(m: np.ndarray) -> np.ndarray:
    return (m - m.mean()) / (m.std() + 1e-9)


def costs(ea, eb, pa, pb) -> dict[str, np.ndarray]:
    pos = np.linalg.norm(pa[:, None] - pb[None], axis=2)
    out = {"position": pos}
    if ea is not None:
        na = ea / (np.linalg.norm(ea, axis=1, keepdims=True) + 1e-12)
        nb = eb / (np.linalg.norm(eb, axis=1, keepdims=True) + 1e-12)
        emb = 1.0 - na @ nb.T
        out["embedding"] = emb
        out["embedding+position"] = z(emb) + z(pos)
    return out


def idf1(gt_ids: list[np.ndarray], pred_ids: list[np.ndarray]) -> float:
    g = np.concatenate(gt_ids)
    p = np.concatenate(pred_ids)
    ug, gi = np.unique(g, return_inverse=True)
    up, pi = np.unique(p, return_inverse=True)
    m = np.zeros((len(ug), len(up)))
    np.add.at(m, (gi, pi), 1)
    from scipy.optimize import linear_sum_assignment

    r, c = linear_sum_assignment(-m)
    return float(m[r, c].sum() / len(g))


def main() -> int:
    t_start = time.time()
    movies = sorted((REPO / "data" / "raw" / "s-biad2196" / "Celltrack_data").glob("*.zip"))
    data = {}
    for mv in movies:
        raw, lab = sbiad.load_movie(mv)
        r = raw.astype(np.float32)
        h = CROP // 2
        data[mv.name] = {
            "raw_p": np.pad(r, ((0, 0), (h, h), (h, h)), mode="edge"),
            "cells": cells_by_frame(lab),
            "T": raw.shape[0],
        }
        del raw, lab
        print(f"  loaded {mv.name}")
    frames_needed = {}
    for name, d in data.items():
        chain = list(range(T0, d["T"], CHAIN_K))
        anchors = [t for t in range(T0, d["T"] - max(GAPS), ANCHOR_EVERY)]
        fr = sorted(set(chain) | {a + k for a in anchors for k in (0, *GAPS)})
        frames_needed[name] = {"frames": fr, "chain": chain, "anchors": anchors}

    def embed_movie(fn_clips, name) -> dict[int, tuple[list[int], np.ndarray]]:
        d = data[name]
        out = {}
        for t in frames_needed[name]["frames"]:
            ids = sorted(d["cells"].get(t, {}))
            cl = np.stack([clip_at(d["raw_p"], t, *d["cells"][t][i]) for i in ids])
            out[t] = (ids, fn_clips(cl))
        return out

    # encoder adapters: clips (N, 16, 96, 96) float -> (N, D)
    enc = {"position_only": None}
    recs: dict = {}

    def img(fn):
        return lambda cl: fn(ENC.minmax(cl[:, -1]))

    enc["E2_imagenet_vit_s"] = img(lambda X: ENC.embed_timm(ENC.TIMM["E2_imagenet_vit_s"], X))
    enc["E3_dinov2_vit_s"] = img(lambda X: ENC.embed_timm(ENC.TIMM["E3_dinov2_vit_s"], X))
    from tijepa.eval import vjepa2 as VJ2

    enc[VJ2.KEY] = lambda cl: VJ2.embed_clips(
        ENC.minmax(cl.reshape(-1, CROP, CROP)).reshape(cl.shape), ENC.DEV
    )
    recs[VJ2.KEY] = {"model": VJ2.record()}

    @torch.no_grad()
    def jep(m, shuffle=False):
        def f(cl):
            if shuffle:
                rng = np.random.default_rng(SEED)
                cl = np.stack([c[rng.permutation(NF)] for c in cl])
            outs = []
            for i in range(0, len(cl), 128):
                t = torch.from_numpy(cl[i : i + 128])[:, None]
                mu = t.mean(dim=(1, 2, 3, 4), keepdim=True)
                sd = t.std(dim=(1, 2, 3, 4), keepdim=True)
                c = ((t - mu) / (sd + 1e-6)).to(ENC.DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ENC.DEV == "cuda"):
                    outs.append(m.embed(c).float().cpu().numpy())
            return np.concatenate(outs)

        return f

    models = []
    for run, fn in ENC.JEPA_RUNS:
        m, rec = ENC.load_jepa(run, fn)
        models.append((PV.key_for(rec), rec, m))
    from tijepa.models.video_jepa import VideoJEPA

    for s in ENC.RANDOM_SEEDS:
        torch.manual_seed(s)
        models.append(
            (
                f"E6_random_init_draw{s - ENC.RANDOM_SEEDS[0]}",
                PV.random_init_record(s),
                VideoJEPA(clip=(NF, CROP, CROP)).eval().to(ENC.DEV),
            )
        )
    for key, rec, m in models:
        PV.label_for_key(key, rec)
        enc[key] = jep(m)
        recs[key] = {"checkpoint": rec}
        if key == "E4_tijepa1":
            enc["E5_frame_shuffled"] = jep(m, shuffle=True)
            recs["E5_frame_shuffled"] = {
                "checkpoint": rec,
                "note": "TiJepa-1 with the 16 frames of every clip shuffled",
            }

    res: dict = {
        "phase": "PLAN_v3 H4 (finding R9)",
        "gaps": list(GAPS),
        "chain_gap_for_idf1": CHAIN_K,
        "anchor_every": ANCHOR_EVERY,
        "cost_rule": "z-scored embedding cosine distance + z-scored centroid "
        "distance, equal weights, fixed in advance",
        "encoders": recs,
        "per_movie": {},
        "summary": {},
        "frames": {k: {kk: len(vv) for kk, vv in v.items()} for k, v in frames_needed.items()},
    }
    for key, fn in enc.items():
        res["per_movie"][key] = {}
        for name, d in data.items():
            if key == "E0_tijepa0" and name == E0_TWIN:
                continue
            emb = None if fn is None else embed_movie(fn, name)
            pos = {
                t: (
                    sorted(d["cells"].get(t, {})),
                    np.array([d["cells"][t][i] for i in sorted(d["cells"].get(t, {}))]),
                )
                for t in frames_needed[name]["frames"]
            }
            acc: dict[str, dict[str, list]] = {}
            for a in frames_needed[name]["anchors"]:
                for k in GAPS:
                    ia, pa = pos[a]
                    ib, pb = pos[a + k]
                    ea = None if emb is None else emb[a][1]
                    eb = None if emb is None else emb[a + k][1]
                    common = set(ia) & set(ib)
                    for cname, c in costs(ea, eb, pa, pb).items():
                        asg = hungarian(c)
                        ok = [
                            ib[asg[j]] == ia[j]
                            for j in range(len(ia))
                            if ia[j] in common and asg[j] >= 0
                        ]
                        acc.setdefault(cname, {}).setdefault(str(k), []).append(
                            float(np.mean(ok)) if ok else np.nan
                        )
            row = {c: {k: float(np.nanmean(v)) for k, v in g.items()} for c, g in acc.items()}
            # IDF1 over the k = 9 chain
            chain = frames_needed[name]["chain"]
            for cname in row:
                nxt = 0
                prev_ids = pos[chain[0]][0]
                lab_pred = {cid: i for i, cid in enumerate(prev_ids)}
                nxt = len(prev_ids)
                gts, prs = [np.array(prev_ids)], [np.array([lab_pred[c] for c in prev_ids])]
                for t0, t1 in zip(chain[:-1], chain[1:], strict=False):
                    ia, pa = pos[t0]
                    ib, pb = pos[t1]
                    c = costs(
                        None if emb is None else emb[t0][1],
                        None if emb is None else emb[t1][1],
                        pa,
                        pb,
                    )[cname]
                    asg = hungarian(c)
                    inv = {int(asg[j]): j for j in range(len(ia)) if asg[j] >= 0}
                    new = {}
                    for jb, cid in enumerate(ib):
                        if jb in inv:
                            new[cid] = lab_pred[ia[inv[jb]]]
                        else:
                            new[cid] = nxt
                            nxt += 1
                    lab_pred = new
                    gts.append(np.array(ib))
                    prs.append(np.array([lab_pred[c] for c in ib]))
                row[cname]["idf1_k9_chain"] = idf1(gts, prs)
            res["per_movie"][key][name] = row
        summ = {}
        for cname in next(iter(res["per_movie"][key].values())):
            summ[cname] = {
                m: float(np.mean([v[cname][m] for v in res["per_movie"][key].values()]))
                for m in (*map(str, GAPS), "idf1_k9_chain")
            }
        res["summary"][key] = summ
        print(
            f"  {key:24s} "
            + "  ".join(
                f"{c}: k1 {s['1']:.3f} k9 {s['9']:.3f} idf1 {s['idf1_k9_chain']:.3f}"
                for c, s in summ.items()
            )
        )
    # tests: embedding+position vs position-only, per encoder, k = 9 and IDF1
    tests = {}
    pos_only = res["per_movie"]["position_only"]
    for key in enc:
        if key == "position_only":
            continue
        pm = res["per_movie"][key]
        ms = sorted(pm)
        tests[key] = {
            m: FC.sign_flip(
                [pm[n]["embedding+position"][m] - pos_only[n]["position"][m] for n in ms]
            )
            for m in ("9", "idf1_k9_chain")
        }
        tests[key]["embedding_only_k9_vs_position"] = FC.sign_flip(
            [pm[n]["embedding"]["9"] - pos_only[n]["position"]["9"] for n in ms]
        )
    res["tests_vs_position_only"] = tests
    t1 = res["per_movie"].get("E4_tijepa1", {})
    for other in (
        "E5_frame_shuffled",
        "E4s_tijepa1_spatial",
        "E6_random_init_draw0",
        "E3_dinov2_vit_s",
        VJ2.KEY,
    ):
        o = res["per_movie"].get(other, {})
        ms = sorted(set(t1) & set(o))
        res.setdefault("tijepa1_vs", {})[other] = FC.sign_flip(
            [t1[n]["embedding"]["9"] - o[n]["embedding"]["9"] for n in ms]
        )
    res["seconds"] = round(time.time() - t_start, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
