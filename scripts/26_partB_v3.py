"""H5.2: Part B refocused (PLAN_v3 §9; findings R14, R15).

1. **Ligand-pair identity from NF-kB dynamics (B3.2), promoted to a primary Part B result.**
   Same task, cells and well grouping as v2 (11 resolvable (first, second) ligand pairs,
   125 wells, up to 120 cells per well per seed, StratifiedGroupKFold(4) by well, three
   seeds that vary the subsample and the folds). Representations:
   raw trace, hand-crafted trace features (src/tijepa/models/trace_jepa.trace_features),
   a self-supervised **trace JEPA** and the same architecture at random initialisation.
   The trace JEPA is pretrained (no labels) only on traces from wells that are **not** among
   the 125 evaluation wells or the 4 tolerance wells, so no evaluated cell is seen even
   unlabelled. 125 wells >= 10, so 95% cluster-bootstrap CIs over wells are reported, with
   paired well-bootstrap differences and BH correction.

2. **Power analysis for tolerance from dynamics (B3.1, 4 wells).** Well-level, model-free
   summaries of the response to the second LPS (peak amplitude and AUC of the baseline-
   corrected response window): observed tolerant-minus-naive difference and pooled
   between-well SD (df = 2), the standardised effect d, and the wells per arm a two-sided
   two-sample t-test needs for 80% power at alpha = 0.05 (non-central t). Because the SD
   rests on 4 wells, the same calculation is repeated with the SD at the upper limit of
   its one-sided 80% chi-square interval.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.eval import probe as P  # noqa: E402
from tijepa.io import tay  # noqa: E402
from tijepa.models.trace_jepa import TraceJEPA, random_patch_mask, trace_features  # noqa: E402

_spec = importlib.util.spec_from_file_location("partB_v1", REPO / "scripts" / "25_partB_v1.py")
B1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B1)

OUT = REPO / "results" / "v3" / "partB" / "partB_v3.json"
CKPT = REPO / "checkpoints" / "trace_jepa_v3_seed0.pt"
SEEDS = (0, 1, 2)
L = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PRE_STEPS, PRE_BATCH = 6000, 512


def pad(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)
    if x.shape[1] < L:
        x = np.pad(x, ((0, 0), (0, L - x.shape[1])))
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True) + 1e-6
    return (x - mu) / sd


def b32_cells(ts: tay.TraceSet):
    sub = ts.subset(ts.layout_verified)
    named = dict(tay.LIGAND_CODES)

    def name(c: float) -> str:
        return "naive" if c == 0.0 else named.get(c, f"code{int(c)}")

    labels = np.array([f"{name(f)}->{name(s)}" for f, s in zip(sub.first, sub.second,
                                                                 strict=False)])
    keep = np.array(["code" not in lb for lb in labels])
    return sub.subset(keep), labels[keep]


def pretrain(pool: np.ndarray) -> tuple[TraceJEPA, dict]:
    torch.manual_seed(0)
    m = TraceJEPA(length=L).to(DEV)
    params = [p for n, p in m.named_parameters() if not n.startswith("target.")]
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.05)
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    x_all = torch.from_numpy(pool)
    losses = []
    t0 = time.time()
    for step in range(PRE_STEPS):
        lr = 5e-4 * min(1.0, (step + 1) / 300) * 0.5 * (1 + np.cos(np.pi * step / PRE_STEPS))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.integers(0, len(x_all), PRE_BATCH)
        x = x_all[idx].to(DEV)
        mask = random_patch_mask(m.n_patches, 0.5, gen).to(DEV)
        loss = m(x, mask)["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        m.update_target((step + 1) / PRE_STEPS)
        losses.append(float(loss))
    info = {"steps": PRE_STEPS, "batch": PRE_BATCH, "n_pool_cells": int(len(pool)),
            "final_loss_last200": float(np.mean(losses[-200:])),
            "seconds": round(time.time() - t0, 1), "mask_ratio": 0.5}
    torch.save({"model": m.state_dict(), "info": info}, CKPT)
    return m.eval(), info


@torch.no_grad()
def embed(m: TraceJEPA, x: np.ndarray) -> np.ndarray:
    return np.concatenate([m.embed(torch.from_numpy(x[i:i + 4096]).to(DEV)).cpu().numpy()
                           for i in range(0, len(x), 4096)])


def b32(ts, feats: dict[str, np.ndarray], labels, groups) -> dict:
    classes, y = np.unique(labels, return_inverse=True)
    out = {"n_classes": int(len(classes)), "classes": classes.tolist(),
           "n_cells": int(len(y)), "n_wells": int(len(np.unique(groups))),
           "chance_balanced_accuracy": 1.0 / len(classes), "results": {}}
    preds0 = {}
    for key, x in feats.items():
        per = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            sel = B1.subsample_per_group(groups, rng, 120)
            xs, ys, gs = x[sel], y[sel], groups[sel]
            cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            pred = np.zeros(len(ys), dtype=int)
            for tr, te in cv.split(xs, ys, gs):
                sc = StandardScaler().fit(xs[tr])
                clf = LogisticRegression(max_iter=1500, C=1.0, class_weight="balanced",
                                         random_state=seed)
                clf.fit(sc.transform(xs[tr]), ys[tr])
                pred[te] = clf.predict(sc.transform(xs[te]))
            per.append(float(balanced_accuracy_score(ys, pred)))
            if seed == 0:
                preds0[key] = (ys, pred, gs)
        ys, pred, gs = preds0[key]
        out["results"][key] = {"balanced_accuracy_mean": float(np.mean(per)),
                               "balanced_accuracy_sd": float(np.std(per, ddof=1)),
                               "per_seed": per,
                               "ci95_seed0_well_bootstrap":
                                   P.cluster_bootstrap(ys, pred, gs)["ci95"]}
        print(f"  B3.2 {key:22s} BA {np.mean(per):.3f} +/- {np.std(per, ddof=1):.3f} "
              f"CI {out['results'][key]['ci95_seed0_well_bootstrap']}")
    ref = "trace_jepa"
    cmp = {k: P.paired_bootstrap(preds0[ref][0], preds0[ref][1], preds0[k][1],
                                 preds0[ref][2]) for k in feats if k != ref}
    q = P.bh([c["p_two_sided"] for c in cmp.values()])
    for c, qq in zip(cmp.values(), q, strict=False):
        c["q_bh"] = qq
    out["paired_vs_trace_jepa"] = cmp
    return out


def power(ts: tay.TraceSet) -> dict:
    from scipy import stats
    v = ts.layout_verified
    sel = v & (ts.second == B1.LPS) & np.isin(ts.first, [B1.LPS, B1.NAIVE]) & (ts.gap == 8.0)
    sub = ts.subset(sel)
    win, _, _ = tay.response_window(sub, 8.0)
    win = np.nan_to_num(win, nan=0.0)
    g = sub.group
    tol = sub.first == B1.LPS
    summ = {"peak_amplitude": win.max(1), "auc": np.trapezoid(win, dx=tay.FRAME_MIN, axis=1)}
    out = {"wells": {}, "metrics": {}}
    for name, val in summ.items():
        wm = {w: float(val[g == w].mean()) for w in np.unique(g)}
        wt = {w: bool(tol[g == w][0]) for w in wm}
        a = np.array([wm[w] for w in wm if wt[w]])
        b = np.array([wm[w] for w in wm if not wt[w]])
        sd = float(np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                           / (len(a) + len(b) - 2)))
        df = len(a) + len(b) - 2
        sd_hi = float(sd * np.sqrt(df / stats.chi2.ppf(0.2, df)))
        diff = float(a.mean() - b.mean())

        def n_for(d: float) -> int | None:
            if d == 0 or not np.isfinite(d):
                return None
            for n in range(2, 10_000):
                dfn = 2 * n - 2
                tc = stats.t.ppf(0.975, dfn)
                nc = abs(d) * np.sqrt(n / 2)
                pw = 1 - stats.nct.cdf(tc, dfn, nc) + stats.nct.cdf(-tc, dfn, nc)
                if pw >= 0.8:
                    return n
            return None
        out["metrics"][name] = {
            "per_well_mean": wm, "tolerant_wells": [w for w in wm if wt[w]],
            "tolerant_minus_naive": diff, "pooled_between_well_sd": sd, "df": df,
            "effect_size_d": diff / sd if sd > 0 else None,
            "wells_per_arm_80pct_power": n_for(diff / sd) if sd > 0 else None,
            "sd_upper_80pct": sd_hi,
            "wells_per_arm_80pct_power_at_sd_upper": n_for(diff / sd_hi) if sd_hi > 0 else None}
        print(f"  power {name}: diff {diff:.3f} sd {sd:.3f} d {diff / sd:.2f} -> n/arm "
              f"{out['metrics'][name]['wells_per_arm_80pct_power']} "
              f"(sd upper: {out['metrics'][name]['wells_per_arm_80pct_power_at_sd_upper']})")
    out["test"] = "two-sided two-sample t-test on well means, alpha 0.05, power 0.8"
    out["caveat"] = ("the between-well SD rests on 2 wells per arm (df 2); the sd-upper row "
                     "shows how sensitive the answer is to that")
    return out


def main() -> int:
    t0 = time.time()
    ts = tay.load(REPO)
    sub, labels = b32_cells(ts)
    groups = sub.group
    eval_wells = set(groups)
    tol = ts.layout_verified & (ts.second == B1.LPS) & np.isin(ts.first, [B1.LPS, B1.NAIVE]) \
        & (ts.gap == 8.0)
    excl = eval_wells | set(ts.group[tol])
    pool_mask = ~np.isin(ts.group, list(excl))
    pool = pad(ts.traces[pool_mask])
    print(f"B3.2: {len(sub)} cells, {len(eval_wells)} wells; pretraining pool "
          f"{len(pool)} cells from {len(set(ts.group[pool_mask]))} other wells")
    m, info = pretrain(pool)
    info["pool_wells"] = int(len(set(ts.group[pool_mask])))
    info["excluded_wells"] = int(len(excl))
    x = pad(sub.traces)
    torch.manual_seed(1)
    rnd = TraceJEPA(length=L).to(DEV).eval()
    feats = {"raw_trace": np.nan_to_num(sub.traces, nan=0.0),
             "handcrafted": trace_features(sub.traces)[0],
             "trace_jepa": embed(m, x), "trace_jepa_random_init": embed(rnd, x)}
    res = {"phase": "PLAN_v3 H5.2 (findings R14, R15)",
           "trace_jepa_pretraining": info,
           "b32_ligand_identity": b32(ts, feats, labels, groups),
           "b31_power_analysis": power(ts),
           "seconds": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
