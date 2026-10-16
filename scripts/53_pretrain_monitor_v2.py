"""G1.3 monitor: collapse/regression check on the latest checkpoint of each run.

Appends one row per run to results/v2/pretrain_monitor.csv: step, embedding std and
effective rank on a fixed BBBC054 subset, the latest spatial and future losses from the
training log, and a fast 3-class BBBC054 linear-probe balanced accuracy on a fixed
~1,000-cell subset (FOV-grouped 5-fold CV).

This is a monitor only: it never chooses a checkpoint, and its numbers are not results.
It reads the checkpoint through a hard link, so the trainer's atomic replace cannot
change the file under it.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from tijepa.models.video_jepa import VideoJEPA, collapse_diagnostics  # noqa: E402

OUT = REPO / "results" / "v2" / "pretrain_monitor.csv"
SUBSET = REPO / "data" / "interim" / "monitor_bbbc054_subset.npz"
CK = REPO / "checkpoints"
FIELDS = ["utc", "run", "step", "std_mean", "effective_rank", "loss_spatial",
          "loss_future", "probe3_balanced_accuracy", "n_cells"]


def bbbc():
    spec = importlib.util.spec_from_file_location(
        "bbbc", REPO / "scripts" / "62_bbbc054_v1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def subset(mod) -> dict:
    if SUBSET.exists():
        d = np.load(SUBSET, allow_pickle=False)
        return {k: d[k] for k in d.files}
    d = mod.build_dataset(1100, seed=12345)
    out = {"X": d["X"], "labels": d["labels"].astype(str), "fovs": d["fovs"]}
    SUBSET.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SUBSET, **out)
    return out


def last_losses(run: str, step: int) -> tuple[str, str]:
    p = CK / f"pretrain_log_{run}_seed0.csv"
    if not p.exists():
        return "", ""
    rows = [r for r in csv.DictReader(p.open()) if int(r["step"]) <= step]
    return (rows[-1]["loss_spatial"], rows[-1]["loss_future"]) if rows else ("", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["tijepa1", "tijepa1_spatial"])
    a = ap.parse_args()
    mod = bbbc()
    d = subset(mod)
    clips = mod.static_clips(d["X"])
    new = not OUT.exists()
    with OUT.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for run in a.runs:
            src = CK / f"{run}_T1_seed0.pt"
            if not src.exists():
                continue
            link = CK / f".monitor_{run}.pt"
            if link.exists():
                link.unlink()
            os.link(src, link)
            try:
                sd = torch.load(link, map_location="cpu", weights_only=False)
            finally:
                link.unlink()
            m = VideoJEPA(clip=(16, 96, 96))
            m.load_state_dict(sd["model"])
            m = m.eval().to(mod.DEV)
            z = mod.embed_jepa(m, clips, batch=64)
            diag = collapse_diagnostics(torch.from_numpy(z))
            y, g = d["labels"], d["fovs"]
            pred = np.empty(len(y), dtype=object)
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(z, y, g):
                clf = make_pipeline(StandardScaler(), LogisticRegression(
                    max_iter=2000, class_weight="balanced"))
                clf.fit(z[tr], y[tr])
                pred[te] = clf.predict(z[te])
            ls, lf = last_losses(run, int(sd["step"]))
            row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run": run,
                   "step": int(sd["step"]), "std_mean": round(diag["std_mean"], 5),
                   "effective_rank": round(diag["effective_rank"], 3),
                   "loss_spatial": ls, "loss_future": lf,
                   "probe3_balanced_accuracy": round(float(
                       balanced_accuracy_score(y, pred.astype(str))), 4),
                   "n_cells": len(y)}
            w.writerow(row)
            fh.flush()
            print(row, flush=True)
            del m, sd
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
