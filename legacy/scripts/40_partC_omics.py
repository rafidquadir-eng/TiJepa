"""M3 / Part C omics side (PLAN.md 7, C1-C2). HYPOTHESIS-GENERATING BRIDGE.

C1  human anchor signatures from GSE46903 (GM-CSF macrophages, 72 h: LPS, IFN-g, IL-4
    vs matched control)
C2  rank-based scoring of beta-glucan-trained and LPS-tolerised states in GSE58310,
    expressed relative to that dataset's own macrophage baseline, with donor bootstrap

Nothing here pairs images with omics. Part C is a condition-level, cross-study,
cross-species bridge and is labelled as such everywhere it appears.
"""
from __future__ import annotations

import csv
import functools
import gzip
import json
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.omics import signatures as S  # noqa: E402

OUT = REPO / "results" / "partC"
GEO = REPO / "data" / "raw" / "geo"
ANCHORS = {"upLPS": "S_LPS", "IFNg": "S_IFNg", "IL4": "S_IL4"}
ANCHOR_TIME = "72h"
DIFFERENTIATION = "GM-CSF"
N_SIG = 100
FDR = 0.05


def c1_anchors() -> dict:
    mat = next((GEO / "GSE46903" / "matrix").glob("*series_matrix.txt.gz"))
    ids, gsms, vals = S.read_series_matrix_table(mat)
    print(f"C1: GSE46903 matrix {vals.shape[0]} probes x {vals.shape[1]} samples")

    meta = {r["gsm"]: r for r in csv.DictReader(
        open(REPO / "results" / "omics" / "GSE46903_samples.csv"))}
    groups: dict[str, list[int]] = {}
    donors: dict[str, list[str]] = {}
    for j, g in enumerate(gsms):
        ch = S.parse_characteristics(meta.get(g, {}).get("characteristics_ch1", ""))
        if ch.get("initial differentiation") != DIFFERENTIATION:
            continue
        if ch.get("time") != ANCHOR_TIME:
            continue
        stim = ch.get("activation stimuli")
        if stim in ANCHORS or stim == "con":
            groups.setdefault(stim, []).append(j)
            donors.setdefault(stim, []).append(ch.get("donor", "?"))

    annot = S.load_gpl_annotation(GEO / "platforms" / "GPL6947.annot.gz")
    syms, X = S.collapse_to_symbols(ids, vals, annot)
    print(f"C1: {len(syms)} gene symbols after probe collapse")

    sigs, detail = {}, {}
    ctrl = groups.get("con", [])
    for stim, name in ANCHORS.items():
        idx = groups.get(stim, [])
        if len(idx) < 2 or len(ctrl) < 2:
            detail[name] = {"status": "skipped", "n_stim": len(idx), "n_ctrl": len(ctrl)}
            continue
        t, p, diff = S.welch_de(X[:, idx], X[:, ctrl])
        sig = S.top_up_signature(syms, t, p, diff, n=N_SIG, fdr=FDR)
        sigs[name] = sig
        detail[name] = {
            "status": "ok", "stimulus": stim, "n_stim": len(idx), "n_ctrl": len(ctrl),
            "donors_stim": sorted(set(donors.get(stim, []))),
            "donors_ctrl": sorted(set(donors.get("con", []))),
            "time": ANCHOR_TIME, "differentiation": DIFFERENTIATION,
            "n_signature_genes": len(sig),
            "top20": sig[:20],
            "median_log2_diff_of_signature": float(np.nanmedian(
                [diff[syms.index(g)] for g in sig])) if sig else None,
        }
        print(f"C1: {name} from {len(idx)} {stim} vs {len(ctrl)} con -> "
              f"{len(sig)} genes; top: {', '.join(sig[:6])}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c1_signatures.json").write_text(json.dumps({
        "source": "GSE46903 (in SuperSeries GSE47189), Xue et al. 2014, platform GPL6947",
        "selection": {"initial differentiation": DIFFERENTIATION, "time": ANCHOR_TIME,
                      "baseline_stimulus": "con"},
        "selection_note": (
            "72 h is the only time point at which ultrapure LPS (upLPS) samples exist in "
            "this series, so all three anchors are taken at 72 h for a matched comparison."),
        "method": "Welch t per gene on log2 array values, top 100 up-regulated by t",
        "signatures": sigs, "detail": detail,
    }, indent=2) + "\n")
    return {"signatures": sigs, "symbols": syms, "detail": detail}


def c2_score_trained_tolerant(sigs: dict[str, list[str]]) -> dict:
    path = GEO / "GSE58310" / "suppl" / "GSE58310_GeneExpression.csv.gz"
    with gzip.open(path, "rt") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        ens_ids, rows = [], []
        for r in rd:
            ens_ids.append(r[0])
            rows.append([np.nan if v in ("", "NA") else float(v) for v in r[1:]])
    X = np.asarray(rows, dtype=np.float64)
    cols = S.parse_gse58310_columns(header[1:])
    print(f"C2: GSE58310 {X.shape[0]} genes x {X.shape[1]} samples")

    emap = S.load_hgnc_ensembl_map(REPO / "data" / "raw" / "maps" / "hgnc_complete_set.txt")
    if not emap:
        return {"status": "failed", "reason": "no Ensembl->symbol map available"}
    syms, Xs = S.collapse_to_symbols(ens_ids, X, emap)
    print(f"C2: {len(syms)} symbols after Ensembl mapping")

    scores = {name: S.singscore(Xs, syms, sig) for name, sig in sigs.items()}
    states = [c.get("state", "unparsed") for c in cols]
    donors = [c.get("donor", "?") for c in cols]

    # express relative to this dataset's own 6d macrophage control baseline
    base_idx = [i for i, s in enumerate(states) if s == "control"]
    per_sample = []
    for i, c in enumerate(cols):
        row = {"column": c["column"], "state": states[i], "donor": donors[i]}
        for name in sigs:
            row[name] = float(scores[name][i])
            row[f"{name}_rel_baseline"] = float(
                scores[name][i] - np.nanmean(scores[name][base_idx]))
        per_sample.append(row)

    with (OUT / "c2_scores_per_sample.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        w.writerows(per_sample)

    # state-level coordinates with donor bootstrap
    rng = np.random.default_rng(0)
    coords = {}
    for st in sorted({s for s in states if s != "unparsed"}):
        idx = [i for i, s in enumerate(states) if s == st]
        m1 = np.nanmean([[scores[n][i] for i in idx] for n in ("S_LPS", "S_IFNg")
                         if n in scores], axis=0)
        m2 = np.asarray([scores["S_IL4"][i] for i in idx]) if "S_IL4" in scores else None
        b1 = np.nanmean([[scores[n][i] for i in base_idx] for n in ("S_LPS", "S_IFNg")
                         if n in scores], axis=0)
        b2 = np.asarray([scores["S_IL4"][i] for i in base_idx]) if m2 is not None else None
        a = float(np.nanmean(m1) - np.nanmean(b1))
        b = float(np.nanmean(m2) - np.nanmean(b2)) if m2 is not None else float("nan")
        boots_a, boots_b = [], []
        for _ in range(2000):
            k = rng.choice(len(idx), len(idx), replace=True)
            kb = rng.choice(len(base_idx), len(base_idx), replace=True)
            boots_a.append(np.nanmean(m1[k]) - np.nanmean(b1[kb]))
            if m2 is not None:
                boots_b.append(np.nanmean(m2[k]) - np.nanmean(b2[kb]))
        coords[st] = {
            "n_samples": len(idx), "donors": sorted(set(donors[i] for i in idx)),
            "M1_like": a, "M1_like_ci": [float(np.quantile(boots_a, 0.025)),
                                         float(np.quantile(boots_a, 0.975))],
            "M2_like": b,
            "M2_like_ci": [float(np.quantile(boots_b, 0.025)),
                           float(np.quantile(boots_b, 0.975))] if boots_b else None,
        }
        print(f"C2: {st:22s} M1-like {a:+.4f} M2-like {b:+.4f} (n={len(idx)})")

    out = {
        "status": "ok",
        "framing": "HYPOTHESIS-GENERATING: condition-level, cross-study, cross-platform; "
                   "no sample is paired with any image",
        "source": "GSE58310 (Saeed et al. 2014) processed GeneExpression.csv",
        "n_genes_mapped": len(syms),
        "baseline_state": "control (6d RPMI macrophage)",
        "axes": {"M1_like": "mean of S_LPS and S_IFNg rank scores",
                 "M2_like": "S_IL4 rank score"},
        "coordinates": coords,
        "per_sample_csv": "results/partC/c2_scores_per_sample.csv",
        "replication": {
            "GSE85243": "NOT USABLE: the GEO series matrix contains no expression table "
                        "(table_begin immediately followed by table_end) and no processed "
                        "supplementary matrix is deposited",
            "GSE172116": "NOT USABLE for gene-level scoring: only bigWig/bedGraph coverage "
                         "tracks are deposited, no gene-level matrix",
        },
    }
    (OUT / "c2_scores.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> int:
    c1 = c1_anchors()
    if not c1["signatures"]:
        print("C1 produced no signatures; skipping C2")
        return 1
    c2_score_trained_tolerant(c1["signatures"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
