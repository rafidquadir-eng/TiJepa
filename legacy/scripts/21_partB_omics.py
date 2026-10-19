"""M2b / PLAN.md B3.3 + B4: dynamics -> transcriptome, and the cross-species overlap.

B3.3 regresses condition-level expression from condition-mean trace embeddings with
     leave-one-condition-out ridge, against a condition-label permutation null.
B4   derives mouse tolerance genes from the Tay deposit and human tolerance genes from
     GSE58310, maps them with the MGI ortholog table, and tests the overlap.

The deposited Tay RNA-seq (logCPM.csv, 18 samples = 9 conditions x 2 replicates) has
NO LPS->LPS condition, so B3.3 covers heterologous priming only; the homologous LPS
tolerance contrast that B3.1 classifies has no matching transcriptome in the deposit.
That limitation is recorded in the output and the paper.
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

from tijepa.eval.stats import benjamini_hochberg  # noqa: E402
from tijepa.io import tay  # noqa: E402
from tijepa.models.trace_jepa import trace_features  # noqa: E402
from tijepa.omics import signatures as S  # noqa: E402

OUT = REPO / "results" / "partB"
GEO = REPO / "data" / "raw" / "geo"

# logCPM.csv column codes: first letter = first stimulus, second = second stimulus.
# M = media (no first stimulus), C = CpG, P = PIC (poly(I:C)), L = LPS, T = TNF.
RNA_CODE = {"M": "media", "C": "CpG", "P": "PIC", "L": "LPS", "T": "TNF"}


def load_tay_logcpm() -> tuple[list[str], list[str], np.ndarray]:
    path = next((REPO / "data" / "interim" / "tay").glob("*/logCPM.csv"))
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        genes, rows = [], []
        for r in rd:
            genes.append(r[0].strip('"'))
            rows.append([np.nan if v in ("", "NA") else float(v) for v in r[1:]])
    return genes, [h.strip('"') for h in header[1:]], np.asarray(rows, float)


def parse_rna_condition(col: str) -> dict:
    core = "".join(ch for ch in col if ch.isalpha())
    rep = "".join(ch for ch in col if ch.isdigit())
    if len(core) != 2:
        return {"column": col, "parsed": False}
    return {"column": col, "parsed": True, "replicate": rep,
            "first": RNA_CODE.get(core[0], core[0]),
            "second": RNA_CODE.get(core[1], core[1]), "code": core}


def b33_dynamics_to_expression() -> dict:
    genes, cols, X = load_tay_logcpm()
    conds = [parse_rna_condition(c) for c in cols]
    print(f"B3.3: logCPM {X.shape[0]} genes x {X.shape[1]} samples; "
          f"codes {sorted({c['code'] for c in conds if c['parsed']})}")

    ts = tay.load(REPO)
    ts = ts.subset(ts.layout_verified)
    lig_name = {2.0: "LPS", 6.0: "TNF", 3.0: "CpG"}

    # Build condition-mean trace embeddings for conditions that exist in BOTH modalities.
    # Trace side: (first ligand name, second ligand name) with first==0 -> "media".
    emb_by_cond: dict[tuple[str, str], np.ndarray] = {}
    ncells: dict[tuple[str, str], int] = {}
    for gap in sorted(set(ts.gap)):
        sub0 = ts.subset(ts.gap == gap)
        if len(sub0) == 0:
            continue
        win, i0, i1 = tay.response_window(sub0, gap)
        ok = np.isfinite(win).all(axis=1)
        sub0, win = sub0.subset(ok), win[ok]
        if len(sub0) == 0:
            continue
        feats, _ = trace_features(win)
        for f in np.unique(sub0.first):
            for s in np.unique(sub0.second):
                m = (sub0.first == f) & (sub0.second == s)
                if m.sum() < 30:
                    continue
                fname = "media" if f == tay.NAIVE_CODE else lig_name.get(f)
                sname = lig_name.get(s)
                if fname is None or sname is None:
                    continue
                key = (fname, sname)
                if key in emb_by_cond:
                    continue
                emb_by_cond[key] = feats[m].mean(axis=0)
                ncells[key] = int(m.sum())

    rna_by_cond: dict[tuple[str, str], list[int]] = {}
    for j, c in enumerate(conds):
        if c["parsed"]:
            rna_by_cond.setdefault((c["first"], c["second"]), []).append(j)

    shared = sorted(set(emb_by_cond) & set(rna_by_cond))
    print(f"B3.3: trace conditions {len(emb_by_cond)}, RNA conditions "
          f"{len(rna_by_cond)}, shared {len(shared)}: {shared}")
    if len(shared) < 4:
        return {
            "status": "underpowered_skipped",
            "reason": f"only {len(shared)} conditions have both traces and RNA-seq; "
                      "leave-one-condition-out ridge needs at least 4",
            "shared_conditions": [list(k) for k in shared],
            "trace_conditions": [list(k) for k in sorted(emb_by_cond)],
            "rna_conditions": [list(k) for k in sorted(rna_by_cond)],
            "no_lps_lps_note": "the deposited RNA-seq has no LPS->LPS sample, so the "
                               "homologous tolerance contrast of B3.1 cannot be "
                               "regressed onto expression",
        }

    Xe = np.stack([emb_by_cond[k] for k in shared])
    Y = np.stack([np.nanmean(X[:, rna_by_cond[k]], axis=1) for k in shared])
    var = np.nanvar(Y, axis=0)
    top = np.argsort(-var)[:2000]
    Yt = Y[:, top]
    gene_names = [genes[i] for i in top]

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    def loco_r(Xin: np.ndarray, Yin: np.ndarray) -> np.ndarray:
        pred = np.zeros_like(Yin)
        for tr, te in LeaveOneOut().split(Xin):
            sc = StandardScaler().fit(Xin[tr])
            m = Ridge(alpha=10.0).fit(sc.transform(Xin[tr]), Yin[tr])
            pred[te] = m.predict(sc.transform(Xin[te]))
        return np.array([
            np.corrcoef(Yin[:, g], pred[:, g])[0, 1]
            if np.std(Yin[:, g]) > 1e-9 and np.std(pred[:, g]) > 1e-9 else np.nan
            for g in range(Yin.shape[1])])

    r_obs = loco_r(Xe, Yt)
    rng = np.random.default_rng(0)
    null_counts = []
    for _ in range(200):
        perm = rng.permutation(len(shared))
        r_null = loco_r(Xe, Yt[perm])
        null_counts.append(int(np.nansum(r_null > 0.5)))
    n_obs = int(np.nansum(r_obs > 0.5))
    p = (1.0 + sum(1 for v in null_counts if v >= n_obs)) / (1.0 + len(null_counts))

    order = np.argsort(-np.nan_to_num(r_obs, nan=-1))
    out = {
        "status": "ok",
        "n_conditions": len(shared),
        "conditions": [list(k) for k in shared],
        "cells_per_condition": {f"{a}->{b}": ncells[(a, b)] for a, b in shared},
        "n_genes": int(Yt.shape[1]),
        "mean_r": float(np.nanmean(r_obs)),
        "median_r": float(np.nanmedian(r_obs)),
        "n_genes_r_above_0.5": n_obs,
        "permutation_null_mean_count": float(np.mean(null_counts)),
        "permutation_p": float(p),
        "n_permutations": len(null_counts),
        "top50_best_predicted_genes": [gene_names[i] for i in order[:50]],
        "underpowered_note": (
            f"leave-one-condition-out over {len(shared)} conditions is severely "
            "underpowered; reported as specified in PLAN.md B3.3"),
        "no_lps_lps_note": "the deposited RNA-seq contains no LPS->LPS sample",
    }
    print(f"B3.3: mean r {out['mean_r']:+.3f}, {n_obs} genes r>0.5 "
          f"(null {np.mean(null_counts):.1f}), p={p:.3f}")
    return out


def b4_cross_species() -> dict:
    # --- human tolerance genes: GSE58310 LPS-tolerised vs 6d RPMI control
    path = GEO / "GSE58310" / "suppl" / "GSE58310_GeneExpression.csv.gz"
    with gzip.open(path, "rt") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        ens, rows = [], []
        for r in rd:
            ens.append(r[0])
            rows.append([np.nan if v in ("", "NA") else float(v) for v in r[1:]])
    Xh = np.asarray(rows, float)
    cols = S.parse_gse58310_columns(header[1:])
    emap = S.load_hgnc_ensembl_map(REPO / "data" / "raw" / "maps" / "hgnc_complete_set.txt")
    syms, Xs = S.collapse_to_symbols(ens, Xh, emap)
    # The design is donor-paired (the same four buffy-coat donors appear in every
    # condition), so a paired test is the correct analysis; an unpaired Welch test
    # throws away the pairing and is badly underpowered at n=4.
    tol_by_donor = {c["donor"]: i for i, c in enumerate(cols)
                    if c.get("state") == "LPS_tolerised"}
    ctl_by_donor = {c["donor"]: i for i, c in enumerate(cols)
                    if c.get("state") == "control"}
    donors = sorted(set(tol_by_donor) & set(ctl_by_donor))
    tol = [tol_by_donor[d] for d in donors]
    ctl = [ctl_by_donor[d] for d in donors]
    # Standard expressed-gene prefilter: 46% of values in this matrix sit at the
    # detection floor, and testing undetected genes only inflates the FDR burden.
    expressed = (np.nanmean(Xs[:, tol], axis=1) > -6.0) | (
        np.nanmean(Xs[:, ctl], axis=1) > -6.0)
    keep = np.flatnonzero(expressed)
    syms = [syms[i] for i in keep]
    Xs = Xs[keep]
    print(f"B4: expressed-gene filter keeps {len(keep)} of {len(expressed)} genes")
    from scipy import stats as _sps
    t, p = _sps.ttest_rel(Xs[:, tol], Xs[:, ctl], axis=1, nan_policy="omit")
    t, p = np.asarray(t, float), np.asarray(p, float)
    diff = np.nanmean(Xs[:, tol] - Xs[:, ctl], axis=1)
    print(f"B4: paired on {len(donors)} donors: {donors}")
    q = np.ones_like(p)
    fin = np.isfinite(p)
    q[fin] = benjamini_hochberg(p[fin])
    human_up = {syms[i] for i in np.flatnonzero((q < 0.05) & (diff > 1))}
    human_dn = {syms[i] for i in np.flatnonzero((q < 0.05) & (diff < -1))}
    # Secondary, NOMINAL (not FDR-controlled) set. With 4 donors the FDR-controlled set
    # is near-empty, so the overlap test is also run on this set and is labelled as
    # exploratory wherever it is reported.
    human_up_nom = {syms[i] for i in np.flatnonzero((p < 0.01) & (diff > 1))}
    human_dn_nom = {syms[i] for i in np.flatnonzero((p < 0.01) & (diff < -1))}
    print(f"B4: nominal p<0.01 set: {len(human_up_nom)} up, {len(human_dn_nom)} down")
    human_measured = set(syms)
    print(f"B4: human GSE58310 tolerised vs control: {len(human_up)} up, "
          f"{len(human_dn)} down (FDR<0.05, |log2FC|>1), n={len(tol)}v{len(ctl)}")

    # --- mouse: Tay logCPM, heterologous primed->LPS vs media->LPS
    genes, mcols, Xm = load_tay_logcpm()
    conds = [parse_rna_condition(c) for c in mcols]
    primed = [j for j, c in enumerate(conds)
              if c["parsed"] and c["second"] == "LPS" and c["first"] != "media"]
    naive = [j for j, c in enumerate(conds)
             if c["parsed"] and c["second"] == "LPS" and c["first"] == "media"]
    mouse_up: set[str] = set()
    mouse_dn: set[str] = set()
    mouse_up_nom: set[str] = set()
    mouse_note = ""
    if len(primed) >= 2 and len(naive) >= 2:
        tm, pm, dm = S.welch_de(Xm[:, primed], Xm[:, naive])
        # (unpaired: the two replicates are separate experiments, not matched donors)
        qm = np.ones_like(pm)
        finm = np.isfinite(pm)
        qm[finm] = benjamini_hochberg(pm[finm])
        mouse_up = {genes[i].upper() for i in np.flatnonzero((qm < 0.05) & (dm > 1))}
        mouse_dn = {genes[i].upper() for i in np.flatnonzero((qm < 0.05) & (dm < -1))}
        mouse_up_nom = {genes[i].upper() for i in np.flatnonzero((pm < 0.01) & (dm > 1))}
        mouse_note = (f"heterologous primed->LPS (n={len(primed)}) vs media->LPS "
                      f"(n={len(naive)}); the deposit has no LPS->LPS sample")
    else:
        mouse_note = "too few samples for a mouse contrast"
    print(f"B4: mouse {len(mouse_up)} up, {len(mouse_dn)} down; {mouse_note}")

    # --- orthologs via MGI
    rpt = REPO / "data" / "raw" / "maps" / "HOM_MouseHumanSequence.rpt"
    by_key: dict[str, dict[str, str]] = {}
    with open(rpt, encoding="utf-8", errors="replace") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        i_key, i_org, i_sym = (head.index("DB Class Key"),
                               head.index("Common Organism Name"), head.index("Symbol"))
        for line in fh:
            pr = line.rstrip("\n").split("\t")
            if len(pr) <= max(i_key, i_org, i_sym):
                continue
            org = "mouse" if "mouse" in pr[i_org].lower() else (
                "human" if "human" in pr[i_org].lower() else None)
            if org:
                by_key.setdefault(pr[i_key], {})[org] = pr[i_sym].upper()
    m2h = {v["mouse"]: v["human"] for v in by_key.values()
           if "mouse" in v and "human" in v}
    print(f"B4: {len(m2h)} mouse->human ortholog pairs from MGI")

    mouse_up_h = {m2h[g] for g in mouse_up if g in m2h}
    universe = human_measured & set(m2h.values())
    overlap = mouse_up_h & human_up & universe
    from scipy import stats as sps

    a = len(overlap)
    b = len(mouse_up_h & universe) - a
    c = len(human_up & universe) - a
    d = len(universe) - a - b - c
    odds, pv = (sps.fisher_exact([[a, b], [c, d]], alternative="greater")
                if min(a, b, c, d) >= 0 and len(universe) > 0 else (float("nan"),) * 2)

    def overlap_test(mouse_set: set[str], human_set: set[str]) -> dict:
        mh = {m2h[g] for g in mouse_set if g in m2h}
        uni = human_measured & set(m2h.values())
        ov = mh & human_set & uni
        aa = len(ov)
        bb = len(mh & uni) - aa
        cc = len(human_set & uni) - aa
        dd = len(uni) - aa - bb - cc
        if min(aa, bb, cc, dd) < 0 or len(uni) == 0:
            return {"status": "not_testable"}
        o, pp = sps.fisher_exact([[aa, bb], [cc, dd]], alternative="greater")
        return {"universe_size": len(uni), "mouse_up_mapped": len(mh & uni),
                "human_up_in_universe": len(human_set & uni), "overlap": aa,
                "odds_ratio": float(o), "p_value": float(pp),
                "overlapping_genes": sorted(ov)}

    nominal = overlap_test(mouse_up_nom, human_up_nom)
    print(f"B4: NOMINAL overlap {nominal.get('overlap')} genes, "
          f"OR {nominal.get('odds_ratio')}, p {nominal.get('p_value')}")

    out = {
        "status": "ok",
        "human": {
            "source": "GSE58310 LPS-tolerised vs 6d RPMI macrophage control",
            "n_tolerised": len(tol), "n_control": len(ctl),
            "paired_donors": donors,
            "test": "paired t-test on donor-matched samples (the design is paired)",
            "criteria": "BH-FDR < 0.05 and |log2FC| > 1",
            "n_up": len(human_up), "n_down": len(human_dn),
            "up_examples": sorted(human_up)[:30],
            "n_up_nominal_p01": len(human_up_nom),
            "n_down_nominal_p01": len(human_dn_nom),
            "n_genes_tested_after_expression_filter": len(syms),
        },
        "mouse": {
            "source": "Tay-lab logCPM.csv (deposit of GSE260996)",
            "contrast": mouse_note,
            "n_primed_samples": len(primed), "n_naive_samples": len(naive),
            "n_up": len(mouse_up), "n_down": len(mouse_dn),
            "up_examples": sorted(mouse_up)[:30],
        },
        "ortholog_source": "MGI HOM_MouseHumanSequence.rpt (downloaded this run)",
        "n_ortholog_pairs": len(m2h),
        "overlap_test": {
            "universe_size": len(universe),
            "mouse_up_mapped": len(mouse_up_h & universe),
            "human_up_in_universe": len(human_up & universe),
            "overlap": a, "odds_ratio": float(odds), "p_value": float(pv),
            "overlapping_genes": sorted(overlap),
            "test": "one-sided Fisher exact, universe = genes measured in both and "
                    "having an MGI ortholog",
        },
        "overlap_test_nominal_exploratory": {
            **nominal,
            "label": "EXPLORATORY: both gene sets are nominal p<0.01, NOT FDR-controlled; "
                     "with 4 human donors and 2 mouse replicates the FDR-controlled sets "
                     "are near-empty, so this test is reported as exploratory only",
        },
        "limitation": (
            "the mouse contrast is heterologous priming (CpG or poly(I:C) then LPS), "
            "not homologous LPS tolerance, because the deposited RNA-seq has no "
            "LPS->LPS sample; the human contrast is homologous LPS tolerance. The two "
            "are therefore not the same biological comparison."),
    }
    print(f"B4: overlap {a} genes, OR {odds:.2f}, p {pv:.3g}")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b33_dynamics_to_expression.json").write_text(
        json.dumps(b33_dynamics_to_expression(), indent=2, default=float) + "\n")
    (OUT / "b4_cross_species.json").write_text(
        json.dumps(b4_cross_species(), indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
