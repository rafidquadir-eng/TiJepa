"""F4 7.4 (B3.3) and 7.5 (B4): dynamics -> omics, and the cross-species contrast.

### 7.4 Two analyses, each labelled for what it actually is

v0 ran one analysis and called it "dynamics -> transcriptome" while feeding it the ATAC
accessibility matrix (review finding D1). v1 runs both, separately and correctly named:

  (a) condition-level NF-kB dynamics -> **RNA expression** (GSE260996), and
  (b) condition-level NF-kB dynamics -> **chromatin accessibility** (GSE261647).

**Condition matching uses stimulus identity and timing**, as 7.4 requires. The GEO
treatment protocol for GSE260996 states "Stimulus A ... After 4 hours, Stimulus A was
aspirated and Stimulus B ... added", so the omics inter-stimulus interval is **4 h** and
only the 4 h trace arm may be matched to it. That excludes LPS->LPS, which exists in the
traces only at 8 h and has no counterpart in the RNA series at all.

GSE261647's protocol independently confirms the ligand code mapping of F1.2: "LPS
(1ng/mL), CpG (100nM), polyI:C (1ug/mL), or TNF (10ng/mL)" matches the deposit's own
code comments (code 2 = 1 ng LPS, 1 = 100 nM CpG, 3 = 1000 ng/mL PIC, 6 = 10 ng/mL TNF)
exactly. Two independent sources agreeing on all four codes is much stronger than the
v0 inference.

The time at which RNA and ATAC were collected after stimulus B is **not stated** in
either GEO record, so matching is on stimulus identity and interval only; the residual
timing uncertainty is recorded rather than papered over.

With a handful of matched conditions this is **descriptive**, and is labelled so. A
pre-specified NF-kB target panel is analysed separately, and the condition-label
permutation is exact.

### 7.5 B4 rebuilt on RNA

Mouse side, from **RNA** (not ATAC, as in v0): genes induced by LPS in the naive arm
(media->LPS vs media->media) whose induction is *reduced* when the cells were primed
first (CpG->LPS and polyI:C->LPS), i.e. an interaction contrast, via DESeq2 on counts.
Human side: GSE58310 LPS-tolerised vs control with a moderated **paired** test, donors
shared. Orthologs from the MGI table, overlap tested with Fisher's exact test.

The two contrasts are not the same biology and that is pre-registered here rather than
discovered afterwards: mouse is **heterologous priming** (CpG or polyI:C, then LPS, 4 h
apart) while human is **homologous tolerance** (LPS, then LPS, 6 days apart). The test
is exploratory and a null result is the expected outcome of comparing them.
"""
from __future__ import annotations

import functools
import gzip
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sklearn.linear_model import Ridge  # noqa: E402

from tijepa.io import tay  # noqa: E402
from tijepa.omics import guard, moderated  # noqa: E402
from tijepa.omics import signatures as S  # noqa: E402

OUT = REPO / "results" / "v1" / "partB"
GEO = REPO / "data" / "raw" / "geo"
INTERIM = REPO / "data" / "interim" / "tay"
DEPOSIT = INTERIM / "tay-lab-macrophage_memory-5242e14"
MAPS = REPO / "data" / "raw" / "maps"

OMICS_INTERVAL_H = 4.0

# trace (first, second) code pair -> (RNA condition, ATAC condition)
# codes: 0 naive/media, 1 CpG, 2 LPS, 3 polyI:C, 6 TNF   (sourced; see F1.2)
CONDITION_MAP = {
    (0.0, 2.0): ("FL", "ML"),      # media -> LPS
    (0.0, 6.0): ("FT", "MT"),      # media -> TNF
    (1.0, 2.0): ("CL", "CL"),      # CpG   -> LPS
    (1.0, 6.0): ("CT", "CT"),      # CpG   -> TNF
    (3.0, 2.0): ("PL", "PL"),      # polyIC-> LPS
    (3.0, 6.0): ("PT", "PT"),      # polyIC-> TNF
}

NFKB_PANEL = ["Tnf", "Il1b", "Il6", "Cxcl2", "Ccl4", "Nfkbia", "Tnfaip3",
              "Il12b", "Cxcl10", "Ifit1", "Irf7"]
HUMAN_PANEL = [g.upper() for g in NFKB_PANEL]


def dynamics_features(ts: tay.TraceSet) -> dict[tuple, np.ndarray]:
    """Condition-level mean dynamics features at the 4 h interval."""
    v = ts.layout_verified & (ts.gap == OMICS_INTERVAL_H)
    sub = ts.subset(v)
    out: dict[tuple, np.ndarray] = {}
    for pair in CONDITION_MAP:
        m = (sub.first == pair[0]) & (sub.second == pair[1])
        if m.sum() < 20:
            continue
        s2 = sub.subset(m)
        win, _, _ = tay.response_window(s2, OMICS_INTERVAL_H, length=30)
        w = np.nan_to_num(win, nan=0.0)
        feats = np.concatenate([
            w.mean(axis=0),                       # the mean trajectory
            [w.max(axis=1).mean(), w.sum(axis=1).mean(),
             w.argmax(axis=1).mean(), w.std(axis=1).mean(),
             float(m.sum())],
        ])
        out[pair] = feats
    return out


def leave_one_condition_out_ridge(x: np.ndarray, y: np.ndarray) -> dict:
    """Predict condition-level omics from condition-level dynamics, LOO over conditions."""
    n = x.shape[0]
    pred = np.zeros_like(y)
    for i in range(n):
        tr = np.arange(n) != i
        mdl = Ridge(alpha=10.0).fit(x[tr], y[tr])
        pred[i] = mdl.predict(x[i:i + 1])[0]
    r = []
    for j in range(y.shape[1]):
        if np.std(y[:, j]) < 1e-9 or np.std(pred[:, j]) < 1e-9:
            continue
        r.append(float(np.corrcoef(y[:, j], pred[:, j])[0, 1]))
    return {"n_conditions": int(n), "n_features_out": int(y.shape[1]),
            "per_target_r": r,
            "mean_r": float(np.mean(r)) if r else None,
            "median_r": float(np.median(r)) if r else None,
            "n_targets_r_gt_0.5": int(sum(1 for v in r if v > 0.5))}


def exact_condition_permutation(x: np.ndarray, y: np.ndarray, max_perm: int = 10000
                                ) -> dict:
    """Permute condition labels exactly when n! is small enough."""
    n = x.shape[0]
    perms = list(itertools.permutations(range(n)))
    exact = len(perms) <= max_perm
    if not exact:
        rng = np.random.default_rng(0)
        perms = [tuple(rng.permutation(n)) for _ in range(max_perm)]
    obs = leave_one_condition_out_ridge(x, y)["n_targets_r_gt_0.5"]
    null = []
    for p in perms:
        if list(p) == list(range(n)):
            continue
        null.append(leave_one_condition_out_ridge(x, y[list(p)])["n_targets_r_gt_0.5"])
    null_arr = np.asarray(null, dtype=float)
    return {
        "statistic": "number of targets with LOO r > 0.5",
        "observed": int(obs),
        "exact": bool(exact),
        "n_permutations": int(len(null)),
        "null_mean": float(null_arr.mean()),
        "null_p95": float(np.quantile(null_arr, 0.95)),
        "p_value": float((np.sum(null_arr >= obs) + 1) / (len(null_arr) + 1)),
        "minimum_achievable_p": float(1.0 / (len(null_arr) + 1)),
    }


def load_rna():
    counts = pd.read_csv(INTERIM / "rnaseq_counts.csv", index_col=0)
    sym = counts.pop("symbol")
    return counts, sym


def load_atac():
    p = GEO / "GSE261647" / "suppl" / "GSE261647_logCPM.csv.gz"
    return pd.read_csv(p, index_col=0)


def b33(ts: tay.TraceSet) -> dict:
    feats = dynamics_features(ts)
    counts, sym = load_rna()
    atac = load_atac()
    guard.check_matrix(counts.index, counts.columns, guard.GSE260996_RNA)
    guard.check_matrix(atac.index, atac.columns, guard.GSE261647_ATAC)

    # condition-level means
    cpm = np.log2(counts / counts.sum(axis=0) * 1e6 + 1.0)
    out: dict = {
        "name": "B3.3 condition-level NF-kB dynamics -> omics",
        "primary_endpoint": False,
        "labelled": "DESCRIPTIVE: the matched-condition count is small",
        "interval_matching": (
            "The GSE260996 treatment protocol states stimulus A for 4 h then stimulus "
            "B, so the omics interval is 4 h and only the 4 h trace arm is matched. "
            "LPS->LPS exists in the traces only at 8 h and is absent from the RNA "
            "series entirely, so it cannot enter this analysis."),
        "collection_time_after_stimulus_b": (
            "NOT STATED in either GEO record; matching is on stimulus identity and "
            "interval only, and this residual timing uncertainty is a limitation."),
        "ligand_code_cross_check": (
            "GSE261647's protocol gives LPS 1 ng/mL, CpG 100 nM, polyI:C 1 ug/mL, TNF "
            "10 ng/mL, matching the deposit's own code comments exactly and "
            "independently confirming the F1.2 mapping."),
        "analyses": {},
    }

    for label, mat, kind in (("rna_expression", cpm, "RNA expression (GSE260996)"),
                             ("chromatin_accessibility", atac,
                              "chromatin accessibility (GSE261647)")):
        key = 0 if label == "rna_expression" else 1
        pairs = [p for p in feats if CONDITION_MAP[p][key] in
                 {c.split("_")[0] if "_" in c else c[:2] for c in mat.columns}]
        # condition-level mean across replicates
        cols_for = {}
        for p in feats:
            cond = CONDITION_MAP[p][key]
            cs = [c for c in mat.columns if c.startswith(cond)]
            if cs:
                cols_for[p] = cs
        pairs = sorted(cols_for)
        if len(pairs) < 3:
            out["analyses"][label] = {
                "status": "too_few_matched_conditions",
                "n_matched": len(pairs)}
            continue
        x = np.vstack([feats[p] for p in pairs])
        y = np.vstack([mat[cols_for[p]].to_numpy(dtype=float).mean(axis=1)
                       for p in pairs])
        # keep the most variable targets, so the LOO correlation is meaningful
        var = y.var(axis=0)
        top = np.argsort(-var)[:2000]
        res = leave_one_condition_out_ridge(x, y[:, top])
        perm = exact_condition_permutation(x, y[:, top])

        # pre-specified panel, analysed separately
        if label == "rna_expression":
            names = [str(sym.get(i, "")) for i in mat.index]
            rows = [i for i, nm in enumerate(names) if nm in set(NFKB_PANEL)]
        else:
            rows = []
        panel = None
        if rows:
            panel = leave_one_condition_out_ridge(x, y[:, rows])
            panel["genes"] = sorted({names[i] for i in rows})

        out["analyses"][label] = {
            "status": "ok",
            "target": kind,
            "matched_conditions": [
                {"trace_codes": [p[0], p[1]],
                 "trace_label": f"{tay.LIGAND_CODES.get(p[0], 'naive')}->"
                                f"{tay.LIGAND_CODES.get(p[1])}",
                 "omics_condition": CONDITION_MAP[p][key],
                 "omics_columns": cols_for[p],
                 "n_cells": int(feats[p][-1])}
                for p in pairs],
            "n_matched_conditions": len(pairs),
            "loo_ridge_top2000_variable_targets": res,
            "exact_condition_permutation": perm,
            "prespecified_nfkb_panel": panel,
            "confidence_interval": None,
            "confidence_interval_reason":
                f"{len(pairs)} conditions is far below the 10-group minimum (rule 3)",
        }
        print(f"  B3.3 {label}: {len(pairs)} matched conditions, "
              f"{res['n_targets_r_gt_0.5']} of {res['n_features_out']} targets r>0.5 "
              f"(null mean {perm['null_mean']:.1f}), p={perm['p_value']:.3f}")
        if panel:
            print(f"        pre-specified NF-kB panel ({len(panel['genes'])} genes): "
                  f"mean r {panel['mean_r']:+.3f}")
    return out


def b4() -> dict:
    """Cross-species: mouse priming-suppressed LPS genes vs human LPS tolerance."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    counts, sym = load_rna()
    samples = pd.read_csv(INTERIM / "rnaseq_samples.csv")

    def contrast(a: str, b: str) -> pd.DataFrame:
        sel = samples[samples["condition"].isin([a, b])]
        sub = counts[sel["sample"].tolist()]
        meta = pd.DataFrame({"condition": sel["condition"].to_numpy()},
                            index=sel["sample"].to_numpy())
        dds = DeseqDataSet(counts=sub.T.astype(int), metadata=meta,
                           design="~condition", refit_cooks=True, quiet=True)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["condition", a, b], quiet=True)
        st.summary()
        r = st.results_df.copy()
        r.index = sub.index
        return r

    fl_ff = contrast("FL", "FF")          # LPS induction in the naive arm
    cl_cf = contrast("CL", "CF")          # LPS induction after CpG priming
    pl_pf = contrast("PL", "PF")          # LPS induction after polyI:C priming

    induced = fl_ff[(fl_ff["padj"] < 0.05) & (fl_ff["log2FoldChange"] > 1)].index
    reduced = [g for g in induced
               if g in cl_cf.index and g in pl_pf.index
               and float(cl_cf.loc[g, "log2FoldChange"]) <
               float(fl_ff.loc[g, "log2FoldChange"])
               and float(pl_pf.loc[g, "log2FoldChange"]) <
               float(fl_ff.loc[g, "log2FoldChange"])]
    mouse_syms = sorted({str(sym.get(g, "")) for g in reduced} - {"", "nan"})
    print(f"  B4 mouse: {len(induced)} LPS-induced genes, {len(reduced)} with reduced "
          f"induction after BOTH primings -> {len(mouse_syms)} symbols")

    # ---- human side: GSE58310 LPS-tolerised vs day-6 control, paired by donor
    expr = GEO / "GSE58310" / "suppl" / "GSE58310_GeneExpression.csv.gz"
    with gzip.open(expr, "rt") as fh:
        hdf = pd.read_csv(fh, index_col=0)
    cols = S.parse_gse58310_columns(list(hdf.columns))
    ens2sym = S.load_hgnc_ensembl_map(MAPS / "hgnc_complete_set.txt")
    tol = {c["donor"]: c["column"] for c in cols
           if c.get("state") == "LPS_tolerised"}
    ctl = {c["donor"]: c["column"] for c in cols if c.get("state") == "control"}
    donors = sorted(set(tol) & set(ctl))
    lv = np.log2(hdf.to_numpy(dtype=float) + 1.0)
    cmap = {c: i for i, c in enumerate(hdf.columns)}
    res = moderated.paired_moderated_test(
        lv[:, [cmap[tol[d]] for d in donors]],
        lv[:, [cmap[ctl[d]] for d in donors]])
    hsym = np.array([str(ens2sym.get(str(i).split(".")[0], "")).upper()
                     for i in hdf.index])
    human_down = sorted({s for s, q, c in zip(hsym, res.q_value, res.coefficient,
                                              strict=False)
                         if s and q < 0.05 and c < 0})
    human_up = sorted({s for s, q, c in zip(hsym, res.q_value, res.coefficient,
                                            strict=False)
                       if s and q < 0.05 and c > 0})
    print(f"  B4 human: {len(donors)} donor pairs, prior df {res.d0:.1f}, "
          f"{len(human_down)} down / {len(human_up)} up at FDR<0.05")

    # ---- orthologs and overlap
    orth: dict[str, set[str]] = {}
    hom = MAPS / "HOM_MouseHumanSequence.rpt"
    if hom.exists():
        h = pd.read_csv(hom, sep="\t", low_memory=False)
        key = next((c for c in h.columns if "DB Class Key" in c), None)
        sym_c = next((c for c in h.columns if c.strip() == "Symbol"), None)
        org_c = next((c for c in h.columns if "Common Organism Name" in c), None)
        if key and sym_c and org_c:
            for _k, grp in h.groupby(key):
                mice = {str(s) for s, o in zip(grp[sym_c], grp[org_c], strict=False)
                        if "mouse" in str(o)}
                hums = {str(s).upper() for s, o in zip(grp[sym_c], grp[org_c],
                                                       strict=False)
                        if "human" in str(o)}
                for m in mice:
                    orth.setdefault(m, set()).update(hums)
    mouse_as_human = sorted({h for m in mouse_syms for h in orth.get(m, set())})
    overlap = sorted(set(mouse_as_human) & set(human_down))
    universe = len({h for m in sym.dropna().astype(str) for h in orth.get(m, set())})
    table = [[len(overlap), len(mouse_as_human) - len(overlap)],
             [len(human_down) - len(overlap),
              max(0, universe - len(mouse_as_human) - len(human_down) + len(overlap))]]
    odds, p = stats.fisher_exact(table, alternative="greater")

    print(f"  B4 overlap: {len(overlap)} genes, Fisher p={p:.3g} (OR {odds:.2f})")
    return {
        "name": "B4 cross-species: mouse priming-suppressed LPS genes vs human LPS tolerance",
        "primary_endpoint": False,
        "labelled": "EXPLORATORY",
        "preregistered_mismatch": (
            "The two contrasts are deliberately not the same biology: mouse is "
            "HETEROLOGOUS priming (CpG or polyI:C, then LPS, 4 h apart, BMDM) while "
            "human is HOMOLOGOUS tolerance (LPS, then LPS, 6 days apart, monocyte-"
            "derived macrophages). A null result is the expected outcome of comparing "
            "them and is not evidence about either one."),
        "mouse_source": "GSE260996 RNA-seq counts, DESeq2 (v0 used the ATAC matrix here)",
        "mouse_n_lps_induced": int(len(induced)),
        "mouse_n_reduced_by_both_primings": int(len(reduced)),
        "mouse_symbols": mouse_syms[:500],
        "human_source": "GSE58310, moderated paired test (limma method), donor-paired",
        "human_n_donor_pairs": len(donors),
        "human_donors": donors,
        "human_prior_df": float(res.d0),
        "human_n_down_fdr05": len(human_down),
        "human_n_up_fdr05": len(human_up),
        "n_mouse_genes_with_human_ortholog": len(mouse_as_human),
        "ortholog_universe": int(universe),
        "overlap_genes": overlap,
        "n_overlap": len(overlap),
        "fisher_odds_ratio": float(odds),
        "fisher_p_one_sided": float(p),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = tay.load(REPO)
    res = {"phase": "F4 7.4-7.5", "b33": b33(ts), "b4": b4()}
    (OUT / "b33_b4.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(f"wrote {OUT/'b33_b4.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
