"""G4.2-G4.3 (PLAN_v2.md §7; findings N9, N10): B3.3 statistic and B4 rebuilt.

### B3.3 - a statistic that is not biased against itself

v1 counted targets with leave-one-condition-out (LOO) r > 0.5. With 5 conditions LOO
correlations are negatively biased (each held-out value pulls the training mean the
other way), visible as a below-zero null in the accessibility analysis. v2 reports,
per target, the **predictive R-squared against the LOO mean predictor**

    Q2 = 1 - sum_i (y_i - yhat_i)^2 / sum_i (y_i - mean(y_{-i}))^2,

so a target scores above 0 only if the ridge prediction beats predicting the mean of
the other conditions under the *same* folds. The summary is the fraction of targets
with Q2 > 0, against the exact permutation null over condition labels (5! = 120).
Both analyses stay descriptive.

### B4 - an interaction test, the closer human contrast, and an ortholog control

* Mouse (GSE260996 counts): DESeq2 with `~ priming + challenge + priming:challenge` over
  FF, FL, CF, CL, PF, PL. A **priming-blunted LPS gene** is LPS-induced in the naive arm
  (padj < 0.05, log2FC > 1) **and** has a significantly negative priming x LPS
  interaction (padj < 0.05). Counted for CpG, for polyI:C, and for both.
* Human primary: **GSE185433**, LPS-tolerised -> LPS against untrained -> LPS, donor-paired
  moderated test (3 donors): homologous tolerance with restimulation. Resting GSE58310
  (LPS-tolerised vs control, no restimulation) is the secondary.
* Ortholog positive control: Tnf, Il1b, Il6, Cxcl2, Ccl4, Nfkbia, Tnfaip3 must map to their
  human symbols, or the analysis stops.
* Fisher's exact test in both directions, with the expected overlap printed. Exploratory.
"""
from __future__ import annotations

import functools
import gzip
import importlib.util
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

OUT = REPO / "results" / "v2" / "partB"
GEO = REPO / "data" / "raw" / "geo"
INTERIM = REPO / "data" / "interim" / "tay"
RESTIM = REPO / "data" / "interim" / "restim"
MAPS = REPO / "data" / "raw" / "maps"

ORTHOLOG_CONTROL = {"Tnf": "TNF", "Il1b": "IL1B", "Il6": "IL6", "Cxcl2": "CXCL2",
                    "Ccl4": "CCL4", "Nfkbia": "NFKBIA", "Tnfaip3": "TNFAIP3"}


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V1 = _load("b33_v1", "26_partB_dynamics_omics.py")
RESTIM_MOD = _load("restim_v1", "43_partC_restim.py")


# ------------------------------------------------------------------------- B3.3


def loo_q2(x: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """Per-target Q2 of LOO ridge against the LOO mean predictor (same folds)."""
    n = x.shape[0]
    pred = np.zeros_like(y)
    mean_pred = np.zeros_like(y)
    for i in range(n):
        tr = np.arange(n) != i
        pred[i] = Ridge(alpha=alpha).fit(x[tr], y[tr]).predict(x[i:i + 1])[0]
        mean_pred[i] = y[tr].mean(axis=0)
    ss_model = ((y - pred) ** 2).sum(axis=0)
    ss_mean = ((y - mean_pred) ** 2).sum(axis=0)
    ok = ss_mean > 1e-12
    q2 = np.full(y.shape[1], np.nan)
    q2[ok] = 1.0 - ss_model[ok] / ss_mean[ok]
    return q2


def q2_summary(x: np.ndarray, y: np.ndarray) -> dict:
    q2 = loo_q2(x, y)
    ok = np.isfinite(q2)
    obs = float((q2[ok] > 0).mean())
    perms = [p for p in itertools.permutations(range(x.shape[0]))
             if list(p) != list(range(x.shape[0]))]
    null = []
    for p in perms:
        qp = loo_q2(x, y[list(p)])
        okp = np.isfinite(qp)
        null.append(float((qp[okp] > 0).mean()))
    null_arr = np.asarray(null)
    return {
        "statistic": "fraction of targets with Q2 > 0 (LOO ridge vs LOO mean predictor)",
        "n_targets": int(ok.sum()),
        "fraction_q2_positive": obs,
        "median_q2": float(np.median(q2[ok])),
        "exact_permutation": {
            "n_permutations": len(null), "exact": True,
            "null_mean": float(null_arr.mean()),
            "null_p95": float(np.quantile(null_arr, 0.95)),
            "p_value": float((np.sum(null_arr >= obs) + 1) / (len(null_arr) + 1)),
            "minimum_achievable_p": float(1.0 / (len(null_arr) + 1))},
    }


def b33(ts: tay.TraceSet) -> dict:
    feats = V1.dynamics_features(ts)
    counts, sym = V1.load_rna()
    atac = V1.load_atac()
    guard.check_matrix(counts.index, counts.columns, guard.GSE260996_RNA)
    guard.check_matrix(atac.index, atac.columns, guard.GSE261647_ATAC)
    cpm = np.log2(counts / counts.sum(axis=0) * 1e6 + 1.0)
    out = {"name": "B3.3 condition-level NF-kB dynamics -> omics",
           "labelled": "DESCRIPTIVE: five matched conditions", "analyses": {}}
    for label, mat in (("rna_expression", cpm), ("chromatin_accessibility", atac)):
        key = 0 if label == "rna_expression" else 1
        cols_for = {p: [c for c in mat.columns if c.startswith(V1.CONDITION_MAP[p][key])]
                    for p in feats}
        pairs = sorted(p for p, cs in cols_for.items() if cs)
        x = np.vstack([feats[p] for p in pairs])
        y = np.vstack([mat[cols_for[p]].to_numpy(dtype=float).mean(axis=1) for p in pairs])
        top = np.argsort(-y.var(axis=0))[:2000]
        res = q2_summary(x, y[:, top])
        rec = {"target": label, "n_matched_conditions": len(pairs),
               "matched_conditions": [V1.CONDITION_MAP[p][key] for p in pairs],
               "top2000_variable_targets": res, "confidence_interval": None,
               "confidence_interval_reason": f"{len(pairs)} conditions (rule 3)"}
        if label == "rna_expression":
            names = [str(sym.get(i, "")) for i in mat.index]
            rows = [i for i, nm in enumerate(names) if nm in set(V1.NFKB_PANEL)]
            q = loo_q2(x, y[:, rows])
            rec["prespecified_nfkb_panel"] = {
                "per_gene_q2": {names[i]: (None if not np.isfinite(v) else float(v))
                                for i, v in zip(rows, q, strict=True)},
                "fraction_q2_positive": float((q[np.isfinite(q)] > 0).mean())}
        out["analyses"][label] = rec
        print(f"  B3.3 {label}: {len(pairs)} conditions, fraction Q2>0 "
              f"{res['fraction_q2_positive']:.3f} (null mean "
              f"{res['exact_permutation']['null_mean']:.3f}), "
              f"p={res['exact_permutation']['p_value']:.3f}")
    return out


# --------------------------------------------------------------------------- B4


def orthologs() -> dict[str, set[str]]:
    orth: dict[str, set[str]] = {}
    h = pd.read_csv(MAPS / "HOM_MouseHumanSequence.rpt", sep="\t", low_memory=False)
    key = next(c for c in h.columns if "DB Class Key" in c)
    sym_c = next(c for c in h.columns if c.strip() == "Symbol")
    org_c = next(c for c in h.columns if "Common Organism Name" in c)
    for _k, grp in h.groupby(key):
        mice = {str(s) for s, o in zip(grp[sym_c], grp[org_c], strict=False)
                if "mouse" in str(o)}
        hums = {str(s).upper() for s, o in zip(grp[sym_c], grp[org_c], strict=False)
                if "human" in str(o)}
        for m in mice:
            orth.setdefault(m, set()).update(hums)
    return orth


def ortholog_control(orth: dict[str, set[str]]) -> dict:
    rows = {m: sorted(orth.get(m, set())) for m in ORTHOLOG_CONTROL}
    ok = all(h in orth.get(m, set()) for m, h in ORTHOLOG_CONTROL.items())
    rec = {"genes": rows, "status": "PASS" if ok else "FAIL"}
    if not ok:
        raise SystemExit(f"ortholog positive control failed: {rows}")
    return rec


def mouse_interaction() -> dict:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    counts, sym = V1.load_rna()
    counts.index = counts.index.astype(str)
    sym.index = sym.index.astype(str)
    samples = pd.read_csv(INTERIM / "rnaseq_samples.csv")
    use = samples[samples["condition"].isin(["FF", "FL", "CF", "CL", "PF", "PL"])]
    # level names chosen so the formula's alphabetical reference is the naive arm
    prim = {"F": "0none", "C": "CpG", "P": "polyIC"}
    chal = {"F": "0none", "L": "LPS"}
    meta = pd.DataFrame({"priming": [prim[c[0]] for c in use["condition"]],
                         "challenge": [chal[c[1]] for c in use["condition"]]},
                        index=use["sample"].to_numpy())
    sub = counts[use["sample"].tolist()].T.astype(int)
    sub = sub.loc[:, sub.sum(axis=0) >= 10]
    dds = DeseqDataSet(counts=sub, metadata=meta,
                       design="~priming + challenge + priming:challenge",
                       refit_cooks=True, quiet=True, n_cpus=4)
    dds.deseq2()
    cols = list(dds.obsm["design_matrix"].columns)

    def coef(name: str) -> pd.DataFrame:
        v = np.zeros(len(cols))
        v[cols.index(name)] = 1.0
        st = DeseqStats(dds, contrast=v, quiet=True, n_cpus=4)
        st.summary()
        return st.results_df

    lps = coef("challenge[T.LPS]")
    inter = {"CpG": coef("priming[T.CpG]:challenge[T.LPS]"),
             "polyIC": coef("priming[T.polyIC]:challenge[T.LPS]")}
    induced = lps[(lps["padj"] < 0.05) & (lps["log2FoldChange"] > 1)].index
    blunted = {}
    for k, r in inter.items():
        rr = r.reindex(induced)
        blunted[k] = set(rr[(rr["padj"] < 0.05) & (rr["log2FoldChange"] < 0)].index)
    both = blunted["CpG"] & blunted["polyIC"]
    to_sym = {g: str(sym.get(g, "")) for g in sub.columns}
    syms = {k: sorted({to_sym[g] for g in v} - {"", "nan"})
            for k, v in {**blunted, "both": both}.items()}
    print(f"  B4 mouse: {len(induced)} LPS-induced (naive arm); blunted by CpG "
          f"{len(blunted['CpG'])}, polyI:C {len(blunted['polyIC'])}, both {len(both)}")
    return {"design": "~ priming + challenge + priming:challenge (pydeseq2), conditions "
                      "FF FL CF CL PF PL, 3 replicates each",
            "design_columns": cols,
            "rule": "LPS-induced in the naive arm (padj < 0.05, log2FC > 1) AND priming x "
                    "LPS interaction padj < 0.05 with log2FC < 0",
            "n_lps_induced": int(len(induced)),
            "n_blunted": {k: len(v) for k, v in {**blunted, "both": both}.items()},
            "symbols": syms,
            "induced_symbols": sorted({to_sym[g] for g in induced} - {"", "nan"}),
            "expressed_universe_symbols": sorted({to_sym[g] for g in sub.columns}
                                                 - {"", "nan"})}


def human_gse185433(ens2sym: dict[str, str]) -> dict:
    files = {}
    for d in (1, 2, 3):
        for arm in ("LPS_Restim", "RPMI_Restim"):
            hit = sorted((RESTIM / "GSE185433").glob(f"GSM*_{arm}_don{d}.gene.mmseq.txt.gz"))
            if hit:
                files[(arm, d)] = hit[0]
    donors = sorted({d for (_a, d) in files if ("LPS_Restim", d) in files
                     and ("RPMI_Restim", d) in files})
    mat = pd.DataFrame({f"{a}_{d}": RESTIM_MOD.read_mmseq(p) for (a, d), p in files.items()})
    mat = mat / np.log(2.0)  # MMSEQ log_mu is a natural log
    expressed = (mat > np.log2(1.0)).all(axis=1) & np.isfinite(mat).all(axis=1)
    mat = mat[expressed]
    res = moderated.paired_moderated_test(
        mat[[f"LPS_Restim_{d}" for d in donors]].to_numpy(),
        mat[[f"RPMI_Restim_{d}" for d in donors]].to_numpy())
    sym = np.array([str(ens2sym.get(str(i).split(".")[0], "")).upper() for i in mat.index])
    down = sorted({s for s, q, c in zip(sym, res.q_value, res.coefficient, strict=True)
                   if s and q < 0.05 and c < 0})
    up = sorted({s for s, q, c in zip(sym, res.q_value, res.coefficient, strict=True)
                 if s and q < 0.05 and c > 0})
    print(f"  B4 human GSE185433: {len(donors)} donors, prior df {res.d0:.1f}, "
          f"{len(down)} down / {len(up)} up at FDR < 0.05")
    return {"dataset": "GSE185433", "contrast": "LPS-tolerised -> LPS restimulation vs "
            "untrained (RPMI) -> LPS restimulation, donor-paired moderated test",
            "samples": {f"{a}_{d}": p.name for (a, d), p in files.items()},
            "donors": donors, "prior_df": float(res.d0),
            "n_expressed_genes": int(len(mat)), "down": down, "up": up,
            "universe": sorted(set(sym) - {""})}


def human_gse58310(ens2sym: dict[str, str]) -> dict:
    with gzip.open(GEO / "GSE58310" / "suppl" / "GSE58310_GeneExpression.csv.gz", "rt") as fh:
        hdf = pd.read_csv(fh, index_col=0)
    cols = S.parse_gse58310_columns(list(hdf.columns))
    tol = {c["donor"]: c["column"] for c in cols if c.get("state") == "LPS_tolerised"}
    ctl = {c["donor"]: c["column"] for c in cols if c.get("state") == "control"}
    donors = sorted(set(tol) & set(ctl))
    lv = np.log2(hdf.to_numpy(dtype=float) + 1.0)
    ci = {c: i for i, c in enumerate(hdf.columns)}
    res = moderated.paired_moderated_test(lv[:, [ci[tol[d]] for d in donors]],
                                          lv[:, [ci[ctl[d]] for d in donors]])
    sym = np.array([str(ens2sym.get(str(i).split(".")[0], "")).upper() for i in hdf.index])
    down = sorted({s for s, q, c in zip(sym, res.q_value, res.coefficient, strict=True)
                   if s and q < 0.05 and c < 0})
    print(f"  B4 human GSE58310 (secondary, resting): {len(donors)} donors, "
          f"{len(down)} down")
    return {"dataset": "GSE58310", "contrast": "LPS-tolerised vs control at day 6, resting "
            "(no restimulation), donor-paired moderated test", "donors": donors,
            "prior_df": float(res.d0), "down": down, "universe": sorted(set(sym) - {""})}


def overlap(mouse_syms: list[str], mouse_universe: list[str], human: dict,
            orth: dict[str, set[str]]) -> dict:
    """Fisher test on the ortholog universe expressed on both sides, both directions."""
    m_h = {h for m in mouse_syms for h in orth.get(m, set())}
    uni = {h for m in mouse_universe for h in orth.get(m, set())} & set(human["universe"])
    a_set = m_h & uni
    b_set = set(human["down"]) & uni
    k = len(a_set & b_set)
    table = [[k, len(a_set) - k], [len(b_set) - k, len(uni) - len(a_set) - len(b_set) + k]]
    odds, p_enr = stats.fisher_exact(table, alternative="greater")
    _, p_dep = stats.fisher_exact(table, alternative="less")
    expected = len(a_set) * len(b_set) / max(1, len(uni))
    return {"universe": len(uni), "n_mouse_set_in_universe": len(a_set),
            "n_human_down_in_universe": len(b_set), "n_overlap": k,
            "expected_overlap": float(expected), "odds_ratio": float(odds),
            "p_enrichment": float(p_enr), "p_depletion": float(p_dep),
            "overlap_genes": sorted(a_set & b_set)}


def b4() -> dict:
    orth = orthologs()
    ctrl = ortholog_control(orth)
    ens2sym = S.load_hgnc_ensembl_map(MAPS / "hgnc_complete_set.txt")
    mouse = mouse_interaction()
    prim = human_gse185433(ens2sym)
    sec = human_gse58310(ens2sym)
    tests = {}
    for hname, h in (("GSE185433_primary", prim), ("GSE58310_secondary", sec)):
        for mset in ("CpG", "polyIC", "both"):
            t = overlap(mouse["symbols"][mset], mouse["expressed_universe_symbols"], h, orth)
            tests[f"{hname}:{mset}"] = t
            print(f"    {hname:20s} {mset:7s} overlap {t['n_overlap']} "
                  f"(expected {t['expected_overlap']:.1f}), OR {t['odds_ratio']:.2f}, "
                  f"p_enr {t['p_enrichment']:.3g}, p_dep {t['p_depletion']:.3g}")
    for h in (prim, sec):
        h.pop("universe")
    mouse.pop("expressed_universe_symbols")
    return {"name": "B4 cross-species: priming-blunted mouse LPS genes vs human tolerance",
            "labelled": "EXPLORATORY",
            "preregistered_mismatch": (
                "Mouse is HETEROLOGOUS priming (CpG or polyI:C, then LPS, 4 h apart, BMDM); "
                "human is HOMOLOGOUS tolerance (LPS then LPS, monocyte-derived "
                "macrophages). The two contrasts are not the same biology."),
            "ortholog_positive_control": ctrl, "mouse": mouse,
            "human_primary": prim, "human_secondary": sec, "fisher": tests}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = tay.load(REPO)
    res = {"phase": "G4.2-G4.3", "b33": b33(ts), "b4": b4()}
    (OUT / "b33_b4.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(f"wrote {OUT / 'b33_b4.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
