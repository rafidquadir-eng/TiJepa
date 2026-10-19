"""H5.1.3-5: replication of the beta-glucan TNF observation and the G-BetaGlucan gate
(PLAN_v3 §9, §2; finding R13).

Observation (GSE212282, v2): beta-glucan-trained human macrophages restimulated with LPS
for 4 h at day 6 have lower TNF mRNA than untrained ones in all 3 donors (about -2.1 log2).

Replication criteria (fixed before looking at the data): public bulk RNA, beta-glucan (or
BCG) training, >= 5-day rest, restimulation with a TLR ligand or microbe at a stated
time point, >= 3 donors. Candidates named in the plan, verified from GEO:

* GSE141656 (Radboud; 3 donors x {RPMI, beta-glucan} x {medium, M. tuberculosis 4 h} at
  day 6; MMSEQ gene files) - meets every criterion; the restimulus is Mtb, not LPS.
* GSE242947 (beta-glucan "ETSU", zymosan, depleted zymosan, untrained; LPS 0/1/3 h; 2
  replicates) - fails the >= 3 donor criterion; reported descriptively only.
* GSE85246 - a SuperSeries of epigenome/RNA maps of differentiation; its RNA subseries
  (GSE85243, already used in v1) has no restimulation arm, so it cannot test this.

Each dataset gets the restimulation positive control (restimulated untrained cells must
exceed unstimulated untrained cells for TNF, IL6 and IL1B in every donor/replicate) and the
same TNF/IL6/IL1B trained-minus-untrained differences. GSE141656's donors are compared
with GSE212282's by the HLA-DRB5 / GSTM1 genotype calls, because both use donor letters A-C
from the same group and a shared donor would make the replication non-independent.

G-BetaGlucan: pass iff the GSE212282 samples pass identity QC (sample_qc.json) AND the TNF
decrease is consistent (trained < untrained in every donor) in at least one independent
dataset meeting the criteria, or the source paper's own RNA data report it.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.omics import signatures as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("restim_v1", REPO / "scripts" /
                                               "43_partC_restim.py")
V1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V1)
_spec2 = importlib.util.spec_from_file_location("qc_v3", REPO / "scripts" /
                                                "46_partC_sample_qc_v3.py")
QC = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(QC)

RESTIM = REPO / "data" / "interim" / "restim"
GEO = REPO / "data" / "raw" / "geo"
HGNC = REPO / "data" / "raw" / "maps" / "hgnc_complete_set.txt"
OUT = REPO / "results" / "v3" / "partC" / "beta_glucan_replication.json"
GENES = ["TNF", "IL6", "IL1B"]


def diffs(mat, pairs) -> dict:
    out = {}
    for g in GENES:
        d = np.array([mat.loc[g, a] - mat.loc[g, b] for a, b in pairs])
        out[g] = {"per_donor_log2_difference": d.tolist(),
                  **{k: v for k, v in V1.sign_flip_test(d).items()
                     if k != "per_donor_difference"},
                  "all_same_sign_negative": bool((d < 0).all()),
                  "all_same_sign_positive": bool((d > 0).all())}
    return out


def gse141656(ens2sym) -> dict:
    paths = {re.sub(r"^GSM\d+_", "", p.name).split(".gene.mmseq")[0]: p
             for p in sorted((RESTIM / "GSE141656").glob("GSM*.gz"))}
    mat = V1.build_matrix(paths) / np.log(2.0)
    mat.index = [str(ens2sym.get(str(i).split(".")[0], "")).upper() for i in mat.index]
    mat = mat[mat.index != ""].groupby(level=0).max()
    donors = ["A", "B", "C"]
    pc = diffs(mat, [(f"{d}_RPMI_MTB", f"{d}_RPMI_RPMI") for d in donors])
    pc_pass = all(v["all_same_sign_positive"] for v in pc.values())
    meta = pd.DataFrame({"donor": [c.split("_")[0] for c in mat.columns],
                         "time": "d6", "arm": ["_".join(c.split("_")[1:]) for c in
                                               mat.columns]}, index=mat.columns)
    geno = QC.genotype_check(mat, meta)
    return {"accession": "GSE141656", "licence": "GEO (public, no restrictions stated)",
            "design": "3 donors x {RPMI, beta-glucan 24 h} x {medium, Mtb 4 h} at day 6",
            "files_md5": {p.name: __import__("hashlib").md5(p.read_bytes()).hexdigest()
                          for p in paths.values()},
            "positive_control_restim_vs_unstim_untrained": pc,
            "positive_control_pass": bool(pc_pass),
            "trained_minus_untrained_after_restim": diffs(
                mat, [(f"{d}_BG_MTB", f"{d}_RPMI_MTB") for d in donors]),
            "trained_minus_untrained_unstimulated": diffs(
                mat, [(f"{d}_BG_RPMI", f"{d}_RPMI_RPMI") for d in donors]),
            "donor_genotype": geno, "meets_criteria": True, "_mat": mat,
            "restimulus": "M. tuberculosis 4 h (not LPS)"}


OVERLAP_GENES = ["HLA-DRB5", "NPIPB15", "RPSAP15", "GSTM1"]


def donor_overlap(m_a, donors_a, m_b, donors_b) -> dict:
    """Presence calls (> -1 log2 after flooring at -5) per donor for genes found, by a
    scan of both datasets, to be on/off within donors (HLA-DRB5, NPIPB15, RPSAP15 with a
    >= 3 log2 gap in both; GSTM1 the classic deletion). A shared donor must match on all."""
    def calls(m, donors):
        out = {}
        for d in sorted(set(donors)):
            cols = [c for c, dd in zip(m.columns, donors, strict=False) if dd == d]
            out[d] = [bool(m.loc[g, cols].clip(lower=-5).median() > -1) for g in OVERLAP_GENES]
        return out
    ca, cb = calls(m_a, donors_a), calls(m_b, donors_b)
    pairs = {f"{x}~{y}": int(sum(u == v for u, v in zip(ca[x], cb[y], strict=False)))
             for x in ca for y in cb}
    return {"genes": OVERLAP_GENES, "GSE141656": ca, "GSE212282": cb,
            "agreeing_genes_per_pair": pairs,
            "possible_shared_donors": [k for k, v in pairs.items()
                                       if v == len(OVERLAP_GENES)]}


def gse242947(ens2sym) -> dict:
    with gzip.open(GEO / "GSE242947" / "GSE242947_Cheng_RNAseqCounts.csv.gz", "rt") as fh:
        df = pd.read_csv(fh)
    counts = df.set_index("Geneid").filter(regex=r"\.(0h|1h|3h)\.R\d$")
    lib = counts.sum()
    logcpm = np.log2(counts / lib * 1e6 + 1)
    logcpm.index = [str(ens2sym.get(str(i).split(".")[0], "")).upper() for i in logcpm.index]
    logcpm = logcpm[logcpm.index != ""].groupby(level=0).max()
    reps = ["R1", "R2"]
    out = {"accession": "GSE242947", "design": "untrained / beta-glucan (ETSU) / zymosan / "
           "depleted zymosan training; LPS 0, 1, 3 h; 2 replicates",
           "meets_criteria": False, "reason": "2 replicates (< 3 donors)",
           "positive_control_restim_vs_unstim_untrained": {
               g: [float(logcpm.loc[g, f"Untrained.3h.{r}"] - logcpm.loc[g, f"Untrained.0h.{r}"])
                   for r in reps] for g in GENES},
           "beta_glucan_minus_untrained": {
               t: {g: [float(logcpm.loc[g, f"ETSU.{t}.{r}"] - logcpm.loc[g, f"Untrained.{t}.{r}"])
                       for r in reps] for g in GENES} for t in ("0h", "1h", "3h")},
           "units": "log2(CPM + 1) differences per replicate"}
    return out


def main() -> int:
    ens2sym = S.load_hgnc_ensembl_map(HGNC)
    qc = json.loads((REPO / "results" / "v3" / "partC" / "sample_qc.json").read_text())
    src = json.loads((REPO / "results" / "v3" / "partC" / "source_rna.json").read_text())
    a = gse141656(ens2sym)
    b = gse242947(ens2sym)
    gq = qc["datasets"]["GSE212282"]["donor_genotype"]["genes"]
    ga = a["donor_genotype"]["genes"]
    common = sorted(g for g in ga if ga[g].get("informative") and gq.get(g, {}).get("informative"))
    g212 = {d: [gq[g]["present_by_donor"][d] for g in common]
            for d in gq[common[0]]["present_by_donor"]} if common else {}
    g141 = {d: [ga[g]["present_by_donor"][d] for g in common]
            for d in ga[common[0]]["present_by_donor"]} if common else {}
    shared = sorted({f"{d1}~{d2}" for d1, c1 in g141.items() for d2, c2 in g212.items()
                     if c1 == c2})
    a["genotype_calls_vs_GSE212282"] = {
        "genes_informative_in_both": common, "GSE141656": g141, "GSE212282": g212,
        "note": "calls over the informative genes of each dataset; identical call vectors "
                "are compatible with, not proof of, a shared donor",
        "donors_with_identical_calls": shared}
    m141 = a.pop("_mat")
    m212, meta212 = QC.load_all("GSE212282", ens2sym)
    a["donor_overlap_with_GSE212282"] = donor_overlap(
        m141, [c.split("_")[0] for c in m141.columns], m212, list(meta212["donor"]))
    shared = a["donor_overlap_with_GSE212282"]["possible_shared_donors"]
    tnf = a["trained_minus_untrained_after_restim"]["TNF"]
    rep_consistent = a["positive_control_pass"] and tnf["all_same_sign_negative"]
    qc_pass = qc["datasets"]["GSE212282"]["identity_qc_pass"]
    gate = {"identity_qc_pass": qc_pass,
            "independent_replication": {"dataset": "GSE141656", "tnf_lower_in_every_donor":
                                        tnf["all_same_sign_negative"],
                                        "positive_control_pass": a["positive_control_pass"],
                                        "consistent": bool(rep_consistent),
                                        "independence_caveat": bool(shared)},
            "source_rna_reports_it": False,
            "source_rna_note": src["what_the_source_reports_for_TNF_mRNA_after_restimulation"]}
    if not qc_pass:
        dec = "withdraw: sample-identity QC failed"
    elif rep_consistent:
        dec = ("pass: report as a replicated transcript-level observation (GSE141656, Mtb "
               "restimulation)")
    else:
        dec = "fail: report as a single-dataset observation (QC passed)"
    gate["decision"] = dec
    res = {"phase": "PLAN_v3 H5.1.3-5", "GSE141656": a, "GSE242947": b,
           "GSE85246": {"meets_criteria": False, "reason": "SuperSeries; its RNA subseries "
                        "GSE85243 has no restimulation arm"},
           "kinetics_hypothesis": "Trained macrophages may reach the TNF transcript peak "
                                  "earlier and resolve it faster, so that a 4 h transcript "
                                  "can be lower while 24 h secreted protein is higher. This "
                                  "is a hypothesis; the test is a 0.5-24 h restimulation "
                                  "time course with paired RNA and protein.",
           "G_BetaGlucan": gate}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"pc141656": a["positive_control_pass"],
                      "tnf141656": tnf["per_donor_log2_difference"],
                      "il6": a["trained_minus_untrained_after_restim"]["IL6"]
                      ["per_donor_log2_difference"],
                      "shared_genotypes": shared,
                      "gse242947": b["beta_glucan_minus_untrained"], "decision": dec},
                     indent=0, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
