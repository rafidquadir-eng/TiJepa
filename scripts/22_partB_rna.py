"""F1.1 + F4 7.1: load the real GSE260996 RNA-seq and run its positive control.

Three things happen here, in this order, and the order matters:

  1. the gene x sample count matrix is assembled from the GEO processed files;
  2. the **data-type guard** (PLAN_v1.md rule 1) checks it against the GEO record, and
     the ATAC matrix is checked too, so the v0 confusion of the two cannot recur;
  3. the **positive control** (PLAN_v1.md 7.1) tests whether `media->LPS` versus
     `media->media` recovers canonical LPS/NF-kB targets. Per rule 2, nothing
     downstream may report a result unless this passes.

Differential expression uses DESeq2 (`pydeseq2`) on raw counts, which is the test
PLAN_v1.md Section 9 requires for count data.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.io import geo_rna  # noqa: E402
from tijepa.omics import guard  # noqa: E402

OUT = REPO / "results" / "v1" / "partB"
INTERIM = REPO / "data" / "interim" / "tay"
RAW_TAR = REPO / "data" / "raw" / "geo" / "GSE260996" / "suppl" / "GSE260996_RAW.tar"
DEPOSIT = INTERIM / "tay-lab-macrophage_memory-5242e14"
GENE_INFO = REPO / "data" / "raw" / "annotation" / "Mus_musculus.gene_info.gz"

# PLAN_v1.md 7.1: canonical LPS / NF-kB targets. >= 5 of 7 must be up with
# log2FC > 1 and FDR < 0.05 for the control to pass.
POSITIVE_CONTROL_GENES = ["Tnf", "Il1b", "Il6", "Cxcl2", "Ccl4", "Nfkbia", "Tnfaip3"]
PC_MIN_PASS = 5
PC_LFC = 1.0
PC_FDR = 0.05

# PLAN_v1.md 7.4: the pre-specified NF-kB target panel for the dynamics analyses.
NFKB_PANEL = POSITIVE_CONTROL_GENES + ["Il12b", "Cxcl10", "Ifit1", "Irf7"]


def deseq2(counts: pd.DataFrame, samples: pd.DataFrame, group_a: str,
           group_b: str) -> pd.DataFrame:
    """DESeq2 contrast group_a vs group_b (a = numerator, i.e. positive = up in a)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    sel = samples[samples["condition"].isin([group_a, group_b])]
    sub = counts[sel["sample"].tolist()]
    meta = pd.DataFrame({"condition": sel["condition"].to_numpy()},
                        index=sel["sample"].to_numpy())
    # DESeq2 wants samples as rows.
    dds = DeseqDataSet(counts=sub.T.astype(int), metadata=meta,
                       design="~condition", refit_cooks=True, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", group_a, group_b], quiet=True)
    st.summary()
    res = st.results_df.copy()
    res.index = sub.index
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit: dict = {"phase": "F1.1 + F4 7.1"}

    # ---------------------------------------------------------------- 1. load RNA
    if not RAW_TAR.exists():
        (OUT / "rna_positive_control.json").write_text(json.dumps(
            {"status": "skipped", "reason": f"{RAW_TAR} absent"}, indent=2) + "\n")
        print(f"SKIP: {RAW_TAR} absent")
        return 0

    print("loading GSE260996 featureCounts objects ...")
    rna = geo_rna.load_counts(RAW_TAR, INTERIM / "rnaseq_rdata")
    print(f"  counts {rna.counts.shape[0]} genes x {rna.counts.shape[1]} samples")

    # ------------------------------------------------------- 2. data-type guard
    audit["guard_rna"] = guard.check_matrix(
        rna.counts.index, rna.counts.columns, guard.GSE260996_RNA)
    print(f"  guard RNA: PASS ({audit['guard_rna']['row_id_class_found']} row IDs, "
          f"{audit['guard_rna']['n_samples_found']} samples)")

    atac_csv = DEPOSIT / "logCPM.csv"
    if atac_csv.exists():
        atac = pd.read_csv(atac_csv, index_col=0, nrows=2000)
        full_cols = pd.read_csv(atac_csv, index_col=0, nrows=0).columns
        audit["guard_atac"] = guard.check_matrix(
            atac.index, full_cols, guard.GSE261647_ATAC)
        audit["guard_atac"]["note"] = (
            "This is the file the v0 run analysed as if it were RNA-seq. The guard "
            "classifies its row IDs as genomic intervals and its sample count as 18, "
            "both matching GSE261647 ATAC-seq and neither matching GSE260996.")
        print(f"  guard ATAC: PASS ({audit['guard_atac']['row_id_class_found']} row IDs, "
              f"{audit['guard_atac']['n_samples_found']} samples) - correctly ATAC")

    # ------------------------------------------- 2b. independent cross-check
    audit["cross_check_merged_fc"] = geo_rna.cross_check_merged_fc(
        rna.counts, DEPOSIT / "figure6_RNAseq" / "FC" / "merged_FC.csv")
    print(f"  cross-check vs deposit merged_FC.csv: "
          f"{audit['cross_check_merged_fc']['status']} "
          f"({audit['cross_check_merged_fc'].get('n_differing_cells')} differing cells)")

    # symbols
    sym = geo_rna.entrez_to_symbol(GENE_INFO)
    symbols = pd.Series(rna.counts.index.map(sym), index=rna.counts.index)
    audit["n_genes_with_symbol"] = int(symbols.notna().sum())

    interim_dir = REPO / "data" / "interim" / "tay"
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_counts = rna.counts.copy()
    out_counts.insert(0, "symbol", symbols)
    out_counts.to_csv(interim_dir / "rnaseq_counts.csv")
    rna.samples.to_csv(interim_dir / "rnaseq_samples.csv", index=False)
    (interim_dir / "rnaseq_schema.md").write_text(
        "# GSE260996 RNA-seq count matrix\n\n"
        f"- rows: {rna.counts.shape[0]} genes, index = **Entrez GeneID** (string); "
        "column `symbol` = NCBI official symbol from "
        "`Mus_musculus.gene_info.gz`\n"
        f"- columns: {rna.counts.shape[1]} samples named `<condition>_<replicate>`\n"
        "- values: **raw read counts** from Rsubread featureCounts (union exon model), "
        "as deposited by GEO; not normalised\n"
        "- condition codes: first stimulus then second, 4 h apart - "
        "F=media, C=CpG, L=LPS, T=TNF, P=polyI:C\n"
        "- assay: RNA-Seq (GEO library_strategy); this is NOT the ATAC matrix that the "
        "Tay deposit ships as `logCPM.csv`\n"
        f"- cross-check against the deposit's own `merged_FC.csv`: "
        f"{audit['cross_check_merged_fc']['status']}\n")

    # -------------------------------------------------- 3. positive control
    print("\npositive control: media->LPS (FL) vs media->media (FF) by DESeq2 ...")
    res = deseq2(rna.counts, rna.samples, "FL", "FF")
    res["symbol"] = symbols
    res_sym = res.dropna(subset=["symbol"]).set_index("symbol")

    rows = []
    for g in POSITIVE_CONTROL_GENES:
        if g not in res_sym.index:
            rows.append({"gene": g, "status": "absent"})
            continue
        r = res_sym.loc[g]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        lfc = float(r["log2FoldChange"])
        padj = float(r["padj"]) if np.isfinite(r["padj"]) else None
        rows.append({
            "gene": g, "log2FC": lfc, "padj": padj,
            "baseMean": float(r["baseMean"]),
            "passes": bool(lfc > PC_LFC and padj is not None and padj < PC_FDR),
        })
    n_pass = sum(1 for r in rows if r.get("passes"))
    passed = n_pass >= PC_MIN_PASS

    sig = res[(res["padj"] < 0.05) & (res["log2FoldChange"].abs() > 1)]
    pc = {
        "status": "ok",
        "analysis": "GSE260996 RNA-seq, DESeq2, contrast FL (media->LPS) vs FF (media->media)",
        "criterion": f">= {PC_MIN_PASS} of {len(POSITIVE_CONTROL_GENES)} canonical "
                     f"LPS/NF-kB targets with log2FC > {PC_LFC} and FDR < {PC_FDR}",
        "genes": rows,
        "n_passing": n_pass,
        "passed": passed,
        "n_de_genes_total": int(len(sig)),
        "n_up": int((sig["log2FoldChange"] > 0).sum()),
        "n_down": int((sig["log2FoldChange"] < 0).sum()),
        "consequence": (
            "Part B RNA analyses may report results."
            if passed else
            "PLAN_v1.md rule 2: the pipeline failed its positive control, so no Part B "
            "RNA result is reported; the failure is reported instead."),
    }
    (OUT / "rna_positive_control.json").write_text(
        json.dumps(pc, indent=2, default=float) + "\n")
    (OUT / "rna_guard.json").write_text(json.dumps(audit, indent=2, default=float) + "\n")
    res.to_csv(OUT / "rna_FL_vs_FF_deseq2.csv")

    print(f"\n  {'gene':10s} {'log2FC':>8s} {'padj':>10s}  pass")
    for r in rows:
        if r.get("status") == "absent":
            print(f"  {r['gene']:10s} {'absent':>8s}")
        else:
            print(f"  {r['gene']:10s} {r['log2FC']:8.2f} {r['padj']:10.2e}  "
                  f"{'yes' if r['passes'] else 'no'}")
    print(f"\n  POSITIVE CONTROL: {n_pass}/{len(POSITIVE_CONTROL_GENES)} -> "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"  {len(sig)} DE genes total ({pc['n_up']} up, {pc['n_down']} down)")
    print(f"wrote {OUT/'rna_positive_control.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
