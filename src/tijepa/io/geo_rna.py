"""Loader for the GSE260996 mouse BMDM RNA-seq counts (PLAN_v1.md F1.1.3).

GEO ships the processed data as one Rsubread `featureCounts` object per sample inside
`GSE260996_RAW.tar` (`GSM*_<cond><rep>Aligned.RData.gz`). Each object is an R list with
`counts` (genes x 1), `annotation` (GeneID, Chr, Start, End, Strand, Length), `targets`
and `stat`. Counts are keyed by **Entrez GeneID**, which is what distinguishes this
matrix from the ATAC matrix of the same nominal name (interval row IDs); the data-type
guard in `tijepa.omics.guard` enforces that distinction.

These are processed per-sample outputs, not raw reads.

**Independent cross-check.** The Tay-lab code deposit (Zenodo 14337874) also contains
`figure6_RNAseq/FC/merged_FC.csv`, the merged count matrix its own analysis consumed.
`cross_check_merged_fc` compares the matrix assembled from GEO against that file
gene-for-gene and sample-for-sample. Two independently published copies agreeing is a
much stronger integrity statement than either alone, so the check is run and recorded
rather than assumed.

Condition codes (first stimulus then second, 4 h apart per the GEO treatment protocol):
`F` media, `C` CpG, `L` LPS, `T` TNF, `P` polyI:C.
"""
from __future__ import annotations

import gzip
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RE = re.compile(r"^(GSM\d+)_([A-Z]{2})(\d)Aligned\.RData\.gz$")

#: stimulus letter -> human-readable name, from the GEO treatment protocol
STIMULUS = {
    "F": "media",
    "C": "CpG",
    "L": "LPS",
    "T": "TNF",
    "P": "polyIC",
}


@dataclass
class RnaMatrix:
    counts: pd.DataFrame          # Entrez GeneID (str) x sample
    samples: pd.DataFrame         # per-sample metadata
    gene_length: pd.Series        # featureCounts union-exon length, for TPM if needed


def _read_one(path: Path) -> tuple[pd.Series, pd.Series]:
    """Read a single `*Aligned.RData.gz` featureCounts object.

    Returns `(counts, lengths)` indexed by Entrez GeneID as strings.
    """
    import rdata

    with tempfile.NamedTemporaryFile(suffix=".RData", delete=True) as tmp:
        with gzip.open(path, "rb") as fh:
            shutil.copyfileobj(fh, tmp)
        tmp.flush()
        obj = rdata.conversion.convert(rdata.parser.parse_file(tmp.name))

    if "fc" not in obj:
        raise ValueError(f"{path.name}: expected an 'fc' featureCounts object, "
                         f"found keys {list(obj)}")
    fc = obj["fc"]
    arr = np.asarray(fc["counts"])
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError(f"{path.name}: counts has shape {arr.shape}, expected (genes, 1)")
    ann = fc["annotation"]
    gene_ids = [str(g) for g in np.asarray(ann["GeneID"])]
    counts = pd.Series(arr[:, 0].astype(np.int64), index=gene_ids)
    lengths = pd.Series(np.asarray(ann["Length"]).astype(np.int64), index=gene_ids)
    return counts, lengths


def load_counts(raw_tar: Path, work_dir: Path) -> RnaMatrix:
    """Assemble the gene x sample count matrix from `GSE260996_RAW.tar`."""
    work_dir.mkdir(parents=True, exist_ok=True)
    members = sorted(p.name for p in work_dir.glob("GSM*Aligned.RData.gz"))
    if not members:
        with tarfile.open(raw_tar) as tf:
            tf.extractall(work_dir, filter="data")
        members = sorted(p.name for p in work_dir.glob("GSM*Aligned.RData.gz"))

    cols: dict[str, pd.Series] = {}
    rows: list[dict] = []
    lengths: pd.Series | None = None
    for name in members:
        m = SAMPLE_RE.match(name)
        if not m:
            raise ValueError(f"unexpected member name {name!r} in {raw_tar.name}")
        gsm, cond, rep = m.groups()
        label = f"{cond}_{rep}"
        counts, lens = _read_one(work_dir / name)
        if lengths is None:
            lengths = lens
        elif not lengths.index.equals(lens.index):
            raise ValueError(f"{name}: gene order differs from the first sample")
        cols[label] = counts
        rows.append({
            "sample": label,
            "gsm": gsm,
            "condition": cond,
            "replicate": int(rep),
            "stim1": STIMULUS[cond[0]],
            "stim2": STIMULUS[cond[1]],
            "label": f"{STIMULUS[cond[0]]}->{STIMULUS[cond[1]]}",
        })

    counts_df = pd.DataFrame(cols)
    counts_df = counts_df[sorted(counts_df.columns)]
    samples = pd.DataFrame(rows).set_index("sample").loc[counts_df.columns]
    # .loc with a plain Index drops the index name, which would silently rename the
    # column to "index" on reset_index().
    samples.index.name = "sample"
    samples = samples.reset_index()
    assert lengths is not None
    return RnaMatrix(counts=counts_df, samples=samples, gene_length=lengths)


def cross_check_merged_fc(counts: pd.DataFrame, merged_fc_csv: Path) -> dict:
    """Compare the GEO-assembled matrix against the deposit's own `merged_FC.csv`."""
    if not merged_fc_csv.exists():
        return {"status": "skipped", "reason": f"{merged_fc_csv} not present"}
    dep = pd.read_csv(merged_fc_csv, index_col=0)
    dep["gene"] = dep["gene"].astype(str)
    dep = dep.set_index("gene")
    dep.columns = [c.replace("_", "_") for c in dep.columns]

    shared_genes = counts.index.intersection(dep.index)
    shared_cols = [c for c in counts.columns if c in dep.columns]
    if not len(shared_genes) or not shared_cols:
        return {"status": "no_overlap",
                "n_shared_genes": int(len(shared_genes)),
                "n_shared_samples": len(shared_cols)}
    a = counts.loc[shared_genes, shared_cols].to_numpy(dtype=np.int64)
    b = dep.loc[shared_genes, shared_cols].to_numpy(dtype=np.int64)
    n_diff = int((a != b).sum())
    return {
        "status": "identical" if n_diff == 0 else "mismatch",
        "source": str(merged_fc_csv),
        "n_shared_genes": int(len(shared_genes)),
        "n_shared_samples": len(shared_cols),
        "n_differing_cells": n_diff,
        "max_abs_difference": int(np.abs(a - b).max()) if n_diff else 0,
    }


def entrez_to_symbol(gene_info_gz: Path) -> pd.Series:
    """Entrez GeneID -> official symbol, from the NCBI `Mus_musculus.gene_info` table."""
    df = pd.read_csv(gene_info_gz, sep="\t", compression="gzip",
                     usecols=["GeneID", "Symbol"], dtype={"GeneID": str})
    return pd.Series(df["Symbol"].to_numpy(), index=df["GeneID"].to_numpy())
