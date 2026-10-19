"""Data-type guard for omics matrices (PLAN_v1.md Section 1, rule 1).

Every omics matrix is checked on load against the GEO record it claims to come from:

  * **row-ID class** - gene symbols / Ensembl / Entrez IDs (expression) versus genomic
    intervals `chrN.start.end` or `chrN:start-end` (accessibility);
  * **sample count** against the GEO series;
  * **assay type** against the GEO `library_strategy` (RNA-Seq vs ATAC-seq).

A mismatch raises `DataTypeError`. This is a hard error by design: the v0 run analysed
an ATAC-seq accessibility matrix as if it were RNA-seq expression, because the Tay
deposit ships the ATAC matrix as a bare `logCPM.csv` at its root while the RNA matrix of
the same name lives in `figure6_RNAseq/`. Nothing but a mechanical check on the row IDs
would have caught it, so that check is now mandatory and runs before any analysis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A genomic interval row ID: chr1.4414143.4414644 or chr1:4414143-4414644.
INTERVAL_RE = re.compile(r"^(chr)?[0-9XYMT]+[.:][0-9]+[.\-][0-9]+$", re.IGNORECASE)
ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G[0-9]{6,}", re.IGNORECASE)
ENTREZ_RE = re.compile(r"^[0-9]{1,9}$")
SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._\-]{0,24}$")

#: row-ID classes that represent a gene-level expression measurement
EXPRESSION_CLASSES = frozenset({"symbol", "ensembl_gene", "entrez"})
#: row-ID classes that represent a genomic-interval measurement
INTERVAL_CLASSES = frozenset({"interval"})

#: GEO library_strategy -> the row-ID classes that assay may legitimately produce
ASSAY_ROWS: dict[str, frozenset[str]] = {
    "rna-seq": EXPRESSION_CLASSES,
    " atac-seq".strip(): INTERVAL_CLASSES,
    "dnase-hypersensitivity": INTERVAL_CLASSES,
    "chip-seq": INTERVAL_CLASSES,
    # Microarray series carry no library_strategy; GEO reports them by platform.
    "array": EXPRESSION_CLASSES,
}


class DataTypeError(AssertionError):
    """Raised when a matrix does not match the GEO record it claims to come from."""


def classify_row_id(row_id: str) -> str:
    """Classify a single row identifier. Returns one of the *_CLASSES members."""
    s = str(row_id).strip().strip('"')
    if INTERVAL_RE.match(s):
        return "interval"
    if ENSEMBL_RE.match(s):
        return "ensembl_gene"
    if ENTREZ_RE.match(s):
        return "entrez"
    if SYMBOL_RE.match(s):
        return "symbol"
    return "unknown"


def classify_row_ids(row_ids) -> tuple[str, dict[str, float]]:
    """Classify a whole index by majority vote.

    Returns `(dominant_class, fractions)`. Symbols are only reported as the dominant
    class when they are an outright majority, because a handful of symbol-looking
    strings can otherwise mask an interval matrix.
    """
    ids = [str(r) for r in row_ids]
    if not ids:
        raise DataTypeError("matrix has no rows")
    counts: dict[str, int] = {}
    for r in ids:
        k = classify_row_id(r)
        counts[k] = counts.get(k, 0) + 1
    fracs = {k: v / len(ids) for k, v in counts.items()}
    dominant = max(fracs, key=lambda k: fracs[k])
    return dominant, fracs


@dataclass
class GeoExpectation:
    """What a GEO record says a matrix must look like."""

    accession: str
    library_strategy: str
    n_samples: int
    row_id_class: str
    organism: str | None = None
    notes: list[str] = field(default_factory=list)


def check_matrix(
    row_ids,
    sample_names,
    expect: GeoExpectation,
    *,
    min_row_fraction: float = 0.8,
) -> dict:
    """Validate a matrix against its GEO record. Raises `DataTypeError` on mismatch.

    Returns an audit dict that is written next to the matrix so the check is auditable
    after the fact, not merely performed.
    """
    dominant, fracs = classify_row_ids(row_ids)
    n = len(list(sample_names))
    problems: list[str] = []

    allowed = ASSAY_ROWS.get(expect.library_strategy.lower())
    if allowed is None:
        problems.append(
            f"unknown library_strategy {expect.library_strategy!r}; "
            f"add it to ASSAY_ROWS before loading this matrix")
    elif expect.row_id_class not in allowed:
        problems.append(
            f"the expectation itself is inconsistent: library_strategy "
            f"{expect.library_strategy!r} cannot produce row_id_class "
            f"{expect.row_id_class!r} (allowed: {sorted(allowed)})")

    if dominant != expect.row_id_class:
        problems.append(
            f"row IDs look like {dominant!r} but {expect.accession} "
            f"({expect.library_strategy}) requires {expect.row_id_class!r}; "
            f"class fractions {fracs}")
    elif fracs.get(dominant, 0.0) < min_row_fraction:
        problems.append(
            f"only {fracs.get(dominant, 0.0):.1%} of row IDs are {dominant!r}, "
            f"below the required {min_row_fraction:.0%}; class fractions {fracs}")

    if n != expect.n_samples:
        problems.append(
            f"matrix has {n} samples but GEO {expect.accession} has "
            f"{expect.n_samples}")

    if problems:
        raise DataTypeError(
            f"data-type guard failed for {expect.accession}:\n  - "
            + "\n  - ".join(problems))

    return {
        "accession": expect.accession,
        "library_strategy": expect.library_strategy,
        "organism": expect.organism,
        "n_samples_expected": expect.n_samples,
        "n_samples_found": n,
        "row_id_class_expected": expect.row_id_class,
        "row_id_class_found": dominant,
        "row_id_class_fractions": fracs,
        "n_rows": len(list(row_ids)),
        "passed": True,
        "notes": expect.notes,
    }


# ---------------------------------------------------------------------------
# The records used in this run. Values are taken from the GEO pages, not guessed.
# ---------------------------------------------------------------------------
GSE260996_RNA = GeoExpectation(
    accession="GSE260996",
    library_strategy="RNA-Seq",
    n_samples=33,
    row_id_class="entrez",
    organism="Mus musculus",
    notes=[
        "11 conditions x 3 replicates of RelA-mVenus BMDMs.",
        "Condition codes are two letters, first stimulus then second: "
        "F=media, C=CpG, L=LPS, T=TNF, P=polyI:C "
        "(CF CL CT FC FF FL FP FT PF PL PT).",
        "Processed per-sample files are Rsubread featureCounts objects "
        "(GSM*Aligned.RData.gz) whose counts are keyed by Entrez GeneID.",
    ],
)

GSE261647_ATAC = GeoExpectation(
    accession="GSE261647",
    library_strategy="ATAC-seq",
    n_samples=18,
    row_id_class="interval",
    organism="Mus musculus",
    notes=[
        "9 conditions x 2 replicates; condition codes use M for media "
        "(CL CT MC ML MM MP MT PL PT), unlike the RNA series which uses F.",
        "This is the matrix the Tay deposit ships as a bare logCPM.csv at its root, "
        "and the one the v0 run mistook for RNA-seq.",
    ],
)
