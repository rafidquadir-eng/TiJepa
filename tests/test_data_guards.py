"""Data-type guard, BBBC054 coordinates and Tay ligand codes (PLAN_v1.md 11.4).

Three checks the plan names explicitly, each of which exists because v0 got the
corresponding thing wrong:

  * the **data-type guard** must classify gene-keyed matrices as expression and
    interval-keyed matrices as accessibility, and must reject a sample count that does
    not match the GEO record - this is what makes the v0 "ATAC analysed as RNA" bug
    (D1) impossible to repeat;
  * the **BBBC054 coordinate convention** must be the one the data support: every
    annotation falls inside the image under the chosen axis order, and a large fraction
    would fall outside under the transposed order;
  * the **Tay ligand mapping** must be the one sourced from the deposit, and the
    tolerance sanity check must point the way tolerance does.

Tests that need downloaded data are marked `@pytest.mark.data` and skip when it is
absent, so this file passes on a clean checkout. The guard tests themselves use
synthetic fixtures and always run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.omics import guard  # noqa: E402

RESULTS = REPO / "results" / "v1"


# --------------------------------------------------------------- the guard itself
def test_row_id_classes():
    assert guard.classify_row_id("chr1.4414143.4414644") == "interval"
    assert guard.classify_row_id("chr1:4414143-4414644") == "interval"
    assert guard.classify_row_id("ENSG00000000003") == "ensembl_gene"
    assert guard.classify_row_id("497097") == "entrez"
    assert guard.classify_row_id("Tnf") == "symbol"
    assert guard.classify_row_id("") == "unknown"


def test_guard_accepts_a_matching_rna_matrix():
    rows = [str(i) for i in range(100, 100 + 500)]          # Entrez-like
    cols = [f"S{i}" for i in range(33)]
    audit = guard.check_matrix(rows, cols, guard.GSE260996_RNA)
    assert audit["passed"] and audit["row_id_class_found"] == "entrez"


def test_guard_rejects_an_interval_matrix_offered_as_rna():
    """The v0 bug, as a test: ATAC row IDs must not pass as expression."""
    rows = [f"chr1.{i}.{i + 500}" for i in range(1000, 1000 + 500)]
    cols = [f"S{i}" for i in range(33)]
    with pytest.raises(guard.DataTypeError) as e:
        guard.check_matrix(rows, cols, guard.GSE260996_RNA)
    assert "interval" in str(e.value)


def test_guard_rejects_a_wrong_sample_count():
    rows = [str(i) for i in range(100, 600)]
    with pytest.raises(guard.DataTypeError) as e:
        guard.check_matrix(rows, [f"S{i}" for i in range(18)], guard.GSE260996_RNA)
    assert "18 samples" in str(e.value)


def test_guard_accepts_the_atac_matrix_as_atac():
    rows = [f"chr1.{i}.{i + 500}" for i in range(1000, 1000 + 400)]
    audit = guard.check_matrix(rows, [f"S{i}" for i in range(18)],
                               guard.GSE261647_ATAC)
    assert audit["passed"] and audit["row_id_class_found"] == "interval"


def test_guard_rejects_a_mixed_matrix_below_the_purity_threshold():
    rows = ([f"chr1.{i}.{i + 500}" for i in range(300)]
            + [str(i) for i in range(200)])
    with pytest.raises(guard.DataTypeError):
        guard.check_matrix(rows, [f"S{i}" for i in range(18)],
                           guard.GSE261647_ATAC)


def test_guard_rejects_an_internally_inconsistent_expectation():
    """An expectation that asks ATAC to produce gene IDs is itself the bug."""
    bad = guard.GeoExpectation(accession="GSEX", library_strategy="ATAC-seq",
                               n_samples=4, row_id_class="symbol")
    with pytest.raises(guard.DataTypeError) as e:
        guard.check_matrix(["Tnf", "Il6", "Il1b", "Ccl4"],
                           ["a", "b", "c", "d"], bad)
    assert "cannot produce" in str(e.value)


def test_guard_rejects_an_empty_matrix():
    with pytest.raises(guard.DataTypeError):
        guard.check_matrix([], ["a"], guard.GSE260996_RNA)


# ------------------------------------------------------- BBBC054 coordinates
@pytest.mark.data
def test_bbbc054_annotations_fall_inside_the_image_under_the_chosen_axis_order():
    from tijepa.io import bbbc054 as B
    csv = REPO / "data" / "raw" / "bbbc054" / "Replicate1annotation.csv"
    if not csv.exists():
        pytest.skip("BBBC054 annotations not downloaded")
    ann = B.load_annotations(csv)
    assert ann, "no annotations parsed"
    h, w = 1080, 1280           # rows, columns
    y = np.array([a["y"] for a in ann])
    x = np.array([a["x"] for a in ann])
    inside = ((y >= -8) & (y <= h + 8) & (x >= -8) & (x <= w + 8)).mean()
    # transposed: treat axis-1 as the column index and axis-2 as the row index
    inside_T = ((y >= -8) & (y <= w + 8) & (x >= -8) & (x <= h + 8)).mean()
    assert inside > 0.999, f"only {inside:.3%} inside under the chosen order"
    assert inside_T < 0.95, (
        "the transposed order is not clearly worse, so this test would not have "
        f"caught the v0 bug ({inside_T:.3%} nominally inside)")


@pytest.mark.data
def test_bbbc054_has_three_classes_including_ramified():
    """v0's label tuple contained 'stratified', which does not occur in the file."""
    from tijepa.io import bbbc054 as B
    csv = REPO / "data" / "raw" / "bbbc054" / "Replicate1annotation.csv"
    if not csv.exists():
        pytest.skip("BBBC054 annotations not downloaded")
    assert set(B.LABELS) == {"round", "amoeboid", "ramified"}
    labels = {a["label"] for a in B.load_annotations(csv)}
    assert labels == {"round", "amoeboid", "ramified"}


# ------------------------------------------------------------ Tay ligand codes
def test_tay_ligand_codes_are_the_sourced_mapping():
    from tijepa.io import tay
    assert tay.LIGAND_CODES == {1.0: "CpG", 2.0: "LPS", 3.0: "polyIC", 6.0: "TNF"}
    # the v0 inference had 3 = CpG; that must not come back
    assert tay.LIGAND_CODES[3.0] != "CpG"
    assert 4.0 in tay.UNRESOLVED_CODES and 17.0 in tay.UNRESOLVED_CODES


@pytest.mark.data
def test_tolerance_sanity_direction_is_recorded_and_correct():
    p = RESULTS / "partB" / "ligand_codes.json"
    if not p.exists():
        pytest.skip("ligand-code results not generated")
    d = json.loads(p.read_text())
    san = d["tolerance_sanity_check"]
    if san.get("status") != "ok":
        pytest.skip(f"sanity check not performed: {san.get('reason')}")
    assert san["mean_amplitude_lps_to_lps"] < san["mean_amplitude_naive_to_lps"], (
        "the NF-kB second-response amplitude must be LOWER after LPS priming; if it is "
        "not, the ligand-code mapping has to be flagged unconfirmed in the paper")
    assert san["expected_direction_observed"] is True
    assert d["code_mapping"]["status"] == "sourced_from_deposit_code"


@pytest.mark.data
def test_no_confidence_interval_is_reported_below_ten_groups():
    """Rule 3, as a test: the v0 four-well bootstrap CI must not reappear."""
    p = RESULTS / "partB" / "b3_dynamics.json"
    if not p.exists():
        pytest.skip("Part B results not generated")
    d = json.loads(p.read_text())
    tol = d["b31_tolerance"]
    if tol.get("status") != "ok":
        pytest.skip("B3.1 not performed")
    assert tol["n_wells"] < 10
    for kind, res in tol["results"].items():
        assert res["confidence_interval"] is None, (
            f"{kind} reports a CI over {tol['n_wells']} groups")
        assert res["exact_permutation"]["minimum_achievable_p"] > 0
