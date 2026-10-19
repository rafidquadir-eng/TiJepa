"""Tests for the table writer and the claims checker (PLAN_v1.md 11.4).

Both exist to stop a specific v0 failure, so each failure is reproduced here as a test:
a table whose header and rows disagree (E1), and a directional sentence that contradicts
the numbers it describes (E5, B6).

All fixtures are synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tijepa.viz.claims import (  # noqa: E402
    Comparison,
    check_text,
    compare_phrase,
    describe,
    render_placeholders,
)
from tijepa.viz.tables import TableShapeError, markdown_table  # noqa: E402

NAMES = {"E4": ["TiJepa-1", "E4"], "E3": ["DINOv2", "E3"], "E6": ["random init", "E6"]}


# ------------------------------------------------------------------ table writer
def test_table_renders_and_aligns():
    t = markdown_table(["encoder", "BA"], [["E4", "0.83"], ["E3", "0.77"]])
    lines = t.splitlines()
    assert lines[0] == "| encoder | BA |"
    assert lines[1] == "|:---|---:|"
    assert lines[2] == "| E4 | 0.83 |"
    assert len(lines) == 4


def test_table_rejects_a_row_that_does_not_match_the_header():
    """The exact v0 bug: a header of 2 columns and a row of 3."""
    with pytest.raises(TableShapeError) as e:
        markdown_table(["encoder", "BA"], [["E4", "0.83", "extra"]])
    assert "row 0 has 3 cells" in str(e.value)


def test_table_rejects_an_empty_table_unless_asked():
    with pytest.raises(TableShapeError):
        markdown_table(["a", "b"], [])
    assert markdown_table(["a", "b"], [], allow_empty=True).count("\n") == 1


def test_table_rejects_a_bad_align_spec_and_no_columns():
    with pytest.raises(TableShapeError):
        markdown_table(["a", "b"], [["1", "2"]], align=["l"])
    with pytest.raises(TableShapeError):
        markdown_table([], [])


def test_table_renders_none_as_empty_cell():
    t = markdown_table(["a", "b"], [["x", None]])
    assert t.splitlines()[2] == "| x |  |"


# ---------------------------------------------------------------- claim wording
def test_phrase_follows_the_interval_not_the_point_estimate():
    # a positive difference whose CI straddles zero is NOT "higher"
    c = Comparison("E4", "E3", "bbbc054", difference=0.02, ci=(-0.01, 0.05))
    assert compare_phrase(c) == "not distinguishable from"
    c2 = Comparison("E4", "E3", "bbbc054", difference=0.02, ci=(0.01, 0.03))
    assert compare_phrase(c2) == "higher than"
    c3 = Comparison("E4", "E3", "bbbc054", difference=-0.02, ci=(-0.03, -0.01))
    assert compare_phrase(c3) == "lower than"


def test_phrase_falls_back_to_the_permutation_p_when_there_is_no_ci():
    c = Comparison("E4", "E3", "tracking", difference=0.06, p_value=0.33)
    assert compare_phrase(c) == "not distinguishable from"
    c2 = Comparison("E4", "E3", "tracking", difference=0.06, p_value=0.01)
    assert compare_phrase(c2) == "higher than"


def test_phrase_is_not_compared_when_nothing_is_available():
    assert compare_phrase(Comparison("E4", "E3", "x")) == "not compared"
    assert compare_phrase(
        Comparison("E4", "E3", "x", difference=0.1)) == "not compared"


def test_describe_includes_the_number_and_its_uncertainty():
    s = describe(Comparison("E4", "E6", "bbbc054", difference=0.056,
                            ci=(0.045, 0.067)))
    assert "higher than" in s and "+0.056" in s and "95% CI" in s
    s2 = describe(Comparison("E4", "E6", "t", difference=0.01, p_value=0.33))
    assert "exact p = 0.330" in s2


def test_render_placeholders_substitutes_and_reports_missing():
    comps = {"E4:E3:bbbc054": Comparison("E4", "E3", "bbbc054", difference=0.06,
                                         ci=(0.04, 0.08))}
    text = "TiJepa-1 is {{CMP:E4:E3:bbbc054}} DINOv2, and {{CMP:E4:E6:other}} too."
    out, missing = render_placeholders(text, comps)
    assert "is higher than DINOv2" in out
    assert missing == ["E4:E6:other"]
    assert "{{CMP:E4:E6:other}}" in out  # left visible rather than silently dropped


# -------------------------------------------------------------- claims checker
def test_checker_flags_a_sentence_that_contradicts_the_numbers():
    comps = {"E4:E3:bbbc054": Comparison("E4", "E3", "bbbc054", difference=-0.05,
                                         ci=(-0.08, -0.02))}
    text = "On BBBC054, TiJepa-1 beats DINOv2 by a clear margin."
    v = check_text(text, comps, NAMES)
    assert len(v) == 1
    assert v[0]["word"] == "beats"
    assert v[0]["results_say"] == -1


def test_checker_accepts_a_sentence_that_agrees():
    comps = {"E4:E3:bbbc054": Comparison("E4", "E3", "bbbc054", difference=0.05,
                                         ci=(0.02, 0.08))}
    text = "On BBBC054, TiJepa-1 is higher than DINOv2."
    assert check_text(text, comps, NAMES) == []


def test_checker_flags_a_directional_claim_when_the_ci_straddles_zero():
    comps = {"E4:E6:bbbc054": Comparison("E4", "E6", "bbbc054", difference=0.01,
                                         ci=(-0.02, 0.04))}
    text = "TiJepa-1 is better than random init."
    v = check_text(text, comps, NAMES)
    assert len(v) == 1 and v[0]["results_say"] == 0


def test_checker_accepts_indistinguishable_when_the_ci_straddles_zero():
    comps = {"E4:E6:bbbc054": Comparison("E4", "E6", "bbbc054", difference=0.01,
                                         ci=(-0.02, 0.04))}
    text = "TiJepa-1 is indistinguishable from random init."
    assert check_text(text, comps, NAMES) == []


def test_checker_respects_the_order_the_sentence_mentions_encoders():
    """A claim stated the other way round must be judged the other way round."""
    comps = {"E4:E3:bbbc054": Comparison("E4", "E3", "bbbc054", difference=0.05,
                                         ci=(0.02, 0.08))}
    assert check_text("DINOv2 is lower than TiJepa-1.", comps, NAMES) == []
    v = check_text("DINOv2 is higher than TiJepa-1.", comps, NAMES)
    assert len(v) == 1


def test_checker_ignores_table_rows_and_single_encoder_sentences():
    comps = {"E4:E3:bbbc054": Comparison("E4", "E3", "bbbc054", difference=-0.05,
                                         ci=(-0.08, -0.02))}
    assert check_text("| TiJepa-1 | DINOv2 | higher |", comps, NAMES) == []
    assert check_text("TiJepa-1 is higher than chance.", comps, NAMES) == []
