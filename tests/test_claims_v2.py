"""Provenance guard, banned strings and dangling comparatives (PLAN_v2.md G2.1, G7.1).

All fixtures are synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tijepa.eval import provenance as PV  # noqa: E402
from tijepa.viz.claims import check_banned, check_grammar, check_labels  # noqa: E402

V0_REC = {"file": "tijepa0_T1_seed0.pt", "run_name": "tijepa0", "step": 55000,
          "sha256": "x"}
V1_REC = {"file": "tijepa1_T1_seed0.pt", "run_name": "tijepa1", "step": 117000,
          "sha256": "y"}


def test_label_is_derived_from_the_checkpoint_record():
    assert PV.label_for_key("E4_tijepa1", V1_REC) == "TiJepa-1"
    assert PV.label_for_key("E0_tijepa0", V0_REC) == "TiJepa-0"
    assert PV.label_for_key("E6_random_init_draw1", PV.random_init_record(1)) == \
        "random init"
    assert PV.label_for_key("E3_dinov2_vit_s", None) == "DINOv2 ViT-S/14"


def test_n1_a_tijepa0_result_under_a_tijepa1_key_is_a_hard_error():
    with pytest.raises(PV.ProvenanceError):
        PV.label_for_key("E4_tijepa1", V0_REC)
    with pytest.raises(PV.ProvenanceError):
        PV.label_for_key("E4_tijepa1", None)  # learned encoder without a record
    assert PV.check_results({"E4_tijepa1": {"checkpoint": V0_REC}})


def test_check_labels_catches_the_v1_mislabel():
    bad = {"E4 TiJepa-1": {"key": "E0_tijepa0", "checkpoint": V0_REC,
                           "label": "TiJepa-1"}}
    assert check_labels(bad, PV.label_for_key)
    good = {"TiJepa-0": {"key": "E0_tijepa0", "checkpoint": V0_REC, "label": "TiJepa-0"}}
    assert check_labels(good, PV.label_for_key) == []


def test_banned_strings_are_found():
    synthetic_paper = ("As PLAN_v2 says, see results/v1/partA. In v0 the e5 control "
                       "and the bbbc054 probe were NOT USABLE (2 blocking requirements "
                       "recorded.)")
    found = {h["match"] for h in check_banned(synthetic_paper)}
    assert {"PLAN_v", "results/v1/", "NOT USABLE", "blocking requirements recorded",
            "v0", " e5 ", " bbbc054 "} <= found
    assert check_banned("TiJepa-1 was evaluated on BBBC054 and S-BIAD2196 (E5).") == []


def test_dangling_comparative_is_found():
    assert check_grammar("TiJepa-1 is higher than (+0.082, 95% CI [0.01, 0.15]).")
    assert check_grammar("TiJepa-1 is higher than DINOv2 (+0.082).") == []


def test_per_seed_random_init_records_are_accepted():
    rec = {"checkpoint": None,
           "checkpoint_per_seed": {"0": PV.random_init_record(0),
                                   "1": PV.random_init_record(1)}}
    assert PV.check_results({"E6_random_init": rec}) == []
    assert PV.check_results({"E6_random_init": {"checkpoint": None}})
