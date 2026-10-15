"""M6 / Part C3: project omics coordinates into the learned image space.

HYPOTHESIS-GENERATING ONLY. Nothing here pairs an image with a molecular sample.

PLAN.md C3 asks for image-space centroids c_M0, c_M1, c_M2 from the Dryad BMDM movies
and a bilinear map from the omics (M1-like, M2-like) coordinates onto them. Those movies
were unreachable in this run (reports/MANUAL_DOWNLOADS.md), and **no accessible dataset
provides images labelled M0/M1/M2**, so the planned two-axis projection cannot be done.

Rather than silently substitute a different label set, this script:
  * refuses the two-axis projection and records why, and
  * computes a clearly-labelled REDUCED one-axis analysis instead: BBBC054 supplies a
    single activation axis (LPS exposure time, 0-30 h), so the M1-like omics coordinate
    of each state is expressed as an "equivalent LPS exposure time" by anchoring the
    unstimulated macrophage baseline at 0 h and the strongest observed activation at the
    longest exposure. This is a one-dimensional, cross-species, cross-modality
    interpolation and is labelled as such everywhere it appears.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "partC"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c2 = json.loads((OUT / "c2_scores.json").read_text()) if (
        OUT / "c2_scores.json").exists() else None
    bb_path = REPO / "results" / "partA" / "bbbc054_probes.json"
    bb = json.loads(bb_path.read_text()) if bb_path.exists() else None

    if not c2 or c2.get("status") != "ok":
        (OUT / "c3_projection.json").write_text(json.dumps({
            "status": "not_performed",
            "reason": "Part C2 omics coordinates are not available.",
        }, indent=2) + "\n")
        print("C3 not performed: no C2 coordinates")
        return 0

    coords = c2["coordinates"]
    result: dict = {
        "status": "not_performed",
        "framing": "HYPOTHESIS-GENERATING ONLY; no image is paired with any molecular "
                   "sample; cross-study, cross-platform and cross-species",
        "planned_analysis": "bilinear map of (M1-like, M2-like) omics coordinates onto "
                            "image-space centroids c_M0, c_M1, c_M2 (PLAN.md C3)",
        "reason": (
            "The planned two-axis projection requires image-space centroids for M0, M1 "
            "and M2 cells. The only public dataset providing images with those labels is "
            "Dryad doi:10.5061/dryad.m0cfxpp8v, which was unreachable in this run "
            "(an Anubis proof-of-work challenge on every official download endpoint; see "
            "reports/MANUAL_DOWNLOADS.md). No accessible dataset substitutes for it: "
            "BBBC054 carries microglial morphology classes and LPS exposure time, not "
            "macrophage polarization states. Substituting it and presenting the result "
            "as the planned analysis would misrepresent what was measured, so the "
            "two-axis projection is left undone."),
        "omics_coordinates_available": coords,
    }

    # ---- reduced one-axis analysis, clearly labelled
    if bb and bb["tasks"]["lps_exposure_time_regression"]:
        base = coords.get("control", {}).get("M1_like", 0.0)
        states = {k: v["M1_like"] - base for k, v in coords.items()}
        span = max(abs(v) for v in states.values()) or 1.0
        max_h = 29.5  # BBBC054: 60 FOVs at 30 min intervals, 0-29.5 h
        reduced = {
            k: {"m1_like_rel_baseline": float(v),
                "equivalent_lps_exposure_hours": float(abs(v) / span * max_h)
                if v > 0 else 0.0,
                "direction": "toward LPS activation" if v > 0 else
                             "away from LPS activation"}
            for k, v in states.items()
        }
        result["reduced_one_axis_analysis"] = {
            "label": "REDUCED, ONE-AXIS, HYPOTHESIS ONLY - not the analysis PLAN.md C3 "
                     "specifies. It maps a human monocyte omics score onto a mouse "
                     "microglia imaging axis; the two share no donors, no species, no "
                     "platform and no experiment.",
            "image_axis": "BBBC054 LPS exposure time, 0-29.5 h (60 FOVs, 30 min apart)",
            "anchors": "unstimulated macrophage baseline = 0 h; the largest observed "
                       "M1-like displacement = 29.5 h",
            "states": reduced,
            "testable_prediction": (
                "If LPS-tolerised and beta-glucan-trained monocytes sit near the "
                "unstimulated baseline on the M1-like axis, a label-free imaging assay "
                "should NOT separate them from unstimulated macrophages on an "
                "activation axis trained on acute LPS exposure alone. Separating them "
                "would require a readout sensitive to the primed state rather than to "
                "acute activation. Testing this needs a prime-rest-restimulate imaging "
                "experiment, which no public dataset provides."),
        }
        print("C3: two-axis projection NOT performed (no M0/M1/M2 image anchors)")
        for k, v in reduced.items():
            print(f"   reduced axis: {k:22s} rel {v['m1_like_rel_baseline']:+.4f} "
                  f"-> {v['equivalent_lps_exposure_hours']:.1f} h equivalent")
    else:
        result["reduced_one_axis_analysis"] = {
            "status": "not_performed",
            "reason": "BBBC054 probe results are not available either",
        }
        print("C3 not performed and no reduced analysis possible")

    (OUT / "c3_projection.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n")
    print(f"wrote {OUT/'c3_projection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
