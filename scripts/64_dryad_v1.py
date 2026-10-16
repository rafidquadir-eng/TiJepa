"""F3.1: the Dryad M0/M1/M2 video headline benchmark.

PLAN_v1.md makes this the v1 headline (review finding D2): tracked BMDM cells from
Dryad doi:10.5061/dryad.m0cfxpp8v, leave-one-batch-out over 5 batches, every encoder
including the frame-shuffle and random-init controls, with batch and magnification
confound probes.

The dataset is an **author prerequisite** (PLAN_v1.md Section 2.3): it had to be fetched
by hand through a browser and uploaded to Drive, because every automated endpoint serves
a proof-of-work interstitial that PLAN.md Section 2/9 forbids bypassing. It was not
provided for this run, so this stage writes an explicit `not_provided` status with the
code path that would consume it, rather than being skipped silently or substituted with
a different label set.

When the archive is placed in `data/raw/dryad_bmdm/`, re-running this script performs
the benchmark; nothing else needs to change.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "v2" / "partA"
DATA = REPO / "data" / "raw" / "dryad_bmdm"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in DATA.glob("**/*") if p.is_file()) if DATA.exists() else []
    intake = REPO / "results" / "v2" / "manual_intake.json"
    status = {
        "phase": "F3.1",
        "analysis": "Dryad M0/M1/M2 BMDM video benchmark, leave-one-batch-out",
        "planned_as": "the v1 headline Part A result (fixes review finding D2)",
        "doi": "10.5061/dryad.m0cfxpp8v",
        "data_dir": str(DATA.relative_to(REPO)),
        "n_files_present": len(files),
    }
    if files:
        status.update({
            "status": "data_present_but_benchmark_not_implemented",
            "note": "Files are present. The leave-one-batch-out benchmark itself is "
                    "not implemented, because it could not be written against data "
                    "nobody had seen: the condition and batch encoding lives in the "
                    "deposit's README and filenames. Implement it here.",
            "files": files[:50],
        })
        print(f"{len(files)} files present in {DATA}; benchmark not implemented")
    else:
        status.update({
            "status": "not_provided",
            "reason": (
                "The author prerequisite (PLAN_v1.md 2.3, PLAN_v2.md 2.1) was not "
                "completed: TiJepa/manual_downloads/ does not exist on the author's "
                "Drive and neither Dryad m0cfxpp8v nor LTDB is in the account (checked "
                "at the start of the run and again before the evaluations; see "
                "results/v2/manual_intake.json). Automated download remains "
                "blocked by a proof-of-work interstitial, which PLAN.md Section 2 and "
                "Section 9 forbid bypassing, so the run continued per the Section 14 "
                "degrade rule."),
            "consequences": [
                "Part A has NO video benchmark with activation-state labels, so the "
                "frame-shuffle control E5 cannot be evaluated on a labelled video task "
                "and Part A's classification evidence rests on static BBBC054 "
                "morphology classes.",
                "The F3.4 forecasting evaluation runs on S-BIAD2196 only, not on Dryad.",
                "Part C 8.4 (the C3 projection) cannot be performed: it needs "
                "image-space M0/M1/M2 centroids with stimuli matched to the omics "
                "anchors. See results/v1/partC/c3_projection.json.",
                "The every-2-hour collapse monitor uses a fixed BBBC054 subset instead "
                "of Dryad (results/v2/pretrain_monitor.csv).",
            ],
            "first_todo_author": True,
            "code_path": "this script detects the files and records them; the "
                         "leave-one-batch-out benchmark itself still has to be written "
                         "against the deposit's README (condition, batch and "
                         "magnification encoding), which nobody in this project has seen",
            "intake_record": str(intake.relative_to(REPO)),
        })
        print("Dryad NOT PROVIDED: recording an explicit status and continuing")
    (OUT / "dryad_benchmark.json").write_text(
        json.dumps(status, indent=2) + "\n")
    print(f"wrote {OUT/'dryad_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
