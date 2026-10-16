"""F5 8.4: the C3 projection from omics coordinates into the learned image space.

HYPOTHESIS-GENERATING ONLY. Nothing here pairs an image with a molecular sample.

8.4 requires image anchors that are TiJepa-1 centroids of Dryad M0/M1/M2 cells, with
the **stimulus matched** across the two modalities: image M1 = LPS 100 ng/ml against an
LPS omics anchor, image M2 = IL-4 20 ng/ml against an IL-4 omics anchor.

Two independent requirements both fail in this run, and either alone would be enough to
stop it:

  1. **No image anchors.** The Dryad movies were not provided, and no other accessible
     dataset carries M0/M1/M2 image labels (see results/v1/partA/dryad_benchmark.json).
  2. **No matching omics anchor.** Of the anchors 8.1 could build, the LPS anchor
     cannot be formed at all - GSE46903 has only 2 donor-matched sLPS/con pairs at
     GM-CSF 72 h against a minimum of 3 - and the IL-4 anchor fails its canonical-marker
     sanity rule. The only surviving anchor is IFN-gamma, whose stimulus has no
     counterpart among the image labels 8.4 specifies.

So the projection is **not performed**, and the v0 "reduced one-axis analysis" is
**deleted** rather than carried forward: PLAN_v1.md 8.5 removes it because it mapped a
human monocyte omics score onto a mouse microglia imaging axis anchored at its own
endpoints, which is circular. This script records the refusal and the reasons.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "results" / "v1" / "partC"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    anchors_p = OUT / "c1_anchors.json"
    anchors = json.loads(anchors_p.read_text()) if anchors_p.exists() else {}
    dryad_p = REPO / "results" / "v1" / "partA" / "dryad_benchmark.json"
    dryad = json.loads(dryad_p.read_text()) if dryad_p.exists() else {}

    usable = anchors.get("anchors_usable", [])
    per_anchor = {k: v.get("status") for k, v in (anchors.get("anchors") or {}).items()}

    res = {
        "phase": "F5 8.4",
        "framing": "HYPOTHESIS-GENERATING ONLY; no image is paired with any molecular "
                   "sample; cross-study, cross-platform and cross-species",
        "status": "not_performed",
        "planned_analysis": (
            "map omics activation coordinates onto TiJepa-1 image-space centroids for "
            "Dryad M0/M1/M2 cells, using only axes whose anchor stimulus matches the "
            "image stimulus (LPS<->LPS, IL-4<->IL-4)"),
        "blocking_reasons": [
            {
                "requirement": "image-space M0/M1/M2 anchors",
                "met": False,
                "detail": (
                    f"Dryad status: {dryad.get('status', 'unknown')}. No other "
                    "accessible dataset provides images labelled M0/M1/M2, and "
                    "substituting a different label set - BBBC054's microglial "
                    "morphology classes - and presenting it as this analysis would "
                    "misrepresent what was measured."),
            },
            {
                "requirement": "an omics anchor whose stimulus matches an image label",
                "met": False,
                "detail": (
                    f"anchor statuses: {per_anchor}. The LPS anchor cannot be formed "
                    "(2 donor-matched pairs, minimum 3) and the IL-4 anchor fails its "
                    "canonical-marker sanity rule, so neither stimulus 8.4 names "
                    f"survives. The only usable anchor is {usable}, whose stimulus has "
                    "no counterpart among the specified image labels."),
            },
        ],
        "v0_reduced_one_axis_analysis": {
            "status": "DELETED",
            "why": (
                "PLAN_v1.md 8.5. v0, unable to run C3, substituted a one-axis analysis "
                "that expressed a human monocyte omics score as an 'equivalent LPS "
                "exposure time' on a mouse microglia imaging axis, anchoring that axis "
                "at the extremes of the very quantity being mapped. It is circular, it "
                "shares no donor, species, platform or experiment across the two "
                "sides, and it is removed from code, results, figures and paper rather "
                "than relabelled."),
        },
        "what_would_unblock_it": (
            "Fetching Dryad m0cfxpp8v by hand (the first TODO(author)) supplies the "
            "image anchors. The omics side needs an LPS or IL-4 anchor built from at "
            "least 3 donor-matched pairs that passes 8.1.4; GSE46903 cannot supply one "
            "at GM-CSF 72 h, so that would need a different macrophage atlas."),
    }
    (OUT / "c3_projection.json").write_text(json.dumps(res, indent=2) + "\n")
    print("C3 not performed: no image anchors AND no stimulus-matched omics anchor")
    print(f"wrote {OUT/'c3_projection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
