"""F7 10.2: verify the paper's directional claims against results/ (fixes E5, B6).

Run in `run_all.sh` and in the tests. Exit code 1 means the prose contradicts the
numbers, which fails the build - the whole point being that a conclusion cannot survive
in the paper once the result it describes has changed.

Comparisons are loaded from the result JSONs rather than restated here, so this script
has no numbers of its own to drift out of date:

  * `results/v2/partA/bbbc054.json` -> `pairwise_comparisons`, each with a paired
    cluster-bootstrap CI over FOVs and a BH-corrected p (60 FOVs, so a CI is allowed);
  * `results/v2/partA/tracking_forecast.json` -> each method against the baseline by
    exact sign-flip permutation over movies (6 movies, so no CI, per rule 3);
  * `results/v2/partB/b3_dynamics.json` -> raw trace against hand-crafted features by
    exact permutation over wells.

A claim about a pair with no stored comparison is not checked, and the script says how
many sentences it checked so that a silent "0 violations" cannot be mistaken for
"0 claims examined".

v2 (PLAN_v2.md G2.1; findings N1, N11, N14) adds three checks that each fail the build:
  * provenance: every encoder record in results/v2/partA passes
    `tijepa.eval.provenance.check_results`, and every label the renderer wrote to
    `paper/labels.json` equals the label derived from its checkpoint record;
  * banned strings: no internal process language in the paper body (rule 5);
  * grammar: no dangling comparative such as "higher than (".
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.eval import provenance as PV  # noqa: E402
from tijepa.viz.claims import (  # noqa: E402
    Comparison,
    check_banned,
    check_grammar,
    check_labels,
    check_text,
)

V1 = REPO / "results" / "v2"
PAPER = REPO / "paper" / "tijepa.md"
LABELS = REPO / "paper" / "labels.json"

#: how each results key is referred to in prose
ALIASES: dict[str, list[str]] = {
    "E1_handcrafted_logreg": ["hand-crafted", "handcrafted", "E1"],
    "E2_imagenet_vit_s": ["ImageNet", "E2"],
    "E3_dinov2_vit_s": ["DINOv2", "E3"],
    "E4_tijepa1": ["TiJepa-1 ", "TiJepa-1,", "TiJepa-1.", "E4 "],
    "E4s_tijepa1_spatial": ["spatial-only", "TiJepa-1-spatial", "E4s"],
    "E6_random_init": ["random init", "random-init", "E6"],
    "E0_tijepa0": ["TiJepa-0", "E0"],
    "baseline_nearest_detection": ["nearest-detection", "baseline"],
    "E5_frame_shuffled": ["frame-shuffled", "frame shuffle", "E5"],
    "raw_trace": ["raw trace"],
    "handcrafted": ["hand-crafted features"],
}


def load(rel: str):
    p = V1 / rel
    return json.loads(p.read_text()) if p.exists() else None


def gather() -> dict[str, Comparison]:
    comps: dict[str, Comparison] = {}

    bb = load("partA/bbbc054.json")
    if bb:
        for key, rec in bb.get("pairwise_comparisons", {}).items():
            task, _pair = key.split(":", 1)
            a, b = rec["encoder_a"], rec["encoder_b"]
            comps[f"{a}:{b}:{task}"] = Comparison(
                a=a, b=b, task=task,
                difference=rec["difference_mean_over_seeds"],
                ci=tuple(rec["ci95_widest_over_seeds"]),
                p_value=rec.get("p_value_bh"))

    tr = load("partA/tracking_forecast.json")
    if tr:
        agg = tr.get("tracking", {}).get("aggregate", {})
        for meth, rec in agg.items():
            perm = rec.get("vs_baseline_exact_permutation")
            if perm:
                comps[f"{meth}:baseline_nearest_detection:tracking"] = Comparison(
                    a=meth, b="baseline_nearest_detection", task="tracking",
                    difference=perm["mean_difference"],
                    p_value=perm["exact_sign_flip_p_two_sided"])

    b3 = load("partB/b3_dynamics.json")
    if b3:
        res = b3.get("b31_tolerance", {}).get("results", {})
        if "raw_trace" in res and "handcrafted" in res:
            d = (res["raw_trace"]["cell_balanced_accuracy_mean"]
                 - res["handcrafted"]["cell_balanced_accuracy_mean"])
            comps["raw_trace:handcrafted:b31"] = Comparison(
                a="raw_trace", b="handcrafted", task="b31", difference=d,
                p_value=max(res["raw_trace"]["exact_permutation"]["p_exact"],
                            res["handcrafted"]["exact_permutation"]["p_exact"]))
    return comps


def main() -> int:
    if not PAPER.exists():
        print(f"no paper at {PAPER}; nothing to check")
        return 0
    full = PAPER.read_text()
    text = full.split("\n## References")[0]
    other: dict[str, list] = {"banned": check_banned(text), "grammar": check_grammar(text),
                              "provenance": []}
    for rel in ("partA/bbbc054.json", "partA/tracking_forecast.json",
                "partA/dryad_benchmark.json"):
        r = load(rel)
        if r and isinstance(r.get("encoders"), dict):
            other["provenance"] += [f"{rel}: {p}" for p in PV.check_results(r["encoders"])]
    if LABELS.exists():
        other["provenance"] += check_labels(json.loads(LABELS.read_text()),
                                            PV.label_for_key)
    else:
        other["provenance"].append("paper/labels.json absent: labels cannot be verified")
    bad = {k: v for k, v in other.items() if v}
    if bad:
        print("FAIL: manuscript checks")
        for k, v in bad.items():
            for item in v[:20]:
                print(f"  [{k}] {item}")
        (V1 / "claims_check.json").write_text(json.dumps(
            {"status": "fail", "manuscript_checks": other}, indent=2, default=str) + "\n")
        return 1
    comps = gather()
    if not comps:
        print("WARNING: no comparisons could be loaded from results/v2; "
              "the paper's directional claims were NOT verified")
        return 1

    violations = check_text(text, comps, ALIASES)
    n_sent = len([s for s in text.split(". ") if s.strip()])
    print(f"checked {n_sent} sentences against {len(comps)} stored comparisons")
    for k, c in sorted(comps.items()):
        print(f"  {k:62s} d={c.difference:+.4f} "
              f"{'CI ' + str([round(x, 4) for x in c.ci]) if c.ci else ''} "
              f"{'p=' + format(c.p_value, '.4f') if c.p_value is not None else ''} "
              f"-> {c.direction}")
    if violations:
        print(f"\nFAIL: {len(violations)} directional claim(s) contradict the results:")
        for v in violations:
            print(f"  [{v['comparison']}] word {v['word']!r} implies "
                  f"{v['word_implies']:+d} but the results say {v['results_say']:+d}")
            print(f"    difference {v['difference']:+.4f} "
                  f"ci {v['ci']} p {v['p_value']}")
            print(f"    sentence: {v['sentence']}")
        (V1 / "claims_check.json").write_text(
            json.dumps({"status": "fail", "violations": violations},
                       indent=2, default=float) + "\n")
        return 1

    print("\nOK: every directional claim near an encoder name agrees with results/v2")
    (V1 / "claims_check.json").write_text(json.dumps({
        "status": "ok", "n_comparisons": len(comps),
        "manuscript_checks": {"banned_strings": "pass", "grammar": "pass",
                              "provenance": "pass",
                              "n_labels_verified": len(json.loads(LABELS.read_text()))},
        "n_sentences_scanned": n_sent,
        "comparisons": {k: {"difference": c.difference, "ci": list(c.ci) if c.ci
                            else None, "p_value": c.p_value,
                            "direction": c.direction}
                        for k, c in comps.items()},
    }, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
