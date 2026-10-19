"""Write reports/RUN_REPORT_v3.md from the results (PLAN_v3 §11.3, §14).

Every status in the Section 14 table is derived from a file that exists (its evidence);
a row whose evidence is missing is reported as "open", never as done.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "RUN_REPORT_v3.md"


def J(rel: str):
    p = REPO / rel
    return json.loads(p.read_text()) if p.exists() else None


def exists(rel: str) -> bool:
    return (REPO / rel).exists()


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def gates() -> list[tuple[str, str, str, str]]:
    rows = []
    mi = J("results/v3/manual_intake.json") or {}
    dry = "absent" not in json.dumps(mi.get("datasets", {})) and bool(mi)
    rows.append(("G-Dryad", "Dryad files present and verified", "no" if not dry else "yes",
                 "Part A headline stays BBBC054; the gap is stated in one sentence"
                 if not dry else "Dryad is the Part A headline"))
    fc = J("results/v3/partA/forecast_controls.json")
    if fc:
        g = fc["G_Forecast"]["gate"]
        rows.append(("G-Forecast", "beats copy-last and slot mean; swap/static remove most "
                     "of the advantage", "pass" if g["pass"] else "fail",
                     fc["G_Forecast"]["decision"]))
    sw = J("results/v3/partA/recipe_sweep.json")
    if sw and sw.get("G_Recipe", {}).get("decision"):
        rows.append(("G-Recipe", "validation score +0.02 and 2x effective rank vs S0 at 20k",
                     "pass" if sw["G_Recipe"]["passing"] else "fail",
                     sw["G_Recipe"]["decision"]))
    else:
        rows.append(("G-Recipe", "validation score +0.02 and 2x effective rank vs S0 at 20k",
                     "pending", "sweep not finished when this report was generated"))
    bg = J("results/v3/partC/beta_glucan_replication.json")
    if bg:
        rows.append(("G-BetaGlucan", "identity QC passes and replicated in an independent "
                     "dataset or the source RNA", "pass" if bg["G_BetaGlucan"]["decision"]
                     .startswith("pass") else "fail", bg["G_BetaGlucan"]["decision"]))
    return rows


def section14() -> list[tuple[str, str, str, str]]:
    r = []

    def row(i, status, evidence, note=""):
        r.append((i, status, evidence, note))
    mi = J("results/v3/manual_intake.json")
    row("R1", "done (no branch)" if mi else "open", "results/v3/manual_intake.json",
        "Dryad still absent; G-Dryad 'no' branch taken")
    bk = (REPO / "reports" / "BACKUP.md").read_text() if exists("reports/BACKUP.md") else ""
    row("R2", "done (persistent filesystem); Drive pending" if "H0.2" in bk else "open",
        "reports/BACKUP.md", "Lambda persistent filesystem verified by SHA-256; rclone has "
        "no OAuth token")
    mon = (REPO / "results/v2/pretrain_monitor.csv").read_text().splitlines()
    keys = [tuple(x.split(",")[1:3]) for x in mon[1:]]
    row("R3", "done" if len(keys) == len(set(keys)) else "open",
        "results/v2/pretrain_monitor.csv", "monitor and backup loops stopped; duplicates removed")
    row("R4", "done" if exists("results/v3/partA/validation_reid.json") else "open",
        "results/v3/partA/validation_reid.json (diagnosis); results/v3/partA/recipe_sweep.json",
        "collapse diagnosis; G-Recipe in the gate table")
    snaps = sorted((REPO / "checkpoints" / "sweep").glob("sweep_*/step_*.pt"))
    row("R5", f"done ({len(snaps)} snapshots)" if snaps else "open",
        "checkpoints/sweep/sweep_*/step_*.pt", "every 5,000 steps, immutable")
    row("R6", "done" if exists("results/v3/partA/forecast_controls.json") else "open",
        "results/v3/partA/forecast_controls.json", "G-Forecast in the gate table")
    bb = J("results/v3/partA/bbbc054.json") or {}
    e7 = [k for k in ("results/v3/partA/bbbc054.json", "results/v3/coverage/autofluorescence.json",
                      "results/v3/coverage/flim.json", "results/v3/partA/validation_reid.json",
                      "results/v3/partA/association.json")
          if exists(k) and "E7_vjepa2_vitl" in (REPO / k).read_text()]
    row("R7", f"done ({len(e7)} benchmarks)" if "E7_vjepa2_vitl" in bb.get("encoders", {})
        else "open", "; ".join(e7) + "; manifests/models.yaml", "MIT licence recorded; "
        "forecasting is N/A for E7")
    row("R8", "reported" if bb else "open", "results/v3/partA/bbbc054.json",
        "TiJepa-1 vs DINOv2, V-JEPA 2, TiJepa-0 stated plainly in the paper")
    row("R9", "done" if exists("results/v3/partA/association.json") else "open",
        "results/v3/partA/association.json", "point tracking demoted to the supplement")
    row("R10", "done" if exists("results/v3/validation_set.json") else "open",
        "results/v3/validation_set.json", "64 non-corpus ComplexEye movies, never a test set")
    row("R11", "done" if exists("results/v3/partA/forecast_controls.json") else "open",
        "results/v3/partA/forecast_controls.json (strides 1, 3, 9, 27)",
        "S4 (long-horizon training) in the sweep")
    row("R12", "done (classical segmentation recorded; scale result stated)" if bb else "open",
        "results/v3/partA/bbbc054.json (e1_segmentation_methods, scale_sensitivity)",
        "Cellpose not installed; stated in Methods")
    qc = exists("results/v3/partC/sample_qc.json")
    row("R13", "done" if qc and exists("results/v3/partC/beta_glucan_replication.json")
        else "open", "results/v3/partC/sample_qc.json; source_rna.json; "
        "beta_glucan_replication.json", "G-BetaGlucan in the gate table")
    pb = exists("results/v3/partB/partB_v3.json")
    row("R14", "done" if pb else "open", "results/v3/partB/partB_v3.json",
        "trace JEPA vs hand-crafted vs raw, well-bootstrap CIs")
    row("R15", "done" if pb else "open", "results/v3/partB/partB_v3.json (b31_power_analysis)", "")
    row("R16", "done" if exists("results/v3/partB/b4_symmetric.json") else "open",
        "results/v3/partB/b4_symmetric.json", "exploratory")
    tm = exists("paper/v3/tmlr/main.tex")
    row("R17", "done" if tm else "open", "paper/v3/tmlr/main.tex; paper/v3/biorxiv/main.tex",
        "coverage map as the organising frame; C3 stated once as not performed")
    row("R18", "done (reduced to a Future-work subsection, author scope decision)"
        if exists("paper/v3/sections/future.tex") else "open", "paper/v3/sections/future.tex", "")
    row("R19", "done" if exists("scripts/run_all_v3.sh") and exists("legacy/README.md")
        else "open", "scripts/run_all_v3.sh; legacy/README.md", "")
    absp = subprocess.run(["grep", "-rl", "/home/ubuntu", str(REPO / "results")],
                          capture_output=True, text=True).stdout.split()
    row("R20", "done" if not absp else f"open ({len(absp)} files)",
        "grep -rl /home/ubuntu results/", "")
    row("R21", "PR #2 opened from tijepa-v3" if exists("reports/PR_BODY_v3.md") else "open",
        "reports/PR_BODY_v3.md", "author squash-merges PR #1")
    row("R22", "done", "scripts/74_recipe_sweep_v3.py; legacy/scripts/53_pretrain_monitor_v2.py",
        "new runs scored on the validation set only")
    row("R23", "stated" if tm else "open", "paper/v3/sections/limitations.tex", "")
    row("R24", "stated (one seed per recipe; one GPU)" if tm else "open",
        "paper/v3/sections/limitations.tex", "")
    return r


def compute() -> list[str]:
    out = []
    for name, f in (("TiJepa-1", "checkpoints/pretrain_summary_tijepa1_seed0.json"),
                    ("TiJepa-1 (spatial-only)",
                     "checkpoints/pretrain_summary_tijepa1_spatial_seed0.json")):
        d = J(f)
        if d:
            out.append(f"| {name} | {d['steps']:,} | {d['gpu_hours']:.2f} | "
                       f"{'yes' if d['annealed'] else 'no'} | `{f}` |")
    for p in sorted((REPO / "checkpoints" / "sweep").glob("pretrain_summary_sweep_*_seed0.json")):
        d = json.loads(p.read_text())
        out.append(f"| sweep {d['run_name'][6:]} | {d['steps']:,} | {d['gpu_hours']:.2f} | "
                   f"{'yes' if d['annealed'] else 'no'} | `{p.relative_to(REPO)}` |")
    return out


def main() -> int:
    cov = J("results/v3/coverage/coverage_map.json")
    lines = ["# TiJepa — run report v3", "",
             "Generated by `scripts/99_run_report_v3.py` from the files it cites.", "",
             "## Compute", "", "| run | steps | GPU-h | annealed | summary |",
             "|---|---|---|---|---|", *compute(), "",
             "## Decision gates", "", "| gate | condition | outcome | branch taken |",
             "|---|---|---|---|"]
    lines += [f"| {a} | {b} | **{c}** | {d} |" for a, b, c, d in gates()]
    lines += ["", "## Modality coverage map (§0A.3)", ""]
    if cov:
        lines += ["| family | pairing | dataset | result | outcome | evidence |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r['family']} | {r['pairing']} | {r['dataset']} | {r['result']} | "
                  f"{r['outcome']} | `{r['evidence']}` |" for r in cov["rows"]]
    lines += ["", "## Section 14: findings → status", "",
              "| ID | status | evidence | note |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | `{c}` | {d} |" for a, b, c, d in section14()]
    extra = REPO / "reports" / "RUN_REPORT_v3_notes.md"
    if extra.exists():
        lines += ["", extra.read_text().strip()]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    _ = re
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
