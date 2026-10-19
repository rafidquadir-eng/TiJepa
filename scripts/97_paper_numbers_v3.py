"""Manuscript numbers for paper/v3 (both versions), generated from results only.

Writes `paper/v3/numbers.tex` (one `\\val{key}` macro per number) and
`paper/v3/labels.json` (every encoder label shown in the papers, with the checkpoint
record it was derived from, for the claims checker). A key that the manuscript uses but
this script does not define fails `scripts/98_check_claims_v3.py`, so a number cannot be
typed into the prose by hand or survive a change in the results.

A result file that does not exist yet contributes no keys; the manuscripts are built only
after every analysis has run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from tijepa.eval import provenance as PV  # noqa: E402

OUT = REPO / "paper" / "v3"
V: dict[str, str] = {}
LABELS: dict[str, dict] = {}


def load(rel: str):
    p = REPO / "results" / rel
    return json.loads(p.read_text()) if p.exists() else None


def put(key: str, value, fmt: str = "{:.3f}") -> None:
    if value is None:
        return
    if isinstance(value, str):
        V[key] = value
    elif isinstance(value, int) and fmt == "{:.3f}":
        V[key] = f"{value:,}".replace(",", "{,}")
    else:
        V[key] = fmt.format(value)


def ci(key: str, lohi, fmt: str = "{:.3f}") -> None:
    if lohi:
        V[key] = f"{fmt.format(lohi[0])}--{fmt.format(lohi[1])}"


def lab(key: str, rec) -> None:
    LABELS[PV.label_for_key(key, rec)] = {"key": key, "checkpoint": rec,
                                          "label": PV.label_for_key(key, rec)}


def pretraining() -> None:
    for run, f in (("t1", "pretrain_summary_tijepa1_seed0.json"),
                   ("ts", "pretrain_summary_tijepa1_spatial_seed0.json")):
        p = REPO / "checkpoints" / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        put(f"pre.{run}.steps", int(d["steps"]))
        put(f"pre.{run}.gpuh", d["gpu_hours"], "{:.1f}")
        put(f"pre.{run}.epochs", d["effective_epochs"], "{:.1f}")
        put(f"pre.{run}.clips", int(d["n_clips_in_corpus"]))
    mon = REPO / "results" / "v2" / "pretrain_monitor.csv"
    if mon.exists():
        import csv
        rows = list(csv.DictReader(mon.open()))
        t1 = [r for r in rows if r["run"] == "tijepa1"]
        ts = [r for r in rows if r["run"] == "tijepa1_spatial"]
        put("mon.t1.rank.first", float(t1[0]["effective_rank"]), "{:.1f}")
        put("mon.t1.rank.last", float(t1[-1]["effective_rank"]), "{:.1f}")
        put("mon.ts.rank.last", float(ts[-1]["effective_rank"]), "{:.1f}")
        put("mon.t1.std.first", float(t1[0]["std_mean"]), "{:.2f}")
        put("mon.t1.std.last", float(t1[-1]["std_mean"]), "{:.2f}")


def bbbc054() -> None:
    bb = load("v3/partA/bbbc054.json")
    if not bb:
        return
    for k, e in bb["encoders"].items():
        if not isinstance(e, dict) or "three_class_mean" not in e:
            continue
        cp = e.get("checkpoint") or (next(iter(e["checkpoint_per_seed"].values()))
                                     if e.get("checkpoint_per_seed") else None)
        if not k.startswith("E1"):
            lab(k, cp)
        put(f"bb.{k}.3c", e["three_class_mean"])
        put(f"bb.{k}.3c.sd", e["three_class_sd"])
        put(f"bb.{k}.ra", e["round_vs_amoeboid_mean"])
    put("bb.time.3c", bb["time_only_baseline"]["three_class"]["mean"])
    put("bb.ncells", int(bb["n_cells_per_seed"][0] if isinstance(bb["n_cells_per_seed"], list)
                         else list(bb["n_cells_per_seed"].values())[0]))
    for key, rec in bb["pairwise_comparisons"].items():
        task, pair = key.split(":", 1)
        a, b = rec["encoder_a"], rec["encoder_b"]
        t = "3c" if task == "three_class" else "ra"
        put(f"bb.d.{a}.{b}.{t}", rec["difference_mean_over_seeds"], "{:+.3f}")
        ci(f"bb.ci.{a}.{b}.{t}", rec["ci95_widest_over_seeds"], "{:+.3f}")
    sc = bb.get("scale_sensitivity", {})
    for s, v in (sc.get("three_class_balanced_accuracy") or {}).items():
        put(f"bb.scale.{s}", v)


def autofluorescence() -> None:
    af = load("v3/coverage/autofluorescence.json")
    if not af:
        return
    for t, short in (("A_all_fovs", "A"), ("A_mixed_fovs_only", "Am"), ("B_six_types", "B")):
        put(f"af.{short}.n", int(af["tasks"][t]["n_cells"]))
        put(f"af.{short}.fovs", int(af["tasks"][t]["n_fovs"]))
        for k, r in af["results"][t].items():
            enc, fs = k.split("|")
            put(f"af.{short}.{enc}.{fs}", r["balanced_accuracy_mean"])
            ci(f"af.{short}.{enc}.{fs}.ci", r["ci95_seed0"])
    for enc, e in af["encoders"].items():
        if e.get("checkpoint") or enc.startswith(("E2", "E3", "E7")):
            lab(enc, e.get("checkpoint"))


def forecasting() -> None:
    fc = load("v3/partA/forecast_controls.json")
    if not fc:
        return
    for k, per in fc["results"].items():
        for s, e in per.items():
            if "summary" not in e:
                continue
            sm = e["summary"]
            for m in ("ratio_vs_copy_last", "ratio_vs_slot_mean", "swap_ratio_vs_copy_last",
                      "swap_degradation", "static_ratio_vs_copy_last",
                      "pixel_ratio_decoded_pred_vs_pixel_copy_last"):
                put(f"fc.{k}.{s}.{m}", sm[m], "{:.2f}")
            g = e["gate"]
            put(f"fc.{k}.{s}.adv", g["advantage_real"], "{:.2f}")
            put(f"fc.{k}.{s}.adv.swap", g["advantage_context_swap"], "{:.2f}")
            put(f"fc.{k}.{s}.adv.static", g["advantage_static"], "{:.2f}")
            put(f"fc.{k}.{s}.p.copy", g["log_ratio_vs_copy_last"]["p_two_sided"], "{:.3f}")
            put(f"fc.{k}.{s}.p.slot", g["log_ratio_vs_slot_mean"]["p_two_sided"], "{:.3f}")
        lab(k, fc["models"][k]["checkpoint"])
    put("fc.pass", "passes" if fc["G_Forecast"]["gate"]["pass"] else "fails")


def reid() -> None:
    r = load("v3/partA/validation_reid.json")
    if not r:
        return
    for k, v in r["reid"].items():
        put(f"re.{k}", v["validation_score_mean"])
        ci(f"re.{k}.ci", v["ci95_over_movies"])
        put(f"re.{k}.rank", v["effective_rank_query"], "{:.1f}")
        for g, x in v["per_gap_mean"].items():
            put(f"re.{k}.k{g}", x)
    put("re.chance", r["reid"]["raw_pixels"]["chance_mean"])
    for k, d in r["diagnosis"].items():
        for w in ("target", "context", "token"):
            s = d["validation_clips"][w]
            put(f"dg.{k}.{w}.rank", s["effective_rank"], "{:.1f}")
            put(f"dg.{k}.{w}.top1", s["variance_in_top1"], "{:.2f}")
            put(f"dg.{k}.{w}.top8", s["variance_in_top8"], "{:.2f}")
        s = d["bbbc054_monitor_crops"]["target"]
        put(f"dg.{k}.bb.rank", s["effective_rank"], "{:.1f}")
    rc = r.get("rank_correlations")
    if rc:
        put("dg.rho.bb", rc["spearman_rank_vs_bbbc054"], "{:+.2f}")
        put("dg.rho.val", rc["spearman_rank_vs_validation"], "{:+.2f}")
    vs = load("v3/validation_set.json")
    if vs:
        put("val.movies", int(vs["n_ok"]))
        put("val.cells", int(vs["n_cells_total"]))


def association() -> None:
    a = load("v3/partA/association.json")
    if not a:
        return
    for k, s in a["summary"].items():
        for cost, v in s.items():
            c = {"position": "pos", "embedding": "emb", "embedding+position": "both"}[cost]
            for m, x in v.items():
                put(f"as.{k}.{c}.{m.replace('idf1_k9_chain', 'idf1')}", x)
    for k, t in a["tests_vs_position_only"].items():
        put(f"as.p.{k}.both9", t["9"]["p_two_sided"], "{:.3f}")
        put(f"as.d.{k}.both9", t["9"]["mean"], "{:+.3f}")


def recipe() -> None:
    r = load("v3/partA/recipe_sweep.json")
    if not r:
        return
    for vid, e in r["variants"].items():
        s = e["snapshots"].get("20000")
        if s:
            put(f"sw.{vid}.score", s["validation_score"])
            put(f"sw.{vid}.rank", s["effective_rank"], "{:.1f}")
        if e.get("summary"):
            put(f"sw.{vid}.gpuh", e["summary"]["gpu_hours"], "{:.1f}")
    g = r.get("G_Recipe") or {}
    for vid, x in (g.get("per_variant") or {}).items():
        put(f"sw.{vid}.gain", x["score_gain"], "{:+.3f}")
        put(f"sw.{vid}.rr", x["rank_ratio"], "{:.2f}")
    if g.get("decision"):
        put("sw.pass", "passes" if g.get("passing") else "fails")


def flim() -> None:
    f = load("v3/coverage/flim.json")
    if not f:
        return
    put("fl.cells", int(f["n_cells"]))
    put("fl.fovs", int(f["n_fovs"]))
    put("fl.tau.mean", f["target_summary"]["tau_m_ns"]["mean"], "{:.2f}")
    put("fl.tau.sd", f["target_summary"]["tau_m_ns"]["sd"], "{:.2f}")
    for t, short in (("tau_m_ns", "tau"), ("bound_fraction", "bf")):
        for k, r in f["regression"][t].items():
            put(f"fl.{short}.{k}", r["r2_mean"], "{:+.2f}")
            ci(f"fl.{short}.{k}.ci", r["r2_ci95_seed0"], "{:+.2f}")
    c = f["classification"]["ars_vs_dmso"]
    for k, r in c["results"].items():
        put(f"fl.ars.{k.split(' ')[0]}", r["balanced_accuracy_mean"])
    put("fl.ars.n", int(c["n_cells"]))
    put("fl.ars.fovs", int(c["n_fovs"]))


def partB() -> None:
    p = load("v3/partB/partB_v3.json")
    if p:
        b = p["b32_ligand_identity"]
        put("pb.lig.n", int(b["n_cells"]))
        put("pb.lig.wells", int(b["n_wells"]))
        put("pb.lig.classes", int(b["n_classes"]))
        put("pb.lig.chance", b["chance_balanced_accuracy"])
        for k, r in b["results"].items():
            put(f"pb.lig.{k}", r["balanced_accuracy_mean"])
            ci(f"pb.lig.{k}.ci", r["ci95_seed0_well_bootstrap"])
        for k, c in b["paired_vs_trace_jepa"].items():
            put(f"pb.lig.d.{k}", c["difference"], "{:+.3f}")
            ci(f"pb.lig.d.{k}.ci", c["ci95"], "{:+.3f}")
        put("pb.pre.cells", int(p["trace_jepa_pretraining"]["n_pool_cells"]))
        put("pb.pre.wells", int(p["trace_jepa_pretraining"]["pool_wells"]))
        for m, x in p["b31_power_analysis"]["metrics"].items():
            put(f"pb.pow.{m}.d", x["effect_size_d"], "{:.2f}")
            put(f"pb.pow.{m}.n", x["wells_per_arm_80pct_power"], "{}")
            put(f"pb.pow.{m}.nhi", x["wells_per_arm_80pct_power_at_sd_upper"], "{}")
    b4 = load("v3/partB/b4_symmetric.json")
    if b4:
        for m, t in b4["tests"].items():
            put(f"b4.{m}.uni", int(t["universe"]))
            put(f"b4.{m}.k", int(t["n_overlap"]))
            put(f"b4.{m}.exp", t["expected"], "{:.1f}")
            put(f"b4.{m}.or", t["odds_ratio"], "{:.2f}")
            put(f"b4.{m}.p", t["p_two_sided"], "{:.2f}")
        put("b4.h.induced", int(b4["human"]["induced"]))
        put("b4.h.blunted", int(b4["human"]["blunted"]))


def partC() -> None:
    qc = load("v3/partC/sample_qc.json")
    rep = load("v3/partC/beta_glucan_replication.json")
    if qc:
        d = qc["datasets"]["GSE212282"]
        put("qc.212.n", int(d["n_samples"]))
        put("qc.212.misassigned", int(d["donor_identity"]["n_misassigned"]))
        put("qc.185.n", int(qc["datasets"]["GSE185433"]["n_samples"]))
        tr = d["cytokine_trajectories"]["TNF"]
        for t in ("4H", "d1", "d6", "d7"):
            vals = [tr[t][x]["BG_minus_RPMI"] for x in "ABC"]
            put(f"pc.212.tnf.{t}.min", min(vals), "{:+.1f}")
            put(f"pc.212.tnf.{t}.max", max(vals), "{:+.1f}")
            put(f"pc.212.tnf.{t}.mean", sum(vals) / 3, "{:+.2f}")
        arm = d["arm_labels"]["d7"]
        put("qc.212.d7.match", f"{arm['n_best_match_same_arm']}/{arm['n_samples']}")
    if rep:
        a = rep["GSE141656"]
        t = a["trained_minus_untrained_after_restim"]
        for g in ("TNF", "IL6", "IL1B"):
            v = t[g]["per_donor_log2_difference"]
            put(f"pc.141.{g}.min", min(v), "{:+.2f}")
            put(f"pc.141.{g}.max", max(v), "{:+.2f}")
            put(f"pc.141.{g}.mean", sum(v) / len(v), "{:+.2f}")
        u = a["trained_minus_untrained_unstimulated"]["TNF"]["per_donor_log2_difference"]
        put("pc.141.TNF.unstim.mean", sum(u) / len(u), "{:+.2f}")
        put("pc.141.pc.tnf.min", min(a["positive_control_restim_vs_unstim_untrained"]["TNF"]
                                     ["per_donor_log2_difference"]), "{:.1f}")
        put("pc.gate", "passes" if rep["G_BetaGlucan"]["decision"].startswith("pass")
            else "does not pass")
    v2 = load("v2/partC/restim.json") or load("v2/partC/partC_restim.json")
    if v2:
        put("pc.v2.present", "yes")


def main() -> int:
    for f in (pretraining, bbbc054, autofluorescence, forecasting, reid, association,
              recipe, flim, partB, partC):
        f()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["% generated by scripts/97_paper_numbers_v3.py from results/; do not edit",
             r"\makeatletter",
             r"\newcommand{\val}[1]{\@ifundefined{val@#1}{\textbf{??#1??}}"
             r"{\csname val@#1\endcsname}}"]
    for k in sorted(V):
        if not re.fullmatch(r"[A-Za-z0-9_.+\-]+", k):
            raise SystemExit(f"bad key {k!r}")
        lines.append(rf"\expandafter\def\csname val@{k}\endcsname{{{V[k]}}}")
    lines.append(r"\makeatother")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(V, indent=1, sort_keys=True))
    (OUT / "labels.json").write_text(json.dumps(LABELS, indent=1, default=str))
    print(f"wrote {len(V)} numbers and {len(LABELS)} labels to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
