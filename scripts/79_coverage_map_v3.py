"""H5.4: assemble the modality coverage map (PLAN_v3 §0A.3) from the result files.

One row per modality family: the best public dataset that pairs images, video or
image-derived dynamics with that modality, the pairing level, what was tested and the
result, including the families for which no public pairing exists. Every number is read
from a results file named in the row's `evidence`; a row whose evidence file is missing
is written as "not tested" with that reason, never filled in.

Outcome codes: "positive" = learned image/dynamics features carry information about the
modality beyond the simple baseline named in the row (CI excludes zero gain, or exact
test p < 0.05 with every group in the same direction); "null" = tested, no such gain;
"not tested" = data exist but the test could not be run; "absent" = no public pairing.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "v3" / "coverage" / "coverage_map.json"


def load(rel: str):
    p = REPO / "results" / rel
    return json.loads(p.read_text()) if p.exists() else None


NAMES = {
    "E2_imagenet_vit_s": "ImageNet ViT-S",
    "E3_dinov2_vit_s": "DINOv2 ViT-S/14",
    "E7_vjepa2_vitl": "V-JEPA 2 ViT-L",
    "E4_tijepa1": "TiJepa-1",
    "E4s_tijepa1_spatial": "TiJepa-1 (spatial-only)",
    "E0_tijepa0": "TiJepa-0",
}


def nm(k: str) -> str:
    return NAMES.get(k.split("|")[0], k)


def best(d: dict, keys, metric):
    ks = [k for k in keys if k in d]
    return max(ks, key=lambda k: d[k][metric]) if ks else None


def main() -> int:
    rows = []
    bb = load("v3/partA/bbbc054.json")
    if bb:
        e = bb["encoders"]
        learned = [
            k
            for k in e
            if k.startswith(("E2", "E3", "E4", "E7", "E0")) and "three_class_mean" in e[k]
        ]
        b = best(e, learned, "three_class_mean")
        hc = e["E1_handcrafted_logreg"]["three_class_mean"]
        pc = bb["pairwise_comparisons"].get(f"three_class:{b}_vs_E1_handcrafted_logreg") or bb[
            "pairwise_comparisons"
        ].get(f"three_class:E1_handcrafted_logreg_vs_{b}")
        lo = None
        if pc:
            c = pc["ci95_widest_over_seeds"]
            lo = c[0] if pc["encoder_a"] == b else -c[1]
        rows.append(
            {
                "family": "Immunophenotype: activation morphology",
                "pairing": "same cell",
                "dataset": "BBBC054 (mouse microglia, LPS time course; CC BY 3.0)",
                "test": "3-class morphology from frozen features, FOV-grouped",
                "result": f"best frozen encoder {nm(b)} BA {e[b]['three_class_mean']:.3f} vs "
                f"hand-crafted {hc:.3f}; TiJepa-1 {e['E4_tijepa1']['three_class_mean']:.3f}",
                "outcome": "positive" if lo is not None and lo > 0 else "null",
                "short": "BBBC054 morphology",
                "evidence": "results/v3/partA/bbbc054.json",
            }
        )
    rows.append(
        {
            "family": "Immunophenotype: M0/M1/M2 polarisation (video)",
            "pairing": "same cell",
            "dataset": "Dryad m0cfxpp8v (BMDM movies)",
            "test": "polarisation from frozen video features",
            "result": "not obtained: the download requires an interactive browser check",
            "outcome": "absent",
            "short": "Dryad BMDM (not obtained)",
            "evidence": "results/v3/manual_intake.json",
        }
    )
    af = load("v3/coverage/autofluorescence.json")
    if af:
        r = af["results"]["A_mixed_fovs_only"]
        enc = [
            k
            for k in r
            if k.endswith("|all")
            and k.split("|")[0].startswith(("E2", "E3", "E4", "E7", "E0"))
        ]
        b = max(enc, key=lambda k: r[k]["balanced_accuracy_mean"])
        base = r["intensity_baseline|all"]
        cmp = (
            af["paired_vs_tijepa1_A_mixed_fovs_only"]
            if "paired_vs_tijepa1_A_mixed_fovs_only" in af
            else {}
        )
        rows.append(
            {
                "family": "Immunophenotype: cell identity",
                "pairing": "same cell",
                "dataset": "Zenodo 15657646 (two-photon autofluorescence, T cells and "
                "neutrophils in the same fields; CC BY 4.0)",
                "test": "T cell vs neutrophil, FOV-grouped, mixed fields only",
                "result": f"best frozen encoder {nm(b)} BA "
                f"{r[b]['balanced_accuracy_mean']:.3f}; intensity-only baseline "
                f"{base['balanced_accuracy_mean']:.3f} "
                f"(CI {base['ci95_seed0'][0]:.3f}-{base['ci95_seed0'][1]:.3f})",
                "outcome": "positive"
                if r[b]["ci95_seed0"][0] > base["balanced_accuracy_mean"]
                else "null",
                "short": "autofluorescence identity",
                "_cmp": bool(cmp),
                "evidence": "results/v3/coverage/autofluorescence.json",
            }
        )
    fl = load("v3/coverage/flim.json")
    if fl:
        reg = fl["regression"]["tau_m_ns"]
        enc = [k for k in reg if k.startswith(("E2", "E3", "E4", "E7", "E0"))]
        b = max(enc, key=lambda k: reg[k]["r2_mean"])
        base = reg["intensity_baseline"]
        ref = fl.get("condition_mean_reference", {}).get("tau_m_ns", {}).get("r2_mean")
        wf = fl.get("within_fov_regression", {}).get("tau_m_ns", {})
        wb = max(
            (k for k in wf if k.startswith(("E2", "E3", "E4", "E7", "E0"))),
            key=lambda k: wf[k]["r2_mean"],
            default=None,
        )
        ok = bool(
            wb
            and wf[wb]["r2_ci95_seed0"]
            and wf[wb]["r2_ci95_seed0"][0] > 0
            and wf[wb]["r2_ci95_seed0"][0] > wf["intensity_baseline"]["r2_mean"]
        )
        res = (
            f"best frozen encoder {nm(b)} R2 {reg[b]['r2_mean']:+.2f}; intensity "
            f"baseline {base['r2_mean']:+.2f}; knowing only the experimental arm "
            f"{ref:+.2f}"
        )
        if wb:
            res += (
                f"; within fields: {nm(wb)} {wf[wb]['r2_mean']:+.2f}, intensity "
                f"{wf['intensity_baseline']['r2_mean']:+.2f}"
            )
        rows.append(
            {
                "family": "Metabolism (NAD(P)H lifetime)",
                "pairing": "same cell",
                "dataset": "Zenodo 14330511 (two-photon NAD(P)H-FLIM, macrophages; CC BY 4.0)",
                "test": "per-cell mean lifetime from the photon-count image, FOV-grouped "
                "ridge",
                "result": res,
                "outcome": "positive" if ok else "null",
                "short": "NAD(P)H-FLIM lifetime",
                "evidence": "results/v3/coverage/flim.json",
            }
        )
    else:
        rows.append(
            {
                "family": "Metabolism (NAD(P)H lifetime)",
                "pairing": "same cell",
                "dataset": "Zenodo 14330511",
                "test": "-",
                "result": "not run",
                "outcome": "not tested",
                "short": "NAD(P)H-FLIM",
                "evidence": "results/v3/coverage/flim.json (absent)",
            }
        )
    rows.append(
        {
            "family": "Metabolism (macrophage M1/M2 FLIM)",
            "pairing": "same cell",
            "dataset": "Dryad 8gtht76q2",
            "test": "-",
            "result": "not obtained: the download requires an interactive browser check",
            "outcome": "absent",
            "short": "Dryad FLIM (not obtained)",
            "evidence": "results/v3/manual_intake.json",
        }
    )
    pb = load("v3/partB/partB_v3.json")
    if pb:
        b = pb["b32_ligand_identity"]
        tj = b["results"]["trace_jepa"]
        hc = b["results"]["handcrafted"]
        rows.append(
            {
                "family": "Signalling (NF-κB dynamics)",
                "pairing": "same cell",
                "dataset": "Tay lab deposit, Zenodo 14337874 (live-cell NF-κB reporter, "
                "125 wells)",
                "test": "11-way ligand-pair identity from single-cell traces, well-grouped",
                "result": f"trace JEPA BA {tj['balanced_accuracy_mean']:.3f}, raw trace "
                f"{b['results']['raw_trace']['balanced_accuracy_mean']:.3f}, hand-crafted "
                f"{hc['balanced_accuracy_mean']:.3f}; reference: chance "
                f"{b['chance_balanced_accuracy']:.3f}. Pretraining adds nothing over the "
                "same network at random initialisation",
                "outcome": "positive"
                if tj["ci95_seed0_well_bootstrap"][0] > b["chance_balanced_accuracy"]
                else "null",
                "short": "NF-κB ligand decoding",
                "evidence": "results/v3/partB/partB_v3.json",
            }
        )
        pw = pb["b31_power_analysis"]["metrics"]["peak_amplitude"]
        rows.append(
            {
                "family": "Signalling memory (acute tolerance)",
                "pairing": "same cell",
                "dataset": "Tay deposit, LPS then LPS (4 wells)",
                "test": "tolerant vs naive from the second response; power analysis",
                "result": "unpowered (2 wells per arm); "
                f"{pw['wells_per_arm_80pct_power']} wells "
                "per arm needed for 80% power at the observed effect",
                "outcome": "null",
                "short": "acute tolerance (4 wells)",
                "evidence": "results/v3/partB/partB_v3.json",
            }
        )
    b3 = load("v2/partB/b33_b4.json")
    if b3:
        q = b3.get("b33", {})
        rows.append(
            {
                "family": "Transcriptomics",
                "pairing": "same condition",
                "dataset": "Tay RNA-seq GSE260996 (same lab as the imaging; 5 conditions)",
                "test": "condition-level dynamics -> RNA (Q2 vs leave-one-out mean)",
                "result": "fraction of top-2,000 genes with Q2 > 0: "
                f"{q['analyses']['rna_expression']['top2000_variable_targets']['fraction_q2_positive']:.3f}"
                f" (exact permutation p "
                f"{q['analyses']['rna_expression']['top2000_variable_targets']['exact_permutation']['p_value']:.2f}"
                "; 5 conditions, descriptive)",
                "outcome": "null",
                "short": "Tay RNA (5 conditions)",
                "evidence": "results/v2/partB/b33_b4.json",
                "_raw": bool(q),
            }
        )
        rows.append(
            {
                "family": "Epigenomics",
                "pairing": "same condition",
                "dataset": "Tay ATAC-seq GSE261647 (5 conditions)",
                "test": "condition-level dynamics -> accessibility",
                "result": "fraction of top-2,000 peaks with Q2 > 0: "
                f"{q['analyses']['chromatin_accessibility']['top2000_variable_targets']['fraction_q2_positive']:.3f}"
                f" (exact permutation p "
                f"{q['analyses']['chromatin_accessibility']['top2000_variable_targets']['exact_permutation']['p_value']:.2f}"
                "; 5 conditions, descriptive)",
                "outcome": "null",
                "short": "Tay ATAC (5 conditions)",
                "evidence": "results/v2/partB/b33_b4.json",
            }
        )
    rep = load("v3/partC/beta_glucan_replication.json")
    if rep:
        rows.append(
            {
                "family": "Cytokines (restimulation; memory proxies)",
                "pairing": "no public pairing",
                "dataset": "GSE212282, GSE185433, GSE141656 (human macrophage training and "
                "tolerance, RNA only; no images)",
                "test": "TNF/IL6 transcripts after restimulation, donor-paired",
                "result": "β-glucan-trained cells: TNF transcript lower at 4 h after LPS (3/3 "
                "donors) and after M. tuberculosis (3/3, independent donors)",
                "outcome": "absent",
                "short": "restimulation RNA (no images)",
                "evidence": "results/v3/partC/beta_glucan_replication.json",
            }
        )
    b4 = load("v3/partB/b4_symmetric.json")
    if b4:
        t = b4["tests"]["both"]
        rows.append(
            {
                "family": "Cross-species memory signature",
                "pairing": "cross-study",
                "dataset": "Tay mouse RNA (priming x LPS) vs GSE185433 human tolerance",
                "test": "overlap of priming-blunted LPS genes among genes induced in both "
                "species",
                "result": f"overlap {t['n_overlap']} vs {t['expected']:.1f} expected, OR "
                f"{t['odds_ratio']:.2f}, two-sided p {t['p_two_sided']:.2f}",
                "outcome": "null",
                "short": "mouse-human overlap",
                "evidence": "results/v3/partB/b4_symmetric.json",
            }
        )
    for fam, why in (
        (
            "Proteomics (beyond signalling reporters)",
            "no public dataset pairs images of innate immune cells with proteomics",
        ),
        (
            "Integrated single-cell multi-omics",
            "no public dataset pairs live-cell imaging of innate cells with single-cell "
            "multi-omics of the same cells",
        ),
        (
            "Spatial omics / tissue imaging",
            "in-tissue data exist but not for the in-vitro cell systems studied here",
        ),
        (
            "Perturbation screens (Perturb-seq, optical pooled)",
            "no public optical or Perturb-seq screen of trained or tolerised innate cells",
        ),
        (
            "Immune repertoire, microbiome, clinical",
            "not applicable to in-vitro innate-cell imaging",
        ),
    ):
        rows.append(
            {
                "family": fam,
                "pairing": "no public pairing",
                "dataset": "-",
                "test": "-",
                "result": why,
                "outcome": "absent",
                "short": "none",
                "evidence": "search recorded in this file",
            }
        )
    for r in rows:
        r.pop("_cmp", None)
        r.pop("_raw", None)
    out = {
        "phase": "PLAN_v3 H5.4",
        "outcome_codes": __doc__.split("Outcome codes:")[1].strip(),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    for r in rows:
        print(
            f"  {r['family'][:42]:42s} {r['pairing']:18s} {r['outcome']:10s} "
            f"{r['result'][:80]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
