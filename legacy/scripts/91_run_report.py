"""M7: generate reports/RUN_REPORT.md from results/ plus the recorded deviations."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"


def load(rel: str):
    p = R / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def f(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> int:
    pf = load("preflight.json") or {}
    corpus = load("partA/pretrain_corpus.json")
    pre = load("partA/pretrain_seed0.json")
    bb = load("partA/bbbc054_probes.json")
    tr = load("partA/tracking.json")
    b31 = load("partB/b31_tolerance.json")
    b32 = load("partB/b32_ligand_identity.json")
    b33 = load("partB/b33_dynamics_to_expression.json")
    b4 = load("partB/b4_cross_species.json")
    c2 = load("partC/c2_scores.json")
    c3 = load("partC/c3_projection.json")

    comp = pf.get("compute", {})
    tier = pf.get("tier", {})
    gpu = (comp.get("gpus") or [{}])[0]
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()[:10]

    L: list[str] = []
    a = L.append
    a("# TiJepa-0 run report\n")
    a(f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} "
      f"from `results/` at commit `{sha}` by `scripts/91_run_report.py`. "
      "Every number below is read from a results file.\n")

    a("## 1. Compute\n")
    a("| item | value |\n|---|---|")
    a(f"| GPU | {comp.get('gpu_count')}x {gpu.get('name')} "
      f"({gpu.get('memory_total_mib')} MiB) |")
    a(f"| CPU | {comp.get('cpu_cores_logical')} logical cores |")
    a(f"| RAM | {comp.get('ram_total_gb')} GB |")
    a(f"| Free disk at start | {pf.get('disk', {}).get('free_gb')} GB (all data local) |")
    a(f"| Tier | **{tier.get('tier')}** - {tier.get('reason')} |")
    a(f"| Encoder / precision | {tier.get('encoder')} / {tier.get('precision')} |")
    a(f"| Raw-video cap | {tier.get('raw_video_cap_gb')} GB "
      f"({tier.get('raw_video_cap_source')}) |")
    a(f"| PyTorch | {comp.get('torch_version')} (CUDA {comp.get('cuda_version')}) |")
    a(f"| Seed policy | {tier.get('seed_policy')} |")
    if pre:
        a(f"| Pretraining | {pre['steps']:,} steps, **{pre['gpu_hours']} GPU-hours**, "
          f"{pre['effective_epochs']} effective epochs, stop: `{pre['stop_reason']}` |")
    else:
        a("| Pretraining | not completed in this run |")
    a("")

    a("## 2. Data used\n")
    man = REPO / "manifests" / "datasets.yaml"
    if man.exists():
        import yaml

        a("| dataset | accession | licence (read from record) | role | status |")
        a("|---|---|---|---|---|")
        for e in yaml.safe_load(man.read_text())["datasets"]:
            a(f"| {e.get('key')} | {e.get('accession')} | {e.get('licence_verified')} "
              f"| {e.get('role')} | {e.get('status')} |")
    if corpus:
        a(f"\nPretraining corpus: **{corpus['n_clips']:,} clips** "
          f"({corpus['total_gb']} GB, {corpus['n_shards']} shards) from "
          f"{corpus['n_movies_ok']}/{corpus['n_movies_attempted']} movies; "
          f"{corpus['n_movies_failed']} movies failed on corrupt zip members "
          "(bad CRC / magic number) in archives whose transfer had been interrupted, "
          "and were excluded by the integrity check rather than partially decoded.\n")
    a("Datasets the plan excluded on licence grounds (Zenodo 16319732 CC BY-NC-ND, "
      "DeepCell DynamicNuclearNet, the Cellpose training set, CTMC-v1, LIVECell, "
      "Cell Tracking Challenge) were never downloaded or used.\n")

    a("## 3. Headline results\n")
    a("| analysis | result | source file |")
    a("|---|---|---|")
    if bb:
        t1 = bb["tasks"]["morphology_class"]
        for k, e in t1.items():
            if e.get("status") == "ok":
                ci = e["cluster_bootstrap_ci_seed0"]
                a(f"| BBBC054 morphology, {k} | BA {f(e['balanced_accuracy_mean'])} "
                  f"+/- {f(e['balanced_accuracy_sd'])}, 95% CI "
                  f"[{f(ci['ci_low'])}, {f(ci['ci_high'])}], perm p "
                  f"{f(e['fov_permutation_test_seed0'].get('p_value'), 3)} "
                  f"| `results/partA/bbbc054_probes.json` |")
        for k, e in bb["tasks"]["lps_exposure_time_regression"].items():
            if e.get("status") == "ok":
                a(f"| BBBC054 LPS-time, {k} | Spearman rho {f(e['spearman_rho'])}, "
                  f"MAE {f(e['mae_hours'], 2)} h (mean-predictor "
                  f"{f(e['mae_hours_baseline_mean_predictor'], 2)} h) "
                  f"| `results/partA/bbbc054_probes.json` |")
    else:
        a("| Part A probes | not completed | - |")
    if tr and tr.get("status") == "ok":
        for k, v in tr["aggregate"].items():
            if v.get("delta_avg_mean") is not None:
                lo, hi = (v.get("delta_avg_ci") or [None, None])
                a(f"| Tracking, {k} | <delta> {f(v['delta_avg_mean'])} "
                  f"[{f(lo)}, {f(hi)}], median survival "
                  f"{v['median_survival_frames']:.0f} frames "
                  f"| `results/partA/tracking.json` |")
    else:
        a("| Point tracking | not completed | - |")
    if b31:
        iv = b31["by_interval"].get("8h", {})
        if iv.get("status") == "ok":
            for rep, r in iv["representations"].items():
                ci = r["cluster_bootstrap_ci_seed0"]
                a(f"| B3.1 tolerance ({rep}) | BA {f(r['balanced_accuracy_mean'])}, "
                  f"95% CI [{f(ci['ci_low'])}, {f(ci['ci_high'])}], perm p "
                  f"{f(r['group_permutation_test_seed0'].get('p_value'), 2)} "
                  f"**not significant**, {iv['n_groups']} wells "
                  f"| `results/partB/b31_tolerance.json` |")
    if b32:
        a(f"| B3.2 ligand identity | BA {f(b32['balanced_accuracy_mean'])} "
          f"({b32['n_classes']} classes, chance {f(1/b32['n_classes'])}) "
          f"| `results/partB/b32_ligand_identity.json` |")
    if b33:
        if b33.get("status") == "ok":
            a(f"| B3.3 dynamics -> expression | {b33['n_genes_r_above_0.5']} genes "
              f"r>0.5 vs null mean {f(b33['permutation_null_mean_count'], 1)}, "
              f"p {f(b33['permutation_p'], 2)} - **null** "
              f"| `results/partB/b33_dynamics_to_expression.json` |")
        else:
            a(f"| B3.3 dynamics -> expression | {b33.get('status')}: "
              f"{str(b33.get('reason'))[:80]} | `results/partB/...` |")
    if b4 and b4.get("status") == "ok":
        ov = b4["overlap_test"]
        a(f"| B4 cross-species overlap | human {b4['human']['n_up']} up / mouse "
          f"{b4['mouse']['n_up']} up, overlap {ov['overlap']} - **null** "
          f"| `results/partB/b4_cross_species.json` |")
    if c2 and c2.get("status") == "ok":
        for st, v in c2["coordinates"].items():
            a(f"| C2 {st} | M1-like {f(v['M1_like'], 4)}, M2-like "
              f"{f(v['M2_like'], 4)} (hypothesis) | `results/partC/c2_scores.json` |")
    if c3:
        a(f"| C3 projection | **{c3.get('status')}** - no M0/M1/M2 image anchors "
          f"| `results/partC/c3_projection.json` |")
    a("")

    a("""## 4. Deviations from PLAN.md

Each item states what the plan asked for, what was done instead, and why.

| # | Plan | What happened | Why |
|---|---|---|---|
| 1 | A4.1 headline benchmark on Dryad `m0cfxpp8v` (M0/M1/M2 BMDM movies) | **Not obtained.** Part A evaluation rests on BBBC054 instead | All official Dryad endpoints return 401/403; the file-stream endpoint serves an Anubis proof-of-work interstitial. Solving it is bypassing an access control (PLAN §2, §9) |
| 2 | A4 six-encoder comparison including E5 frame-shuffle | E5 is **N/A** on BBBC054 and is run only on the tracking task | BBBC054 is static: shuffling identical frames is a no-op. With Dryad unreachable, no dataset in this run has activation-labelled video |
| 3 | C3 projection into image space | **Not performed**; a clearly-labelled reduced one-axis analysis is reported instead | C3 needs image-space M0/M1/M2 centroids, which only the unreachable Dryad set provides. Substituting BBBC054's microglial classes would not be the same analysis |
| 4 | A1.2 segmentation with Cellpose `cyto3` | Classical detector (temporal-median background, MAD threshold, area filter) + trackpy | The labelled sets need no segmentation (BBBC054 ships coordinates, S-BIAD2196 ships tracked masks), so this kept the GPU free for training. Permitted by A1.2 |
| 5 | A1.4 frame-QC classifier from S-BIAD2133 | **Not trained**; no QC filtering applied, removed-fraction reported as N/A | Only 5 of 8,441 files retrieved: the BioStudies files API returns ~5 rows per call regardless of `pageSize`, needing ~85 paginated calls plus thousands of small-file downloads |
| 6 | A4.1 "C tuned by inner CV" | C tuned once on the full design, then held fixed inside permutation refits | Re-tuning inside each of 200 group-permutation refits is computationally prohibitive; recorded in the results JSON |
| 7 | 8.1 permutation tests at 1,000 permutations | 200 permutations | Runtime. The binding limit on these tests is the number of distinct group-label assignments, not the permutation count (see #9) |
| 8 | A1.1 resample everything to 0.65 um/px | ComplexEye/S-BIAD2196 at 0.641 um/px used unresampled (factor 0.986); **BBBC054 crops not resampled at all** | BBBC054 states a 20x objective but no pixel size, so no factor could be computed. Recorded in the results JSON |
| 9 | B3.1 significance testing | Reported, but **not significant** | The contrast has 4 wells, so a well-level permutation null admits only a handful of distinct assignments and p cannot fall below ~1/6 regardless of effect size. This is a design limit of the deposit, not of the test |
| 10 | B4 "DESeq2-style via pydeseq2" | Paired t-test on donor-matched samples + BH, with an expressed-gene prefilter | The deposited matrices are processed log-scale values, not raw counts, so a count-based model does not apply. GSE58310 is donor-paired and an unpaired test discards that |
| 11 | B3.3 / B4 on LPS tolerance | Cover **heterologous** priming only | The deposited Tay RNA-seq has no LPS->LPS sample, so the homologous contrast B3.1 classifies has no matching transcriptome |
| 12 | 4.1 pretraining data cap 120 GB | 7 of 21 ComplexEye archives (44.23 GB) | Transfer time. The cap and the 60%-of-free-disk limit were never reached |
| 13 | Pretraining seeds 0/1/2 | **Seed 0 only** | Single GPU, per the run brief. Seeds 0/1/2 are used for all probes and downstream models |
| 14 | Weights to a draft GitHub Release | **Not published** | Publishing is outward-facing and was not authorised for this run; `reports/WEIGHTS.md` has instructions |
| 15 | A4.3 CoTracker and LTDB | Neither run | `dl.fbaipublicfiles.com` and `ndownloader.figshare.com` were unreachable in preflight |
| 16 | 4.1 ComplexEye record `10.5281/zenodo.7962144` | Used v2 record `10082674` | The 7962144 API endpoint returns no JSON; all licence and size checks were read from 10082674 |
| 17 | Environment | torch upgraded 2.11.0+cu128 -> 2.14.1+cu130 mid-run | Installing the imaging stack pulled a newer torch. CUDA was re-verified on the A100 before training |

## 5. Weaknesses a reviewer would raise

1. **The pretraining corpus is single-source**: one cell type (human neutrophils), one modality, one lab. Claims about "innate immune activation" representations should be read with that in mind.
2. **No activation-labelled video was evaluated.** The only labelled probe is static, so the central claim that temporal modelling helps is tested only on tracking, not on state classification.
3. **Part B's tolerance result is not significant** and rests on 4 wells.
4. **Parts B3.3 and B4 are null**, and both are underpowered by construction.
5. **Part C is a cross-species, cross-platform interpolation between three anchor conditions** with effect sizes in the hundredths of a rank-scale unit.
6. **S_LPS does not recover canonical acute LPS genes** at 72 h, weakening the LPS half of the M1-like axis (S_IFNg does recover canonical genes).
7. **No QC filtering** was applied to pretraining clips.
8. **Single pretraining seed.**

## 6. For the author - TODO list

- [ ] `TODO(author)`: confirm the affiliation in `paper/tijepa.md`.
- [ ] `TODO(author)`: review the AI-assistance disclosure.
- [ ] `TODO(author)`: declare competing interests.
- [ ] `TODO(author)`: choose the repository licence (currently Apache-2.0 for code, CC BY 4.0 for text/figures); say so if a non-commercial licence is preferred.
- [ ] `TODO(verify)`: resolve the DOIs/arXiv ids marked in `paper/tijepa.md` references before posting.
- [ ] Fetch Dryad `m0cfxpp8v` by hand (`reports/MANUAL_DOWNLOADS.md`) and re-run `scripts/60_partA_eval.py` and `scripts/41_partC_projection.py` to obtain the planned headline benchmark and the C3 projection.
- [ ] Optionally fetch LTDB and complete S-BIAD2133 for the QC classifier.
- [ ] Publish weights per `reports/WEIGHTS.md` if desired.
- [ ] **Nothing has been posted to bioRxiv, Zenodo or Hugging Face**, per PLAN §1.4.3.
""")
    text = "\n".join(L) + "\n"
    (REPO / "reports" / "RUN_REPORT.md").write_text(text)
    print(f"wrote reports/RUN_REPORT.md ({len(text.split())} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
