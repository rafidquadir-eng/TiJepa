# v0 errata

Corrections to the v0 run report and paper, established while executing v1
(`PLAN_v1.md` 11.1). Each entry says what v0 asserted, what the evidence shows, and
where that evidence is. Nothing here is a re-interpretation: in every case a machine-
written record contradicts a hand-written claim, and the machine-written record wins.

The v0 branch history is **not** rewritten to fix these; `PLAN_v1.md` forbids
force-pushing, and the record of what was claimed is itself worth keeping.

---

## E1. The torch version during v0 pretraining

**v0 claimed** (`reports/RUN_REPORT.md` §4, deviation 17): "torch upgraded
2.11.0+cu128 → 2.14.1+cu130 mid-run — installing the imaging stack pulled a newer
torch. CUDA was re-verified on the A100 before training."

**The evidence.** `results/v0/partA/pretrain_seed0.json` is written by the training code
itself, from `torch.__version__` inside the process that trained the model, and records

    "torch_version": "2.11.0+cu128"

The environment on this instance still reports `torch==2.11.0+cu128`
(`requirements-lock.txt`).

**Correction.** v0 pretraining ran on **torch 2.11.0+cu128**. The claimed upgrade either
did not take effect or was not present in the training environment; the deviation note
was written from the intention rather than from the log. All of v1 runs on
2.11.0+cu128, pinned in `requirements-lock.txt`, and every v1 stage records its own
torch version.

## E2. BBBC054 annotations: a fifth of the data was silently discarded

**v0 behaviour.** `src/tijepa/io/bbbc054.py` defined
`LABELS = ("round", "amoeboid", "stratified")` and dropped any annotation whose label
was not in that tuple.

**The evidence.** `Replicate1annotation.csv` contains exactly three labels:
`round` (43,813), `ramified` (11,143) and `amoeboid` (3,230) — 58,186 rows. The string
`stratified` does not occur in the file.

**Correction.** Every `ramified` annotation — 11,143 rows, 19.2% of the file — was
silently filtered out, so v0 ran a two-class task while believing the third class was
simply absent from the annotated replicate. This is the real content of review finding
D5b and of `PLAN_v1.md` F3.2.1: not a missing class, a wrong class name. v1 uses all
three classes and reports the three-class task as primary.

## E3. The Part B expression matrix was an accessibility matrix

**v0 behaviour.** Part B's dynamics-to-expression and cross-species analyses read
`logCPM.csv` from the root of the Tay-lab deposit and treated it as RNA-seq.

**The evidence.** That file has 18 columns (`CL1 CL2 CT1 CT2 MC1 MC2 ML1 ML2 MM1 MM2
MP1 MP2 MT1 MT2 PL1 PL2 PT1 PT2`) and row IDs of the form `chr1.4414143.4414644`. Both
match **GSE261647 ATAC-seq**. The deposit's RNA matrix is a different file,
`figure6_RNAseq/logCPM.csv`, which `F7_atac_code.R` line 38 reads as `logCPMRNA`; the
RNA series GSE260996 has 33 samples and gene-level IDs.

**Correction.** v0's "dynamics → transcriptome" and B4 cross-species analyses were run
on chromatin accessibility, not expression. This is review finding D1. v1 loads the real
RNA-seq from GEO, guards every matrix against its record
(`src/tijepa/omics/guard.py`), and analyses accessibility and expression separately and
under their own names.

## E4. The ligand code 3 was mislabelled

**v0 claimed** code 3 = CpG, inferred by intersecting code sets with file names.

**The evidence.** `figure4_5plotting.m` lines 8-9 and 148-149 assign the codes with
doses in trailing comments: `lig1 = 1; %100 nM CpG`, `lig2 = 2; %1 ng LPS`,
`lig1 = 3; %1000 ng/mL PIC`, `lig2 = 6; %10 ng/mL TNF`. The GSE261647 treatment
protocol independently states LPS 1 ng/mL, CpG 100 nM, polyI:C 1 µg/mL, TNF 10 ng/mL.

**Correction.** 1 = CpG, 2 = LPS, 3 = **polyI:C**, 6 = TNF. v0 was right about LPS and
TNF — so its tolerance analysis, which depends only on code 2, used the correct code —
and wrong about code 3, which affected the class names in its ligand-identity task.

## E5. A bootstrap CI was reported over four groups

**v0 reported** B3.1 tolerance as balanced accuracy 0.701 with a 95% CI of
[0.643, 0.752].

**The evidence.** The contrast has 4 wells, 2 per group.

**Correction.** A cluster bootstrap over 4 units does not support an interval; v1
reports per-well predictions, the exact permutation distribution over all 6 distinct
well-label assignments, and the resulting floor of p = 0.167, with no CI
(`PLAN_v1.md` rule 3). Under the v1 leave-one-well-out protocol the point estimate is
also lower, 0.547.

## E6. An unresolvable comparison was stated directionally

**v0 wrote**, in the abstract: TiJepa-0 "reaches a balanced accuracy of 0.828 … ahead of
DINOv2-S (0.769) and of the identical architecture with random weights (0.772)".

**The evidence.** On the v1 paired FOV bootstrap, DINOv2 and the random-weight control
differ by −0.007 with a 95% CI of [−0.068, 0.056] and a BH-corrected p of 0.72.

**Correction.** Those two encoders are not distinguishable on this task, so a sentence
that ranks all three in sequence overstates what the data support. v1 generates every
directional phrase from the interval and verifies the rendered paper with
`scripts/92_check_claims.py`, which is what caught this one.

---

## Not errata

Two things that look like v0 errors and are not:

- **R06.zip.** v0 recorded corrupt members in some ComplexEye archives. For R06 the
  defect is in the **published** file: its md5 matches the Zenodo record exactly and
  there is no end-of-central-directory record, with the last 57.5% of the file zeros.
  See `results/v1/complexeye_archive_defect.json`.
- **The BBBC054 axis convention.** v0 found that the BBBC054 page's statement
  ("axis-1 is x") contradicts its own data and followed the data. That was correct, and
  v1 keeps it; only the stale comment in `scripts/60_partA_eval.py` needed fixing.

---

# v1 errata

Corrections to the v1 record (`reports/RUN_REPORT_v1.md`, `paper/tijepa.md` as of commit
`94576a3`), established while executing `PLAN_v2.md`. Same rule as above: where a
machine-written record contradicts a hand-written claim, the record wins.

## V1-E1. TiJepa-0 results were labelled TiJepa-1 (finding N1)

`scripts/91_make_paper_v1.py` mapped the results key `E4_tijepa0_only_available` to the
label "E4 TiJepa-1". The v1 BBBC054 and tracking numbers printed as TiJepa-1 came from
`tijepa0_T1_seed0.pt`, because the TiJepa-1 checkpoint was still at step 14,751 of an
unfinished run. v2 derives every label from the checkpoint record stored with each result
(`src/tijepa/eval/provenance.py`), refuses unfinished TiJepa-1 checkpoints, and the claims
checker fails on any mismatch.

## V1-E2. The paper, figures and run report were generated before pretraining finished (N2)

The v1 report marked C3 "partially fixed - pretraining ran to its GPU-hour budget" when
the run had used 1.51 of 12 GPU-hours. v2 regenerates the paper, figures and report only
after both pretraining runs and all evaluations have finished.

## V1-E3. The GSE212282 "4-OI" arm is dimethyl itaconate (DMI)

`scripts/43_partC_restim.py` labelled the files `*_d7_DMI` as `4OI_trained_restim` and
`*_d7_BG_DMI` as `beta_glucan_plus_4OI_restim`. The file names, the GEO overall design
("LPS, BG, DMI, or Itaconate") and the source paper (Ferreira et al., *Cell Rep* 2023,
"Dimethyl itaconate induces long-term innate immune responses") all say **DMI**; 4-octyl
itaconate is not an arm of this series.

## V1-E4. Part C "expected directions" were assumed, not sourced (N8)

The v1 restimulation code set "expected higher" for any arm whose name contained `train`,
`BG` or `IL4` and "expected lower" otherwise, so the itaconate, DMI and β-glucan + DMI
arms received expectations no publication gave them. v2 sources each expectation from the
dataset's own publication (`results/v2/partC/arm_expectations.json`) or states none.

## V1-E5. The BBBC054 numbers came from a smoke run (N6)

`RUN_REPORT_v1.md` and the paper reported BBBC054 results computed on 867-879 cells per
seed (`logs/v1_bbbc_smoke.log`), not the ~9,000 the run command specifies. The E1
exposure-time MAE of 1.0 x 10^8 h came from `np.nan_to_num` turning infinite feature
values into 1.8 x 10^308 before the regressor. The note that E1 was "probed with logistic
regression and a random forest, better reported" was not what the code did: only
logistic regression ran. v2 reports both probes.

## V1-E6. The B3.1 sensitivity "seeds" were identical (N10)

Each seed refitted the same cells with the same deterministic folds, so the three values
were equal and the sd was exactly 0. v2 draws a per-well cell subsample per seed.

## V1-E7. The leakage check covered 8 of the 10 corpus archives (N5)

R05 and R12 had been deleted before the check ran. v2 re-fetches all 10 archives and
adds a full-resolution comparison (`results/v2/leakage_check_full.json`).
