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
