# TiJepa v1: corrected computational preprint

This replaces the v0 preprint on the same branch. It executes `PLAN_v1.md`, the fix plan
written from the v0 review, and it is a **correction** rather than an extension: the
headline numbers of v0 do not survive correct statistics, and we say so rather than
quietly restating them.

Full detail: [`reports/RUN_REPORT_v1.md`](reports/RUN_REPORT_v1.md).
Corrections to the v0 record: [`reports/V0_ERRATA.md`](reports/V0_ERRATA.md).
Manuscript: [`paper/tijepa.md`](paper/tijepa.md).

**Squash-merge recommended** — it also drops the 39 MB embeddings blob that v0 left in
this branch's history, which cannot be removed any other way without a force-push.

---

## What changed, and why it matters

### Four bugs in v0 that changed its results

Each was found mechanically, and each is now pinned by a test so it cannot return.

1. **An accessibility matrix was analysed as expression.** The Tay deposit ships its
   ATAC matrix as a bare `logCPM.csv` at its root; the RNA matrix of the same name lives
   in `figure6_RNAseq/`. v0 read the wrong one, so its dynamics-to-transcriptome and
   cross-species analyses were run on chromatin accessibility. v1 loads the real RNA-seq
   from GEO, and a **data-type guard** now checks every matrix's row-ID class, sample
   count and assay against its GEO record before anything else touches it.
2. **A fifth of the BBBC054 annotations were silently discarded.** The loader filtered
   on `LABELS = ("round", "amoeboid", "stratified")`, but `stratified` does not occur in
   the annotation file — the third class is `ramified`. 11,143 of 58,186 annotations
   were dropped and v0 ran a two-class task believing the third class was absent. This
   is the real content of review finding D5b: not a missing class, a wrong name.
3. **A bootstrap CI was reported over four groups.** B3.1's tolerance contrast has 4
   wells; v0 gave it a 95% CI. Under leave-one-well-out the point estimate is also lower.
4. **The abstract ranked two encoders that are not distinguishable.** DINOv2 and the
   random-weight control differ by −0.007 with a CI of [−0.068, 0.056]. The new claims
   checker caught this on its first run against the v0 text.

### The results that replace them

- **Part B's tolerance contrast cannot reach significance, and we now say so.** With 2
  wells per group the exact permutation null has 6 distinct assignments, so **no p below
  0.167 exists**, whatever the classifier does. We report per-well predictions, the
  exact distribution and that floor, and no interval. v0 reported 0.701 with a CI.
- **The v0 LPS anchor cannot be built at all.** Under donor matching, GSE46903 supplies
  only 2 donor-matched sLPS/control pairs at GM-CSF 72 h against a minimum of 3. That is
  the mechanical explanation of v0's "LPS signature" of zinc-finger genes containing no
  canonical LPS target: it was built across different donors in a 73-donor study. Only
  the IFN-γ anchor survives, and the two-axis M1/M2 framing collapses to one axis.
- **The new restimulation test is the one that bears on trained immunity**, and it is
  directionally mixed: tolerance points the way tolerance should and three of four
  training arms point the way training should, but **β-glucan, the canonical training
  stimulus, points the other way**. Nothing is significant with three donors.
- **Both dynamics-to-omics analyses and the cross-species comparison are null**, now
  computed on the correct matrices and reported as nulls.

### Real leakage, found

All six S-BIAD2196 evaluation movies were fingerprinted against every ComplexEye movie.
`08_H4` matches `R08.zip:H4.tiff` at **r = 0.9975** with identical dimensions: the
evaluation movies are drawn from the pretraining collection. The matching *pretraining*
movie is excluded, so the evaluation movie stays usable.

The deposit also holds **6 movies, not 12** — the BioStudies API serves only 5 rows
however it is paged, and the hidden sixth is precisely the one that is duplicated in the
corpus. v0, using 5 movies, never saw it.

### A defect in the published ComplexEye deposit

`R06.zip` has **no end-of-central-directory record and is 57.5% zero padding**, while
its md5 matches the Zenodo record exactly. The defect is in the deposited file, not in
our transfer, and is reproducible by anyone
(`results/v1/complexeye_archive_defect.json`). This is distinct from `R08.zip`, which
genuinely failed its checksum here and was re-fetched and verified.

### The objective change

TiJepa-1 mixes the v0 spatial tube mask with a **future mask** whose targets are the last
four frames, making the predictor a latent forecaster, with a matched-step spatial-only
ablation to isolate it. A new forecasting evaluation scores each model against copying
its own last observed latent.

---

## What is NOT in this version

- **The planned headline benchmark.** Dryad `doi:10.5061/dryad.m0cfxpp8v` is an author
  prerequisite: every automated endpoint serves a proof-of-work interstitial, which we
  treat as an access control not to be bypassed, so it had to be fetched by hand and was
  not provided. **No dataset here provides video of innate immune cells with
  activation-state labels** — the paper's own thesis restated as a limitation. Fetching
  it is the first TODO(author) and unblocks three analyses at once.
- **The C3 projection**, blocked twice over: no image anchors, and no surviving omics
  anchor whose stimulus matches an image label. v0's one-axis substitute was circular
  and is **deleted** from code, results and paper rather than relabelled.
- **A second cell type in the corpus.** PhagoStat microglia video did not fit the
  remaining budget after Zenodo averaged 56 minutes per 10 GB archive, which also capped
  the corpus at 10 of 21 archives.

---

## Engineering

- `requirements-lock.txt` pins 122 packages, and settles a v0 contradiction: training
  ran on **torch 2.11.0+cu128**, which is what the training process itself recorded, not
  the upgrade the v0 report claimed.
- `scripts/run_all_v1.sh` reproduces what ran, with no `|| true`: optional stages check
  their inputs and write an explicit status.
- **54 tests**, ruff clean. The new ones pin the data-type guard, the BBBC054 coordinate
  convention *and* that the transposed convention would be clearly worse, the sourced
  Tay ligand codes, the tolerance direction, the mask generators, the table writer and
  the claims checker.
- Every reference resolved against Crossref or DataCite, each resolved title
  keyword-checked — because the MGI DOI the plan suggested resolves cleanly to an
  unrelated correction notice, and a title query for another reference returned a
  confidently wrong paper.
- The v0 checkpoint is backed up to the author's private Drive with matching md5s.

## Section 12 status

PLAN_v1.md's traceability table is reproduced in the run report with a status and an
evidence file for **every** one of its 36 findings, derived from the result files rather
than asserted: **30 fixed, 3 partially fixed, 3 not fixed**, each of the six with its
reason stated in the cell.

## TODO(author)

1. **Fetch Dryad `doi:10.5061/dryad.m0cfxpp8v` by hand** and re-run
   `scripts/64_dryad_v1.py` and `scripts/44_partC_projection_v1.py`. Highest value by
   far: it unblocks the Part A video headline, the frame-shuffle control on labelled
   video, and the C3 projection.
2. Optionally fetch the LTDB collection for the tracking evaluation.
3. Confirm the affiliation in `paper/tijepa.md`.
4. Review the AI-assistance disclosure (§4.10) and declare competing interests (§4.11).
5. Confirm the licence choice: Apache-2.0 for code, CC BY 4.0 for text and figures.
6. Decide whether to publish model weights (`reports/WEIGHTS.md`).

Nothing was posted to bioRxiv, Zenodo or Hugging Face, no GitHub Release was created, no
weights were published, `main` was never pushed to and nothing was force-pushed. Google
Drive backups are the author's private storage, authorised by the plan, and no
credential appears anywhere in this repository.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
