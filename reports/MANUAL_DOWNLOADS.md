# Datasets the author must fetch manually

These were specified in `PLAN.md` but could not be retrieved in this run. In every case
the block is an access control or an unreachable host, and PLAN.md §2 and §9 forbid
working around either. Nothing here was substituted, simulated, or filled in.

---

## 1. Dryad `doi:10.5061/dryad.m0cfxpp8v` — the planned headline Part A benchmark

**This is the most consequential gap in the run.** It was to be the headline
polarization benchmark (PLAN.md A4.1) and the image-side anchor for the Part C
projection (A/C3).

| | |
|---|---|
| Record | <https://datadryad.org/dataset/doi:10.5061/dryad.m0cfxpp8v> |
| Licence | **CC0-1.0**, read from the Dryad API (`license` field) |
| Size | 2,230,930,533 bytes, 19 files (18 multi-page TIFFs + README.md) |
| Content | BMDM phase-contrast movies, 5 min/frame, 10× = 2.155 px/µm and 20× = 4.31 px/µm; M0 (4 files), M1 = LPS 100 ng/ml (6), M2 = IL-4 20 ng/ml (8); 20 mice, 5 batches |
| Paper | Sun & Kesapragada et al., "A data-driven approach to establishing cell motility patterns as predictors of macrophage subtypes and their relation to cell morphology" |

**Why it failed.** Every official endpoint refused an automated client:

```
GET https://datadryad.org/api/v2/files/<id>/download                  -> HTTP 401
GET https://datadryad.org/api/v2/datasets/<doi>/download              -> HTTP 401
GET https://datadryad.org/downloads/file_stream/<id>                  -> HTTP 403
GET https://datadryad.org/downloads/file_stream/<id>  (browser UA)    -> HTTP 200, but the
    body is an Anubis proof-of-work interstitial ("Validating...",
    /.within.website/x/xess/xess.min.css), not the file.
```

Solving a proof-of-work challenge is bypassing an access control, so the run stopped and
recorded the dataset as unreachable. The file IDs and their published SHA-256 digests are
in `results/downloads/` logic and the Dryad API; `scripts/11_download_labelled.py` already
verifies downloads against those digests.

**How to fetch it.** Open the record in a normal browser and use "Download dataset", then:

```bash
mkdir -p data/raw/dryad_bmdm
# unzip the downloaded archive into that directory, then:
python3 -I scripts/11_download_labelled.py --only dryad   # re-verifies checksums
```

**What it unlocks once present.** `scripts/60_partA_eval.py` gains the M0/M1/M2
leave-one-batch-out probe with all six encoders (including E5 frame-shuffle, which is
undefined on the static BBBC054 data), and Part C3 gains the image-space anchors
`c_M0, c_M1, c_M2` needed to project the omics coordinates. Both are currently absent,
not approximated.

---

## 2. LTDB — Leukocyte Tracking Database (point-tracking evaluation)

| | |
|---|---|
| Record | Figshare collection <https://doi.org/10.6084/m9.figshare.c.3827890>, also `ltdb.info` |
| Licence | article CC BY 4.0, metadata CC0 (per the data descriptor) |
| Content | 38 4D two-photon intravital videos; 728 consensus tracks / 44,722 annotations |

**Why it failed.** `ndownloader.figshare.com` — Figshare's file-delivery host — was
unreachable from this instance in preflight (`results/preflight.json`,
`unreachable_hosts`). `figshare.com` itself resolved, so this is a delivery-host
block, not a missing record.

**How to fetch it.** Download the collection from the Figshare page in a browser into
`data/raw/ltdb/`, then extend `scripts/61_tracking_eval.py` with an LTDB loader (the
plan specifies max-projections for 2D tracking).

---

## 3. CoTracker weights — off-the-shelf point tracker (PLAN.md A4.3 iii)

**Why it failed.** `dl.fbaipublicfiles.com` was unreachable in preflight. The tracking
evaluation therefore compares the nearest-detection baseline against patch-token
correlation tracking for each encoder, and reports no third-party tracker number.

---

## 4. Immunemap

Deferred as future work, which PLAN.md A4.3 explicitly permits ("If time is short, run
S-BIAD2196 + LTDB and mark Immunemap as future work"). Not attempted, so no claim is
made about its availability.

---

## 5. Partially retrieved, usable as-is

| Dataset | State | Effect |
|---|---|---|
| ComplexEye Zenodo `10082674` | 7 of 21 archives (44.23 GB, 302 movies). `R06.zip` was left mid-transfer and is excluded by a zip-integrity check | Pretraining corpus is ComplexEye-only; the 120 GB cap was never reached |
| S-BIAD2196 | 5 of 12 movies | Tracking evaluation uses 5 held-out movies |
| S-BIAD2133 (frame QC) | 5 of 8,441 files | **The QC classifier of PLAN.md A1.4 was not trained**; no QC filtering was applied to pretraining clips, and the removed-fraction figure is reported as not applicable. The BioStudies files API returns ~5 rows per request regardless of `pageSize`, so the full listing needs ~85 paginated calls followed by thousands of small-file downloads; this exceeded the time it was worth in this run. `scripts/11_download_labelled.py` now paginates correctly, so `--only s2133` will complete given time |
| GSE85243 | Series matrix contains no expression table | Excluded from Part C replication |
| GSE172116 | Only bigWig/bedGraph coverage tracks deposited | Excluded from gene-level Part C replication |
