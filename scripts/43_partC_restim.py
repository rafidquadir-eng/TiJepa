"""F5 8.3: the restimulation test - the biologically meaningful version of Part C.

Trained immunity is defined by the response to **restimulation**, not by the resting
state (review finding B2). The v0 run scored only resting day-6 cells, where proximity
to baseline is the expected result and says nothing about training.

PLAN_v1.md 8.3 lists candidate datasets in order and says to use the first that
qualifies: a processed public matrix with naive->LPS and trained->LPS (and/or
tolerant->LPS) on **matched donors**. The search, in the plan's order:

  * **GSE212282 - QUALIFIES, and is used as the primary.** Human monocytes trained with
    beta-glucan, itaconate or 4-octyl-itaconate, differentiated for 5 days, then
    re-exposed to LPS for 4 h. `Restim_RPMI` is untrained->LPS and `Restim_BG` is
    beta-glucan-trained->LPS, on 3 matched donors (A, B, C). Per-sample MMSEQ
    gene-level files are deposited. Note the deposited file names say `d7` where the
    GEO titles say `Restim`; the correspondence was confirmed by accession
    (GSM6523220 = Donor_A_Restim_RPMI = A_d7_RPMI), not assumed.
  * **GSE185433 - also qualifies, and is reported as well** because it supplies the
    *tolerant* arm the primary lacks: `d6_RPMI_Restim`, `d6_LPS_Restim` (LPS-tolerised)
    and `d6_IL4_Restim` (IL-4-trained), 3 matched donors, same restimulation protocol.
  * GSE323311 - does **not** qualify: ATAC-seq only. Scoring an expression signature on
    an accessibility matrix is exactly the type confusion the data-type guard exists to
    prevent.
  * GSE242947 and GSE184241 were not needed, since earlier candidates qualified.

### Two readouts, both pre-specified

The only anchor that survived 8.1's donor-matching and sanity rules is the IFN-gamma
one (the LPS anchor could not be built from 2 donor-matched pairs, and the IL-4 anchor
failed its marker rule). An IFN-gamma anchor is M1-like but is *not* an LPS-response
signature, and the restimulation here is LPS, so this script also reports the
pre-specified canonical LPS/NF-kB panel directly. The panel is the more apt readout for
an LPS challenge; the anchor score is reported because 8.3 asks for it. Both are
labelled, and neither is chosen after seeing the result.

With 3 donors the exact paired sign-flip null has 8 assignments, so the smallest
attainable two-sided p is 0.25. No CI is reported (rule 3). The direction is what this
test establishes.
"""
from __future__ import annotations

import functools
import gzip
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.omics import signatures as S  # noqa: E402

OUT = REPO / "results" / "v1" / "partC"
RESTIM = REPO / "data" / "interim" / "restim"
HGNC = REPO / "data" / "raw" / "maps" / "hgnc_complete_set.txt"

# Pre-specified canonical human LPS/NF-kB response genes (the human orthologs of the
# Part B positive-control panel, plus the interferon-stimulated genes of 7.4).
LPS_PANEL = ["TNF", "IL1B", "IL6", "CXCL2", "CCL4", "NFKBIA", "TNFAIP3",
             "IL12B", "CXCL10", "IFIT1", "IRF7"]


def read_mmseq(path: Path) -> pd.Series:
    """One MMSEQ per-sample file -> log expression (`log_mu`) by Ensembl gene ID."""
    with gzip.open(path, "rt") as fh:
        first = fh.readline()
        skip = 1 if first.startswith("#") else 0
    df = pd.read_csv(path, sep="\t", skiprows=skip, compression="gzip",
                     usecols=["feature_id", "log_mu"])
    s = pd.Series(df["log_mu"].to_numpy(dtype=float),
                  index=df["feature_id"].astype(str).to_numpy())
    return s[~s.index.duplicated(keep="first")]


def build_matrix(files: dict[str, Path]) -> pd.DataFrame:
    cols = {name: read_mmseq(p) for name, p in files.items()}
    df = pd.DataFrame(cols)
    return df.dropna(how="all")


def sign_flip_test(d: np.ndarray) -> dict:
    """Exact paired sign-flip permutation on per-donor differences."""
    obs = float(np.mean(d))
    flips = [float(np.mean(d * np.array(sg)))
             for sg in itertools.product([1, -1], repeat=len(d))]
    return {
        "mean_difference": obs,
        "per_donor_difference": [float(x) for x in d],
        "n_donors": int(len(d)),
        "exact_sign_flip_p_two_sided": float(
            np.mean(np.abs(np.asarray(flips)) >= abs(obs))),
        "n_sign_assignments": len(flips),
        "minimum_achievable_p": float(2.0 / len(flips)),
        "all_donors_same_direction": bool(np.all(d > 0) or np.all(d < 0)),
        "confidence_interval": None,
        "confidence_interval_reason":
            f"{len(d)} donors is far below the 10-group minimum (PLAN_v1.md rule 3)",
    }


def score_dataset(name: str, files: dict[str, tuple[str, str]], anchor: list[str] | None,
                  ens2sym: dict[str, str], base_dir: Path) -> dict:
    """Score every sample, then run the donor-paired contrasts.

    `files` maps sample label -> (arm, donor).
    """
    paths = {}
    for label in files:
        hits = sorted(base_dir.glob(f"GSM*_{label}.gene.mmseq.txt.gz"))
        if hits:
            paths[label] = hits[0]
    if not paths:
        return {"status": "no_files", "dataset": name}
    mat = build_matrix(paths)
    syms = [str(ens2sym.get(str(i).split(".")[0], "")).upper() for i in mat.index]
    values = mat.to_numpy(dtype=float)
    n_mapped = sum(1 for s in syms if s)

    scores: dict[str, dict] = {}
    if anchor:
        scores["anchor_IFNg"] = dict(zip(
            mat.columns, S.singscore(values, syms, anchor), strict=False))
    scores["panel_LPS_NFkB"] = dict(zip(
        mat.columns, S.singscore(values, syms, LPS_PANEL), strict=False))
    # A rank-based score over 11 genes among ~55,000 barely moves, so the panel also
    # gets the statistic that actually suits a small pre-specified gene set: the mean
    # log expression of its genes. This is the readout to look at for the panel; the
    # singscore version is kept so both are visible.
    panel_rows = [i for i, sym in enumerate(syms) if sym in set(LPS_PANEL)]
    if panel_rows:
        panel_mean = np.nanmean(values[panel_rows, :], axis=0)
        scores["panel_LPS_NFkB_mean_log_expression"] = dict(zip(
            mat.columns, panel_mean, strict=False))

    arms: dict[str, dict[str, str]] = {}
    for label, (arm, donor) in files.items():
        if label in paths:
            arms.setdefault(arm, {})[donor] = label

    out: dict = {
        "status": "ok", "dataset": name,
        "n_samples_scored": len(paths),
        "n_genes": int(mat.shape[0]),
        "n_ensembl_mapped_to_symbol": n_mapped,
        "arms": {a: sorted(d) for a, d in arms.items()},
        "per_sample_scores": {k: {kk: (float(vv) if np.isfinite(vv) else None)
                                  for kk, vv in v.items()}
                              for k, v in scores.items()},
        "contrasts": {},
    }
    if anchor:
        out["anchor_genes_present"] = int(sum(1 for g in anchor if g in set(syms)))
    out["panel_genes_present"] = int(sum(1 for g in LPS_PANEL if g in set(syms)))

    naive_arm = "untrained_restim"
    for arm in arms:
        if arm == naive_arm:
            continue
        donors = sorted(set(arms[arm]) & set(arms.get(naive_arm, {})))
        if len(donors) < 2:
            out["contrasts"][f"{arm}_vs_{naive_arm}"] = {
                "status": "insufficient_donor_pairs", "n_donors": len(donors)}
            continue
        for sname, sc in scores.items():
            d = np.array([sc[arms[arm][dn]] - sc[arms[naive_arm][dn]]
                          for dn in donors])
            if not np.all(np.isfinite(d)):
                continue
            t = sign_flip_test(d)
            t["donors"] = donors
            t["readout"] = sname
            t["expected_direction"] = (
                "higher than untrained (trained immunity)"
                if "train" in arm or "BG" in arm or "IL4" in arm
                else "lower than untrained (tolerance)")
            t["direction_observed"] = ("higher" if t["mean_difference"] > 0
                                       else "lower")
            t["matches_expectation"] = bool(
                (t["mean_difference"] > 0) == ("higher" in t["expected_direction"]))
            out["contrasts"][f"{arm}_vs_{naive_arm}__{sname}"] = t
            print(f"    {arm:24s} {sname:16s} {t['mean_difference']:+.4f} "
                  f"({t['direction_observed']}, expected "
                  f"{'higher' if 'higher' in t['expected_direction'] else 'lower'}) "
                  f"p={t['exact_sign_flip_p_two_sided']:.3f} "
                  f"floor={t['minimum_achievable_p']:.3f}")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    anchors_path = OUT / "c1_anchors.json"
    anchor = None
    if anchors_path.exists():
        a = json.loads(anchors_path.read_text())
        if a["anchors"].get("IFNg", {}).get("status") == "ok":
            anchor = a["anchors"]["IFNg"]["signature"]
    ens2sym = S.load_hgnc_ensembl_map(HGNC)

    results: dict = {
        "phase": "F5 8.3 restimulation test",
        "framing": "HYPOTHESIS-GENERATING; no image is paired with any molecular "
                   "sample. This is the restimulation design that trained immunity is "
                   "actually defined by, unlike the resting scores of 8.2.",
        "dataset_search": {
            "order_from_plan": ["GSE212282", "GSE185433", "GSE323311", "GSE242947",
                                "GSE184241"],
            "GSE212282": "QUALIFIES (primary): monocytes trained with beta-glucan / "
                         "itaconate / 4-OI, differentiated 5 days, re-exposed to LPS "
                         "for 4 h; 3 matched donors; per-sample MMSEQ files deposited",
            "GSE185433": "QUALIFIES (also reported): adds the TOLERANT arm "
                         "(LPS-trained -> LPS) plus IL-4-trained -> LPS; 3 matched "
                         "donors; same restimulation protocol",
            "GSE323311": "DOES NOT QUALIFY: ATAC-seq only. Scoring an expression "
                         "signature on an accessibility matrix is the type confusion "
                         "the data-type guard exists to prevent.",
            "GSE242947": "not needed; earlier candidates qualified",
            "GSE184241": "not needed; and it has no in vitro LPS restimulation",
        },
        "readouts": {
            "anchor_IFNg": "the only anchor that passed 8.1's donor-matching and "
                           "sanity rules. It is M1-like but NOT an LPS-response "
                           "signature, while the restimulation is LPS - a mismatch "
                           "stated rather than hidden.",
            "panel_LPS_NFkB": "pre-specified canonical human LPS/NF-kB genes "
                              + ", ".join(LPS_PANEL)
                              + ". The more apt readout for an LPS challenge; "
                                "pre-specified, not chosen after seeing results.",
        },
        "statistics": "Exact paired sign-flip permutation over donors. With 3 donors "
                      "there are 8 assignments and the smallest attainable two-sided p "
                      "is 0.25, so this design cannot reach significance; the "
                      "direction is the informative quantity. No CI (rule 3).",
    }

    # ---------------- GSE212282 (primary)
    d1 = RESTIM / "GSE212282"
    if d1.exists():
        files = {}
        for donor in ("A", "B", "C"):
            files[f"{donor}_d7_RPMI"] = ("untrained_restim", donor)
            files[f"{donor}_d7_BG"] = ("beta_glucan_trained_restim", donor)
            files[f"{donor}_d7_Ita"] = ("itaconate_trained_restim", donor)
            files[f"{donor}_d7_DMI"] = ("4OI_trained_restim", donor)
            files[f"{donor}_d7_BG_DMI"] = ("beta_glucan_plus_4OI_restim", donor)
        print("GSE212282 (primary):")
        results["GSE212282"] = score_dataset("GSE212282", files, anchor, ens2sym, d1)
        results["GSE212282"]["filename_title_correspondence"] = (
            "Deposited files use d7 where GEO titles say Restim; confirmed by "
            "accession: GSM6523220 = Donor_A_Restim_RPMI = A_d7_RPMI.")
    else:
        results["GSE212282"] = {"status": "not_on_disk"}

    # ---------------- GSE185433 (tolerant arm)
    d2 = RESTIM / "GSE185433"
    if d2.exists():
        files = {}
        for donor in ("don1", "don2", "don3"):
            files[f"RPMI_Restim_{donor}"] = ("untrained_restim", donor)
            files[f"LPS_Restim_{donor}"] = ("LPS_tolerised_restim", donor)
            files[f"IL4_Restim_{donor}"] = ("IL4_trained_restim", donor)
            files[f"IL4_LPS_Restim_{donor}"] = ("IL4_plus_LPS_restim", donor)
        print("GSE185433 (tolerant arm):")
        results["GSE185433"] = score_dataset("GSE185433", files, anchor, ens2sym, d2)
    else:
        results["GSE185433"] = {"status": "not_on_disk"}

    (OUT / "c3_restimulation.json").write_text(
        json.dumps(results, indent=2, default=float) + "\n")
    print(f"wrote {OUT/'c3_restimulation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
