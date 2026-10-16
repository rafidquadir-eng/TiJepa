"""F1.2: confirm the Tay trace ligand codes and the tolerance sanity check.

v0 inferred the numeric ligand codes by intersecting code sets with file names. This
script replaces that inference with the deposit's own source, checks it against the
data, enumerates every LPS->LPS well, and runs the biological sanity check that
PLAN_v1.md F1.2.2 requires before any tolerance result may be reported.

### The code mapping (now sourced, not inferred)

`figure4_5plotting.m` assigns ligand codes with the dose in a trailing comment:

    line   8:  lig1 = 1; %100 nM CpG          (CpG -> LPS single-cell replicates)
    line   9:  lig2 = 2; %1 ng LPS
    line 148:  lig1 = 3; %1000 ng/mL PIC      (polyI:C -> TNF single-cell replicates)
    line 149:  lig2 = 6; %10 ng/mL TNF

so 1 = CpG, 2 = LPS, 3 = polyI:C, 6 = TNF, with doses. This **confirms** v0's inference
that code 2 is LPS and code 6 is TNF, and **corrects** v0's inference that code 3 is
CpG: code 3 is polyI:C and code 1 is CpG.

### Column 128

`rawtraceprocessing.m` lines 77-81 write column 128 as 8 when `ligs` contains a
repeated value and 4 otherwise, which reads like a homologous/heterologous flag rather
than the inter-stimulus interval. The data decide between the two readings: the pair
(first = 0, second = 2) occurs with column 128 equal to **both** 4 and 8, so the column
cannot be a function of ligand identity, and every heterologous pair appears at both
values while homologous pairs appear only at 8. That is the signature of a crossed
interval design, so column 128 is the inter-stimulus interval in hours, and homologous
restimulation was only performed at 8 h. Both the script's wording and the empirical
resolution are recorded in the output.
"""
from __future__ import annotations

import functools
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.io import tay  # noqa: E402

OUT = REPO / "results" / "v1" / "partB"
TRACES = (REPO / "data" / "interim" / "tay" /
          "tay-lab-macrophage_memory-5242e14" / "processedtraces")

# Sourced from the deposit; see the module docstring.
LIGAND_CODES = {1: "CpG", 2: "LPS", 3: "polyIC", 6: "TNF"}
LIGAND_DOSES = {1: "100 nM", 2: "1 ng/ml", 3: "1000 ng/ml", 6: "10 ng/ml"}
NO_FIRST_LIGAND = 0  # rawtraceprocessing.m: wells with no priming stimulus

LPS = 2
C_FIRST, C_SECOND, C_GAP, C_REP, C_WELL = 126, 127, 128, 129, 130
FRAME_MIN = 6.0
STIM1_IDX = 3


def load_trace_matrix(path: Path) -> np.ndarray | None:
    m = loadmat(path)
    keys = [k for k in m if not k.startswith("__")]
    for k in keys:
        a = m[k]
        if getattr(a, "ndim", 0) == 2 and a.shape[1] >= 131:
            return np.asarray(a, dtype=float)
    return None


def second_response_amplitude(mat: np.ndarray, gap_h: float) -> np.ndarray:
    """Peak NF-kB amplitude in the window after the second stimulus, per cell.

    The second stimulus arrives `gap_h` hours after the first, which itself arrives at
    frame `STIM1_IDX`. Amplitude is the maximum of the trace in the 2 h window after
    the second stimulus, minus the mean of the 3 frames immediately before it, so that
    a cell still elevated from the first stimulus is not scored as responding.
    """
    s2 = STIM1_IDX + int(round(gap_h * 60.0 / FRAME_MIN))
    win = int(round(120.0 / FRAME_MIN))
    if s2 + win > tay.N_TRACE:
        win = tay.N_TRACE - s2
    if win <= 2:
        return np.full(mat.shape[0], np.nan)
    pre = mat[:, max(0, s2 - 3):s2].mean(axis=1)
    peak = mat[:, s2:s2 + win].max(axis=1)
    return peak - pre


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "phase": "F1.2",
        "code_mapping": {
            "source_file": "figure4_5plotting.m",
            "source_lines": {
                "1": "line 8: lig1 = 1; %100 nM CpG",
                "2": "line 9: lig2 = 2; %1 ng LPS",
                "3": "line 148: lig1 = 3; %1000 ng/mL PIC",
                "6": "line 149: lig2 = 6; %10 ng/mL TNF",
            },
            "codes": {str(k): v for k, v in LIGAND_CODES.items()},
            "doses": {str(k): v for k, v in LIGAND_DOSES.items()},
            "status": "sourced_from_deposit_code",
            "relation_to_v0": (
                "CONFIRMS v0's inference that code 2 = LPS and code 6 = TNF. "
                "CORRECTS v0's inference that code 3 = CpG: code 3 is polyI:C and "
                "code 1 is CpG. Every analysis in v1 that depends on a ligand name "
                "uses this sourced mapping."),
            "unresolved_codes": (
                "Codes 4, 5, 7, 17, 19, 20 and 21 also occur. The deposit's "
                "human-readable condition labels live in `single_cell_features/`, "
                "which was NOT deposited, so these are left unresolved rather than "
                "guessed. Code 0 marks wells with no priming stimulus "
                "(rawtraceprocessing.m writes ligs(1) into column 126). The analyses "
                "below use only codes 0 and 2, both of which are resolved."),
        },
        "column_128": {
            "script_reading": (
                "rawtraceprocessing.m lines 77-81 set column 128 to 8 when `ligs` "
                "contains a repeated value and to 4 otherwise, which reads as a "
                "homologous/heterologous flag."),
            "empirical_test": (
                "The pair (first=0, second=2) occurs with column 128 equal to both 4 "
                "and 8, so the column is not a function of ligand identity. Every "
                "heterologous pair occurs at both values; homologous pairs occur only "
                "at 8."),
            "conclusion": (
                "Column 128 is the inter-stimulus interval in hours, as v0 assumed. "
                "Homologous restimulation was performed only at the 8 h interval, "
                "which is why the matched naive-vs-primed comparison is at 8 h."),
        },
    }

    # ----------------------------------------------- enumerate wells in every file
    files, lps_lps, naive_lps = [], [], []
    for path in sorted(TRACES.glob("*.mat")):
        mat = load_trace_matrix(path)
        if mat is None:
            files.append({"file": path.name, "status": "no_131_column_matrix",
                          "note": "excluded: metadata columns absent"})
            continue
        wells: dict[tuple, dict] = {}
        for i in range(mat.shape[0]):
            key = (float(mat[i, C_WELL]), float(mat[i, C_REP]))
            w = wells.setdefault(key, {
                "first": float(mat[i, C_FIRST]), "second": float(mat[i, C_SECOND]),
                "gap_h": float(mat[i, C_GAP]), "n_cells": 0, "rows": []})
            w["n_cells"] += 1
            w["rows"].append(i)
        verified = path.name in tay.VERIFIED_LAYOUT_FILES
        files.append({
            "file": path.name, "n_cells": int(mat.shape[0]), "n_cols": int(mat.shape[1]),
            "n_wells": len(wells), "layout_verified": verified,
            "first_codes": sorted({w["first"] for w in wells.values()}),
            "second_codes": sorted({w["second"] for w in wells.values()}),
            "intervals_h": sorted({w["gap_h"] for w in wells.values()}),
        })
        if not verified:
            continue
        for (well, rep), w in wells.items():
            rec = {"file": path.name, "well": well, "replicate": rep,
                   "gap_h": w["gap_h"], "n_cells": w["n_cells"],
                   "first": w["first"], "second": w["second"]}
            if w["second"] != LPS:
                continue
            amp = second_response_amplitude(mat[w["rows"], :], w["gap_h"])
            rec["second_peak_amplitude_mean"] = float(np.nanmean(amp))
            rec["second_peak_amplitude_median"] = float(np.nanmedian(amp))
            if w["first"] == LPS:
                lps_lps.append(rec)
            elif w["first"] == NO_FIRST_LIGAND:
                naive_lps.append(rec)

    result["files"] = files
    result["lps_to_lps_wells"] = lps_lps
    result["naive_to_lps_wells"] = naive_lps
    print(f"LPS->LPS wells found: {len(lps_lps)}")
    for w in lps_lps:
        print(f"   {w['file']} well {w['well']:.0f} rep {w['replicate']:.0f} "
              f"gap {w['gap_h']:.0f} h  n={w['n_cells']}  "
              f"amp={w['second_peak_amplitude_mean']:.3f}")
    print(f"naive->LPS wells found: {len(naive_lps)}")
    for w in naive_lps:
        print(f"   {w['file']} well {w['well']:.0f} rep {w['replicate']:.0f} "
              f"gap {w['gap_h']:.0f} h  n={w['n_cells']}  "
              f"amp={w['second_peak_amplitude_mean']:.3f}")

    # ------------------------------------------------- tolerance sanity check
    gaps = sorted({w["gap_h"] for w in lps_lps})
    sanity: dict = {
        "test": "PLAN_v1.md F1.2.2: the NF-kB second-response peak amplitude for "
                "LPS->LPS must be LOWER than for naive->LPS (acute endotoxin "
                "tolerance). Compared at a matched inter-stimulus interval.",
        "matched_interval_h": gaps,
    }
    if not lps_lps or not naive_lps:
        sanity["status"] = "not_performed"
        sanity["reason"] = "no LPS->LPS or no naive->LPS well was found"
    else:
        gap = gaps[0]
        tol = [w for w in lps_lps if w["gap_h"] == gap]
        nai = [w for w in naive_lps if w["gap_h"] == gap]
        t = np.array([w["second_peak_amplitude_mean"] for w in tol])
        n = np.array([w["second_peak_amplitude_mean"] for w in nai])
        # Cohen's d on the well means; with 2 vs 2 wells this is descriptive.
        pooled = np.sqrt((t.var(ddof=1) + n.var(ddof=1)) / 2) if len(t) > 1 else np.nan
        d = float((t.mean() - n.mean()) / pooled) if pooled and np.isfinite(pooled) else None
        # exact permutation over well-label assignments (PLAN_v1.md Section 9)
        allv = np.concatenate([t, n])
        k = len(t)
        obs = t.mean() - n.mean()
        splits = list(itertools.combinations(range(len(allv)), k))
        stats = [allv[list(s)].mean() - allv[[i for i in range(len(allv))
                                              if i not in s]].mean() for s in splits]
        p_one = float(np.mean([s <= obs for s in stats]))
        sanity.update({
            "status": "ok",
            "interval_used_h": gap,
            "n_wells_lps_to_lps": len(tol),
            "n_wells_naive_to_lps": len(nai),
            "n_cells_lps_to_lps": int(sum(w["n_cells"] for w in tol)),
            "n_cells_naive_to_lps": int(sum(w["n_cells"] for w in nai)),
            "mean_amplitude_lps_to_lps": float(t.mean()),
            "mean_amplitude_naive_to_lps": float(n.mean()),
            "difference": float(obs),
            "cohens_d_well_level": d,
            "direction": "lower in LPS->LPS (tolerance, as expected)"
                         if obs < 0 else
                         "HIGHER in LPS->LPS (contradicts tolerance)",
            "expected_direction_observed": bool(obs < 0),
            "exact_permutation_n_assignments": len(splits),
            "exact_permutation_p_one_sided": p_one,
            "minimum_achievable_p": float(1.0 / len(splits)),
            "note": "With 2 wells per group the exact permutation null has only "
                    f"{len(splits)} distinct assignments, so no p below "
                    f"{1.0/len(splits):.3f} is achievable. The direction, not the "
                    "p-value, is the informative quantity here.",
        })
        print(f"\nTOLERANCE SANITY CHECK at {gap:.0f} h interval:")
        print(f"   LPS->LPS   amplitude {t.mean():.4f} ({len(tol)} wells)")
        print(f"   naive->LPS amplitude {n.mean():.4f} ({len(nai)} wells)")
        print(f"   difference {obs:+.4f}  d={d if d is None else round(d,2)}  "
              f"-> {sanity['direction']}")
        print(f"   exact permutation p = {p_one:.3f} "
              f"(min achievable {1.0/len(splits):.3f})")

    result["tolerance_sanity_check"] = sanity
    result["ligand_code_confirmed_by_sanity"] = sanity.get(
        "expected_direction_observed")
    if sanity.get("status") == "ok" and not sanity.get("expected_direction_observed"):
        result["code_mapping"]["status"] = (
            "sourced_from_deposit_code_but_sanity_check_CONTRADICTED; "
            "flagged as unconfirmed in the paper per PLAN_v1.md F1.2.2")

    (OUT / "ligand_codes.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nwrote {OUT/'ligand_codes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
