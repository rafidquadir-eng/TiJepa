"""F5 8.1-8.2: donor-matched activation anchors with sanity rules, and resting scores.

**HYPOTHESIS-GENERATING ONLY.** Nothing here pairs an image with a molecular sample.

### 8.1 Anchors (fixes review findings B1 and B3)

v0 built its anchors by comparing stimulated samples against controls **across
different donors**, in a study with 73 donors. Its LPS anchor came out as a set of
zinc-finger genes with no canonical LPS target in it - a donor-confounding artefact
rather than a signature. v1 fixes this in three ways:

1. the GSE46903 design is enumerated by differentiation x stimulus x time x donor and
   the table is saved, so the available contrasts are visible rather than assumed;
2. each anchor is built from **donor-paired** stimulus-vs-control samples with a
   moderated paired test, and a stimulus with fewer than 3 donor-matched pairs
   **cannot form an anchor**;
3. every anchor must pass a **sanity rule**: a minimum number of canonical marker
   genes among its top 100. An anchor that fails is dropped and the failure reported.

What the data allow, at GM-CSF 72 h with `con` as the matched control:
  * IFN-gamma: 7 donor-matched pairs
  * IL-4: 7 donor-matched pairs
  * soluble LPS: only **2** donor-matched pairs, so under rule (2) **no LPS anchor can
    be built**. This is the direct, mechanical explanation of the v0 LPS artefact.

8.1.2 asks for the differentiation closest to GSE58310 (human serum, M-CSF-like).
GSE46903's M-CSF arm has 24 samples whose only control is at 0 h - i.e. undifferentiated
monocytes - so an M-CSF anchor would conflate 72 h of differentiation with the
stimulus. GM-CSF is therefore used and the mismatch is stated, as 8.1.2 permits.

### 8.2 Resting scores (fixes B2's framing)

GSE58310 cells are **resting** day-6 macrophages. Trained immunity is defined by the
response to *restimulation*, so proximity to the unstimulated baseline is the expected
result here and is not evidence against training. Naive monocytes are excluded from the
bridge and shown separately, because their displacement reflects monocyte-to-macrophage
differentiation rather than activation.
"""
from __future__ import annotations

import functools
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.omics import moderated as MOD  # noqa: E402
from tijepa.omics import signatures as S  # noqa: E402

GEO = REPO / "data" / "raw" / "geo"
OUT = REPO / "results" / "v1" / "partC"

SIG_N = 100

# PLAN_v1.md 8.1.4 sanity rules: canonical markers that must appear in the top 100.
SANITY = {
    "IFNg": (["GBP1", "GBP5", "CXCL9", "CXCL10", "IDO1", "IRF1", "STAT1", "TAP1"], 5),
    "IL4": (["CCL17", "CCL22", "MRC1", "CD209", "ALOX15", "FCER2", "CCL26"], 3),
    "LPS": (["TNF", "IL1B", "IL6", "CXCL8", "CCL3", "CCL4", "CCL20", "NFKBIA",
             "TNFAIP3", "PTGS2", "IL1A"], 5),
}
MIN_DONOR_PAIRS = 3

# The stimuli we try to build anchors from, with the GEO label and the control label.
ANCHOR_SPECS = [
    ("IFNg", "IFNg", "con", "GM-CSF", "72h"),
    ("IL4", "IL4", "con", "GM-CSF", "72h"),
    ("LPS", "sLPS", "con", "GM-CSF", "72h"),
]


def field(text: str, key: str) -> str | None:
    m = re.search(rf"{key}:\s*([^|]+)", str(text))
    return m.group(1).strip() if m else None


def load_gse46903():
    mat = next((GEO / "GSE46903" / "matrix").glob("*series_matrix.txt.gz"))
    ids, gsms, vals = S.read_series_matrix_table(mat)
    # sample characteristics come from the series matrix header
    chars: dict[str, str] = {}
    import gzip
    with gzip.open(mat, "rt", errors="replace") as fh:
        acc, ch = [], []
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                acc = [t.strip().strip('"') for t in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                ch.append([t.strip().strip('"') for t in line.split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                break
    for i, g in enumerate(acc):
        chars[g] = " | ".join(c[i] for c in ch if i < len(c))
    return ids, gsms, vals, chars


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ids, gsms, vals, chars = load_gse46903()
    print(f"GSE46903: {vals.shape[0]} probes x {vals.shape[1]} samples")

    gpl = next((GEO / "platforms").glob("GPL6947*"), None)
    annot = S.load_gpl_annotation(gpl) if gpl else {}
    syms, mat = S.collapse_to_symbols(ids, vals, annot)
    print(f"  collapsed to {len(syms)} gene symbols")

    meta = []
    for g in gsms:
        c = chars.get(g, "")
        meta.append({
            "gsm": g,
            "diff": field(c, "initial differentiation"),
            "stim": field(c, "activation stimuli"),
            "time": field(c, "time"),
            "donor": field(c, "donor"),
        })
    idx = {m["gsm"]: i for i, m in enumerate(meta)}

    # ------------------------------------------------- 8.1.1 enumerate the design
    import csv
    with (OUT / "gse46903_design.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gsm", "diff", "stim", "time", "donor"])
        w.writeheader()
        w.writerows(meta)
    print(f"  wrote {OUT/'gse46903_design.csv'}")

    # ------------------------------------------------ 8.1.2 differentiation choice
    mcsf = [m for m in meta if m["diff"] == "M-CSF"]
    mcsf_control_times = sorted({m["time"] for m in mcsf if m["stim"] == "con"})
    differentiation_choice = {
        "used": "GM-CSF",
        "preferred_by_plan": "the differentiation closest to GSE58310 (human serum, "
                             "M-CSF-like)",
        "m_csf_n_samples": len(mcsf),
        "m_csf_control_times": mcsf_control_times,
        "reason": (
            f"GSE46903's M-CSF arm has {len(mcsf)} samples and its only `con` control "
            f"is at {mcsf_control_times}, i.e. undifferentiated monocytes. A 72 h "
            "M-CSF stimulus-vs-control contrast would therefore conflate 72 h of "
            "differentiation with the stimulus. PLAN_v1.md 8.1.2 permits GM-CSF with "
            "the mismatch stated, which is what is done: the anchors are GM-CSF-derived "
            "macrophages while the Part B4/8.2 target cells are human-serum/M-CSF-like, "
            "and that difference is a limitation of the bridge."),
    }
    print(f"  differentiation: GM-CSF (M-CSF control only at {mcsf_control_times})")

    # ------------------------------------------------- 8.1.3 donor-matched anchors
    anchors: dict[str, dict] = {}
    for name, stim, ctrl, diff, time in ANCHOR_SPECS:
        s_don = {m["donor"]: m["gsm"] for m in meta
                 if m["diff"] == diff and m["stim"] == stim and m["time"] == time}
        c_don = {m["donor"]: m["gsm"] for m in meta
                 if m["diff"] == diff and m["stim"] == ctrl and m["time"] == time}
        shared = sorted(set(s_don) & set(c_don) - {None})
        rec: dict = {
            "stimulus_label_in_geo": stim, "control_label_in_geo": ctrl,
            "differentiation": diff, "time": time,
            "n_stimulated_samples": len(s_don), "n_control_samples": len(c_don),
            "donor_matched_pairs": shared,
            "n_donor_matched_pairs": len(shared),
            "min_required_pairs": MIN_DONOR_PAIRS,
        }
        if len(shared) < MIN_DONOR_PAIRS:
            rec.update({
                "status": "cannot_form_anchor",
                "reason": (
                    f"only {len(shared)} donor-matched pair(s) "
                    f"({shared}), below the minimum of {MIN_DONOR_PAIRS} required by "
                    "PLAN_v1.md 8.1.3. An unpaired comparison across different donors "
                    "is what produced the v0 artefact, so no anchor is built."),
            })
            anchors[name] = rec
            print(f"  anchor {name}: CANNOT FORM ({len(shared)} donor-matched pairs)")
            continue

        t_cols = [idx[s_don[d]] for d in shared]
        c_cols = [idx[c_don[d]] for d in shared]
        res = MOD.paired_moderated_test(mat[:, t_cols], mat[:, c_cols])
        order = np.argsort(-res.t)
        top = [syms[i] for i in order[:SIG_N]]
        markers, need = SANITY[name]
        hits = [g for g in markers if g in top]
        passed = len(hits) >= need
        rec.update({
            "status": "ok" if passed else "failed_sanity_rule",
            "test": "moderated paired test (limma method), donor as the pairing unit",
            "prior_df": res.d0, "prior_variance": res.s2_prior,
            "n_genes_tested": int(len(syms)),
            "n_fdr_0.05": int(np.nansum(res.q_value < 0.05)),
            "signature_top_n": SIG_N,
            "signature": top,
            "sanity_rule": {
                "markers": markers, "required": need,
                "found": hits, "n_found": len(hits), "passed": passed,
                "marker_ranks": {g: int(np.where(np.asarray(
                    [syms[i] for i in order]) == g)[0][0]) + 1
                    for g in markers if g in syms},
            },
        })
        if not passed:
            rec["reason"] = (
                f"only {len(hits)} of the required {need} canonical {name} markers "
                f"appear in the top {SIG_N} ({hits}); the anchor is dropped and the "
                "failure reported, per PLAN_v1.md 8.1.4.")
        anchors[name] = rec
        print(f"  anchor {name}: {len(shared)} donor pairs, "
              f"{rec['n_fdr_0.05']} genes FDR<0.05, sanity {len(hits)}/{need} "
              f"-> {'PASS' if passed else 'FAIL'}  markers found: {hits}")

    usable = {k: v for k, v in anchors.items() if v.get("status") == "ok"}

    # --------------------------------------------------------- 8.2 resting scores
    g58_dir = GEO / "GSE58310" / "suppl"
    expr = next(g58_dir.glob("GSE58310_GeneExpression.csv.gz"), None)
    scores: dict = {"status": "not_run"}
    if expr and usable:
        with gzip.open(expr, "rt") as fh:
            df = pd.read_csv(fh, index_col=0)
        cols = S.parse_gse58310_columns(list(df.columns))
        # GSE58310 is keyed by Ensembl gene ID while the anchors are gene symbols, so
        # the rows are mapped through the HGNC table before scoring. Without this the
        # signature overlap is zero and every score is NaN.
        ens2sym = S.load_hgnc_ensembl_map(REPO / "data" / "raw" / "maps"
                                          / "hgnc_complete_set.txt")
        g_syms = [str(ens2sym.get(str(i).split(".")[0], "")).upper()
                  for i in df.index]
        n_mapped = sum(1 for g in g_syms if g)
        print(f"      mapped {n_mapped}/{len(g_syms)} Ensembl IDs to HGNC symbols")
        values = df.to_numpy(dtype=float)
        # singscore takes the whole genes x samples matrix and scores every column
        anchor_scores = {a: S.singscore(values, g_syms, usable[a]["signature"])
                         for a in usable}
        for a in usable:
            n_in = sum(1 for g in usable[a]["signature"] if g in set(g_syms))
            print(f"      anchor {a}: {n_in}/{len(usable[a]['signature'])} "
                  f"signature genes present in GSE58310")
        per_sample = []
        for ci, c in enumerate(cols):
            row = {"column": df.columns[ci], **c}
            for aname in usable:
                v = anchor_scores[aname][ci]
                row[f"score_{aname}"] = float(v) if np.isfinite(v) else None
            per_sample.append(row)
        sdf = pd.DataFrame(per_sample)
        sdf.to_csv(OUT / "gse58310_anchor_scores.csv", index=False)

        # group by condition, excluding naive monocytes from the bridge (8.2)
        groups: dict[str, dict] = {}
        for _, r in sdf.iterrows():
            cond = str(r.get("state") or "unknown")
            groups.setdefault(cond, {"n": 0, **{f"score_{a}": [] for a in usable}})
            groups[cond]["n"] += 1
            for a in usable:
                v = r.get(f"score_{a}")
                if v is not None and np.isfinite(v):
                    groups[cond][f"score_{a}"].append(float(v))
        summary = {}
        for cond, g in groups.items():
            summary[cond] = {"n": g["n"]}
            for a in usable:
                vs = g[f"score_{a}"]
                summary[cond][a] = {
                    "mean": float(np.mean(vs)) if vs else None,
                    "sd": float(np.std(vs, ddof=1)) if len(vs) > 1 else None,
                    "n": len(vs)}
        # ---- donor-paired contrasts against the day-6 macrophage control.
        # 4 donors, so PLAN_v1.md rule 3 forbids a bootstrap CI: the null here is the
        # exact paired sign-flip distribution over 2^4 = 16 sign assignments, whose
        # smallest attainable two-sided p is 2/16 = 0.125.
        import itertools as _it
        contrasts = {}
        for state in ("beta_glucan_trained", "LPS_tolerised", "naive_monocyte"):
            for a in usable:
                rows_s = sdf[(sdf["state"] == state) & sdf[f"score_{a}"].notna()]
                rows_c = sdf[(sdf["state"] == "control") & sdf[f"score_{a}"].notna()]
                pair_donors = sorted(set(rows_s["donor"]) & set(rows_c["donor"]))
                if len(pair_donors) < 2:
                    contrasts[f"{state}_vs_control_{a}"] = {
                        "status": "insufficient_donor_pairs",
                        "n_donor_pairs": len(pair_donors)}
                    continue
                d = np.array([
                    float(rows_s[rows_s["donor"] == dn][f"score_{a}"].iloc[0])
                    - float(rows_c[rows_c["donor"] == dn][f"score_{a}"].iloc[0])
                    for dn in pair_donors])
                obs = float(d.mean())
                flips = [float((d * np.array(sg)).mean())
                         for sg in _it.product([1, -1], repeat=len(d))]
                p_two = float(np.mean(np.abs(np.asarray(flips)) >= abs(obs)))
                contrasts[f"{state}_vs_control_{a}"] = {
                    "status": "ok",
                    "anchor": a, "state": state,
                    "n_donor_pairs": len(pair_donors), "donors": pair_donors,
                    "per_donor_difference": d.tolist(),
                    "mean_difference": obs,
                    "direction": "higher than control" if obs > 0
                                 else "lower than control",
                    "exact_sign_flip_p_two_sided": p_two,
                    "n_sign_assignments": len(flips),
                    "minimum_achievable_p": float(2.0 / len(flips)),
                    "confidence_interval": None,
                    "confidence_interval_reason":
                        f"{len(pair_donors)} donors is below the 10-group minimum "
                        "(PLAN_v1.md rule 3)",
                }
                print(f"      {state} vs control on {a}: "
                      f"{obs:+.4f} (n={len(pair_donors)} donors, exact p={p_two:.3f}, "
                      f"floor {2.0/len(flips):.3f})")

        scores = {
            "status": "ok",
            "donor_paired_contrasts": contrasts,
            "framing": (
                "RESTING day-6 macrophages. Trained immunity is defined by the "
                "RESTIMULATION response, so proximity to the unstimulated baseline is "
                "the expected result here and is not evidence against training. The "
                "restimulation test is 8.3."),
            "naive_monocytes": (
                "EXCLUDED from the bridge and reported separately: their displacement "
                "reflects monocyte-to-macrophage differentiation, not activation."),
            "anchors_used": sorted(usable),
            "per_condition": summary,
            "per_sample_file": "results/v1/partC/gse58310_anchor_scores.csv",
        }
        print(f"  8.2 scored {len(sdf)} GSE58310 samples on "
              f"{len(usable)} sanity-passed anchor(s)")
        for cond, s in summary.items():
            bits = ", ".join(f"{a} {s[a]['mean']:+.4f}" for a in usable
                             if s[a]["mean"] is not None)
            print(f"      {cond:28s} n={s['n']:2d}  {bits}")
    elif not usable:
        scores = {"status": "not_run",
                  "reason": "no anchor passed both the donor-matching requirement and "
                            "the sanity rule, so there is nothing to score against"}

    result = {
        "phase": "F5 8.1-8.2",
        "framing": "HYPOTHESIS-GENERATING ONLY; no image is paired with any molecular "
                   "sample; cross-study, cross-platform and cross-donor",
        "differentiation_choice": differentiation_choice,
        "anchors": anchors,
        "anchors_usable": sorted(usable),
        "n_anchors_usable": len(usable),
        "resting_scores": scores,
        "relation_to_v0": (
            "v0 built anchors from unpaired across-donor comparisons and reported an "
            "S_LPS of zinc-finger genes containing no canonical LPS target. Under the "
            "v1 donor-matching rule that anchor cannot be built at all, because "
            "GSE46903 provides only 2 donor-matched sLPS/con pairs at GM-CSF 72 h. The "
            "v0 signature was therefore an artefact of donor confounding, and the v1 "
            "rule catches it mechanically."),
    }
    (OUT / "c1_anchors.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n")
    print(f"wrote {OUT/'c1_anchors.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
