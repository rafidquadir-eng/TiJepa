"""G5 (PLAN_v2.md §8; finding N8): Part C restimulation test with sourced expectations.

v1 assumed an "expected direction" for every arm from its name (anything trained ->
higher, anything else -> lower), including itaconate and DMI, which are often reported
to blunt inflammatory responses. It also labelled the GSE212282 DMI arm "4-OI". v2:

1. **Expectations are sourced** (rule 6): `results/v2/partC/arm_expectations.json`
   records, per arm, what the dataset's own publication reports for TNF/IL-6 after LPS
   restimulation, with figure and quote. An arm without one gets "no sourced
   expectation" and no "as predicted" verdict.
2. **Restimulation is checked** from the GEO sample characteristics of every scored GSM
   (field values recorded), and by a pre-specified **positive control**: in the
   untrained arm, LPS-restimulated samples must exceed the non-restimulated day-6
   samples of the same donor for TNF, IL6 and IL1B (every donor, mean log2 difference
   > 1). A failed control blocks that dataset's results.
3. **Readouts:** primary per-gene TNF and IL6 log2-expression differences (arm minus
   untrained, per donor); secondary the pre-specified NF-kB panel without the
   interferon-stimulated genes, and the ISGs separately. Donor-level values reported.

Hypothesis-generating; no image is paired with any molecular sample. With 3 donors the
exact sign-flip null has 8 assignments, so the smallest two-sided p is 0.25.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.omics import signatures as S  # noqa: E402

OUT = REPO / "results" / "v2" / "partC"
RESTIM = REPO / "data" / "interim" / "restim"
GEO = REPO / "data" / "raw" / "geo"
HGNC = REPO / "data" / "raw" / "maps" / "hgnc_complete_set.txt"

PRIMARY = ["TNF", "IL6"]
CONTROL_GENES = ["TNF", "IL6", "IL1B"]
NFKB_PANEL = ["TNF", "IL1B", "IL6", "CXCL2", "CCL4", "NFKBIA", "TNFAIP3", "IL12B"]
ISG_PANEL = ["CXCL10", "IFIT1", "IRF7"]

_spec = importlib.util.spec_from_file_location("restim_v1", REPO / "scripts" /
                                               "43_partC_restim.py")
V1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V1)

# dataset -> {arm: file label template}, plus the untrained restimulated and the
# untrained non-restimulated (day 6) arms used by the positive control
DESIGN = {
    "GSE212282": {
        "donors": ["A", "B", "C"],
        "untrained_restim": "{d}_d7_RPMI", "untrained_rest": "{d}_d6_RPMI",
        "arms": {"beta_glucan": "{d}_d7_BG", "itaconate": "{d}_d7_Ita",
                 "DMI": "{d}_d7_DMI", "beta_glucan_plus_DMI": "{d}_d7_BG_DMI"},
    },
    "GSE185433": {
        "donors": ["don1", "don2", "don3"],
        "untrained_restim": "RPMI_Restim_{d}", "untrained_rest": "RPMI_d6_{d}",
        "arms": {"LPS_tolerised": "LPS_Restim_{d}", "IL4": "IL4_Restim_{d}",
                 "IL4_plus_LPS": "IL4_LPS_Restim_{d}"},
    },
}


def gsm_characteristics(gse: str) -> dict[str, dict]:
    """Title, source and characteristics of every GSM in the series (cached)."""
    cache = GEO / gse / "gsm_brief.txt"
    if not cache.exists():
        url = (f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}"
               "&targ=gsm&form=text&view=brief")
        with urllib.request.urlopen(url, timeout=60) as r:
            cache.write_bytes(r.read())
    out: dict[str, dict] = {}
    cur = None
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith("^SAMPLE"):
            cur = line.split("=", 1)[1].strip()
            out[cur] = {"characteristics": []}
        elif cur and line.startswith("!Sample_title"):
            out[cur]["title"] = line.split("=", 1)[1].strip()
        elif cur and line.startswith("!Sample_source_name_ch1"):
            out[cur]["source"] = line.split("=", 1)[1].strip()
        elif cur and line.startswith("!Sample_characteristics_ch1"):
            out[cur]["characteristics"].append(line.split("=", 1)[1].strip())
        elif cur and line.startswith("!Sample_treatment_protocol_ch1"):
            out[cur].setdefault("treatment_protocol", line.split("=", 1)[1].strip()[:300])
    return out


def load(gse: str, labels: list[str], ens2sym: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    paths, gsm = {}, {}
    for lab in labels:
        hit = sorted((RESTIM / gse).glob(f"GSM*_{lab}.gene.mmseq.txt.gz"))
        if hit:
            paths[lab] = hit[0]
            gsm[lab] = hit[0].name.split("_")[0]
    mat = V1.build_matrix(paths) / np.log(2.0)  # MMSEQ log_mu is a natural log
    sym = [str(ens2sym.get(str(i).split(".")[0], "")).upper() for i in mat.index]
    mat.index = sym
    mat = mat[mat.index != ""]
    mat = mat.groupby(level=0).max()  # one row per symbol: the highest-expressed ID
    return mat, gsm


def gene_diffs(mat: pd.DataFrame, a: list[str], b: list[str], genes: list[str]) -> dict:
    out = {}
    for g in genes:
        if g not in mat.index:
            out[g] = {"status": "absent"}
            continue
        d = np.array([mat.loc[g, x] - mat.loc[g, y] for x, y in zip(a, b, strict=True)])
        out[g] = {"per_donor_log2_difference": [float(v) for v in d],
                  **{k: v for k, v in V1.sign_flip_test(d).items()
                     if k not in ("per_donor_difference",)}}
    return out


def panel_diff(mat: pd.DataFrame, a: list[str], b: list[str], genes: list[str]) -> dict:
    g = [x for x in genes if x in mat.index]
    d = np.array([mat.loc[g, x].mean() - mat.loc[g, y].mean()
                  for x, y in zip(a, b, strict=True)])
    return {"genes": g, "per_donor_mean_log2_difference": [float(v) for v in d],
            **{k: v for k, v in V1.sign_flip_test(d).items()
               if k not in ("per_donor_difference",)}}


def verdict(expected, observed_sign: int) -> str:
    if expected in (None, "no sourced expectation"):
        return "no sourced expectation"
    want = 1 if expected == "higher" else -1
    return "matches the sourced direction" if want == observed_sign else \
        "opposite to the sourced direction"


def run(gse: str, ens2sym: dict[str, str], expect: dict) -> dict:
    des = DESIGN[gse]
    donors = des["donors"]
    labels = ([des["untrained_restim"].format(d=d) for d in donors]
              + [des["untrained_rest"].format(d=d) for d in donors]
              + [t.format(d=d) for t in des["arms"].values() for d in donors])
    mat, gsm = load(gse, labels, ens2sym)
    chars = gsm_characteristics(gse)
    restim = [des["untrained_restim"].format(d=d) for d in donors]
    rest = [des["untrained_rest"].format(d=d) for d in donors]

    sample_records = {lab: {"gsm": g, **chars.get(g, {})} for lab, g in gsm.items()}
    restim_text_ok = {}
    for lab, rec in sample_records.items():
        blob = " ".join([rec.get("title", ""), rec.get("source", "")]
                        + rec.get("characteristics", [])).lower()
        is_rest = lab in rest
        says_restim = bool(re.search(r"then exposed to lps|restim", blob))
        restim_text_ok[lab] = {"scored_as": "non-restimulated day-6 control" if is_rest
                               else "LPS-restimulated",
                               "geo_text_mentions_restimulation": says_restim}

    label_ok = all(v["geo_text_mentions_restimulation"] == (v["scored_as"] ==
                                                            "LPS-restimulated")
                   for v in restim_text_ok.values())
    ctrl = gene_diffs(mat, restim, rest, CONTROL_GENES)
    passed = label_ok and all(c.get("status") != "absent"
                 and all(v > 0 for v in c["per_donor_log2_difference"])
                 and c["mean_difference"] > 1 for c in ctrl.values())
    res = {"dataset": gse, "donors": donors,
           "samples": sample_records, "restimulation_labels": restim_text_ok,
           "positive_control": {
               "rule": "untrained LPS-restimulated minus untrained non-restimulated day-6, "
                       "same donor: TNF, IL6 and IL1B higher in every donor with mean "
                       "log2 difference > 1",
               "geo_labels_consistent": label_ok,
               "genes": ctrl, "status": "PASS" if passed else "FAIL"}}
    print(f"{gse}: positive control {res['positive_control']['status']} "
          + ", ".join(f"{g} {c.get('mean_difference', float('nan')):+.2f}"
                      for g, c in ctrl.items()))
    if not passed:
        res["status"] = "blocked: restimulation positive control failed"
        return res

    arms = {}
    for arm, tmpl in des["arms"].items():
        a = [tmpl.format(d=d) for d in donors]
        if not all(x in mat.columns for x in a):
            arms[arm] = {"status": "files missing"}
            continue
        e = expect.get(f"{gse}:{arm}", {})
        exp = e.get("expected")
        primary = gene_diffs(mat, a, restim, PRIMARY)
        for g, rec in primary.items():
            ex = exp.get(g) if isinstance(exp, dict) else exp
            rec["sourced_expectation"] = ex or "no sourced expectation"
            if "mean_difference" in rec:
                rec["verdict"] = verdict(ex, int(np.sign(rec["mean_difference"])))
        arms[arm] = {"primary_TNF_IL6": primary,
                     "secondary_nfkb_panel": panel_diff(mat, a, restim, NFKB_PANEL),
                     "secondary_isg_panel": panel_diff(mat, a, restim, ISG_PANEL),
                     "expectation_source": {k: e.get(k) for k in ("where", "quote", "note")
                                            if e.get(k)}}
        print(f"  {arm:22s} " + "  ".join(
            f"{g} {primary[g].get('mean_difference', float('nan')):+.2f} "
            f"[{primary[g].get('verdict', '-')}]" for g in PRIMARY))
    res["arms"] = arms
    bg = arms.get("beta_glucan", {}).get("primary_TNF_IL6", {})
    if any(r.get("verdict") == "opposite to the sourced direction" for r in bg.values()):
        res["beta_glucan_opposite"] = {
            "statement": "beta-glucan points opposite to its sourced direction on the "
                         "primary readouts",
            "candidate_explanations": [
                "readout and timing: the sourced result is cytokine protein after 24 h of "
                "LPS; the deposited samples are RNA after 4 h of LPS, when TNF transcripts "
                "may already be past their peak",
                "readout depth: one bulk RNA-seq library per donor and arm, against an "
                "ELISA on pooled independent experiments (n = 9-12) in the source",
                "donor number: 3 donors here, with a two-sided permutation floor of 0.25",
            ],
            "asserted": None}
    res["status"] = "ok"
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    expect = json.loads((OUT / "arm_expectations.json").read_text())["arms"]
    ens2sym = S.load_hgnc_ensembl_map(HGNC)
    res = {"phase": "G5", "framing": "HYPOTHESIS-GENERATING; condition-level, cross-study",
           "expectations": "results/v2/partC/arm_expectations.json",
           "statistics": "exact paired sign-flip over 3 donors (8 assignments, two-sided "
                         "floor p = 0.25); no CI (rule 3)",
           "readouts": {"primary": PRIMARY, "secondary_nfkb_panel": NFKB_PANEL,
                        "secondary_isg_panel": ISG_PANEL,
                        "note": "the v1 panel mixed NF-kB targets with interferon-"
                                "stimulated genes; they are now reported separately"}}
    for gse in DESIGN:
        res[gse] = run(gse, ens2sym, expect)
    (OUT / "c3_restimulation.json").write_text(json.dumps(res, indent=2, default=float)
                                               + "\n")
    print(f"wrote {OUT / 'c3_restimulation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
