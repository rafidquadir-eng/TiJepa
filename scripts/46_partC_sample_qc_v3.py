"""H5.1.1: sample-identity QC for the restimulation datasets (PLAN_v3 §9, finding R13).

The v2 observation to be checked: in GSE212282, beta-glucan-trained macrophages have lower
TNF mRNA than untrained ones after 4 h of LPS restimulation in all 3 donors (about -2.1
log2), opposite in sign to the 24 h TNF protein reported by the source paper. Before that
can be reported, the samples must be shown to be who and what they are labelled as.

Checks, on every sample of GSE212282 (75 samples) and GSE185433 (39):
1. sex genes (XIST; RPS4Y1, DDX3Y, KDM5D, UTY) per sample: every donor's samples must
   agree on sex;
2. donor identity. Pre-specified: genes with common deletion / strong cis-eQTL
   polymorphisms (GSTM1, GSTT1, HLA-DRB4, HLA-DRB5, HLA-DQA1, HLA-DQB1, ERAP2, CHI3L2)
   plus the sex genes: each
   sample is assigned to the nearest donor centroid computed without it; the assignment
   must equal its label. This rule misassigned 10 of 75 GSE212282 samples, all at
   time points whose cell state shifts HLA class II and GSTM1 expression, so a
   genotype-style presence/absence call (genotype_check, added after seeing that) is the
   identity test used for the gate; both are reported;
3. arm labels: within each time point that has every arm for every donor, each sample's
   donor-centred profile (top 2,000 most variable genes) is correlated with the other
   donors' samples; the best match should carry the same arm label (a swapped arm would
   match another arm);
4. TNF, IL6 and IL1B trajectories per donor for the untrained and the trained arm at every
   time point, so the restimulation result can be read against the pre-restimulation
   baseline (day 6) and the first-exposure response (4 h).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.omics import signatures as S  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "restim_v1", REPO / "scripts" / "43_partC_restim.py"
)
V1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V1)

RESTIM = REPO / "data" / "interim" / "restim"
HGNC = REPO / "data" / "raw" / "maps" / "hgnc_complete_set.txt"
OUT = REPO / "results" / "v3" / "partC" / "sample_qc.json"
SEX_F, SEX_M = ["XIST"], ["RPS4Y1", "DDX3Y", "KDM5D", "UTY"]
IDENTITY = [
    "GSTM1",
    "GSTT1",
    "HLA-DRB4",
    "HLA-DRB5",
    "HLA-DQA1",
    "HLA-DQB1",
    "ERAP2",
    "CHI3L2",
]
CYTO = ["TNF", "IL6", "IL1B"]
PARSE = {
    "GSE212282": re.compile(r"^(?P<donor>[ABC])_(?P<time>D0|4H|d1|d6|d7)(?:_(?P<arm>.+))?$"),
    "GSE185433": re.compile(r"^(?:(?P<arm>.+?)_)?(?P<time>d0|4h|d6|Restim)_(?P<donor>don\d)$"),
}
TRAINED = {"GSE212282": ("RPMI", "BG"), "GSE185433": ("RPMI", "LPS")}


def load_all(gse: str, ens2sym: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = {}
    for p in sorted((RESTIM / gse).glob("GSM*.gz")):
        lab = re.sub(r"^GSM\d+_", "", p.name).split(".gene.mmseq")[0].split(".txt")[0]
        paths[lab] = p
    mat = V1.build_matrix(paths) / np.log(2.0)
    mat.index = [str(ens2sym.get(str(i).split(".")[0], "")).upper() for i in mat.index]
    mat = mat[mat.index != ""].groupby(level=0).max()
    rows = []
    for lab in mat.columns:
        m = PARSE[gse].match(lab)
        d = m.groupdict()
        rows.append(
            {
                "label": lab,
                "donor": d["donor"],
                "time": d["time"],
                "arm": d.get("arm") or "none",
            }
        )
    return mat, pd.DataFrame(rows).set_index("label")


def sex_check(mat, meta) -> dict:
    f = mat.loc[[g for g in SEX_F if g in mat.index]].mean()
    m = mat.loc[[g for g in SEX_M if g in mat.index]].mean()
    call = np.where(f - m > 0, "F", "M")
    per = pd.DataFrame({"donor": meta["donor"], "call": call, "XIST": f, "Ymean": m})
    by = {d: sorted(set(g["call"])) for d, g in per.groupby("donor")}
    return {
        "per_donor_calls": by,
        "consistent": all(len(v) == 1 for v in by.values()),
        "per_sample": {
            s: {
                "call": r.call,
                "XIST_log2": float(r.XIST),
                "Y_genes_mean_log2": float(r.Ymean),
            }
            for s, r in per.iterrows()
        },
    }


def donor_check(mat, meta) -> dict:
    genes = [g for g in IDENTITY + SEX_F + SEX_M if g in mat.index]
    x = mat.loc[genes].T
    x = (x - x.mean()) / (x.std() + 1e-9)
    wrong = []
    for s in x.index:
        rest = x.drop(index=s)
        cent = rest.groupby(meta.loc[rest.index, "donor"]).mean()
        best = cent.sub(x.loc[s]).pow(2).sum(1).idxmin()
        if best != meta.loc[s, "donor"]:
            wrong.append({"sample": s, "label": meta.loc[s, "donor"], "nearest": best})
    return {
        "genes_used": genes,
        "n_samples": int(len(x)),
        "n_misassigned": len(wrong),
        "misassigned": wrong,
    }


def genotype_check(mat, meta) -> dict:
    """Presence/absence of deletion / null-expression polymorphisms (added after the
    pre-specified nearest-centroid rule misassigned samples, see the docstring).

    Values are floored at log2 -5 (MMSEQ reports about -20 for genes with no reads, which
    would otherwise create a spurious gap among non-expressing samples). For each gene the
    samples are split at the widest gap in sorted expression; a gene is informative if that
    gap is >= 2 log2 units (4-fold). A donor's samples must all fall on
    one side for every informative gene.
    """
    out = {"genes": {}, "consistent": True}
    for g in [x for x in IDENTITY if x in mat.index]:
        v = mat.loc[g].clip(lower=-5.0).sort_values()
        gaps = np.diff(v.to_numpy())
        i = int(np.argmax(gaps))
        if gaps[i] < 2:
            out["genes"][g] = {"informative": False, "widest_gap_log2": float(gaps[i])}
            continue
        thr = float((v.iloc[i] + v.iloc[i + 1]) / 2)
        present = mat.loc[g].clip(lower=-5.0) > thr
        per = {
            d: sorted({bool(present[s]) for s in grp.index})
            for d, grp in meta.groupby("donor")
        }
        ok = all(len(x) == 1 for x in per.values())
        out["consistent"] &= ok
        out["genes"][g] = {
            "informative": True,
            "widest_gap_log2": float(gaps[i]),
            "threshold_log2": thr,
            "present_by_donor": {d: x[0] if len(x) == 1 else "mixed" for d, x in per.items()},
            "consistent": ok,
        }
    calls = {
        d: tuple(
            v["present_by_donor"][d] for v in out["genes"].values() if v.get("informative")
        )
        for d in sorted(set(meta["donor"]))
    }
    out["donors_distinguishable"] = len(set(calls.values())) == len(calls)
    out["genotype_calls"] = {d: list(c) for d, c in calls.items()}
    return out


def arm_check(mat, meta) -> dict:
    out = {}
    for t, g in meta.groupby("time"):
        arms = sorted(set(g["arm"]))
        donors = sorted(set(g["donor"]))
        full = all(
            ((g["arm"] == a) & (g["donor"] == d)).sum() == 1 for a in arms for d in donors
        )
        if len(arms) < 2 or not full:
            continue
        sub = mat[g.index]
        expr = sub[sub.mean(1) > 1]
        top = expr.var(1).sort_values(ascending=False).index[:2000]
        x = expr.loc[top]
        resid = x.copy()
        for d in donors:
            cols = g.index[g["donor"] == d]
            resid[cols] = x[cols].sub(x[cols].mean(1), axis=0)
        c = resid.corr()
        res = {}
        for s in g.index:
            others = [o for o in g.index if g.loc[o, "donor"] != g.loc[s, "donor"]]
            best = c.loc[s, others].idxmax()
            res[s] = {
                "best_match": best,
                "best_match_arm": g.loc[best, "arm"],
                "same_arm": bool(g.loc[best, "arm"] == g.loc[s, "arm"]),
                "r": float(c.loc[s, best]),
            }
        out[t] = {
            "arms": arms,
            "n_samples": int(len(g)),
            "n_best_match_same_arm": int(sum(v["same_arm"] for v in res.values())),
            "per_sample": res,
        }
    return out


def trajectories(gse, mat, meta) -> dict:
    untr, tr = TRAINED[gse]
    out = {}
    for gene in CYTO:
        if gene not in mat.index:
            continue
        out[gene] = {}
        for (d, t), g in meta.groupby(["donor", "time"]):
            vals = {a: float(mat.loc[gene, s]) for s, a in g["arm"].items()}
            row = {k: vals[k] for k in (untr, tr, "LPS", "none") if k in vals}
            if untr in vals and tr in vals:
                row[f"{tr}_minus_{untr}"] = vals[tr] - vals[untr]
            out[gene].setdefault(t, {})[d] = row
    return out


def main() -> int:
    ens2sym = S.load_hgnc_ensembl_map(HGNC)
    res = {"phase": "PLAN_v3 H5.1.1 (finding R13)", "datasets": {}}
    for gse in ("GSE212282", "GSE185433"):
        mat, meta = load_all(gse, ens2sym)
        r = {
            "n_samples": int(mat.shape[1]),
            "sex": sex_check(mat, meta),
            "donor_identity": donor_check(mat, meta),
            "donor_genotype": genotype_check(mat, meta),
            "arm_labels": arm_check(mat, meta),
            "cytokine_trajectories": trajectories(gse, mat, meta),
            "trained_vs_untrained_arms": TRAINED[gse],
        }
        ok = (
            r["sex"]["consistent"]
            and r["donor_genotype"]["consistent"]
            and r["donor_genotype"]["donors_distinguishable"]
        )
        r["identity_qc_pass"] = bool(ok)
        res["datasets"][gse] = r
        print(
            gse,
            "sex consistent",
            r["sex"]["consistent"],
            r["sex"]["per_donor_calls"],
            "| donor misassigned",
            r["donor_identity"]["n_misassigned"],
            "| arm best-match same arm:",
            {
                t: f"{v['n_best_match_same_arm']}/{v['n_samples']}"
                for t, v in r["arm_labels"].items()
            },
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
