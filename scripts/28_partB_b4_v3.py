"""H5.2: B4 with a symmetric human definition (PLAN_v3 §9; finding R16). EXPLORATORY.

v2 compared a mouse set defined as "LPS-induced in naive cells AND blunted by priming
(interaction)" with a human set of *all* genes lower in tolerised than untrained cells
after restimulation. v3 makes the human side mirror the mouse side, in GSE185433:

* human LPS-induced: untrained restimulated (RPMI_Restim) vs untrained unstimulated day 6
  (RPMI_d6), donor-paired moderated test, q < 0.05 and log2FC > 1 (the mouse rule);
* human blunted: the tolerance x restimulation interaction, i.e. per donor
  (LPS_Restim - LPS_d6) - (RPMI_Restim - RPMI_d6), moderated test, q < 0.05 and < 0, among
  the LPS-induced genes;
* universe: one-to-one-mapped orthologs that are LPS-induced in **both** species, so the
  test asks whether, among genes both species induce, the same ones are blunted.

The mouse side is the v2 pydeseq2 interaction model (CpG, polyI:C, both). The biological
mismatch stays: heterologous priming in mouse BMDM 4 h apart vs homologous LPS tolerance
over 6 days in human macrophages.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.omics import moderated  # noqa: E402
from tijepa.omics import signatures as S  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "b4_v2", REPO / "scripts" / "27_partB_b33_b4_v2.py"
)
V2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V2)

OUT = REPO / "results" / "v3" / "partB" / "b4_symmetric.json"


def human_symmetric(ens2sym) -> dict:
    files = {}
    for d in (1, 2, 3):
        for arm in ("LPS_Restim", "RPMI_Restim", "LPS_d6", "RPMI_d6"):
            hit = sorted(
                (V2.RESTIM / "GSE185433").glob(f"GSM*_{arm}_don{d}.gene.mmseq.txt.gz")
            )
            files[(arm, d)] = hit[0]
    mat = pd.DataFrame(
        {f"{a}_{d}": V2.RESTIM_MOD.read_mmseq(p) for (a, d), p in files.items()}
    )
    mat = mat / np.log(2.0)
    mat = mat[np.isfinite(mat).all(axis=1)]
    mat = mat[(mat[[f"RPMI_Restim_{d}" for d in (1, 2, 3)]] > 0).all(axis=1)]
    sym = np.array([str(ens2sym.get(str(i).split(".")[0], "")).upper() for i in mat.index])
    D = (1, 2, 3)
    ind = moderated.paired_moderated_test(
        mat[[f"RPMI_Restim_{d}" for d in D]].to_numpy(),
        mat[[f"RPMI_d6_{d}" for d in D]].to_numpy(),
    )
    a = (
        mat[[f"LPS_Restim_{d}" for d in D]].to_numpy()
        - mat[[f"LPS_d6_{d}" for d in D]].to_numpy()
    )
    b = (
        mat[[f"RPMI_Restim_{d}" for d in D]].to_numpy()
        - mat[[f"RPMI_d6_{d}" for d in D]].to_numpy()
    )
    inter = moderated.paired_moderated_test(a, b)
    induced = (ind.q_value < 0.05) & (ind.coefficient > 1)
    # BH within the induced set for the interaction, as the mouse rule does within induced
    q_int = np.full(len(sym), np.nan)
    idx = np.flatnonzero(induced)
    from statsmodels.stats.multitest import multipletests

    q_int[idx] = multipletests(inter.p_value[idx], method="fdr_bh")[1]
    blunted = induced & (q_int < 0.05) & (inter.coefficient < 0)
    out = {
        "dataset": "GSE185433",
        "n_expressed": int(len(sym)),
        "induced_rule": "RPMI_Restim vs RPMI_d6, q < 0.05, log2FC > 1",
        "blunted_rule": "(LPS_Restim - LPS_d6) - (RPMI_Restim - RPMI_d6) < 0, BH q < 0.05 "
        "within the induced genes",
        "prior_df_induction": float(ind.d0),
        "prior_df_interaction": float(inter.d0),
        "induced": sorted(set(sym[induced]) - {""}),
        "blunted": sorted(set(sym[blunted]) - {""}),
    }
    print(
        f"  human: {len(out['induced'])} LPS-induced, {len(out['blunted'])} blunted by "
        "tolerance"
    )
    return out


def main() -> int:
    ens2sym = S.load_hgnc_ensembl_map(V2.MAPS / "hgnc_complete_set.txt")
    orth = V2.orthologs()
    ctrl = V2.ortholog_control(orth)
    one2one = {m: next(iter(h)) for m, h in orth.items() if len(h) == 1}
    inv: dict[str, list[str]] = {}
    for m, h in one2one.items():
        inv.setdefault(h, []).append(m)
    one2one = {m: h for m, h in one2one.items() if len(inv[h]) == 1}
    mouse = V2.mouse_interaction()
    # mouse induced set (symbols), recomputed from the v2 model's rule via its outputs
    mouse_induced = set(mouse["induced_symbols"])
    human = human_symmetric(ens2sym)
    res = {
        "phase": "PLAN_v3 H5.2 B4 symmetric (finding R16)",
        "labelled": "EXPLORATORY",
        "ortholog_positive_control": ctrl,
        "orthology": "MGI one-to-one only",
        "human": {k: (v if not isinstance(v, list) else len(v)) for k, v in human.items()},
        "mouse_n_blunted": mouse["n_blunted"],
        "tests": {},
    }
    h_ind, h_bl = set(human["induced"]), set(human["blunted"])
    for mset in ("CpG", "polyIC", "both"):
        m_bl = {one2one[s] for s in mouse["symbols"][mset] if s in one2one}
        m_ind = {one2one[s] for s in mouse_induced if s in one2one} if mouse_induced else None
        uni = (m_ind & h_ind) if m_ind is not None else h_ind
        a_set, b_set = m_bl & uni, h_bl & uni
        k = len(a_set & b_set)
        table = [[k, len(a_set) - k], [len(b_set) - k, len(uni) - len(a_set) - len(b_set) + k]]
        odds, p_enr = stats.fisher_exact(table, alternative="greater")
        _, p_two = stats.fisher_exact(table)
        res["tests"][mset] = {
            "universe": len(uni),
            "universe_rule": (
                "orthologs LPS-induced in both species"
                if m_ind is not None
                else "orthologs LPS-induced in human only (mouse induced list unavailable)"
            ),
            "n_mouse_blunted": len(a_set),
            "n_human_blunted": len(b_set),
            "n_overlap": k,
            "expected": len(a_set) * len(b_set) / max(1, len(uni)),
            "odds_ratio": float(odds),
            "p_enrichment": float(p_enr),
            "p_two_sided": float(p_two),
            "overlap_genes": sorted(a_set & b_set),
        }
        t = res["tests"][mset]
        print(
            f"  {mset:7s} universe {t['universe']}, mouse {t['n_mouse_blunted']}, human "
            f"{t['n_human_blunted']}, overlap {k} (exp {t['expected']:.1f}), OR {odds:.2f}, "
            f"p_two {p_two:.3g}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
