"""Manuscript checks for paper/v3 (both versions); exit 1 on any failure.

1. **Numbers**: every `\\val{key}` used in a manuscript is defined in paper/v3/numbers.tex
   (generated from results by scripts/97_paper_numbers_v3.py), so no number is typed by
   hand.
2. **Directional claims**: a sentence that names two encoders and a directional word is
   checked against the stored paired comparisons (BBBC054 cluster bootstraps over 60
   fields; autofluorescence bootstraps over fields; ligand decoding over wells); the
   comparison's sign must match the word (src/tijepa/viz/claims.py).
3. **Banned strings**: no internal process language (plan sections, gate names, finding
   ids, version tags, results paths) in either manuscript body.
4. **Biology**: no sentence states or implies that trained immunity was detected in images.
5. **Anonymity** (TMLR version only): no author name, e-mail, affiliation or repository
   owner.
6. **Labels**: every model label written to paper/v3/labels.json equals the label derived
   from its checkpoint record (tijepa.eval.provenance).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tijepa.eval import provenance as PV  # noqa: E402
from tijepa.viz.claims import (  # noqa: E402
    Comparison,
    check_banned,
    check_grammar,
    check_labels,
)

P = REPO / "paper" / "v3"
MANUSCRIPTS = {"tmlr": P / "tmlr" / "main.tex", "biorxiv": P / "biorxiv" / "main.tex"}
OUT = REPO / "results" / "v3" / "claims_check.json"

ALIASES = {
    "E1_handcrafted_logreg": ["hand-crafted"],
    "E2_imagenet_vit_s": ["ImageNet"],
    "E3_dinov2_vit_s": ["DINOv2"],
    "E7_vjepa2_vitl": ["V-JEPA~2", "V-JEPA 2"],
    "E4_tijepa1": ["TiJepa-1 ", "TiJepa-1,", "TiJepa-1.", "TiJepa-1)"],
    "E4s_tijepa1_spatial": ["spatial-only"],
    "E0_tijepa0": ["TiJepa-0"],
    "E6_random_init": ["random init"],
    "intensity_baseline": ["intensity-only", "intensity statistics"],
    "trace_jepa": ["trace JEPA"],
    "raw_trace": ["raw trace"],
    "handcrafted": ["hand-crafted trace features"],
}
V3_BANNED = [
    "PLAN_v3",
    "G-Forecast",
    "G-Recipe",
    "G-BetaGlucan",
    "G-Dryad",
    "results/v3",
    "TODO",
    "gate ",
    "the gate",
    "this run",
    "the agent",
    "RUN_REPORT",
    "errata",
]
V3_PATTERNS = [
    r"\bH[0-9](?:\.[0-9])?\b",
    r"\bR(?:[1-9]|1[0-9]|2[0-4])\b(?![-\w])",
    r"\bv[0-3]\b",
    r"\bE[0-7]s?\b",
]
DETECT = [
    r"(detect|identif|recogni[sz]|classif)\w*\s+(\w+\s+){0,4}trained immunity\s+"
    r"(\w+\s+){0,4}(image|imaging|video|movie)",
    r"trained immunity\s+(is|was|can be)\s+(detected|seen|visible|identified)",
]
ANON = ["Quadir", "Rafid", "rafidquadir", "@gmail", "anthropic", "github.com/"]


def strip_tex(t: str) -> str:
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\(begin|end)\{(table|figure|tabular)\*?\}.*?", " ", t)
    t = re.sub(r"\\(cite[pt]?|ref|label|eqref|includegraphics)(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\val\{[^}]*\}", "0.5", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = t.replace("~", " ").replace("{", "").replace("}", "")
    return re.sub(r"[ \t]+", " ", t)


TASK_WORDS = {
    "three_class": ("microglia", "morpholog"),
    "round_vs_amoeboid": ("round", "amoeboid"),
    "af_A_mixed_fovs_only": (
        "autofluorescence",
        "cell identity",
        "t cell",
        "neutrophils imaged",
    ),
    "af_A_all_fovs": ("autofluorescence", "cell identity", "t cell"),
    "ligand": ("ligand",),
}
DIRECTIONAL = {
    "higher": 1,
    "greater": 1,
    "better": 1,
    "beats": 1,
    "outperforms": 1,
    "exceeds": 1,
    "above": 1,
    "lower": -1,
    "worse": -1,
    "below": -1,
    "indistinguishable": 0,
    "not distinguishable": 0,
    "comparable": 0,
    "as well as": 0,
}


def positions(low: str, aliases: list[str]) -> list[int]:
    out = []
    for a in aliases:
        start = 0
        while (i := low.find(a.lower(), start)) >= 0:
            out.append(i)
            start = i + 1
    return sorted(out)


def check_v3(sections: list[tuple[str, str]], comps: dict, names: dict) -> list[dict]:
    """A directional word is checked against an encoder pair only if one encoder is the
    nearest named encoder before the word and the other the nearest after it, and only for
    tasks the sentence (or, failing that, its section heading) is about."""
    viol = []
    for head, txt in sections:
        for sent in re.split(r"(?<=[.!?;])\s+", txt):
            low = sent.lower()
            tasks = [t for t, ws in TASK_WORDS.items() if any(w in low for w in ws)]
            if not tasks:
                tasks = [
                    t for t, ws in TASK_WORDS.items() if any(w in head.lower() for w in ws)
                ]
            if "round" not in low:
                tasks = [t for t in tasks if t != "round_vs_amoeboid"]
            mentions = sorted((i, k) for k, al in names.items() for i in positions(low, al))
            if len({k for _, k in mentions}) < 2:
                continue
            for word, sign in DIRECTIONAL.items():
                for m in re.finditer(re.escape(word), low):
                    before = [k for i, k in mentions if i < m.start()]
                    after = [k for i, k in mentions if i > m.end()]
                    if not before or not after or before[-1] == after[0]:
                        continue
                    a, b = before[-1], after[0]
                    for t in tasks:
                        c = comps.get(f"{a}:{b}:{t}")
                        exp = c.direction if c else None
                        if c is None:
                            c = comps.get(f"{b}:{a}:{t}")
                            exp = (
                                -c.direction
                                if c and c.direction in (1, -1)
                                else (c.direction if c else None)
                            )
                        if exp is None or exp == sign:
                            continue
                        viol.append(
                            {
                                "sentence": sent.strip()[:240],
                                "pair": f"{a} vs {b}",
                                "task": t,
                                "word": word,
                                "results_say": exp,
                                "difference": c.difference,
                                "ci": c.ci,
                            }
                        )
    return viol


def sections_of(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"\\(?:sub)*section\*?\{([^}]*)\}", text)
    out = [("", parts[0])]
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i], parts[i + 1]))
    return out


def body(path: Path) -> str:
    """The manuscript with every \\input file expanded, without the bibliography."""

    def expand(p: Path) -> str:
        s = p.read_text()
        return re.sub(
            r"\\input\{([^}]+)\}",
            lambda m: (
                expand(
                    (
                        p.parent
                        / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
                    ).resolve()
                )
                if "numbers" not in m.group(1)
                else ""
            ),
            s,
        )

    return expand(path)


def gather() -> dict[str, Comparison]:
    comps: dict[str, Comparison] = {}
    bb = json.loads((REPO / "results/v3/partA/bbbc054.json").read_text())
    for key, rec in bb["pairwise_comparisons"].items():
        task = key.split(":", 1)[0]
        a, b = rec["encoder_a"], rec["encoder_b"]
        comps[f"{a}:{b}:{task}"] = Comparison(
            a=a,
            b=b,
            task=task,
            difference=rec["difference_mean_over_seeds"],
            ci=tuple(rec["ci95_widest_over_seeds"]),
            p_value=rec.get("p_value_bh"),
        )
    af = json.loads((REPO / "results/v3/coverage/autofluorescence.json").read_text())
    for t in ("A_all_fovs", "A_mixed_fovs_only"):
        for k, c in af.get(f"paired_vs_tijepa1_{t}", {}).items():
            enc = k.split("|")[0]
            if not k.endswith("|all"):
                continue
            comps[f"E4_tijepa1:{enc}:af_{t}"] = Comparison(
                a="E4_tijepa1",
                b=enc,
                task=f"af_{t}",
                difference=c["difference"],
                ci=tuple(c["ci95"]) if c.get("ci95") else None,
                p_value=c.get("q_bh"),
            )
    pb_path = REPO / "results/v3/partB/partB_v3.json"
    if pb_path.exists():
        pb = json.loads(pb_path.read_text())["b32_ligand_identity"]
        for k, c in pb["paired_vs_trace_jepa"].items():
            comps[f"trace_jepa:{k}:ligand"] = Comparison(
                a="trace_jepa",
                b=k,
                task="ligand",
                difference=c["difference"],
                ci=tuple(c["ci95"]) if c.get("ci95") else None,
                p_value=c.get("q_bh"),
            )
    return comps


def main() -> int:
    numbers = (P / "numbers.tex").read_text() if (P / "numbers.tex").exists() else ""
    defined = set(re.findall(r"\\csname val@([^\\]+)\\endcsname", numbers))
    report: dict = {"manuscripts": {}}
    ok = True
    comps = gather()
    for name, path in MANUSCRIPTS.items():
        if not path.exists():
            report["manuscripts"][name] = {"status": "absent"}
            ok = False
            continue
        raw = body(path)
        raw_main = raw.split("\\begin{document}")[-1].split("\\bibliography")[0]
        used = set(re.findall(r"\\val\{([^}]+)\}", re.sub(r"(?<!\\)%.*", "", raw_main)))
        text = strip_tex(raw_main)
        r: dict = {
            "undefined_numbers": sorted(used - defined),
            "banned": check_banned(text),
            "grammar": check_grammar(text),
            "banned_v3": [],
            "detection_claims": [],
            "anonymity": [],
            "directional_violations": check_v3(
                [(h, strip_tex(t)) for h, t in sections_of(raw_main)], comps, ALIASES
            ),
        }
        low = raw_main.lower()
        for s in V3_BANNED:
            if s.lower() in low:
                r["banned_v3"].append(s)
        for pat in V3_PATTERNS:
            for m in re.finditer(pat, strip_tex(raw_main)):
                r["banned_v3"].append(m.group(0))
        for pat in DETECT:
            r["detection_claims"] += [m.group(0) for m in re.finditer(pat, text, re.I)]
        if name == "tmlr":
            r["anonymity"] = [a for a in ANON if a.lower() in raw.lower()]
        r["n_numbers_used"] = len(used)
        r["n_sentences"] = len(re.split(r"(?<=[.!?])\s+", text))
        bad = {k: v for k, v in r.items() if isinstance(v, list) and v}
        r["status"] = "fail" if bad else "ok"
        ok &= not bad
        report["manuscripts"][name] = r
        print(
            f"{name}: {r['status']}  ({r['n_numbers_used']} numbers, "
            f"{r['n_sentences']} sentences)"
        )
        for k, v in bad.items():
            for item in v[:15]:
                print(f"   [{k}] {item}")
    labels = (
        json.loads((P / "labels.json").read_text()) if (P / "labels.json").exists() else {}
    )
    lp = check_labels(labels, PV.label_for_key)
    report["labels"] = {"n": len(labels), "problems": lp}
    ok &= not lp
    report["n_comparisons"] = len(comps)
    report["status"] = "ok" if ok else "fail"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    print(
        f"labels verified: {len(labels)}; comparisons: {len(comps)}; "
        f"overall {report['status']}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
