"""Directional-claim generation and verification (PLAN_v1.md 10.2, fixes E5, B6).

The v0 paper contained hand-written directional sentences - "beats", "worse than",
"indistinguishable from" - that were typed once and then not re-checked when the
numbers changed. Review findings E5 and B6 are exactly that: conclusions hard-coded
independently of the results they describe.

Two mechanisms here, used together:

1. **Generation.** `compare_phrase` turns a pair of measurements with uncertainty into
   the words for it, so the template writes `{{CMP:E4:E3:bbbc054}}` and the renderer
   substitutes "higher than" / "lower than" / "not distinguishable from" according to
   the interval. No directional word in the paper is chosen by hand.

2. **Verification.** `check_text` scans the rendered paper for directional words near
   encoder names and re-derives each from `results/`, failing on a contradiction. This
   catches a claim that survived in prose from an earlier draft even if nobody
   remembered to regenerate it.

The decision rule is deliberately conservative: a difference counts as directional only
when its interval excludes zero, and otherwise the phrase is "not distinguishable from".
Where there is no interval (fewer than 10 groups, so rule 3 forbids a CI) the comparison
falls back to the exact permutation p-value and the same wording, and when neither is
available the phrase is "not compared", never a bare "higher".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DIRECTIONAL = {
    "higher": +1, "greater": +1, "better": +1, "beats": +1, "outperforms": +1,
    "exceeds": +1, "above": +1, "ahead of": +1,
    "lower": -1, "worse": -1, "below": -1, "underperforms": -1, "behind": -1,
    "indistinguishable": 0, "comparable": 0, "similar": 0, "equivalent": 0,
    "not distinguishable": 0,
}
ALPHA = 0.05


@dataclass
class Comparison:
    a: str
    b: str
    task: str
    difference: float | None = None
    ci: tuple[float, float] | None = None
    p_value: float | None = None

    @property
    def direction(self) -> int | None:
        """+1 a above b, -1 a below b, 0 not distinguishable, None not comparable."""
        if self.difference is None:
            return None
        if self.ci is not None:
            lo, hi = self.ci
            if lo > 0:
                return +1
            if hi < 0:
                return -1
            return 0
        if self.p_value is not None:
            if self.p_value >= ALPHA:
                return 0
            return +1 if self.difference > 0 else -1
        return None


def compare_phrase(c: Comparison) -> str:
    d = c.direction
    if d is None:
        return "not compared"
    return {1: "higher than", -1: "lower than",
            0: "not distinguishable from"}[d]


def describe(c: Comparison, nd: int = 3) -> str:
    """A full phrase with the number and its uncertainty, for use in prose."""
    phrase = compare_phrase(c)
    if c.difference is None:
        return phrase
    bits = [f"{c.difference:+.{nd}f}"]
    if c.ci is not None:
        bits.append(f"95% CI [{c.ci[0]:+.{nd}f}, {c.ci[1]:+.{nd}f}]")
    elif c.p_value is not None:
        bits.append(f"exact p = {c.p_value:.3f}")
    return f"{phrase} ({', '.join(bits)})"


def render_placeholders(text: str, comparisons: dict[str, Comparison]
                        ) -> tuple[str, list[str]]:
    """Replace every `{{CMP:a:b:task}}` with the generated phrase.

    Returns the rendered text and the list of placeholders that had no comparison, so
    an unresolved placeholder is reported rather than left in the paper.
    """
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in comparisons:
            missing.append(key)
            return m.group(0)
        return compare_phrase(comparisons[key])

    return re.sub(r"\{\{CMP:([^}]+)\}\}", sub, text), missing


def check_text(text: str, comparisons: dict[str, Comparison],
               names: dict[str, list[str]]) -> list[dict]:
    """Verify directional words near encoder names against the results.

    `names` maps a results key (e.g. "E4") to the strings that refer to it in prose
    (e.g. ["TiJepa-1", "E4"]). A sentence mentioning two different encoders and a
    directional word is checked against the corresponding comparison.

    Returns a list of violations; an empty list means the prose agrees with the numbers.
    """
    violations: list[dict] = []
    # Drop table rows and block quotes BEFORE splitting into sentences. Skipping only
    # chunks that *begin* with "|" is not enough: a table introduced by a line ending
    # in ":" has no sentence boundary before it, so the whole table lands inside the
    # preceding sentence and every cell is scanned as prose.
    kept = [ln for ln in text.splitlines()
            if not ln.lstrip().startswith(("|", ">"))]
    sentences = re.split(r"(?<=[.!?])\s+", "\n".join(kept))
    for si, sent in enumerate(sentences):
        low = sent.lower()
        present = [k for k, aliases in names.items()
                   if any(a.lower() in low for a in aliases)]
        if len(present) < 2:
            continue
        found = [(w, s) for w, s in DIRECTIONAL.items() if w in low]
        if not found:
            continue

        def first_position(key: str, _low=low) -> int:
            """Where this encoder is first named in the sentence."""
            pos = [_low.find(a.lower()) for a in names[key]
                   if a.lower() in _low]
            return min(pos) if pos else len(_low)

        for task_key, c in comparisons.items():
            parts = task_key.split(":")
            if len(parts) != 3:
                continue
            ca, cb, _task = parts
            if ca not in present or cb not in present:
                continue
            expected = c.direction
            if expected is None:
                continue
            # The comparison is stored as "ca relative to cb". A sentence that names
            # cb first states it the other way round, so the expected sign flips.
            # Iterating over both orderings instead would flag every correct sentence.
            if first_position(cb) < first_position(ca):
                expected = -expected if expected in (1, -1) else 0
            for word, sign in found:
                if sign != expected:
                    violations.append({
                        "sentence_index": si,
                        "sentence": sent.strip()[:300],
                        "comparison": task_key,
                        "word": word,
                        "word_implies": sign,
                        "results_say": expected,
                        "difference": c.difference,
                        "ci": list(c.ci) if c.ci else None,
                        "p_value": c.p_value,
                    })
    # de-duplicate: one sentence can match several aliases of the same pair
    seen, unique = set(), []
    for v in violations:
        k = (v["sentence_index"], v["comparison"], v["word"])
        if k not in seen:
            seen.add(k)
            unique.append(v)
    return unique


# --------------------------------------------------------------- v2 manuscript checks
# PLAN_v2.md rule 5 and G2.1 (findings N11, N14). These run on the paper BODY.

#: case-insensitive substrings that are internal process language
BANNED_SUBSTRINGS = [
    "PLAN_v", "review finding", "D5b", "F3.", "NOT USABLE", "not run in this pass",
    "blocking requirements recorded", "results/v1/", "results/v2/", "previous version",
]
#: case-sensitive patterns: the version tag and identifiers mangled by lowercasing
BANNED_PATTERNS = [r"\bv0\b", r"\bv1\b", r" e[0-6]s? ", r" bbbc054 ", r" s-biad2196 ",
                   r"\b[DCBEN]\d{1,2}[a-e]?\b(?= (?:fix|finding))"]
DANGLING = [r"\b(?:higher|lower|greater|better|worse) than \(",
            r"\bdistinguishable from \(", r"\bthan \)", r"\bfrom \("]


def check_banned(text: str) -> list[dict]:
    hits = []
    low = text.lower()
    for s in BANNED_SUBSTRINGS:
        i = low.find(s.lower())
        if i >= 0:
            hits.append({"rule": "banned substring", "match": s,
                         "context": text[max(0, i - 60): i + 60]})
    for pat in BANNED_PATTERNS:
        for m in re.finditer(pat, text):
            hits.append({"rule": "banned pattern", "match": m.group(0), "pattern": pat,
                         "context": text[max(0, m.start() - 60): m.end() + 60]})
    return hits


def check_grammar(text: str) -> list[dict]:
    hits = []
    for pat in DANGLING:
        for m in re.finditer(pat, text):
            hits.append({"rule": "dangling comparative", "match": m.group(0),
                         "context": text[max(0, m.start() - 80): m.end() + 40]})
    return hits


def check_labels(labels: dict, label_for_key) -> list[str]:
    """Every rendered label must equal the one derived from its checkpoint record.

    `labels` maps rendered label -> {"key": results key, "checkpoint": record|None}, as
    written by the paper renderer next to the paper; `label_for_key` is
    `tijepa.eval.provenance.label_for_key`.
    """
    problems = []
    for shown, rec in labels.items():
        try:
            derived = label_for_key(rec["key"], rec.get("checkpoint"))
        except ValueError as exc:
            problems.append(f"{shown!r}: {exc}")
            continue
        if derived != rec.get("label", shown):
            problems.append(f"{shown!r} rendered for {rec['key']} but its checkpoint "
                            f"says {derived!r}")
    return problems
