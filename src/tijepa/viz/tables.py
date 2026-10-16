"""Markdown table writer with structural assertions (PLAN_v1.md 10.1, fixes E1).

The v0 paper contained a table whose header had a different number of columns from its
rows, so the rendered table silently misaligned every value in it (review finding E1).
That is the kind of error no amount of proofreading reliably catches and that one
assertion prevents for good: every table in the v1 paper is built here, and a row whose
length does not match the header raises instead of rendering.

The writer also refuses a table with no rows, because an empty table in a paper is
always a bug - either the result is missing, in which case the text should say so in
words, or the filter is wrong.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


class TableShapeError(AssertionError):
    """Raised when a row's length does not match the header's."""


def markdown_table(header: Sequence[str], rows: Iterable[Sequence[object]],
                   *, align: Sequence[str] | None = None,
                   allow_empty: bool = False) -> str:
    """Render a Markdown table, asserting that every row matches the header.

    `align` may give "l", "c" or "r" per column; the default is left for the first
    column and right for the rest, which is what a table of numbers wants.
    """
    header = [str(h) for h in header]
    n = len(header)
    if n == 0:
        raise TableShapeError("table has no columns")

    out_rows: list[list[str]] = []
    for i, r in enumerate(rows):
        cells = list(r)
        if len(cells) != n:
            raise TableShapeError(
                f"row {i} has {len(cells)} cells but the header has {n}: "
                f"header={header} row={cells}")
        out_rows.append(["" if c is None else str(c) for c in cells])

    if not out_rows and not allow_empty:
        raise TableShapeError(
            "table has no rows; say so in words rather than rendering an empty table")

    if align is None:
        align = ["l"] + ["r"] * (n - 1)
    if len(align) != n:
        raise TableShapeError(
            f"align has {len(align)} entries but the header has {n}")
    rule = {"l": ":---", "c": ":---:", "r": "---:"}
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(rule[a] for a in align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in out_rows]
    return "\n".join(lines)
