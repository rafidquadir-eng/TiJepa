"""Write paper/v3/refs.bib from results/v3/references.json (verified records only).

Only entries whose status is resolved (Crossref/DataCite DOI with a title check) or
arXiv (title and authors read from the arXiv abstract page) are written; a key cited in a
manuscript but missing here makes the LaTeX build report an undefined citation, which the
build script treats as a failure.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REFS = REPO / "results" / "v3" / "references.json"
OUT = REPO / "paper" / "v3" / "refs.bib"


def esc(s: str) -> str:
    return (s or "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main() -> int:
    d = json.loads(REFS.read_text())
    refs = d.get("references", d)
    out = []
    for key, r in refs.items():
        st = r.get("status", "")
        if not st.startswith(("resolved", "arxiv")):
            continue
        authors = r.get("authors") or ([r["first_author"]] if r.get("first_author") else [])
        if r.get("n_authors", 1) > len(authors) or len(authors) > 6:
            auth = " and ".join(authors[:1]) + " and others"
        else:
            auth = " and ".join(authors)
        fields = {"title": "{" + esc(r.get("title", "")) + "}", "author": esc(auth),
                  "year": str(r.get("year") or "")}
        if st.startswith("arxiv"):
            kind = "misc"
            fields["howpublished"] = f"arXiv:{r['arxiv_id']}"
            fields["eprint"] = r["arxiv_id"]
        else:
            kind = "article"
            fields["journal"] = esc(r.get("container") or "")
            fields["doi"] = r.get("doi") or ""
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
        out.append(f"@{kind}{{{key},\n{body}\n}}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(out) + "\n")
    print(f"wrote {len(out)} entries to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
