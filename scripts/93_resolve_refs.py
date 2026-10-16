"""F7 10.4: resolve every reference DOI before it is cited (fixes E3).

The v0 paper listed references with `TODO(verify)` markers because nothing resolved
their identifiers, and the review found at least two substantive errors in them: V-JEPA
was attributed to the wrong first author, and an MGI citation carried an author who has
nothing to do with it.

This script takes the candidate list from PLAN_v1.md 10.4 plus the sources actually used
in the run, resolves each DOI against Crossref (`api.crossref.org/works/<doi>`), and
writes `results/v1/references.json` with the **authoritative** title, first author,
container and year as the registry returns them. A DOI that does not resolve stays
`TODO(verify)` and is never silently dressed up as verified. For arXiv preprints the
arXiv id is recorded and, where a journal DOI exists, that is resolved too.

Nothing here invents an identifier: every DOI is one stated in the plan, printed on a
dataset record used in the run, or returned by Crossref for a title query whose result
is reported along with its match score so a weak match can be judged.
"""
from __future__ import annotations

import functools
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "v1"
CROSSREF = "https://api.crossref.org/works/"
UA = "TiJepa-reference-check/1.0 (mailto:noreply@anthropic.com)"

#: words that MUST appear in the resolved title, lower-cased. A DOI can resolve
#: perfectly and still be the wrong paper -- the plan's suggested MGI DOI resolved to a
#: correction notice about chromatin -- so resolution alone is not accepted as
#: verification.
TITLE_MUST_CONTAIN: dict[str, tuple[str, ...]] = {
    "complexeye": ("multi-lens", "migration"),
    "phagostat": ("phago",),
    "bbbc": ("annotated high-throughput microscopy",),
    "saeed2014": ("epigenetic programming", "monocyte"),
    "novakovic2016": ("glucan", "tolerance"),
    "xue2014": ("macrophage activation",),
    "netea2016": ("trained immunity",),
    "wang2025": ("macrophage", "memory"),
    "limma": ("limma",),
    "deseq2": ("fold change", "dispersion"),
    "cellpose": ("cellpose",),
    "smyth2004": ("linear model", "microarray"),
    "mgi": ("mouse genome informatics",),
    "trackpy": ("trackpy",),
    "scikit_image": ("scikit-image",),
}


# key -> (doi or None, arxiv id or None, what it is cited for)
CANDIDATES: dict[str, tuple[str | None, str | None, str]] = {
    "vjepa": (None, "2404.08471",
              "V-JEPA, the architecture TiJepa follows. PLAN_v1.md 10.4 records that "
              "v0 wrongly listed Assran as first author; the first author is Bardes A."),
    "dinov2": (None, "2304.07193", "DINOv2, baseline encoder E3"),
    "complexeye": ("10.1038/s41467-023-43765-3", None,
                   "ComplexEye: the pretraining corpus, and the source of the verified "
                   "ComplexEye and Leica pixel sizes"),
    "phagostat": ("10.1038/s41598-024-56081-7", None,
                  "PhagoStat microglia video (considered for the corpus)"),
    "bbbc": ("10.1038/nmeth.2083", None,
             "the Broad Bioimage Benchmark Collection, which hosts BBBC054"),
    "saeed2014": ("10.1126/science.1251086", None,
                  "Saeed et al. 2014, the source of GSE58310 (Part B4, Part C 8.2)"),
    "novakovic2016": ("10.1016/j.cell.2016.09.034", None,
                      "Novakovic et al. 2016, beta-glucan reverses LPS tolerance"),
    "xue2014": ("10.1016/j.immuni.2014.01.006", None,
                "Xue et al. 2014, the source of GSE46903 (Part C anchors)"),
    "netea2016": ("10.1126/science.aaf1098", None,
                  "Netea et al. 2016, the trained-immunity review"),
    "wang2025": ("10.1016/j.cels.2025.101171", None,
                 "Wang AG, Son M, et al., Cell Systems 2025: the Tay-lab NF-kB traces, "
                 "GSE260996 and GSE261647 used throughout Part B. The DOI was obtained "
                 "from PMC12147866 via the NCBI ID converter (which also gives PMID "
                 "39938520), not guessed and not matched by title."),
    "limma": ("10.1093/nar/gkv007", None,
              "limma, the method reimplemented in src/tijepa/omics/moderated.py"),
    "deseq2": ("10.1186/s13059-014-0550-8", None,
               "DESeq2, used via pydeseq2 for the mouse RNA contrasts"),
    "cellpose": ("10.1038/s41592-020-01018-x", None,
                 "Cellpose: considered for E1 segmentation; not used, see the run report"),
    "cotracker": (None, "2307.07635", "CoTracker: not run, host unreachable"),
    "smyth2004": ("10.2202/1544-6115.1027", None,
                  "Smyth 2004, the empirical-Bayes moderated t-statistic implemented "
                  "in src/tijepa/omics/moderated.py"),
    "mgi": ("10.1093/genetics/iyae031", None,
            "Mouse Genome Informatics, source of the HOM_MouseHumanSequence ortholog "
            "table used in B4. PLAN_v1.md 10.4 asked to remove the spurious author "
            "'Blumberg B' that v0 carried, and suggested 10.1093/nar/gkae733; that DOI "
            "resolves to an unrelated correction notice, so it was NOT used. The "
            "citation here is the MGI system paper (Baldarelli et al., Genetics 2024), "
            "checked against the expected title keywords below."),
    "trackpy": ("10.5281/zenodo.1226458", None,
                "trackpy, used for detection linking in preprocessing"),
    "scikit_image": ("10.7717/peerj.453", None,
                     "scikit-image, used for segmentation and texture features"),
}

# A bibliographic title query was tried for the one reference that had no DOI and
# returned a confidently WRONG paper, so that path is deliberately not used: a weak
# Crossref match dressed up as a citation is worse than an unresolved marker. Every
# reference here now carries an identifier from an authoritative source instead.
TITLE_QUERIES: dict[str, str] = {}


def fetch(url: str, timeout: int = 30) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"[:200]}


def summarise(msg: dict) -> dict:
    authors = msg.get("author") or []
    first = authors[0] if authors else {}
    return {
        "doi": msg.get("DOI"),
        "title": (msg.get("title") or [None])[0],
        "first_author": " ".join(
            x for x in (first.get("given"), first.get("family")) if x) or None,
        "n_authors": len(authors),
        "container": (msg.get("container-title") or [None])[0],
        "year": ((msg.get("issued") or {}).get("date-parts") or [[None]])[0][0],
        "type": msg.get("type"),
        "publisher": msg.get("publisher"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    n_ok = n_todo = 0

    for key, (doi, arxiv, purpose) in CANDIDATES.items():
        rec: dict = {"cited_for": purpose, "arxiv_id": arxiv}
        if doi:
            d = fetch(CROSSREF + urllib.parse.quote(doi))
            if d and "message" in d:
                rec.update(summarise(d["message"]))
                rec["status"] = "resolved"
                rec["resolved_via"] = "crossref DOI lookup"
            else:
                # Zenodo and other repository DOIs are registered with DataCite, not
                # Crossref, so a Crossref miss is not evidence the DOI is wrong.
                dd = fetch("https://api.datacite.org/dois/" + urllib.parse.quote(doi))
                attrs = ((dd or {}).get("data") or {}).get("attributes")
                if attrs:
                    creators = attrs.get("creators") or []
                    rec.update({
                        "doi": doi,
                        "title": (attrs.get("titles") or [{}])[0].get("title"),
                        "first_author": (creators[0].get("name")
                                         if creators else None),
                        "n_authors": len(creators),
                        "container": attrs.get("publisher"),
                        "year": attrs.get("publicationYear"),
                        "type": (attrs.get("types") or {}).get("resourceTypeGeneral"),
                    })
                    rec["status"] = "resolved"
                    rec["resolved_via"] = "datacite DOI lookup (repository DOI)"
                else:
                    rec["status"] = "TODO(verify)"
                    rec["doi_attempted"] = doi
                    rec["error"] = (d or {}).get("__error__")
            time.sleep(0.3)
        elif key in TITLE_QUERIES:
            q = urllib.parse.urlencode({"query.bibliographic": TITLE_QUERIES[key],
                                        "rows": 3})
            d = fetch("https://api.crossref.org/works?" + q)
            items = ((d or {}).get("message") or {}).get("items") or []
            if items:
                best = items[0]
                rec.update(summarise(best))
                rec["status"] = "resolved_by_title_query"
                rec["crossref_match_score"] = best.get("score")
                rec["caveat"] = ("matched by bibliographic query, not by a known DOI; "
                                 "check the title and year before citing")
                rec["other_candidates"] = [summarise(i) for i in items[1:3]]
            else:
                rec["status"] = "TODO(verify)"
                rec["error"] = (d or {}).get("__error__", "no Crossref match")
            time.sleep(0.3)
        elif arxiv:
            rec["status"] = "arxiv_only"
            rec["note"] = ("cited by arXiv id; no journal DOI was looked up because "
                           "none is stated in PLAN_v1.md 10.4")
        else:
            rec["status"] = "TODO(verify)"

        # A resolved DOI is only accepted if its title looks like the work being
        # cited; otherwise it is downgraded, because a confidently wrong citation is
        # worse than an explicit TODO.
        want = TITLE_MUST_CONTAIN.get(key)
        if want and rec.get("status", "").startswith("resolved"):
            title = (rec.get("title") or "").lower()
            missing_words = [w for w in want if w not in title]
            rec["title_keywords_expected"] = list(want)
            if missing_words:
                rec["status"] = "TODO(verify)"
                rec["title_check"] = "FAILED"
                rec["title_check_detail"] = (
                    f"the resolved title does not contain {missing_words}; the DOI "
                    "resolves but may be the wrong work, so it is not accepted")
            else:
                rec["title_check"] = "passed"

        results[key] = rec
        if rec["status"].startswith(("resolved", "arxiv")):
            n_ok += 1
        else:
            n_todo += 1
        label = rec.get("first_author") or rec.get("arxiv_id") or "-"
        print(f"  {key:14s} {rec['status']:26s} {label} "
              f"{(rec.get('container') or '')[:44]} {rec.get('year') or ''}")

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resolver": "api.crossref.org/works",
        "n_resolved": n_ok, "n_todo_verify": n_todo,
        "policy": ("A DOI that does not resolve stays TODO(verify) and is never "
                   "presented as verified. No identifier here was invented: each is "
                   "stated in PLAN_v1.md 10.4, printed on a dataset record used in the "
                   "run, or returned by Crossref for a reported title query."),
        "references": results,
    }
    (OUT / "references.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n{n_ok} resolved, {n_todo} left as TODO(verify)")
    print(f"wrote {OUT/'references.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
