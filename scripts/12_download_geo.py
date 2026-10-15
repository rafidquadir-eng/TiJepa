"""M1: download GEO series matrices and processed supplementary files.

Per PLAN.md Section 4.4: series matrices and processed supplementary files only,
never raw reads. Sample metadata is parsed into results/omics/<GSE>_samples.csv so a
human can audit the condition mapping.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "geo"
OUT = REPO / "results" / "omics"
DL = REPO / "results" / "downloads"

SERIES = {
    "GSE260996": "partB_rnaseq_mouse_bmdm_ligand_pairs",
    "GSE261647": "partB_atac_mouse_bmdm_ligand_pairs",
    "GSE46903": "partC_human_macrophage_activation_anchors",
    "GSE58310": "partBC_human_trained_tolerant_saeed2014",
    "GSE85243": "partBC_human_trained_tolerant_novakovic2016",
    "GSE172116": "partB_human_endotoxin_tolerance",
}

# Supplementary files larger than this are skipped (processed matrices are small;
# anything this big is likely per-cell or raw-adjacent).
MAX_SUPPL_MB = 600


def geo_dir(gse: str) -> str:
    """GEO FTP bucket, e.g. GSE260996 -> GSE260nnn."""
    stub = gse[:-3] + "nnn" if len(gse) > 6 else gse[:3] + "nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}"


def listing(url: str) -> list[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TiJepa/0.1"})
        with urllib.request.urlopen(req, timeout=90) as r:
            html = r.read().decode("utf-8", "replace")
        return sorted(set(re.findall(r'href="([^"?/][^"]*)"', html)))
    except Exception as exc:
        print(f"   listing failed {url}: {exc}")
        return []


def fetch(url: str, dest: Path, tries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for i in range(1, tries + 1):
        rc = subprocess.run(
            ["aria2c", "-x", "4", "-s", "4", "--continue=true", "--auto-file-renaming=false",
             "--summary-interval=0", "--console-log-level=warn", "--timeout=90",
             "-d", str(dest.parent), "-o", dest.name, url]
        ).returncode
        if rc == 0 and dest.exists():
            return True
        time.sleep(5 * i)
    return False


def parse_series_matrix(path: Path) -> tuple[list[dict], dict]:
    """Extract per-sample metadata rows and series-level header fields."""
    op = gzip.open if path.suffix == ".gz" else open
    series: dict[str, str] = {}
    cols: dict[str, list[str]] = {}
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Series_"):
                k, _, v = line.partition("\t")
                series.setdefault(k.strip("!").strip(), v.strip().strip('"'))
            elif line.startswith("!Sample_"):
                parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
                key = parts[0].strip("!")
                cols.setdefault(key, [])
                cols[key].append(parts[1:])
            elif line.startswith("!series_matrix_table_begin"):
                break
    gsms = cols.get("Sample_geo_accession", [[]])[0]
    rows = []
    for i, gsm in enumerate(gsms):
        row = {"gsm": gsm}
        for key, blocks in cols.items():
            if key == "Sample_geo_accession":
                continue
            vals = []
            for b in blocks:
                if i < len(b) and b[i]:
                    vals.append(b[i])
            if vals:
                row[key.replace("Sample_", "")] = " | ".join(dict.fromkeys(vals))
        rows.append(row)
    return rows, series


def do_series(gse: str, role: str) -> None:
    base = geo_dir(gse)
    dest_dir = RAW / gse
    got: list[dict] = []

    # 1. series matrix (metadata, and expression for array series)
    mats = [f for f in listing(f"{base}/matrix/") if f.endswith(".txt.gz")]
    for name in mats:
        dest = dest_dir / "matrix" / name
        if fetch(f"{base}/matrix/{name}", dest):
            got.append({"file": f"matrix/{name}", "url": f"{base}/matrix/{name}",
                        "size_bytes_local": dest.stat().st_size, "status": "ok"})

    # 2. processed supplementary files (skip raw/BAM/fastq and oversized files)
    supp = listing(f"{base}/suppl/")
    skip_pat = re.compile(r"(RAW\.tar|\.bam|\.fastq|\.sra|filelist)", re.I)
    for name in supp:
        if skip_pat.search(name):
            got.append({"file": f"suppl/{name}", "status": "skipped_raw_or_filelist"})
            continue
        url = f"{base}/suppl/{name}"
        dest = dest_dir / "suppl" / name
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "TiJepa/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                size = int(r.headers.get("Content-Length") or 0)
        except Exception:
            size = 0
        if size > MAX_SUPPL_MB * 1e6:
            got.append({"file": f"suppl/{name}", "size_bytes_record": size,
                        "status": f"skipped_too_large_over_{MAX_SUPPL_MB}MB"})
            continue
        if fetch(url, dest):
            got.append({"file": f"suppl/{name}", "url": url,
                        "size_bytes_local": dest.stat().st_size, "status": "ok"})

    # 3. audit table of sample metadata
    series_meta: dict = {}
    n_samples = 0
    if mats:
        first = dest_dir / "matrix" / mats[0]
        if first.exists():
            rows, series_meta = parse_series_matrix(first)
            n_samples = len(rows)
            if rows:
                import csv

                OUT.mkdir(parents=True, exist_ok=True)
                keys = sorted({k for r in rows for k in r})
                keys = ["gsm"] + [k for k in keys if k != "gsm"]
                with (OUT / f"{gse}_samples.csv").open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows)

    DL.mkdir(parents=True, exist_ok=True)
    (DL / f"geo_{gse}.json").write_text(json.dumps({
        "key": f"geo_{gse}", "kind": "geo", "accession": gse,
        "record_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}",
        "title": series_meta.get("Series_title"),
        "platform_summary": series_meta.get("Series_platform_id"),
        "licence_verified": "NCBI GEO public domain data (no licence asserted); "
                            "cite the originating publication",
        "licence_source": "GEO is a US-government public-domain repository",
        "role": role, "n_samples": n_samples,
        "sample_table": f"results/omics/{gse}_samples.csv" if n_samples else None,
        "files_used": got, "status": "ok",
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
    ok = sum(1 for g in got if g.get("status") == "ok")
    print(f"{gse}: {ok} files, {n_samples} samples -> {OUT}/{gse}_samples.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="single GSE accession")
    args = ap.parse_args()
    for gse, role in SERIES.items():
        if args.only and gse != args.only:
            continue
        print(f"== {gse}")
        try:
            do_series(gse, role)
        except Exception as exc:
            print(f"  FAILED {gse}: {type(exc).__name__}: {exc}")
            (DL / f"geo_{gse}.json").write_text(
                json.dumps({"key": f"geo_{gse}", "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
