"""M1: download labelled-evaluation and QC datasets (Dryad, BBBC054, BioImage Archive).

Dryad files are verified against the sha-256 digests published in the Dryad API.
Downloaded archives are extracted into their own directories; nothing in them is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
OUT = REPO / "results" / "downloads"

DRYAD_DOI = "doi:10.5061/dryad.m0cfxpp8v"
DRYAD_VERSION = "405404"
BBBC054 = "https://data.broadinstitute.org/bbbc/BBBC054"
EBI_FILES = "https://www.ebi.ac.uk/biostudies/api/v1/files"
# FIRE paths per the plan: .../S-BIAD/<last3 of number>/<accession>/
FIRE = {
    "S-BIAD2196": "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/196/S-BIAD2196/Files",
    "S-BIAD2133": "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/133/S-BIAD2133/Files",
}


def get_json(url: str, timeout: int = 90) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "TiJepa/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while c := fh.read(1 << 22):
            h.update(c)
    return h.hexdigest()


def fetch(url: str, dest: Path, tries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, tries + 1):
        rc = subprocess.run(
            ["aria2c", "-x", "8", "-s", "8", "-k", "16M", "--continue=true",
             "--auto-file-renaming=false", "--summary-interval=0",
             "--console-log-level=warn", "--timeout=60",
             "-d", str(dest.parent), "-o", dest.name, url]
        ).returncode
        if rc == 0 and dest.exists() and dest.stat().st_size > 0:
            return True
        print(f"   attempt {i} failed: {url}")
        time.sleep(5 * i)
    return False


def write(key: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{key}.json").write_text(json.dumps(payload, indent=2) + "\n")


def dryad() -> None:
    ds = get_json(
        "https://datadryad.org/api/v2/datasets/"
        + DRYAD_DOI.replace(":", "%3A").replace("/", "%2F")
    )
    fl = get_json(f"https://datadryad.org/api/v2/versions/{DRYAD_VERSION}/files")
    dest_dir = RAW / "dryad_bmdm"
    files, total = [], 0
    for f in fl.get("_embedded", {}).get("stash:files", []):
        name = f["path"]
        fid = f["_links"]["self"]["href"].rstrip("/").split("/")[-1]
        url = f"https://datadryad.org/api/v2/files/{fid}/download"
        dest = dest_dir / name
        if not (dest.exists() and dest.stat().st_size == f.get("size")):
            print(f"  get {name} ({f.get('size',0)/1e6:.1f} MB)")
            if not fetch(url, dest):
                files.append({"file": name, "status": "download_failed", "url": url})
                continue
        digest = sha256(dest)
        ok = digest == f.get("digest")
        files.append({
            "file": name, "url": url, "size_bytes_record": f.get("size"),
            "size_bytes_local": dest.stat().st_size,
            "sha256_record": f.get("digest"), "sha256_local": digest,
            "checksum_verified": ok, "status": "ok" if ok else "CHECKSUM_MISMATCH",
        })
        if not ok:
            print(f"  !! checksum mismatch for {name}")
        total += dest.stat().st_size
    write("dryad_bmdm", {
        "key": "dryad_bmdm", "kind": "dryad", "doi": "10.5061/dryad.m0cfxpp8v",
        "record_url": "https://datadryad.org/dataset/doi:10.5061/dryad.m0cfxpp8v",
        "title": ds.get("title"),
        "licence_verified": ds.get("license"),
        "licence_source": "datadryad.org API v2 dataset.license",
        "role": "partA_eval_headline", "files_used": files,
        "total_gb": round(total / 1e9, 3),
        "n_checksum_verified": sum(1 for f in files if f.get("checksum_verified")),
        "status": "ok",
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"dryad_bmdm: {len(files)} files, {total/1e9:.2f} GB")


def bbbc054() -> None:
    dest_dir = RAW / "bbbc054"
    want = ["Replicate1annotation.csv", "Replicate_1.zip"]
    files, total = [], 0
    for name in want:
        dest = dest_dir / name
        if not dest.exists():
            print(f"  get {name}")
            if not fetch(f"{BBBC054}/{name}", dest):
                files.append({"file": name, "status": "download_failed"})
                continue
        files.append({
            "file": name, "url": f"{BBBC054}/{name}",
            "size_bytes_local": dest.stat().st_size,
            "sha256_local": sha256(dest), "status": "ok",
            "note": "no publisher checksum available for BBBC files",
        })
        total += dest.stat().st_size
    write("bbbc054", {
        "key": "bbbc054", "kind": "bbbc",
        "record_url": "https://bbbc.broadinstitute.org/BBBC054",
        "licence_verified": "CC BY 3.0 Unported (commercial use allowed)",
        "licence_source": "BBBC054 page, Copyright section",
        "role": "partA_eval_morphology_and_lps_time",
        "imaging": "Leica DMI 6000, 20x objective, Andor AND-DG-152V-C1E-FI camera, "
                   "Metamorph Premiere; TIFF",
        "fov_time_mapping": "each FOV = one LPS exposure time, 30 min intervals, 0-30 h "
                            "(verified from the BBBC054 page)",
        "files_used": files, "total_gb": round(total / 1e9, 3), "status": "ok",
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"bbbc054: {len(files)} files, {total/1e9:.2f} GB")


def biostudies(acc: str, role: str, max_gb: float, include: str | None = None) -> None:
    meta = get_json(f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{acc}")

    def find_licence(sec: dict) -> str | None:
        for a in sec.get("attributes", []) or []:
            if a.get("name", "").lower() in ("license", "licence"):
                return str(a.get("value"))
        for s in sec.get("subsections", []) or []:
            if isinstance(s, dict):
                got = find_licence(s)
                if got:
                    return got
        return None

    lic = find_licence(meta.get("section", {}))
    listing = get_json(f"{EBI_FILES}/{acc}?pageSize=20000")
    rows = listing.get("data", [])
    if include:
        rows = [r for r in rows if include in r.get("path", "")]
    rows = sorted(rows, key=lambda r: r.get("Size", 0))

    dest_dir = RAW / acc.lower()
    files, total = [], 0.0
    for r in rows:
        size_gb = r.get("Size", 0) / 1e9
        if total + size_gb > max_gb:
            continue
        rel = r["path"]
        dest = dest_dir / rel
        if not (dest.exists() and dest.stat().st_size == r.get("Size")):
            if not fetch(f"{FIRE[acc]}/{rel}", dest):
                continue
        files.append({
            "file": rel, "url": f"{FIRE[acc]}/{rel}",
            "size_bytes_record": r.get("Size"),
            "size_bytes_local": dest.stat().st_size,
            "pixel_size": r.get("Pixel_Size__µm_") or r.get("pixel_dimensions"),
            "time_interval_s": r.get("Time_Interval__s_"),
            "annotation_type": r.get("Annotation_type"),
            "source_image": r.get("Source_image"),
            "status": "ok",
        })
        total += size_gb
    write(acc.lower(), {
        "key": acc.lower(), "kind": "biostudies", "accession": acc,
        "record_url": f"https://www.ebi.ac.uk/biostudies/studies/{acc}",
        "title": next((a.get("value") for a in meta.get("section", {}).get("attributes", [])
                       if a.get("name", "").lower() == "title"), meta.get("accno")),
        "licence_verified": lic, "licence_source": f"BioStudies API study {acc} attributes",
        "role": role, "n_files_in_record": listing.get("recordsFiltered"),
        "files_used": files, "total_gb": round(total, 3), "status": "ok",
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"{acc}: {len(files)} files, {total:.2f} GB (licence {lic})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    jobs = {
        "dryad": dryad,
        "bbbc054": bbbc054,
        "s2196": lambda: biostudies("S-BIAD2196", "pretrain_and_tracking_eval", 5.0),
        "s2133": lambda: biostudies("S-BIAD2133", "frame_qc_classifier", 8.0),
    }
    for name, fn in jobs.items():
        if args.only not in ("all", name):
            continue
        print(f"== {name}")
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED {name}: {type(exc).__name__}: {exc}")
            write(name, {"key": name, "status": "failed", "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
