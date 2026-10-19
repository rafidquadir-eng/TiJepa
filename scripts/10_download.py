"""M1: download datasets listed in the plan, verify sizes and record checksums.

Each dataset goes into its own directory under data/raw/<key>/ (downloaded files are
untrusted: nothing inside them is ever executed). Writes/updates
manifests/datasets.yaml entries via results/downloads/<key>.json side-files that
30_build_manifest.py collects.

Usage:
  python3 -I scripts/10_download.py --group small
  python3 -I scripts/10_download.py --group complexeye --cap-gb 60
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
OUT = REPO / "results" / "v1" / "downloads"

ZENODO = "https://zenodo.org/api/records"

# ---------------------------------------------------------------- small datasets
SMALL = {
    # Part B: Tay-lab code release + single-cell NF-kB traces
    "tay_traces": {
        "kind": "zenodo",
        "record": "14337874",
        "files": None,  # all
        "role": "partB_traces",
    },
    # PhagoStat annotations (small companion archive)
    "phagostat_annot": {
        "kind": "zenodo",
        "record": "10803492",
        "files": ["manually_annotated_data.zip"],
        "role": "pretrain_annot",
    },
}


def sha256(path: Path, limit_gb: float | None = None) -> str:
    """Full sha256; for very large files hash the first limit_gb only (recorded as such)."""
    h = hashlib.sha256()
    cap = None if limit_gb is None else int(limit_gb * 1e9)
    read = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
            read += len(chunk)
            if cap is not None and read >= cap:
                return f"sha256-first{limit_gb}GB:{h.hexdigest()}"
    return h.hexdigest()


def zenodo_record(record: str) -> dict:
    import urllib.request

    req = urllib.request.Request(
        f"{ZENODO}/{record}", headers={"User-Agent": "TiJepa/0.1"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def fetch(url: str, dest: Path, tries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, tries + 1):
        cmd = [
            "aria2c", "-x", "8", "-s", "8", "-k", "16M", "--continue=true",
            "--auto-file-renaming=false", "--summary-interval=0",
            "--console-log-level=warn", "-d", str(dest.parent), "-o", dest.name, url,
        ]
        if subprocess.run(cmd).returncode == 0 and dest.exists():
            return
        print(f"  attempt {attempt} failed for {url}")
        time.sleep(5 * attempt)
    raise RuntimeError(f"download failed after {tries} tries: {url}")


def record_result(key: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{key}.json").write_text(json.dumps(payload, indent=2) + "\n")


def do_zenodo(key: str, spec: dict, cap_gb: float | None = None) -> None:
    rec = zenodo_record(spec["record"])
    lic = rec.get("metadata", {}).get("license", {})
    lic_id = lic.get("id") if isinstance(lic, dict) else str(lic)
    files = rec.get("files", [])
    wanted = spec.get("files")
    if wanted:
        files = [f for f in files if f.get("key") in wanted]
    # smallest first, so a cap yields the most archives
    files = sorted(files, key=lambda f: f.get("size", 0))

    dest_dir = RAW / key
    got, total = [], 0.0
    for f in files:
        size_gb = f.get("size", 0) / 1e9
        if cap_gb is not None and total + size_gb > cap_gb:
            print(f"  cap {cap_gb} GB reached; skipping {f['key']} ({size_gb:.2f} GB)")
            continue
        dest = dest_dir / f["key"].replace("/", "_")
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        if dest.exists() and abs(dest.stat().st_size - f.get("size", 0)) < 1024:
            print(f"  have {dest.name} ({size_gb:.2f} GB)")
        else:
            print(f"  get  {dest.name} ({size_gb:.2f} GB)")
            fetch(url, dest)
        actual = dest.stat().st_size
        got.append(
            {
                "file": dest.name,
                "url": url,
                "size_bytes_record": f.get("size"),
                "size_bytes_local": actual,
                "size_match": abs(actual - f.get("size", 0)) < 1024,
                "checksum_record": f.get("checksum"),
                "sha256_local": sha256(dest, limit_gb=2.0),
            }
        )
        total += size_gb

    record_result(
        key,
        {
            "key": key,
            "kind": "zenodo",
            "record": spec["record"],
            "doi": rec.get("doi"),
            "title": rec.get("metadata", {}).get("title"),
            "licence_verified": lic_id,
            "licence_source": f"{ZENODO}/{spec['record']} metadata.license",
            "role": spec["role"],
            "files_used": got,
            "total_gb": round(total, 3),
            "status": "ok" if got else "empty",
            "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(f"{key}: {len(got)} files, {total:.2f} GB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["small", "complexeye"])
    ap.add_argument("--cap-gb", type=float, default=None)
    args = ap.parse_args()

    if args.group == "small":
        for key, spec in SMALL.items():
            print(f"== {key}")
            try:
                do_zenodo(key, spec)
            except Exception as exc:
                print(f"  FAILED {key}: {exc}")
                record_result(key, {"key": key, "status": "failed", "error": str(exc)})
    elif args.group == "complexeye":
        do_zenodo(
            "complexeye",
            {"kind": "zenodo", "record": "10082674", "files": None, "role": "pretrain"},
            cap_gb=args.cap_gb,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
