"""Checkpoint provenance for every encoder result (PLAN_v2.md rule 1, G2.1; finding N1).

An evaluation records, for each learned encoder, the checkpoint file, run name, step and
SHA-256 it came from. Paper and figure labels are derived from that record and nothing
else, so a TiJepa-0 result can never be printed as TiJepa-1 again.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

RANDOM_INIT = "random_init"

#: run name -> (result key, display label). Keys are what results files use.
RUNS = {
    "tijepa0": ("E0_tijepa0", "TiJepa-0"),
    "tijepa1": ("E4_tijepa1", "TiJepa-1"),
    "tijepa1_spatial": ("E4s_tijepa1_spatial", "TiJepa-1 (spatial-only)"),
    RANDOM_INIT: ("E6_random_init", "random init"),
}

#: fixed (non-checkpoint) encoders and their display labels
FIXED = {
    "E1_handcrafted": "hand-crafted",
    "E2_imagenet_vit_s": "ImageNet ViT-S",
    "E3_dinov2_vit_s": "DINOv2 ViT-S/14",
    "E7_vjepa2_vitl": "V-JEPA 2 ViT-L",
    "E5_frame_shuffle": "TiJepa-1, frame-shuffled",
    "copy_last": "copy-last baseline",
    "baseline": "baseline tracker",
}


class ProvenanceError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def run_name_of(ckpt: Path, sd: dict | None = None) -> str:
    """The run name stored in the checkpoint config, or parsed from the file name."""
    if sd is not None and isinstance(sd.get("config"), dict) and sd["config"].get("run_name"):
        return str(sd["config"]["run_name"])
    stem = Path(ckpt).name
    if stem.startswith("tijepa1_spatial"):
        return "tijepa1_spatial"
    if stem.startswith("tijepa1"):
        return "tijepa1"
    if stem.startswith("tijepa0"):
        return "tijepa0"
    raise ProvenanceError(f"cannot determine the run of {ckpt}")


def checkpoint_record(ckpt: Path, sd: dict | None = None) -> dict:
    ckpt = Path(ckpt)
    if sd is None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = sd.get("config") or {}
    run = run_name_of(ckpt, sd)
    # the v0 trainer did not store run_name; its file name is the only evidence
    if run == "tijepa1" and ckpt.name.startswith("tijepa0"):
        run = "tijepa0"
    return {
        "file": ckpt.name,
        "run_name": run,
        "step": int(sd.get("step", -1)),
        "planned_final_step": cfg.get("max_steps"),
        "final": cfg.get("max_steps") is not None
        and int(sd.get("step", -1)) == int(cfg["max_steps"]) and "segments" in sd,
        "sha256": sha256_file(ckpt),
    }


def random_init_record(seed: int) -> dict:
    return {"file": None, "run_name": RANDOM_INIT, "step": 0, "sha256": None,
            "init_seed": seed}


def key_for(record: dict) -> str:
    try:
        return RUNS[record["run_name"]][0]
    except KeyError as exc:
        raise ProvenanceError(f"unknown run {record.get('run_name')!r}") from exc


def label_for(record: dict) -> str:
    try:
        return RUNS[record["run_name"]][1]
    except KeyError as exc:
        raise ProvenanceError(f"unknown run {record.get('run_name')!r}") from exc


def label_for_key(key: str, record: dict | None) -> str:
    """Display label for a results key, checked against its checkpoint record.

    Raises if a key claims one run but its record says another (the N1 failure).
    """
    if record is not None:
        expect = key_for(record)
        if not (key == expect or key.startswith(expect + "_")
                or key.startswith(expect + ":")):
            raise ProvenanceError(
                f"result key {key!r} does not match its checkpoint run "
                f"{record['run_name']!r} (expected key {expect!r})")
        return label_for(record)
    for k, lab in FIXED.items():
        if key == k or key.startswith(k + "_"):
            return lab
    for rk, (k, _lab) in RUNS.items():
        if key == k or key.startswith(k + "_"):
            raise ProvenanceError(f"{key!r} is a learned encoder ({rk}) with no checkpoint "
                                  "record")
    raise ProvenanceError(f"no label rule for {key!r}")


def check_results(encoders: dict) -> list[str]:
    """Return every provenance problem in an `encoders` mapping {key: {checkpoint: …}}."""
    problems = []
    for key, rec in encoders.items():
        if not isinstance(rec, dict) or rec.get("status") == "unavailable":
            continue
        cp = rec.get("checkpoint")
        if cp is None and rec.get("checkpoint_per_seed"):
            per = list(rec["checkpoint_per_seed"].values())
            if len({c["run_name"] for c in per}) != 1:
                problems.append(f"{key!r}: per-seed checkpoints from different runs")
                continue
            cp = per[0]
        try:
            label_for_key(key, cp)
        except ProvenanceError as exc:
            problems.append(str(exc))
    return problems
