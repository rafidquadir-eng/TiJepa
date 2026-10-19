"""Part C1/C2: anchor activation signatures and rank-based single-sample scoring."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np


def parse_characteristics(text: str) -> dict:
    out: dict[str, str] = {}
    for part in (text or "").split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def read_series_matrix_table(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Return (row_ids, sample_ids, values) from a GEO series matrix."""
    op = gzip.open if path.suffix == ".gz" else open
    rows, ids, header = [], [], None
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        in_table = False
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not in_table:
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = [p.strip().strip('"') for p in parts[1:]]
                continue
            ids.append(parts[0].strip().strip('"'))
            rows.append([np.nan if p.strip() in ("", "null", "NA") else float(p)
                         for p in parts[1:]])
    if header is None or not rows:
        raise ValueError(f"no expression table in {path}")
    return ids, header, np.asarray(rows, dtype=np.float64)


def load_gpl_annotation(path: Path) -> dict[str, str]:
    """Probe ID -> gene symbol from a GEO platform .annot.gz."""
    op = gzip.open if path.suffix == ".gz" else open
    mapping: dict[str, str] = {}
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = None
        for line in fh:
            if line.startswith("#") or line.startswith("!") or line.startswith("^"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                try:
                    i_id, i_sym = header.index("ID"), header.index("Gene symbol")
                except ValueError:
                    return {}
                continue
            if len(parts) > i_sym and parts[i_sym]:
                sym = parts[i_sym].split("///")[0].strip()
                if sym:
                    mapping[parts[i_id].strip()] = sym
    return mapping


def load_hgnc_ensembl_map(path: Path) -> dict[str, str]:
    """Ensembl gene ID -> HGNC symbol."""
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            i_sym, i_ens = header.index("symbol"), header.index("ensembl_gene_id")
        except ValueError:
            return {}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) > max(i_sym, i_ens) and p[i_ens]:
                mapping[p[i_ens].strip()] = p[i_sym].strip()
    return mapping


def collapse_to_symbols(ids: list[str], values: np.ndarray,
                        mapping: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """Map row IDs to symbols, keeping the highest-variance row per symbol."""
    var = np.nanvar(values, axis=1)
    best: dict[str, int] = {}
    for i, rid in enumerate(ids):
        sym = mapping.get(rid) or mapping.get(rid.split(".")[0])
        if not sym:
            continue
        if sym not in best or var[i] > var[best[sym]]:
            best[sym] = i
    syms = sorted(best)
    return syms, values[[best[s] for s in syms], :]


def welch_de(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch t-test per gene: (t, p, mean difference x - y)."""
    from scipy import stats

    t, p = stats.ttest_ind(x, y, axis=1, equal_var=False, nan_policy="omit")
    return (np.asarray(t, dtype=float), np.asarray(p, dtype=float),
            np.nanmean(x, axis=1) - np.nanmean(y, axis=1))


def top_up_signature(syms: list[str], t: np.ndarray, p: np.ndarray, diff: np.ndarray,
                     n: int = 100, fdr: float | None = None) -> list[str]:
    """Top-n up-regulated genes, ranked by t statistic among positive differences."""
    from ..eval.stats import benjamini_hochberg

    ok = np.isfinite(t) & np.isfinite(diff) & (diff > 0)
    if fdr is not None:
        q = np.full_like(p, 1.0, dtype=float)
        fin = np.isfinite(p)
        q[fin] = benjamini_hochberg(p[fin])
        ok &= q < fdr
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(-t[idx])]
    return [syms[i] for i in order[:n]]


def singscore(values: np.ndarray, syms: list[str], signature: list[str]) -> np.ndarray:
    """Rank-based single-sample score (singscore-style), one value per sample.

    For each sample the genes are ranked low->high; the score is the mean rank of the
    signature genes, normalised to [0, 1] and centred so 0 means "mean rank = median".
    Rank-based scoring makes the score comparable across platforms (arrays vs RNA-seq).
    """
    pos = {s: i for i, s in enumerate(syms)}
    rows = [pos[g] for g in signature if g in pos]
    if not rows:
        return np.full(values.shape[1], np.nan)
    out = np.empty(values.shape[1], dtype=float)
    n_genes = values.shape[0]
    for j in range(values.shape[1]):
        col = values[:, j]
        finite = np.isfinite(col)
        ranks = np.full(n_genes, np.nan)
        order = np.argsort(col[finite])
        r = np.empty(order.size)
        r[order] = np.arange(1, order.size + 1)
        ranks[np.flatnonzero(finite)] = r
        sel = ranks[rows]
        sel = sel[np.isfinite(sel)]
        if sel.size == 0:
            out[j] = np.nan
            continue
        mean_rank = sel.mean()
        out[j] = (mean_rank - (order.size + 1) / 2.0) / order.size
    return out


def parse_gse58310_columns(cols: list[str]) -> list[dict]:
    """Decode the Saeed 2014 expression column names.

    Format: Monocytes_<day>_<treatment>_<donor>_<id>, e.g. Monocytes_6d_BG_BC8_4642.
    RPMI at 0d = naive monocyte; RPMI at 6d = macrophage control; LPS = LPS-tolerised;
    BG = beta-glucan-trained.
    """
    label = {"RPMI": "control", "LPS": "LPS_tolerised", "BG": "beta_glucan_trained"}
    out = []
    for c in cols:
        m = re.match(r"Monocytes_(\d+)d_([A-Za-z]+)_([A-Za-z0-9]+)_(\d+)", c)
        if not m:
            out.append({"column": c, "parsed": False})
            continue
        day, treat, donor, sid = m.groups()
        state = ("naive_monocyte" if treat == "RPMI" and day == "0"
                 else label.get(treat, treat))
        out.append({"column": c, "parsed": True, "day": int(day), "treatment": treat,
                    "donor": donor, "sample_id": sid, "state": state})
    return out
