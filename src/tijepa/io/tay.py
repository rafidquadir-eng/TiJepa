"""Loader for the Tay-lab single-cell NF-kB traces (Zenodo 10.5281/zenodo.14337874).

Column layout, derived by reading `rawtraceprocessing.m` in the deposit (the deposit
ships no column dictionary, and the `single_cell_features/` folder that holds the
human-readable condition labels was NOT deposited):

    scmat has 131 columns.
      cols 0..125   NF-kB (RelA) nuclear localisation trace, 6 min per frame
      col  126      first ligand code   = ligs(1)    (`scmat(:, end-4) = ligs(1)`)
      col  127      second ligand code  = ligs(end)  (`scmat(:, end-3) = ligs(end)`)
      col  128      inter-stimulus interval in hours, 4 or 8
      col  129      replicate raw-file index (cc)
      col  130      per-well experiment counter (the grouping unit for CV)

Frame interval: `featureextraction.m` treats MATLAB columns 3:42 (40 frames) as the
first 4-hour stimulus window => 6 min per frame. The first ligand is added at
`startframe + 3`, i.e. index 3, leaving indices 0..2 as pre-stimulus baseline.

Ligand codes (v1: SOURCED from the deposit, superseding the v0 inference below).
`figure4_5plotting.m` assigns the codes with their doses in trailing comments:
    line   8  `lig1 = 1; %100 nM CpG`        line   9  `lig2 = 2; %1 ng LPS`
    line 148  `lig1 = 3; %1000 ng/mL PIC`    line 149  `lig2 = 6; %10 ng/mL TNF`
so 1 = CpG, 2 = LPS, 3 = polyI:C, 6 = TNF. This confirms the v0 inference for LPS and
TNF and CORRECTS it for code 3, which is polyI:C and not CpG. Codes 4, 5, 7, 17 and
19-21 remain unresolved: the folder holding the human-readable labels was not
deposited, so they are left unresolved rather than guessed. See
`scripts/23_partB_ligand_codes.py` and `results/v1/partB/ligand_codes.json`.

Column 128 is the inter-stimulus interval in hours. `rawtraceprocessing.m` writes it
from whether the ligand list repeats a value, which reads like a homologous flag, but
the pair (first=0, second=2) occurs at BOTH 4 and 8 and every heterologous pair occurs
at both values, so it cannot be a function of ligand identity. Homologous
restimulation was only performed at the 8 h interval.

The superseded v0 inference, kept for the audit trail:
`col 127` holds the actual second ligand, and its value set per deposit file must match
that file's name. TNFLPSfirst has col127 in {2, 6} = {TNF, LPS}; CPGLPSscrep has
col127 in {2, 3} = {CpG, LPS}. The label sets intersect in LPS and the code sets
intersect in 2, so **code 2 = LPS**, hence 6 = TNF and 3 = CpG. PICTNFscrep is
consistent (col127 == 6 only, a TNF second stimulus). Code 0 in col 126 marks wells
with no first ligand, i.e. the naive -> X controls: in TNFLPSfirst the homologous
wells (first == second) exist only at the 8 h interval, and (0, X) wells exist at both
4 h and 8 h, so naive and primed can be compared at a matched 8 h interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_TRACE = 126
FRAME_MIN = 6.0
STIM1_IDX = 3
C_FIRST, C_SECOND, C_GAP, C_REP, C_WELL = 126, 127, 128, 129, 130

# Ligand codes sourced from the deposit's figure4_5plotting.m; see module docstring.
LIGAND_CODES = {1.0: "CpG", 2.0: "LPS", 3.0: "polyIC", 6.0: "TNF"}
LIGAND_DOSES = {1.0: "100 nM", 2.0: "1 ng/ml", 3.0: "1000 ng/ml", 6.0: "10 ng/ml"}
#: codes that occur in the data but cannot be resolved from the deposited material
UNRESOLVED_CODES = (4.0, 5.0, 7.0, 17.0, 19.0, 20.0, 21.0)
NAIVE_CODE = 0.0

# Files whose 5-column metadata layout we could confirm: the interval column holds only
# the documented values {4, 8} and the second-ligand column holds a small, consistent
# code set matching the file name. The cytokine-block and sepsis experiments have a
# different design (blocking antibodies, patient samples) whose metadata columns do not
# follow the same convention -- they are excluded from all analyses and reported as
# layout-unverified in results/partB/conditions.csv.
VERIFIED_LAYOUT_FILES = {
    "TNFLPSfirst_traces.mat",
    "PICCPGfirst_traces.mat",
    "PAMIL1first_traces.mat",
    "CPGLPSscrep_traces.mat",
    "PICTNFscrep_traces.mat",
}

TRACE_FILES = [
    "TNFLPSfirst_traces.mat",
    "PICCPGfirst_traces.mat",
    "PAMIL1first_traces.mat",
    "CPGLPSscrep_traces.mat",
    "PICTNFscrep_traces.mat",
    "PICfirst_TNFblock_traces.mat",
    "CPGfirst_TNFblock_traces.mat",
    "CpGfirst_IL10block_traces.mat",
    "sepsis_traces.mat",
]


@dataclass
class TraceSet:
    traces: np.ndarray  # (n_cells, 126) float, NaN where the cell was not tracked
    first: np.ndarray
    second: np.ndarray
    gap: np.ndarray
    rep: np.ndarray
    well: np.ndarray
    source: np.ndarray  # deposit file each cell came from

    def __len__(self) -> int:
        return self.traces.shape[0]

    def subset(self, m: np.ndarray) -> TraceSet:
        return TraceSet(*(getattr(self, f)[m] for f in
                          ("traces", "first", "second", "gap", "rep", "well", "source")))

    @property
    def layout_verified(self) -> np.ndarray:
        return np.isin(self.source, list(VERIFIED_LAYOUT_FILES))

    @property
    def group(self) -> np.ndarray:
        """Grouping unit for cross-validation: source + replicate + well."""
        return np.array(
            [f"{s}|rep{int(r)}|well{int(w)}"
             for s, r, w in zip(self.source, self.rep, self.well, strict=False)]
        )


def find_root(repo: Path) -> Path:
    cands = sorted((repo / "data" / "interim" / "tay").glob("*/processedtraces"))
    if not cands:
        raise FileNotFoundError("Tay processedtraces/ not found under data/interim/tay")
    return cands[0]


def load(repo: Path, files: list[str] | None = None) -> TraceSet:
    import scipy.io as sio

    root = find_root(repo)
    parts: list[TraceSet] = []
    for name in files or TRACE_FILES:
        p = root / name
        if not p.exists():
            continue
        d = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
        key = [k for k in d if not k.startswith("__")][0]
        m = np.asarray(d[key], dtype=np.float64)
        if m.shape[1] < 131:
            # sepsis_traces.mat has 109 columns (shorter movie); keep metadata layout
            n = m.shape[1]
            tr = m[:, : n - 5]
            tr = np.pad(tr, ((0, 0), (0, N_TRACE - tr.shape[1])),
                        constant_values=np.nan)
            meta = m[:, n - 5:]
        else:
            tr, meta = m[:, :N_TRACE], m[:, N_TRACE:]
        parts.append(TraceSet(
            traces=tr, first=meta[:, 0], second=meta[:, 1], gap=meta[:, 2],
            rep=meta[:, 3], well=meta[:, 4],
            source=np.array([name] * m.shape[0]),
        ))
    if not parts:
        raise FileNotFoundError("no Tay trace files loaded")
    return TraceSet(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                      ("traces", "first", "second", "gap", "rep", "well", "source")))


def second_stim_index(gap_hours: float) -> int:
    """Frame index at which the second stimulus is delivered."""
    return STIM1_IDX + int(round(gap_hours * 60.0 / FRAME_MIN))


def response_window(ts: TraceSet, gap_hours: float, length: int | None = None
                    ) -> tuple[np.ndarray, int, int]:
    """Baseline-corrected response to the SECOND stimulus, (n_cells, L)."""
    i0 = second_stim_index(gap_hours)
    i1 = N_TRACE if length is None else min(N_TRACE, i0 + length)
    win = ts.traces[:, i0:i1].copy()
    base = np.nanmean(ts.traces[:, max(0, i0 - 3):i0], axis=1, keepdims=True)
    return win - base, i0, i1
