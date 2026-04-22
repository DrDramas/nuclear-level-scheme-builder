#!/usr/bin/env python3
"""
GetMatricesSCA.py

Parses nuclear gamma-ray data files and constructs the level scheme graph,
then generates the Singles (S), Coincidence (C), and Adjacency (A) matrices
used for numerical optimization of gamma-ray decay schemes.

Functions
---------
MakeLevelsAndVertices(fname)
    Parse a .gam file and return a populated LSGraph.
GetGammaEnergies()
    Return list of gamma-ray energies.
GetGammaObjects()
    Return list of Gamma objects.
GetSingles(nc=1)
    Return the singles intensity vector, scaled by nc.
GetCoincidences(Glevel, nc=1)
    Return the gamma-gamma coincidence matrix as a NumPy array.
GetAdjacency()
    Return the adjacency matrix as a NumPy array.
Print_Level_Scheme()
    Print all levels and their outgoing transitions to stdout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from NuclearObjects import LSGraph, Gamma, Level

logger = logging.getLogger(__name__)

# Module-level state — populated by MakeLevelsAndVertices()
gammas: list[Gamma] = []
levels: list[Level] = []
gammaIDs: dict[float, int] = {}   # {gamma energy: index in gammas}
levelIDs: dict[float, int] = {}   # {level energy: index in levels}
vertices: dict[float, list[float]] = {}


# ------------------------------------------------------------------ #
def MakeLevelsAndVertices(fname: str) -> LSGraph:
    """Parse a .gam gamma-ray data file and build the level scheme graph.

    Populates the module-level ``gammas``, ``levels``, ``gammaIDs``,
    ``levelIDs``, and ``vertices`` collections, then returns an
    :class:`LSGraph` object representing the full level scheme.

    Parameters
    ----------
    fname:
        Path to a .gam formatted nuclear data file.

    Returns
    -------
    LSGraph
        Populated graph with one vertex per nuclear level and edges for
        each gamma-ray transition.
    """
    content = _readin_ascii(fname)

    # Create Gamma objects from numeric data lines
    for line in content:
        data = line.split()
        if not data:
            continue
        try:
            float(data[0])          # only process lines that start with a number
            gam = Gamma(line)
            gammas.append(gam)
            logger.debug("Loaded gamma: %s keV (RI=%s)", gam.gE, gam.RI)
        except (ValueError, IndexError):
            pass                    # header / comment / blank lines — skip silently

    # Build energy->index lookup using enumerate
    for i, gam in enumerate(gammas):
        gammaIDs[gam.gE] = i

    # Build Level objects and branching ratios
    _make_levels()
    _add_ghost_levels()
    for i, lvl in enumerate(levels):
        levelIDs[lvl.ExE] = i

    # Build graph vertices and return LSGraph
    _make_vertices()
    Glevel = LSGraph(vertices)
    Glevel.add_vertex(0.0)   # ensure ground state is always present

    return Glevel


# ------------------------------------------------------------------ #
def GetGammaEnergies() -> list[float]:
    """Return a list of all gamma-ray energies in the order they were loaded."""
    return [gam.gE for gam in gammas]


# ------------------------------------------------------------------ #
def GetGammaObjects() -> list[Gamma]:
    """Return the list of Gamma objects populated by MakeLevelsAndVertices."""
    return gammas


# ------------------------------------------------------------------ #
def GetSingles(nc: float = 1.0) -> list[float]:
    """Return the singles intensity vector.

    Parameters
    ----------
    nc:
        Normalization constant used to scale the total number of counts.

    Returns
    -------
    list[float]
        Relative intensities, one entry per gamma-ray transition.
    """
    return [gam.RI * nc for gam in gammas]


# ------------------------------------------------------------------ #
def GetCoincidences(Glevel: LSGraph, nc: float = 1.0) -> np.ndarray:
    """Compute the gamma-gamma coincidence matrix.

    For each nuclear level, all possible decay pathways to the ground
    state are enumerated. For every pair of gammas (i, j) that appear
    in the same pathway, the coincidence weight is accumulated in
    C[i][j] and C[j][i].

    Parameters
    ----------
    Glevel:
        Populated LSGraph returned by MakeLevelsAndVertices.
    nc:
        Normalization constant.

    Returns
    -------
    np.ndarray
        Symmetric coincidence matrix of shape (N, N) where N is the
        number of gamma-ray transitions.
    """
    n = len(gammas)
    # NumPy array — much faster than nested Python lists for arithmetic
    gam_coinc_arr = np.zeros((n, n), dtype=float)

    # endpoints: (start_energy, stop_energy) -> set of subpath tuples
    # Using a set of tuples gives O(1) lookup for deduplication
    endpoints: dict[tuple[float, float], set[tuple[float, ...]]] = {}

    # Wrap with a progress bar if tqdm is available
    level_iter = (
        _tqdm(levels, desc="Computing coincidences", unit="level")
        if _HAS_TQDM else levels
    )

    for level in level_iter:
        paths = Glevel.find_all_paths(level.ExE, 0.0)

        for path in paths:
            if len(path) <= 2:   # need at least 3 levels to define a g-g coincidence
                continue

            # Build ordered list of Gamma objects along this path
            glst: list[Gamma] = []
            for i in range(1, len(path)):
                # O(1) level lookup via dict instead of linear scan
                lvl_id = levelIDs[path[i - 1]]
                for gam in levels[lvl_id].outGammas:
                    if gam.ExB == path[i]:
                        glst.append(gam)
                        break

            # Accumulate coincidences for every (i, j) pair in this path
            ggc = 0.0
            for i in range(len(glst) - 1):
                for j in range(i + 1, len(glst)):
                    if j == i + 1:
                        # Adjacent pair: weight = RI_i x BR_j
                        ggc = glst[i].RI * glst[j].BR
                    else:
                        # Non-adjacent: keep multiplying branching ratios
                        ggc = ggc * glst[j].BR

                    start_stop = (glst[i].gE, glst[j].gE)
                    # Tuple + set for O(1) subpath deduplication
                    subpath = tuple(g.gE for g in glst[i:j + 1])

                    if start_stop not in endpoints:
                        endpoints[start_stop] = set()
                    if subpath in endpoints[start_stop]:
                        continue   # redundant path — skip
                    endpoints[start_stop].add(subpath)

                    k = gammaIDs[glst[i].gE]
                    l = gammaIDs[glst[j].gE]
                    gam_coinc_arr[k, l] += ggc * nc
                    gam_coinc_arr[l, k] = gam_coinc_arr[k, l]

    return gam_coinc_arr


# ------------------------------------------------------------------ #
def GetAdjacency() -> np.ndarray:
    """Build the adjacency matrix A.

    A[i][j] is the branching ratio for the transition from gamma_i's
    final level to gamma_j.

    Returns
    -------
    np.ndarray
        Adjacency matrix of shape (N, N).
    """
    n = len(gammas)
    A = np.zeros((n, n), dtype=float)

    for level in levels:
        for inGamma in level.outGammas:
            i = gammaIDs[inGamma.gE]
            fl = levelIDs[inGamma.ExB]   # O(1) dict lookup
            for outGamma in levels[fl].outGammas:
                j = gammaIDs[outGamma.gE]
                A[i, j] = outGamma.BR

    return A


# ------------------------------------------------------------------ #
def Print_Level_Scheme() -> None:
    """Print all levels and their outgoing gamma transitions to stdout."""
    for level in levels:
        print(f"Level: {level.ExE}")
        for gam in level.outGammas:
            print(f"  Gamma: {gam.gE} keV  BR: {gam.BR:.4f}  Final state: {gam.ExB}")


# ------------------------------------------------------------------ #
def export_matrices(
    S: list[float],
    C: np.ndarray,
    A: np.ndarray,
    output_dir: str = ".",
) -> None:
    """Export S, C, and A matrices to CSV files.

    Rows and columns of C and A are labelled by gamma-ray energy in keV.

    Parameters
    ----------
    S:
        Singles intensity vector from :func:`GetSingles`.
    C:
        Coincidence matrix from :func:`GetCoincidences`.
    A:
        Adjacency matrix from :func:`GetAdjacency`.
    output_dir:
        Directory to write files into (created if it does not exist).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    energies = GetGammaEnergies()

    pd.Series(S, index=energies, name="intensity").to_csv(out / "S_singles.csv", header=True)
    pd.DataFrame(C, index=energies, columns=energies).to_csv(out / "C_coincidence.csv")
    pd.DataFrame(A, index=energies, columns=energies).to_csv(out / "A_adjacency.csv")
    logger.info("Matrices exported to %s", out.resolve())
    print(f"Matrices exported to {out.resolve()}")


# ------------------------------------------------------------------ #
#  Private helpers
# ------------------------------------------------------------------ #

def _readin_ascii(fname: str) -> list[str]:
    """Read a text data file and return its lines.

    Parameters
    ----------
    fname:
        Path to the data file.

    Returns
    -------
    list[str]
        Lines of the file.

    Raises
    ------
    SystemExit
        If the file cannot be opened.
    """
    path = Path(fname)
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError as exc:
        logger.error("Could not open data file '%s': %s", fname, exc)
        raise SystemExit(1) from exc


def _make_levels() -> None:
    """Populate the module-level levels list from the gammas list."""
    levels.append(Level(0.0))

    for gam in gammas:
        ExE = gam.ExT
        if not any(lvl.ExE == ExE for lvl in levels):
            levels.append(Level(ExE))
        for lvl in levels:
            if lvl.ExE == ExE:
                lvl.add_outgoing_gamma(gam)
                break

    # Compute branching ratios for every non-ground-state level
    for lvl in levels:
        if lvl.ExE > 0:
            lvl.compute_BRs()


def _add_ghost_levels() -> None:
    """Add levels for final states that emit no detected gammas.

    These ghost levels are populated by gamma decay but do not
    themselves emit any detected gammas (e.g. long-lived isomers).
    """
    existing_energies = {lvl.ExE for lvl in levels}   # set for O(1) lookup
    for gam in gammas:
        if gam.ExB not in existing_energies:
            logger.debug("Adding ghost level at %.4f keV", gam.ExB)
            levels.append(Level(gam.ExB))
            existing_energies.add(gam.ExB)


def _make_vertices() -> None:
    """Build the vertices adjacency dictionary used by LSGraph."""
    for lvl in levels:
        vlist = [gam.ExB for gam in lvl.outGammas]
        if vlist:
            vertices[lvl.ExE] = vlist
