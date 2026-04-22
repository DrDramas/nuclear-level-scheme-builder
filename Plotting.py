#!/usr/bin/env python3
"""
Plotting.py

Graphical layout and drawing routines for nuclear level scheme and
transition scheme graphs built with NetworkX.

Public API
----------
draw_level_scheme(ls, ax, color_map, branch)
    Draw a LevelScheme object onto a matplotlib Axes.
draw_transition_scheme(ts, ax, branch)
    Draw a TransitionScheme object onto a matplotlib Axes.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.random as rnd

__all__ = ["draw_level_scheme", "draw_transition_scheme"]


# ------------------------------------------------------------------ #
#  Private layout helpers
# ------------------------------------------------------------------ #

def _align_close_levels(g: nx.DiGraph, pos: dict) -> None:
    """Snap horizontally close source nodes onto the same x-coordinate.

    Prevents near-vertical edges from appearing as thin slivers when
    two levels have very similar energies.

    Parameters
    ----------
    g:
        The level scheme directed graph.
    pos:
        Mutable position dict ``{node: [x, y]}``.
    """
    tol = np.max(g.nodes(data="energy")) / 20
    for n in g.nodes():
        for src, _ in g.in_edges(n):
            en = g.nodes()[n]["energy"]
            en_src = g.nodes()[src]["energy"]
            if np.abs(en - en_src) < tol:
                pos[src][0] = pos[n][0]


def _align_single_edges(g: nx.DiGraph, pos: dict) -> None:
    """Position nodes with a single outgoing edge directly above their child.

    Parameters
    ----------
    g:
        The level scheme directed graph.
    pos:
        Mutable position dict ``{node: [x, y]}``.
    """
    for n in g.nodes():
        if g.out_degree(n) == 1:
            nchild = list(g.out_edges(n))[0][1]
            pos[n][0] = pos[nchild][0]


def _node_offset(value: float, xscale: float) -> float:
    """Return a small random horizontal offset for layout jitter.

    Parameters
    ----------
    value:
        Current x-coordinate of the node (unused but kept for API symmetry).
    xscale:
        Horizontal scale of the current layout.

    Returns
    -------
    float
        A signed random offset in the range ``[0.1·xscale, 0.3·xscale]``.
    """
    scl = xscale if xscale != 0 else 1
    return rnd.choice([-1, 1]) * rnd.uniform(low=0.1 * scl, high=0.3 * scl)


def _avoid_vertical_skips(g: nx.DiGraph, pos: dict) -> None:
    """Jitter node positions to reduce vertical edges crossing unrelated nodes.

    Parameters
    ----------
    g:
        The directed graph.
    pos:
        Mutable position dict ``{node: [x, y]}``.
    """
    xcoords = [p[1][0] for p in pos.items()]

    for position in pos.items():
        x = position[1][0]
        v_edges = [e for e in g.edges() if pos[e[0]][0] == x and pos[e[1]][0] == x]

        for ve in v_edges:
            src, des = ve
            for ve_other in v_edges:
                if ve == ve_other:
                    continue

                xcoords = [p[1][0] for p in pos.items()]
                xscale = np.max(xcoords) - np.min(xcoords)
                src_other, des_other = ve_other

                if src_other == src:
                    pos[des_other][0] += _node_offset(pos[des_other][0], xscale)
                    break
                elif des_other == des:
                    pos[src_other][0] += _node_offset(pos[src_other][0], xscale)
                    break
                elif src_other > src and des_other < src:
                    pos[des_other][0] += _node_offset(pos[des_other][0], xscale)
                    break
                elif src_other > des and des_other < des:
                    pos[des_other][0] += _node_offset(pos[des_other][0], xscale)
                    break

            v_edges = [e for e in g.edges() if pos[e[0]][0] == x and pos[e[1]][0] == x]


def _get_node_labels(g: nx.DiGraph, pos: dict) -> dict:
    """Build a node-label dict, omitting labels for nodes that are too close together.

    Parameters
    ----------
    g:
        The directed graph.
    pos:
        Position dict ``{node: [x, y]}``.

    Returns
    -------
    dict
        ``{node: energy_value}`` for nodes whose labels are far enough apart.
    """
    node_label_min_frac = 0.05
    node_labels = dict(g.nodes(data="energy"))

    xcoords = [p[1][0] for p in pos.items()]
    ycoords = [p[1][1] for p in pos.items()]
    xextent = np.max(xcoords) - np.min(xcoords) or 1
    yextent = np.max(ycoords) - np.min(ycoords) or 1

    for n1 in g.nodes():
        for n2 in g.nodes():
            dx = (pos[n1][0] - pos[n2][0]) / xextent
            dy = (pos[n1][1] - pos[n2][1]) / yextent
            dist = np.sqrt(dx * dx + dy * dy)
            if n1 > n2 and dist < node_label_min_frac and n2 in node_labels:
                node_labels.pop(n2)

    return node_labels


def _get_edge_labels(g: nx.DiGraph, pos: dict) -> dict:
    """Build an edge-label dict with branching ratio values for long edges.

    Edges shorter than a threshold fraction of the layout extent are
    unlabelled to avoid clutter.  Edges with weight == 1.0 (no branching)
    are also omitted.

    Parameters
    ----------
    g:
        The directed graph.
    pos:
        Position dict ``{node: [x, y]}``.

    Returns
    -------
    dict
        ``{(u, v): "weight_string"}`` for edges that should be labelled.
    """
    edge_label_min_frac = 0.15
    xcoords = [p[1][0] for p in pos.items()]
    ycoords = [p[1][1] for p in pos.items()]
    xextent = np.max(xcoords) - np.min(xcoords) or 1
    yextent = np.max(ycoords) - np.min(ycoords) or 1

    edge_labels: dict = {}
    for n1 in g.nodes():
        for n2 in g.nodes():
            dx = (pos[n1][0] - pos[n2][0]) / xextent
            dy = (pos[n1][1] - pos[n2][1]) / yextent
            dist = np.sqrt(dx * dx + dy * dy)
            if g.has_edge(n1, n2) and dist >= edge_label_min_frac:
                edge_labels[(n1, n2)] = f'{g[n1][n2]["weight"]:.2f}'

    # Remove weight labels where branching ratio is exactly 1.0
    for u, v, w in g.edges(data="weight"):
        if w == 1.0 and (u, v) in edge_labels:
            edge_labels.pop((u, v))

    return edge_labels


def _draw_level_symbols(g: nx.DiGraph, pos: dict) -> None:
    """Draw short horizontal tick marks at each nuclear level position.

    Parameters
    ----------
    g:
        The level scheme directed graph.
    pos:
        Position dict ``{node: [x, y]}``.
    """
    nl = g.number_of_nodes()
    hline_frac = 1.0 / nl * 1.5
    xcoords = [p[1][0] for p in pos.items()]
    xextent = np.max(xcoords) - np.min(xcoords)
    hline_dist = hline_frac * xextent

    for n in g.nodes():
        enj = g.nodes(data="energy")[n]
        xlj = pos[n][0] - hline_dist / 2
        xhj = pos[n][0] + hline_dist / 2
        plt.hlines(y=enj, xmin=xlj, xmax=xhj, color="black", zorder=3)


# ------------------------------------------------------------------ #
#  Public drawing functions
# ------------------------------------------------------------------ #

def draw_level_scheme(
    ls,
    ax: plt.Axes,
    color_map: list[str] | None = None,
    branch: bool = False,
) -> None:
    """Draw a LevelScheme graph on a matplotlib Axes.

    Nodes are positioned according to their excitation energy on the
    y-axis.  Horizontal positions are assigned by topological layer and
    then adjusted to reduce visual clutter.

    Parameters
    ----------
    ls:
        A :class:`~NuclearObjects.LevelScheme` instance.
    ax:
        Matplotlib Axes to draw on.
    color_map:
        List of edge colours passed to NetworkX.  Defaults to
        ``["black"]``.
    branch:
        If True, annotate edges with their branching ratio weights.
    """
    if color_map is None:
        color_map = ["black"]

    g = ls.g
    emax = ls.emax
    energy_bin_frac = 1 / 10

    for i, layer in enumerate(nx.topological_generations(g)):
        for n in layer:
            g.nodes[n]["layer"] = i

    layer_size = emax * energy_bin_frac
    for n, en in g.nodes(data="energy"):
        g.nodes[n]["energy_layer"] = math.ceil(en / layer_size)

    pos = nx.multipartite_layout(g, subset_key="layer", align="horizontal")
    for k in pos:
        pos[k][1] *= -1

    # Override y-position with true excitation energy
    for k in pos:
        pos[k][1] = g.nodes[k]["energy"]

    _align_single_edges(g, pos)
    _align_close_levels(g, pos)
    _avoid_vertical_skips(g, pos)
    n_labels = _get_node_labels(g, pos)
    e_labels = _get_edge_labels(g, pos)

    n_sizes = [len(str(en)) * 50 for _, en in g.nodes(data="energy")]

    nx.draw_networkx(g, pos=pos, with_labels=False, node_shape="_", node_size=n_sizes)
    nx.draw_networkx_nodes(g, ax=ax, pos=pos, node_shape="", node_size=n_sizes, alpha=0.4)
    nx.draw_networkx_edges(
        g, ax=ax, pos=pos, node_size=n_sizes,
        nodelist=list(g.nodes()), edgelist=list(g.edges()), edge_color=color_map,
    )
    nx.draw_networkx_labels(
        g, pos, ax=ax, labels=n_labels,
        verticalalignment="bottom", horizontalalignment="right",
        bbox={"color": "white", "alpha": 0.5},
    )
    _draw_level_symbols(g, pos)

    if branch:
        nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=e_labels, label_pos=0.8)


def draw_transition_scheme(ts, ax: plt.Axes, branch: bool = False) -> None:
    """Draw a TransitionScheme graph on a matplotlib Axes.

    Nodes represent individual gamma-ray transitions; edges connect
    transitions that occur sequentially in a cascade.

    Parameters
    ----------
    ts:
        A :class:`~NuclearObjects.TransitionScheme` instance.
    ax:
        Matplotlib Axes to draw on.
    branch:
        If True, annotate edges with their weights.
    """
    g = ts.g

    for i, layer in enumerate(nx.topological_generations(g)):
        for n in layer:
            g.nodes[n]["layer"] = i

    pos = nx.multipartite_layout(g, subset_key="layer", align="horizontal")
    for k in pos:
        pos[k][1] *= -1

    _align_single_edges(g, pos)
    _avoid_vertical_skips(g, pos)
    e_labels = _get_edge_labels(g, pos)

    nx.draw_networkx_nodes(g, ax=ax, pos=pos, node_shape="s", alpha=0.4)
    nx.draw_networkx_edges(
        g, ax=ax, pos=pos,
        nodelist=list(g.nodes()), edgelist=list(g.edges()),
    )

    if branch:
        nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=e_labels, label_pos=0.8)
