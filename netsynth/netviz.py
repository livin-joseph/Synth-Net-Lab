"""Interactive network views (pyvis): node colouring, snowball sampling for large graphs, copied-edge highlighting."""
from __future__ import annotations

import random

import EoN
import matplotlib
import networkx as nx
import numpy as np
import plotly.graph_objects as go

from netsynth.sampling import snowball_nodes

COLOR_MODES = ["Degree", "Local clustering", "Community", "k-core"]
COPY_COLOR = "#e34948"      # edge that also exists in the real network
EDGE_COLOR = "#b5b8be"


def viz_subgraph(G: nx.Graph, max_nodes: int, seed: int = 0, nodes: list | None = None) -> tuple:
    """(subgraph with ORIGINAL node labels, was_sampled). Snowball sample above ``max_nodes``."""
    if nodes is not None:
        return G.subgraph(nodes).copy(), True
    if G.number_of_nodes() <= max_nodes:
        return G.copy(), False
    return G.subgraph(snowball_nodes(G, max_nodes, seed)).copy(), True


def real_edge_set_fn(real: nx.Graph, node_map):
    """Return f(u, v) -> True if the synthetic edge maps onto an edge of the real graph."""
    if node_map is None:
        return None
    nm = np.asarray(node_map)
    def f(u, v):
        a, b = int(nm[u]), int(nm[v])
        return a != b and real.has_edge(a, b)
    return f


def _hex(cmap, x):
    return matplotlib.colors.to_hex(matplotlib.colormaps[cmap](float(np.clip(x, 0, 1))))


def node_colors(G: nx.Graph, mode: str) -> tuple:
    """(dict node -> hex colour, legend text)."""
    nodes = list(G.nodes())
    if mode == "Degree":
        v = np.array([G.degree(u) for u in nodes], float)
        cm, lab = "viridis", f"degree {int(v.min())}–{int(v.max())}"
    elif mode == "Local clustering":
        c = nx.clustering(G)
        v = np.array([c[u] for u in nodes], float)
        cm, lab = "magma", f"clustering {v.min():.2f}–{v.max():.2f}"
    elif mode == "k-core":
        c = nx.core_number(G) if G.number_of_edges() else {u: 0 for u in nodes}
        v = np.array([c[u] for u in nodes], float)
        cm, lab = "plasma", f"k-core {int(v.min())}–{int(v.max())}"
    else:
        comms = nx.community.greedy_modularity_communities(G) if G.number_of_edges() else [set(nodes)]
        pal = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6", "#eda100", "#e34948", "#008300"]
        col = {}
        for i, cset in enumerate(sorted(comms, key=len, reverse=True)):
            for u in cset:
                col[u] = pal[i] if i < len(pal) else "#9aa0a6"
        return col, f"{len(comms)} communities (greedy modularity; top 8 coloured)"
    lo, hi = v.min(), v.max()
    z = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return {u: _hex(cm, zi) for u, zi in zip(nodes, z)}, lab


def network_html(G: nx.Graph, color_by: str = "Degree", is_copy=None, height: int = 520,
                 freeze: bool = True, show_labels: bool = False, seed: int = 0) -> tuple:
    """Self-contained pyvis HTML. Returns (html, legend, n_copied_edges)."""
    from pyvis.network import Network

    colors, legend = node_colors(G, color_by)
    deg = dict(G.degree())
    dmax = max(deg.values()) if deg else 1
    clu = nx.clustering(G)
    net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", font_color="#222222",
                  cdn_resources="in_line", notebook=False)
    for u in G.nodes():
        net.add_node(int(u), label=str(u) if show_labels else " ", size=5 + 16 * np.sqrt(deg[u] / max(dmax, 1)),
                     color=colors[u], borderWidth=1,
                     title=f"node {u}<br>degree {deg[u]}<br>clustering {clu[u]:.2f}")
    n_copy = 0
    for u, v in G.edges():
        copied = bool(is_copy(u, v)) if is_copy else False
        n_copy += copied
        net.add_edge(int(u), int(v), color=COPY_COLOR if copied else EDGE_COLOR,
                     width=2.0 if copied else 0.6, title="also a real edge" if copied else "")
    net.set_options("""{
      "nodes": {"font": {"size": 12}},
      "edges": {"smooth": false, "selectionWidth": 2},
      "interaction": {"hover": true, "tooltipDelay": 80, "navigationButtons": false, "keyboard": false},
      "physics": {"solver": "barnesHut",
                  "barnesHut": {"gravitationalConstant": -4500, "centralGravity": 0.6, "springLength": 70, "springConstant": 0.03, "damping": 0.35},
                  "stabilization": {"enabled": true, "iterations": 220, "updateInterval": 25, "fit": true}}
    }""")
    html = net.generate_html(notebook=False)
    hook = "network.once('stabilizationIterationsDone', function () { network.fit({animation: false});"
    hook += " network.setOptions({physics: false});" if freeze else ""
    hook += " });"
    # the page may be rendered inside a hidden tab (zero size): re-fit whenever the frame gets resized
    hook += """
    (function () { var t = null; function refit() { clearTimeout(t); t = setTimeout(function () {
        if (window.innerWidth > 50) { network.redraw(); network.fit({animation: false}); } }, 150); }
      window.addEventListener('resize', refit);
      if (window.ResizeObserver) new ResizeObserver(refit).observe(document.body); })();"""
    tail = "</script>\n</body>"
    html = html.replace(tail, "\n" + hook + "\n" + tail) if tail in html else html.replace("</body>", "<script>" + hook + "</script></body>")
    return html, legend, n_copy
