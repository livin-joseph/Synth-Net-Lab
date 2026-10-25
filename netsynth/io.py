"""Loading user networks (many formats) and preparing them for the generators."""
from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp

from netsynth.sampling import snowball_sample

SUPPORTED = "edge list (.txt/.csv/.tsv/.edges/.el), adjacency matrix (.csv/.npy/.npz), GraphML, GML, GEXF, Pajek (.net), MatrixMarket (.mtx); any of them may be .gz"
_SPLIT = re.compile(r"[,\t ;]+")
_HEADER_WORDS = {"source", "target", "src", "dst", "from", "to", "node1", "node2", "u", "v", "id1", "id2",
                 "fromnodeid", "tonodeid", "node_1", "node_2", "i", "j", "a", "b", "weight", "w"}


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _is_num(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _tokenize(text: str) -> list:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#%/":
            continue
        rows.append([t for t in _SPLIT.split(line) if t != ""] if ("," in line or ";" in line or "\t" in line)
                    else line.split())
    return rows


def _looks_like_adjacency(rows: list) -> bool:
    L = len(rows[0]) if rows else 0
    if L < 3 or any(len(r) != L for r in rows):
        return False
    if len(rows) == L + 1 and not all(_is_num(t) for t in rows[0]):   # header row
        rows = rows[1:]
    return len(rows) == L and all(_is_num(t) for r in rows for t in r)


def _read_text_graph(text: str, fmt: str) -> tuple:
    rows = _tokenize(text)
    if not rows:
        raise ValueError("The file contains no data rows.")
    notes = []
    if fmt == "adjacency" or (fmt == "auto" and _looks_like_adjacency(rows)):
        if len(rows) == len(rows[0]) + 1:
            rows = rows[1:]
        A = np.array([[float(t) for t in r] for r in rows])
        notes.append(f"Read a {A.shape[0]}×{A.shape[1]} adjacency matrix (non-zero = edge).")
        return nx.from_numpy_array((A != 0).astype(int)), notes
    # edge list: optional header
    first = rows[0]
    if len(first) >= 2 and (first[0].lower() in _HEADER_WORDS or first[1].lower() in _HEADER_WORDS
                            or (not _is_num(first[0]) and len(rows) > 1 and _is_num(rows[1][0]))):
        rows = rows[1:]
        notes.append("Skipped a header row.")
    if any(len(r) < 2 for r in rows):
        raise ValueError("Edge list rows need at least two columns (source, target).")
    G = nx.Graph()
    numeric = all(_is_num(r[0]) and _is_num(r[1]) for r in rows)
    for r in rows:
        u, v = (int(float(r[0])), int(float(r[1]))) if numeric else (r[0], r[1])
        G.add_edge(u, v)
    return G, notes


def load_graph(raw: bytes, filename: str, fmt: str = "auto") -> tuple:
    """Parse bytes into a networkx graph. Returns (graph, notes).

    fmt: "auto" | "edgelist" | "adjacency"  (only affects delimited text files)
    """
    name = filename.lower()
    notes: list = []
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
        name = name[:-3]
    ext = Path(name).suffix
    if ext == ".npy":
        A = np.load(io.BytesIO(raw), allow_pickle=False)
        if A.ndim != 2:
            raise ValueError("A .npy file must hold a 2-D adjacency matrix.")
        if A.shape[0] == A.shape[1]:
            G = nx.from_numpy_array((A != 0).astype(int))
        elif A.shape[1] >= 2:                               # (m, 2) edge array
            G = nx.Graph(); G.add_edges_from(map(tuple, A[:, :2].astype(int).tolist()))
        else:
            raise ValueError("Unsupported .npy shape; expected (n, n) or (m, 2).")
        return G, notes
    if ext == ".npz":
        S = sp.load_npz(io.BytesIO(raw))
        return nx.from_scipy_sparse_array((S != 0).astype(int)), notes
    if ext == ".mtx":
        from scipy.io import mmread
        S = sp.csr_matrix(mmread(io.BytesIO(raw)))
        return nx.from_scipy_sparse_array((S != 0).astype(int)), notes
    if ext == ".graphml":
        return nx.read_graphml(io.BytesIO(raw)), notes
    if ext == ".gml":
        try:
            return nx.read_gml(io.BytesIO(raw)), notes
        except nx.NetworkXError:
            return nx.read_gml(io.BytesIO(raw), label="id"), notes
    if ext == ".gexf":
        return nx.read_gexf(io.BytesIO(raw)), notes
    if ext == ".net":
        return nx.read_pajek(io.BytesIO(raw)), notes
    return _read_text_graph(_decode(raw), fmt)


def prepare_graph(G_raw: nx.Graph, take_lcc: bool = True, max_nodes: int | None = None, seed: int = 0) -> tuple:
    """Make the graph simple, undirected, optionally LCC-only and size-capped; relabel to 0..n-1.

    Returns (graph, report) where report["notes"] lists everything that was changed.
    """
    notes = []
    n0, m0 = G_raw.number_of_nodes(), G_raw.number_of_edges()
    directed = G_raw.is_directed()
    G = nx.Graph()
    G.add_nodes_from(G_raw.nodes())
    G.add_edges_from((u, v) for u, v in G_raw.edges() if u != v)
    loops = sum(1 for u, v in G_raw.edges() if u == v)
    if directed:
        notes.append("Directed edges were treated as undirected.")
    if loops:
        notes.append(f"Removed {loops} self-loop(s).")
    multi = (m0 - loops) - G.number_of_edges() if not directed else 0
    if multi > 0:
        notes.append(f"Merged {multi} parallel edge(s).")
    if take_lcc and G.number_of_nodes():
        comp = max(nx.connected_components(G), key=len)
        if len(comp) < G.number_of_nodes():
            notes.append(f"Kept the largest connected component: {len(comp)} of {G.number_of_nodes()} nodes.")
            G = G.subgraph(comp).copy()
    if max_nodes and G.number_of_nodes() > max_nodes:
        notes.append(f"Sub-sampled {G.number_of_nodes()} → {max_nodes} nodes by snowball (BFS) sampling "
                     f"(the SyNGLER fit stores dense n×n matrices).")
        G = snowball_sample(G, max_nodes, seed=seed)
    G = nx.convert_node_labels_to_integers(G)
    if G.number_of_nodes() < 10 or G.number_of_edges() < 10:
        raise ValueError(f"After cleaning only {G.number_of_nodes()} nodes / {G.number_of_edges()} edges are left; "
                         "need at least 10 of each.")
    return G, {"raw_nodes": n0, "raw_edges": m0, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
               "notes": notes}


def sample_networks() -> dict:
    """Bundled example networks: name -> (path or None for networkx built-ins, description)."""
    root = Path(__file__).resolve().parent.parent / "data" / "samples"
    out = {
        "Zachary karate club (34 nodes)": ("nx:karate", "Classic social network; tiny, good for a first test."),
        "Les Misérables co-appearance (77 nodes)": ("nx:lesmis", "Character co-occurrence network."),
    }
    for f, label, desc in [
        ("polblogs.edgelist.gz", "Political blogs (1,222 nodes)", "US political blogs, largest component; a SyNGLER paper dataset."),
        ("dblp.edgelist.gz", "DBLP co-authorship subset (1,481 nodes)", "Co-authorship subset shipped with the SyNGLER repo."),
        ("youtube.edgelist.gz", "YouTube social subset (1,991 nodes)", "Social-network subset shipped with the SyNGLER repo."),
    ]:
        if (root / f).exists():
            out[label] = (str(root / f), desc)
    return out


def load_sample(ref: str) -> nx.Graph:
    if ref == "nx:karate":
        return nx.karate_club_graph()
    if ref == "nx:lesmis":
        return nx.les_miserables_graph()
    G, _ = load_graph(Path(ref).read_bytes(), Path(ref).name)
    return G
