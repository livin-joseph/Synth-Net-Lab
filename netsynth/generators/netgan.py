"""NetGAN (Bojchevski et al., ICML 2018) in PyTorch.

NetGAN does not model the adjacency matrix directly.  It trains a
Wasserstein GAN (with gradient penalty) whose *generator* is an LSTM that
emits random walks over the node set, and whose *critic* tries to tell these
from real random walks on the observed graph.  After training:

1. sample many walks from the generator,
2. count how often each node pair is traversed -> a symmetric *score matrix*,
3. turn the score matrix into a graph with the same number of edges as the
   original (each node first gets one edge, the rest is filled by sampling
   proportionally to the scores).

This is a compact re-implementation that follows the original architecture
(LSTM generator with Gumbel-softmax straight-through, LSTM critic, WGAN-GP,
Adam) but with smaller defaults so it is usable on a CPU.  The original is
TensorFlow 1.x and needs a GPU for anything but tiny graphs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from netsynth.generators.base import GeneratorResult, ProgressFn, graph_from_edges


@dataclass
class NetGANConfig:
    rw_len: int = 16              # nodes per walk (T)
    batch_size: int = 64
    noise_dim: int = 16
    gen_hidden: int = 40
    disc_hidden: int = 30
    emb_dim: int = 64             # W_down size (original: 128)
    n_critic: int = 3
    lr: float = 3e-4
    gp_lambda: float = 10.0
    temperature: float = 0.5
    iters: int = 600
    max_seconds: Optional[float] = None
    eval_every: int = 50
    n_gen_walks: Optional[int] = None
    seed: int = 0


# --------------------------------------------------------------------------- walks
class WalkSampler:
    """Vectorised uniform random walks on a graph (CSR)."""

    def __init__(self, G: nx.Graph, rw_len: int):
        A = nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), format="csr", weight=None)
        self.indptr, self.indices = A.indptr.astype(np.int64), A.indices.astype(np.int64)
        self.deg = np.diff(self.indptr)
        self.valid = np.flatnonzero(self.deg > 0)
        self.n, self.T = G.number_of_nodes(), rw_len
        self.A = A.tocsr()

    def walks(self, B: int, rng: np.random.Generator) -> np.ndarray:
        cur = rng.choice(self.valid, size=B)
        out = np.empty((B, self.T), dtype=np.int64)
        out[:, 0] = cur
        for t in range(1, self.T):
            d = self.deg[cur]
            off = (rng.random(B) * d).astype(np.int64)
            cur = self.indices[self.indptr[cur] + off]
            out[:, t] = cur
        return out


# --------------------------------------------------------------------------- nets
class Generator(nn.Module):
    def __init__(self, N: int, c: NetGANConfig):
        super().__init__()
        self.N, self.c = N, c
        self.h0 = nn.Sequential(nn.Linear(c.noise_dim, c.gen_hidden), nn.Tanh())
        self.c0 = nn.Sequential(nn.Linear(c.noise_dim, c.gen_hidden), nn.Tanh())
        self.W_down = nn.Linear(N, c.emb_dim, bias=False)
        self.cell = nn.LSTMCell(c.emb_dim, c.gen_hidden)
        self.W_up = nn.Linear(c.gen_hidden, N)

    def forward(self, B: int, T: int, hard: bool = True, return_idx: bool = False):
        dev = self.W_up.weight.device
        z = torch.randn(B, self.c.noise_dim, device=dev)
        h, cs = self.h0(z), self.c0(z)
        x = torch.zeros(B, self.c.emb_dim, device=dev)
        outs, idxs = [], []
        for _ in range(T):
            h, cs = self.cell(x, (h, cs))
            logits = self.W_up(h)
            u = torch.rand_like(logits).clamp(1e-10, 1.0 - 1e-10)
            g = -torch.log(-torch.log(u))                                 # Gumbel(0, 1) noise
            y_soft = F.softmax((logits + g) / self.c.temperature, dim=-1)
            idx = y_soft.argmax(-1)
            y = F.one_hot(idx, self.N).float()
            y = y + y_soft - y_soft.detach() if hard else y_soft      # straight-through
            x = self.W_down(y)
            outs.append(y)
            idxs.append(idx)
        walks = torch.stack(outs, 1)
        return (walks, torch.stack(idxs, 1)) if return_idx else walks


class Critic(nn.Module):
    def __init__(self, N: int, c: NetGANConfig):
        super().__init__()
        self.W_down = nn.Linear(N, c.emb_dim, bias=False)
        self.lstm = nn.LSTM(c.emb_dim, c.disc_hidden, batch_first=True)
        self.out = nn.Linear(c.disc_hidden, 1)

    def forward(self, walks):                    # (B, T, N) one-hot / soft
        h, _ = self.lstm(self.W_down(walks))
        return self.out(h[:, -1])


# --------------------------------------------------------------------------- training
class NetGAN:
    def __init__(self, cfg: NetGANConfig | None = None, device: str | None = None):
        self.cfg = cfg or NetGANConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.history: dict = {"iter": [], "critic_loss": [], "gen_loss": [], "val_iter": [], "walk_validity": []}
        self.train_seconds = 0.0

    def fit(self, G: nx.Graph, progress: ProgressFn = None) -> "NetGAN":
        c = self.cfg
        torch.manual_seed(c.seed)
        rng = np.random.default_rng(c.seed)
        self.G, self.N = G, G.number_of_nodes()
        self.m = G.number_of_edges()
        self.sampler = WalkSampler(G, c.rw_len)
        N = self.N
        self.gen, self.critic = Generator(N, c).to(self.device), Critic(N, c).to(self.device)
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=c.lr, betas=(0.5, 0.9))
        opt_d = torch.optim.Adam(self.critic.parameters(), lr=c.lr, betas=(0.5, 0.9))
        t0 = time.perf_counter()
        for it in range(c.iters):
            for _ in range(c.n_critic):
                real_idx = torch.from_numpy(self.sampler.walks(c.batch_size, rng)).to(self.device)
                real = F.one_hot(real_idx, N).float()
                with torch.no_grad():
                    fake = self.gen(c.batch_size, c.rw_len)
                eps = torch.rand(c.batch_size, 1, 1, device=self.device)
                xhat = (eps * real + (1 - eps) * fake).requires_grad_(True)
                with torch.backends.cudnn.flags(enabled=False):     # double-backward through LSTM
                    d_hat = self.critic(xhat)
                    grad = torch.autograd.grad(d_hat.sum(), xhat, create_graph=True)[0]
                gp = ((grad.flatten(1).norm(dim=1) - 1.0) ** 2).mean()
                d_loss = self.critic(fake).mean() - self.critic(real).mean() + c.gp_lambda * gp
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            fake = self.gen(c.batch_size, c.rw_len)
            g_loss = -self.critic(fake).mean()
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            self.history["iter"].append(it)
            self.history["critic_loss"].append(float(d_loss.detach()))
            self.history["gen_loss"].append(float(g_loss.detach()))
            if (it + 1) % c.eval_every == 0 or it == c.iters - 1:
                self.history["val_iter"].append(it + 1)
                self.history["walk_validity"].append(self.walk_validity())
            elapsed = time.perf_counter() - t0
            if progress and (it % 5 == 0):
                progress(f"NetGAN iter {it + 1}/{c.iters}", min(0.9, (it + 1) / c.iters))
            if c.max_seconds and elapsed > c.max_seconds:
                self.history["stopped_early_at"] = it + 1
                if not self.history["val_iter"] or self.history["val_iter"][-1] != it + 1:
                    self.history["val_iter"].append(it + 1)
                    self.history["walk_validity"].append(self.walk_validity())
                break
        self.train_seconds = time.perf_counter() - t0
        return self

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def walk_validity(self, n_walks: int = 256) -> float:
        """Fraction of generated steps that follow a real edge (1.0 = perfect walks)."""
        _, idx = self.gen(n_walks, self.cfg.rw_len, return_idx=True)
        idx = idx.cpu().numpy()
        u, v = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        return float(np.asarray(self.sampler.A[u, v]).ravel().mean())

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def score_matrix(self, n_walks: Optional[int] = None, batch: int = 512, progress: ProgressFn = None) -> sp.csr_matrix:
        c = self.cfg
        n_walks = n_walks or c.n_gen_walks or int(min(max(20 * self.m, 20_000), 200_000))
        S = sp.csr_matrix((self.N, self.N), dtype=np.float32)
        done = 0
        while done < n_walks:
            b = min(batch, n_walks - done)
            _, idx = self.gen(b, c.rw_len, return_idx=True)
            idx = idx.cpu().numpy()
            u, v = idx[:, :-1].ravel(), idx[:, 1:].ravel()
            S = S + sp.coo_matrix((np.ones(len(u), dtype=np.float32), (u, v)), shape=(self.N, self.N)).tocsr()
            done += b
            if progress: progress("NetGAN sampling walks", 0.9 + 0.1 * done / n_walks)
        S = S + S.T
        S.setdiag(0); S.eliminate_zeros()
        return S.tocsr()

    def sample_graphs(self, S: sp.csr_matrix, k: int, seed: int = 0) -> list:
        return [graph_from_scores(S, self.m, np.random.default_rng([seed, b])) for b in range(k)]


def graph_from_scores(S: sp.csr_matrix, m: int, rng: np.random.Generator) -> nx.Graph:
    """Score matrix -> simple graph with (at most) ``m`` edges, as in the NetGAN paper."""
    N = S.shape[0]
    S = S.tocsr()
    chosen = set()
    for i in range(N):                                   # step 1: one edge per node with any score
        a, b = S.indptr[i], S.indptr[i + 1]
        if b > a:
            w = S.data[a:b].astype(np.float64)
            j = int(S.indices[a + rng.choice(b - a, p=w / w.sum())])
            chosen.add((min(i, j), max(i, j)))
    U = sp.triu(S, k=1).tocoo()
    keys = U.row.astype(np.int64) * N + U.col
    first = np.fromiter((u * N + v for u, v in chosen), dtype=np.int64, count=len(chosen))
    mask = ~np.isin(keys, first)
    need = max(0, m - len(chosen))
    if need > 0 and mask.any():
        cand = np.flatnonzero(mask)
        w = U.data[cand].astype(np.float64)
        take = min(need, len(cand))
        pick = rng.choice(len(cand), size=take, replace=False, p=w / w.sum())
        extra = keys[cand[pick]]
        all_keys = np.concatenate([first, extra])
    else:
        all_keys = first
    edges = np.stack([all_keys // N, all_keys % N], axis=1)
    return graph_from_edges(N, edges)


# --------------------------------------------------------------------------- interface
def generate_netgan(G: nx.Graph, k: int, seed: int = 0, progress: ProgressFn = None,
                    netgan_iters: int = 600, netgan_rw_len: int = 16, netgan_batch: int = 64,
                    netgan_max_seconds: Optional[float] = 300, netgan_walks: Optional[int] = None,
                    **_) -> GeneratorResult:
    cfg = NetGANConfig(iters=netgan_iters, rw_len=netgan_rw_len, batch_size=netgan_batch,
                       max_seconds=netgan_max_seconds, n_gen_walks=netgan_walks, seed=seed)
    gan = NetGAN(cfg).fit(G, progress=progress)
    t0 = time.perf_counter()
    S = gan.score_matrix(progress=progress)
    graphs = gan.sample_graphs(S, k, seed=seed)
    dt = (time.perf_counter() - t0) / k
    n = G.number_of_nodes()
    return GeneratorResult(
        "netgan", "NetGAN", graphs, [np.arange(n)] * k, fit_seconds=gan.train_seconds, gen_seconds=dt,
        info={"history": gan.history, "iters_run": len(gan.history["iter"]), "device": str(gan.device),
              "final_walk_validity": gan.history["walk_validity"][-1] if gan.history["walk_validity"] else None},
    )
