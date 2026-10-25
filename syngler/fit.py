"""Fast, covariate-free LSM fitting for SyNGLER (stage 1).

The reference code in :mod:`syngler.lsm.source` implements the projected
gradient descent (PGD) fit of the latent space model

    Theta_ij = Z_i^T Z_j + alpha_i + alpha_j + rho,     P_ij = sigmoid(Theta_ij)

in a fully general form (covariates ``X``, ``beta``, bias-correction helpers).
For *generation* we never use covariates, so this module re-implements exactly
the same update rules in a lean form:

* no (n, n, p) covariate tensor and no ``einsum`` over it,
* float32 dense math with in-place updates (about 8-10x faster at n ~ 1k),
* a *relative* stopping rule (the original absolute ``eps=1e-5`` on a loss that
  is in the thousands practically never triggers),
* no per-iteration ``print`` (the original prints every step),
* a progress callback so a UI can show what is going on.

The recipe follows ``run_cora_syngr.fit_lsm`` from the SyNGLER repo:

1. spectral initialisation of ``G = Z Z^T`` and ``alpha = log(deg + 1)`` (centred),
2. Phase 1  - PGD on the unconstrained Gram matrix ``G`` (then top-r eigen-
   decomposition gives ``Z``),
3. Phase 2  - PGD refinement directly on ``Z`` and ``alpha``,
4. calibrate the global intercept ``rho`` so that the expected density of the
   LSM equals the observed density.

``tests/test_syngler_fit.py`` checks numerically that Phase 1 reproduces the
reference ``Model.PGD_initialization`` trajectory.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from syngler.lsm.source import center_and_rotate

ProgressFn = Optional[Callable[[str, int, int, float], None]]


@dataclass
class LSMFit:
    """Result of fitting the latent space model to one observed graph."""

    Z: np.ndarray                      # (n, r) latent positions
    alpha: np.ndarray                  # (n,) node intercepts (degree heterogeneity), centred
    rho: float                         # global sparsity intercept
    r: int
    n: int
    density: float                     # observed density
    loss_init: list = field(default_factory=list)
    loss_refine: list = field(default_factory=list)
    converged_init: bool = False
    converged_refine: bool = False
    seconds: float = 0.0

    def expected_density(self) -> float:
        return float(_mean_edge_prob(self.Z @ self.Z.T + self.alpha[:, None] + self.alpha[None, :], self.rho))


def _topk_sqrt_eig(M: np.ndarray, k: int) -> np.ndarray:
    """Top-k eigenpairs of a symmetric matrix -> U sqrt(D); negatives clipped to 0."""
    vals, vecs = np.linalg.eigh(M.astype(np.float64))
    vals, vecs = vals[-k:], vecs[:, -k:]
    return vecs * np.sqrt(np.clip(vals, 0.0, None))


def _prob_matrix(G: np.ndarray, alpha: np.ndarray, rho: float, out: np.ndarray) -> np.ndarray:
    """out <- sigmoid(G + alpha_i + alpha_j + rho) with zero diagonal (in place)."""
    np.add(G, alpha[:, None], out=out)
    out += alpha[None, :]
    out += rho
    # expit has no ``out`` for float32 on every scipy version -> assign
    out[:] = expit(out)
    np.fill_diagonal(out, 0.0)
    return out


def _mean_edge_prob(Theta_no_rho: np.ndarray, rho: float) -> float:
    n = Theta_no_rho.shape[0]
    P = expit(Theta_no_rho + rho)
    np.fill_diagonal(P, 0.0)
    return float(P.sum() / (n * (n - 1)))


def _calibrate_rho(Theta_no_rho: np.ndarray, density: float) -> float:
    """Find rho so that the mean of sigmoid(Theta + rho) over off-diagonal pairs = density."""
    if density <= 0.0:
        return -20.0
    if density >= 1.0:
        return 20.0
    f = lambda rho: _mean_edge_prob(Theta_no_rho, rho) - density
    try:
        return float(brentq(f, -25.0, 25.0, xtol=1e-6))
    except ValueError:
        return float(np.log(density / (1.0 - density)))


def fit_lsm(
    A: np.ndarray,
    r: int = 6,
    eta0: float = 0.5,
    n_iter_init: int = 600,
    n_iter_refine: int = 300,
    rtol: float = 1e-6,
    sigma_init: float = 0.01,
    seed: int = 0,
    dtype=np.float32,
    progress: ProgressFn = None,
) -> LSMFit:
    """Fit the (covariate-free) LSM to a symmetric 0/1 adjacency matrix ``A``.

    Parameters
    ----------
    r : latent dimension (paper default for real data: 6; clipped to n-1).
    eta0 : base step size, as in the reference runner (eta_alpha = eta0 / 2n, ...).
    n_iter_init, n_iter_refine : maximum PGD iterations of Phase 1 / Phase 2.
    rtol : stop a phase once the relative change of the loss is below ``rtol``.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    A = np.asarray(A)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    r = int(max(1, min(r, n - 1)))
    A = np.ascontiguousarray(A, dtype=dtype)
    np.fill_diagonal(A, 0.0)

    m = float(np.triu(A, 1).sum())
    density = m / (n * (n - 1) / 2.0)
    deg = A.sum(1).astype(np.float64)

    # ---- initialisation (identical in spirit to run_cora_syngr.fit_lsm) -------------
    Z0 = center_and_rotate(_topk_sqrt_eig(A.astype(np.float64) - A.mean(), r))
    Z0 = Z0 + sigma_init * rng.standard_normal(Z0.shape)
    alpha = np.log(deg + 1.0)
    alpha = (alpha - alpha.mean()) + sigma_init * rng.standard_normal(n)
    alpha = alpha.astype(dtype)
    rho0 = float(np.log(density / (1 - density))) if 0 < density < 1 else 0.0

    eta_alpha = eta0 / (2.0 * n)
    eta_Z = eta0 / (2.0 * max(float(np.sum(Z0 ** 2) / r), 1.0))

    G = (Z0 @ Z0.T).astype(dtype)
    P = np.empty_like(G)
    D = np.empty_like(G)
    eye_scale = dtype(1e-3)

    # ---- Phase 1: PGD on the Gram matrix G ----------------------------------------
    loss_init, conv_init, loss_old = [], False, 0.0
    for it in range(n_iter_init):
        _prob_matrix(G, alpha, rho0, P)
        np.subtract(A, P, out=D)                       # diff = A - P  (zero diag)
        loss = float(np.einsum("ij,ij->", D, D))
        # G <- J (G + 2 eta_Z (D - 1e-3 G - 1e-3 I)) J
        G *= dtype(1.0 - 2.0 * eta_Z * 1e-3)
        G += dtype(2.0 * eta_Z) * D
        G[np.diag_indices(n)] -= dtype(2.0 * eta_Z) * eye_scale
        G -= G.mean(axis=0, keepdims=True)
        G -= G.mean(axis=1, keepdims=True)             # double centring (J G J)
        alpha = alpha + dtype(2.0 * eta_alpha) * D.sum(axis=1)
        loss_init.append(loss)
        if progress and (it % 10 == 0):
            progress("init", it, n_iter_init, loss)
        if it > 2 and abs(loss - loss_old) <= rtol * max(loss_old, 1.0):
            conv_init = True
            break
        loss_old = loss

    Z = center_and_rotate(_topk_sqrt_eig(G, r))

    # ---- Phase 2: PGD on (Z, alpha) -------------------------------------------------
    loss_ref, conv_ref = [], False
    if n_iter_refine > 0:
        eta_Z2 = eta0 / (2.0 * max(float(np.sum(Z ** 2) / r), 1.0))
        eta_alpha2 = eta0 / (2.0 * n)
        Zf = Z.astype(dtype)
        loss_old = 0.0
        for it in range(n_iter_refine):
            np.matmul(Zf, Zf.T, out=G)
            _prob_matrix(G, alpha, rho0, P)
            np.subtract(A, P, out=D)
            loss = float(np.einsum("ij,ij->", D, D))
            Zn = Zf.astype(np.float64) + 2.0 * eta_Z2 * (D @ Zf).astype(np.float64)
            Zf = center_and_rotate(center_and_rotate(Zn)).astype(dtype)
            alpha = alpha + dtype(2.0 * eta_alpha2) * D.sum(axis=1)
            loss_ref.append(loss)
            if progress and (it % 10 == 0):
                progress("refine", it, n_iter_refine, loss)
            if it > 2 and abs(loss - loss_old) <= rtol * max(loss_old, 1.0):
                conv_ref = True
                break
            loss_old = loss
        Z = Zf.astype(np.float64)

    alpha = alpha.astype(np.float64)
    alpha -= alpha.mean()                              # absorbed into rho below

    # ---- rho calibration to the observed density -------------------------------------
    Theta = Z @ Z.T + alpha[:, None] + alpha[None, :]
    rho = _calibrate_rho(Theta, density)

    return LSMFit(
        Z=Z, alpha=alpha, rho=rho, r=r, n=n, density=density,
        loss_init=loss_init, loss_refine=loss_ref,
        converged_init=conv_init, converged_refine=conv_ref,
        seconds=time.perf_counter() - t0,
    )
