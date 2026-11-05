# ------------------------------------------------------------
# kl_gpu.py
# ------------------------------------------------------------
"""
GPU‑accelerated KL divergence between two multivariate Gaussians.

Public API
~~~~~~~~~~
    kl_div_gpu(mu1, mu2, S1, S2, *, rank: int = 100,
               n_iter: int = 6, chunk_size: int = 250_000) -> float

Interpretation of the inputs
-----------------------------
* mu1, mu2 : 1‑D arrays (length d) – the mean vectors.
* S1, S2   : either
    – 1‑D array (length d) → diagonal covariance Σ = diag(S).
    – 2‑D array (d, d)     → dense covariance.
      In this case the routine will **separate the diagonal from the
      off‑diagonal part**, keep the diagonal exactly, and approximate
      the off‑diagonal part by a low‑rank matrix of size (d, rank).

* rank
    – -1 → exact dense KL (requires the full matrices to fit in memory/GPU).
    – 0  → treat the covariances as purely diagonal (U = ∅).
    – >0 → low‑rank + diagonal approximation (the low‑rank part never
          contains diagonal information, so you never have to supply a
          scalar ρ).

* n_iter – power‑iteration steps used to build the low‑rank basis
           (4–6 are enough for most realistic spectra).

* chunk_size – number of rows processed at once when evaluating the
                trace term; adjust if you run out of GPU memory.

The function always returns a **non‑negative** KL divergence.
"""

from __future__ import annotations
import warnings
from typing import Tuple

import torch


# ----------------------------------------------------------------------
# 1️⃣  Helper – log‑det for diagonal + low‑rank (GPU)
# ----------------------------------------------------------------------
def _logdet_diag_lowrank(D: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """log det(D + U Uᵀ) = Σ log D_i + log det(I + Uᵀ D⁻¹ U)."""
    invD = 1.0 / D
    M = torch.eye(U.shape[1], device=D.device, dtype=D.dtype) + U.t() @ (
        invD[:, None] * U
    )
    sign, logdet_M = torch.linalg.slogdet(M)
    if not torch.all(sign > 0):
        raise RuntimeError("Low‑rank correction not positive‑definite.")
    return torch.log(D).sum() + logdet_M


# ----------------------------------------------------------------------
# 2️⃣  Exact KL for dense matrices (rank == -1)
# ----------------------------------------------------------------------
def _exact_kl(
    mu1: torch.Tensor, mu2: torch.Tensor, S1: torch.Tensor, S2: torch.Tensor
) -> float:
    """Exact KL when both covariances are dense (d×d)."""
    d = mu1.shape[0]

    # Cholesky of denominator (S2)
    L2 = torch.linalg.cholesky(S2)  # lower‑triangular
    invS2_mu = torch.cholesky_solve((mu2 - mu1).unsqueeze(-1), L2)
    quad = torch.squeeze((mu2 - mu1).unsqueeze(0) @ invS2_mu)

    # Trace term  tr(S2⁻¹ S1)
    X = torch.cholesky_solve(S1, L2)  # S₂⁻¹ S₁
    trace = torch.trace(X)

    # Log‑determinants
    logdet2 = 2.0 * torch.log(torch.diagonal(L2)).sum()
    L1 = torch.linalg.cholesky(S1)
    logdet1 = 2.0 * torch.log(torch.diagonal(L1)).sum()

    kld = 0.5 * (trace + quad - d + (logdet2 - logdet1))
    if kld < -1e-8:
        warnings.warn("Negative KL encountered (numeric). Returning absolute value.")
        kld = -kld
    return float(kld.item())


# ----------------------------------------------------------------------
# 3️⃣  Low‑rank + diagonal compression that *preserves* the diagonal
# ----------------------------------------------------------------------
def _compress_preserve_diag(
    S: torch.Tensor, rank: int, n_iter: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate a symmetric PSD matrix S as
        S ≈ D + U Uᵀ,
    where D = diag(S) (exact) and U contains only the off‑diagonal structure.
    The algorithm never mixes the diagonal into the low‑rank subspace.
    """
    d = S.shape[0]
    device = S.device
    dtype = S.dtype

    # ------------------------------------------------------------
    # Step 0 – extract the diagonal (exact) and build the residual
    # ------------------------------------------------------------
    D = torch.diagonal(S)  # (d,)

    # Residual with zero diagonal:  R = S - diag(D)
    # We never materialise R fully; we only need a *matrix‑vector* product.
    def mv_residual(x: torch.Tensor) -> torch.Tensor:
        """Compute (S - diag(D)) @ x without forming the dense residual."""
        # (S @ x) - D * x   ;  D is broadcast as (d,1)
        return S @ x - D[:, None] * x

    # ------------------------------------------------------------
    # Step 1 – block power iteration on the residual
    # ------------------------------------------------------------
    torch.manual_seed(0)
    Q = torch.randn(d, rank, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(Q, mode="reduced")

    for _ in range(n_iter):
        Z = mv_residual(Q)  # (d, rank)
        Q, _ = torch.linalg.qr(Z, mode="reduced")

    # ------------------------------------------------------------
    # Step 2 – small Rayleigh matrix B = Qᵀ R Q   (rank × rank)
    # ------------------------------------------------------------
    B = Q.t() @ mv_residual(Q)  # (rank, rank)
    evals, evecs = torch.linalg.eigh(B)  # ascending
    evals = evals.flip(0)  # descending
    evecs = evecs.flip(1)
    evals = torch.clamp(evals, min=0.0)  # guard negative due to round‑off
    sqrt_evals = torch.sqrt(evals)

    # ------------------------------------------------------------
    # Step 3 – low‑rank factor U = Q @ (evecs * sqrt(evals))
    # ------------------------------------------------------------
    U = Q @ (evecs * sqrt_evals)  # (d, rank)

    # ------------------------------------------------------------
    # Step 4 – diagonal correction (remove any tiny diagonal leakage)
    # ------------------------------------------------------------
    diag_UU = torch.sum(U * U, dim=1)  # (d,)
    D_corr = D - diag_UU
    D_corr = torch.clamp(D_corr, min=0.0)  # safety
    return D_corr, U


# ----------------------------------------------------------------------
# 4️⃣  KL for diagonal + low‑rank (GPU)
# ----------------------------------------------------------------------
def _kl_diag_lowrank(
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    D1: torch.Tensor,
    U1: torch.Tensor,
    D2: torch.Tensor,
    U2: torch.Tensor,
    chunk_size: int,
) -> float:
    """
    Compute KL(N(mu1, D1+U1U1ᵀ)‖N(mu2, D2+U2U2ᵀ)).
    All tensors must live on the same device (CUDA or CPU) and share dtype.
    """
    d = mu1.shape[0]
    r2 = U2.shape[1]  # may be 0

    # ------------------------------------------------------------
    # Woodbury pieces for the denominator Σ₂
    # ------------------------------------------------------------
    invD2 = 1.0 / D2
    A = U2 * invD2[:, None]  # A = D₂⁻¹ U₂   (d × r₂)

    M2 = torch.eye(r2, device=mu1.device, dtype=mu1.dtype) + U2.t() @ A
    L2 = torch.linalg.cholesky(M2)  # M₂ = L₂ L₂ᵀ
    invM2 = torch.cholesky_inverse(L2)  # (r₂, r₂)

    # ------------------------------------------------------------
    # Quadratic term (Δᵀ Σ₂⁻¹ Δ)
    # ------------------------------------------------------------
    delta = mu2 - mu1
    quad_diag = (delta * delta * invD2).sum()

    v = U2.t() @ (invD2 * delta)  # (r₂,)
    y = torch.linalg.solve_triangular(L2, v.unsqueeze(-1), upper=False)
    w = torch.linalg.solve_triangular(L2.t(), y, upper=True).squeeze(-1)
    quad_low = torch.dot(v, w)

    quad_term = quad_diag - quad_low

    # ------------------------------------------------------------
    # Trace term tr( Σ₂⁻¹ Σ₁ )
    # ------------------------------------------------------------
    # 1) diagonal of Σ₂⁻¹  (chunked)
    diag_S2inv = torch.empty(d, device=mu1.device, dtype=mu1.dtype)

    for start in range(0, d, chunk_size):
        stop = min(start + chunk_size, d)
        Ac = A[start:stop]  # (c, r₂)
        Bc = Ac @ invM2  # (c, r₂)
        diag_corr = torch.einsum("ij,ij->i", Ac, Bc)
        diag_S2inv[start:stop] = invD2[start:stop] - diag_corr

    trace_diag = torch.dot(D1, diag_S2inv)

    # 2) tr(U₁ᵀ D₂⁻¹ U₁)
    row_norm_sq_U1 = torch.einsum("ij,ij->i", U1, U1)
    trace_U1 = torch.dot(invD2, row_norm_sq_U1)

    # 3) low‑rank correction
    C = A.t() @ U1  # (r₂, r₁)
    Corr_mat = C @ C.t()  # (r₂, r₂)
    trace_corr = torch.trace(invM2 @ Corr_mat)

    trace_term = trace_diag + trace_U1 - trace_corr

    # ------------------------------------------------------------
    # Log‑determinants
    # ------------------------------------------------------------
    logdet2 = _logdet_diag_lowrank(D2, U2)
    logdet1 = _logdet_diag_lowrank(D1, U1)

    # ------------------------------------------------------------
    # Assemble KL
    # ------------------------------------------------------------
    kld = 0.5 * (trace_term + quad_term - d + (logdet2 - logdet1))
    if kld < -1e-8:
        warnings.warn("Negative KL encountered (numeric). Returning absolute value.")
        kld = -kld
    return float(kld.item())


# ----------------------------------------------------------------------
# 5️⃣  Public wrapper – the function you will import
# ----------------------------------------------------------------------
def kl_div_gpu(
    mu1,
    mu2,
    S1,
    S2,
    *,
    rank: int = 100,
    n_iter: int = 6,
    chunk_size: int = 250_000,
    device: str | None = None,
) -> float:
    """
    KL( N(mu1, Σ1) ‖ N(mu2, Σ2) )  – GPU accelerated.

    Parameters
    ----------
    mu1, mu2 : array‑like, shape (d,)
    S1, S2   : array‑like
               – 1‑D → interpreted as a diagonal covariance (Σ = diag(S)).
               – 2‑D → dense covariance; the routine will split it into
                       diagonal + low‑rank off‑diagonal automatically.
    rank    : int (default 100)
               – -1 → exact dense KL (requires the full matrices to fit).
               – 0  → pure diagonal KL (ignore any off‑diagonal entries).
               – >0 → low‑rank + diagonal approximation.  The diagonal is kept
                     exact; the off‑diagonal part is approximated by `rank`
                     basis vectors obtained via a block power iteration (`n_iter`
                     steps).  No explicit scalar ρ is required.
    n_iter  : int (default 6) – power‑iteration steps for the low‑rank basis.
    chunk_size : int (default 250_000) – rows processed at a time when forming
                 the trace term (tune if you hit GPU‑memory limits).

    Returns
    -------
    float – non‑negative KL divergence.
    """
    # ------------------------------------------------------------
    # Move everything to the appropriate device (GPU if available)
    # ------------------------------------------------------------

    # check if the user specified a device; if not, auto‑detect
    if device is not None:
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    mu1_t = torch.as_tensor(mu1, dtype=torch.float32, device=device)
    mu2_t = torch.as_tensor(mu2, dtype=torch.float32, device=device)
    S1_t = torch.as_tensor(S1, dtype=torch.float32, device=device)
    S2_t = torch.as_tensor(S2, dtype=torch.float32, device=device)

    # ------------------------------------------------------------
    # Exact dense KL (rank == -1)
    # ------------------------------------------------------------
    if rank == -1:
        if S1_t.ndim != 2 or S2_t.ndim != 2:
            raise ValueError("Exact KL (rank=-1) needs full (d,d) covariance matrices.")
        return _exact_kl(mu1_t, mu2_t, S1_t, S2_t)

    # ------------------------------------------------------------
    # Pure diagonal case – either input is 1‑D or we discard off‑diagonal
    # ------------------------------------------------------------
    if S1_t.ndim == 1 and S2_t.ndim == 1:
        # Both are already diagonal; low‑rank part is empty regardless of `rank`.
        D1 = S1_t
        D2 = S2_t
        U1 = torch.empty((D1.shape[0], 0), device=device, dtype=D1.dtype)
        U2 = torch.empty((D2.shape[0], 0), device=device, dtype=D2.dtype)
        return _kl_diag_lowrank(mu1_t, mu2_t, D1, U1, D2, U2, chunk_size)

    # ------------------------------------------------------------
    # At this point we have dense matrices (2‑D).  We now split each matrix
    # into its exact diagonal and a low‑rank approximation of the *off‑diagonal*
    # part.  This never mixes diagonal entries into the low‑rank factor,
    # therefore no user‑supplied scalar ρ is needed.
    # ------------------------------------------------------------
    if rank == 0:
        # Only keep the diagonal, ignore everything else.
        D1 = torch.diagonal(S1_t)
        D2 = torch.diagonal(S2_t)
        U1 = torch.empty((D1.shape[0], 0), device=device, dtype=D1.dtype)
        U2 = torch.empty((D2.shape[0], 0), device=device, dtype=D2.dtype)
        return _kl_diag_lowrank(mu1_t, mu2_t, D1, U1, D2, U2, chunk_size)

    # General low‑rank case (rank > 0)
    D1, U1 = _compress_preserve_diag(S1_t, rank, n_iter)
    D2, U2 = _compress_preserve_diag(S2_t, rank, n_iter)

    return _kl_diag_lowrank(mu1_t, mu2_t, D1, U1, D2, U2, chunk_size)


if __name__ == "__main__":
    import numpy as np

    # Small d so we can keep dense matrices on the GPU
    d = 5000
    rng = np.random.default_rng(1)

    # Synthetic diagonal+rank‑1 covariances (the same model you used before)
    var1 = rng.uniform(0.5, 2.0, size=d)
    var2 = rng.uniform(0.5, 2.0, size=d)
    rho1, rho2 = 0.9, 0.8

    # Build the full matrices (CPU → later moved to GPU)
    S1 = np.diag(var1) + rho1 * (np.outer(np.sqrt(var1), np.sqrt(var1)) - np.diag(var1))
    S2 = np.diag(var2) + rho2 * (np.outer(np.sqrt(var2), np.sqrt(var2)) - np.diag(var2))

    mu1 = np.zeros(d)
    mu2 = np.zeros(d)

    # ---- exact KL (GPU) ---------------------------------
    kl_exact = kl_div_gpu(mu1, mu2, S1, S2, rank=-1)
    print(f"Exact KL (GPU) = {kl_exact:.6f}")

    # ---- low‑rank KL (rank=30) -------------------------
    kl_approx = kl_div_gpu(mu1, mu2, S1, S2, rank=100, n_iter=6)
    print(f"Low‑rank KL (rank=30) = {kl_approx:.6f}")

    print(f"Relative error = {abs(kl_exact - kl_approx) / kl_exact:.4%}")
