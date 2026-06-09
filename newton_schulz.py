"""
Newton-Schulz Iterations for Matrix Functions

A collection of Newton-Schulz (NS) variants for computing matrix functions
using only matrix multiplications. All methods support batched inputs for
GPU-efficient computation.

Each variant converges quadratically given proper initialization, and all
operations are pure matmuls — making them ideal for GPU execution compared
to eigendecomposition, SVD, Cholesky, or triangular solves.

Key advantage for optimizers: when the target matrix changes incrementally
(e.g. a running preconditioner), the previous result serves as a warm start,
and 1-2 iterations per step suffice to track the moving target.

Variants included:
  - inverse:              A^{-1}
  - inverse_sqrt:         A^{-1/2}   (A must be PSD)
  - sqrt:                 A^{1/2}    (A must be PSD)
  - inverse_fourth_root:  A^{-1/4}   (A must be PSD)
  - orthogonalize:        polar factor U where A = US
  - sign:                 sign(A) matrix sign function
"""

import torch
from torch import Tensor


# =====================================================================
# Matrix Inversion: A^{-1}
# =====================================================================

def inverse(
    A: Tensor,
    num_iters: int = 10,
    eps: float = 1e-7,
    warm_start: Tensor | None = None,
    warm_start_tol: float | None = 0.5,
) -> Tensor:
    """
    Newton-Schulz iteration for matrix inversion.

        X_{k+1} = X_k @ (2I - A @ X_k)

    Converges to A^{-1} when X_0 is chosen such that ||I - A X_0|| < 1.

    Args:
        A: (..., n, n) batch of square matrices
        num_iters: number of iterations
        eps: regularization added to A for stability
        warm_start: previous A^{-1} estimate for incremental tracking
        warm_start_tol: maximum relative residual for accepting a warm start;
            set to None to trust warm_start unconditionally

    Returns:
        Approximate A^{-1}, same shape as A
    """
    *batch, n, _ = A.shape
    I = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)

    A_work = A + eps * I

    if warm_start is not None:
        if warm_start_tol is not None:
            X = _screen_warm_start_inverse_power(
                A,
                warm_start,
                alpha=1.0,
                eps=eps,
                tol=float(warm_start_tol),
            )
        else:
            X = warm_start
    else:
        # X_0 = alpha * I, alpha = 1 / ||A||_1 per batch element
        # Using trace as cheap norm estimate for PSD, or col-sum norm otherwise
        col_norms = A_work.abs().sum(dim=-2).max(dim=-1).values  # (...,)
        alpha = 1.0 / (col_norms + eps)
        X = alpha[..., None, None] * I

    two_I = 2.0 * I

    for _ in range(num_iters):
        X = X @ (two_I - A_work @ X)

    return X


# =====================================================================
# Inverse Square Root: A^{-1/2}  (A must be PSD)
# =====================================================================

def inverse_sqrt(
    A: Tensor,
    num_iters: int = 10,
    eps: float = 1e-7,
    warm_start: Tensor | None = None,
    warm_start_tol: float | None = 0.5,
) -> Tensor:
    """
    Newton-Schulz iteration for the inverse square root of a PSD matrix.

        X_{k+1} = 0.5 * X_k @ (3I - X_k @ A @ X_k)

    Converges to A^{-1/2}.

    Args:
        A: (..., n, n) batch of PSD matrices
        num_iters: number of iterations
        eps: regularization for stability
        warm_start: previous A^{-1/2} estimate
        warm_start_tol: maximum relative residual for accepting a warm start;
            set to None to trust warm_start unconditionally

    Returns:
        Approximate A^{-1/2}, same shape as A
    """
    *batch, n, _ = A.shape
    I = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)

    A_work = 0.5 * (A + A.transpose(-1, -2)) + eps * I

    if warm_start is not None:
        if warm_start_tol is not None:
            X = _screen_warm_start_inverse_power(
                A,
                warm_start,
                alpha=0.5,
                eps=eps,
                tol=float(warm_start_tol),
            )
        else:
            X = warm_start
    else:
        # X_0 = ||A||^{-1/2} I keeps lambda * x_0^2 <= 1 and is
        # much less conservative than 1 / trace for large matrices.
        col_norms = A_work.abs().sum(dim=-2).max(dim=-1).values
        alpha = torch.rsqrt(col_norms.clamp_min(eps))
        X = alpha[..., None, None] * I

    three_I = 3.0 * I

    for _ in range(num_iters):
        XAX = X @ A_work @ X
        X = 0.5 * X @ (three_I - XAX)
        X = 0.5 * (X + X.transpose(-1, -2))

    return X


# =====================================================================
# Square Root: A^{1/2}  (A must be PSD)
# =====================================================================

def sqrt(
    A: Tensor,
    num_iters: int = 10,
    eps: float = 1e-7,
    warm_start: tuple[Tensor | None, Tensor | None] | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Coupled Newton-Schulz (Denman-Beavers) iteration for the matrix square root.

        Y_{k+1} = 0.5 * Y_k @ (3I - Z_k @ Y_k)
        Z_{k+1} = 0.5 * (3I - Z_k @ Y_k) @ Z_k

    Y converges to A^{1/2}, Z converges to A^{-1/2}.
    Returns both since you often want the inverse sqrt too.

    Args:
        A: (..., n, n) batch of PSD matrices
        num_iters: number of iterations
        eps: regularization for stability
        warm_start: tuple of (previous A^{1/2}, previous A^{-1/2})

    Returns:
        (A^{1/2}, A^{-1/2}) tuple, each same shape as A
    """
    *batch, n, _ = A.shape
    I = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)

    A = A + eps * I

    if warm_start is not None and warm_start[0] is not None:
        Y = warm_start[0]
    else:
        Y = A.clone()

    if warm_start is not None and warm_start[1] is not None:
        Z = warm_start[1]
    else:
        # Z_0 = alpha * I, alpha ~ 1/||A||
        traces = A.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        alpha = 1.0 / (traces + eps)
        Z = alpha[..., None, None] * I

    three_I = 3.0 * I

    for _ in range(num_iters):
        ZY = Z @ Y
        T = three_I - ZY
        Y = 0.5 * Y @ T
        Z = 0.5 * T @ Z

    return Y, Z


# =====================================================================
# Inverse Fourth Root: A^{-1/4}  (A must be PSD)
# =====================================================================

def inverse_fourth_root(
    A: Tensor,
    num_iters_outer: int = 10,
    num_iters_inner: int = 10,
    eps: float = 1e-7,
    warm_start: Tensor | tuple[Tensor, Tensor] | None = None,
    warm_start_tol: float | None = 0.25,
) -> Tensor:
    """Newton iteration for A^{-1/4}.

    Scalar/eigenvalue update:
        x_{k+1} = 1/4 * x_k * (5 - a * x_k^4)

    Args:
        A: (..., n, n) batch of PSD matrices
        num_iters_outer: iterations for the inverse-fourth-root iteration
        num_iters_inner: ignored; kept for call-site compatibility
        eps: regularization for stability
        warm_start: optional previous A^{-1/4}. Tuple form is accepted for
            compatibility with older callers and uses the second entry.
        warm_start_tol: maximum relative residual for accepting a warm start;
            set to None to trust warm_start unconditionally

    Returns:
        Approximate A^{-1/4}, same shape as A
    """
    *batch, n, _ = A.shape
    I = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    A_work = 0.5 * (A + A.transpose(-1, -2)) + eps * I

    if isinstance(warm_start, tuple):
        warm_start = warm_start[1]

    if warm_start is not None:
        if warm_start_tol is not None:
            X = _screen_warm_start_inverse_power(
                A,
                warm_start,
                alpha=0.25,
                eps=eps,
                tol=float(warm_start_tol),
            )
        else:
            X = 0.5 * (warm_start + warm_start.transpose(-1, -2))
    else:
        col_norms = A_work.abs().sum(dim=-2).max(dim=-1).values
        init = torch.rsqrt(torch.sqrt(col_norms.clamp_min(eps)))
        X = init[..., None, None] * I

    five_I = 5.0 * I
    for _ in range(num_iters_outer):
        X2 = X @ X
        X4 = X2 @ X2
        X = 0.25 * X @ (five_I - A_work @ X4)
        X = 0.5 * (X + X.transpose(-1, -2))

    return X


def inverse_three_fourths_root(
    A: Tensor,
    num_iters_outer: int = 10,
    num_iters_inner: int = 10,
    eps: float = 1e-7,
) -> Tensor:
    """A^{-3/4} = A^{-1/2} A^{-1/4} for PSD A."""
    A_inv_sqrt = inverse_sqrt(A, num_iters=num_iters_outer, eps=eps)
    A_inv_fourth = inverse_fourth_root(
        A,
        num_iters_outer=num_iters_outer,
        num_iters_inner=num_iters_inner,
        eps=eps,
    )
    out = A_inv_sqrt @ A_inv_fourth
    return 0.5 * (out + out.transpose(-1, -2))


def _cold_start_inverse_power(A: Tensor, alpha: float, eps: float) -> Tensor:
    """Return the scalar-identity cold start used by the NS inverse-power solvers."""
    *_, n, _ = A.shape
    I = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)

    if alpha == 1.0:
        A_work = A + eps * I
        col_norms = A_work.abs().sum(dim=-2).max(dim=-1).values
        init = 1.0 / (col_norms + eps)
    elif alpha == 0.5:
        A_work = 0.5 * (A + A.transpose(-1, -2)) + eps * I
        col_norms = A_work.abs().sum(dim=-2).max(dim=-1).values
        init = torch.rsqrt(col_norms.clamp_min(eps))
    elif alpha == 0.25:
        A_work = 0.5 * (A + A.transpose(-1, -2)) + eps * I
        col_norms = A_work.abs().sum(dim=-2).max(dim=-1).values
        init = torch.rsqrt(torch.sqrt(col_norms.clamp_min(eps)))
    else:
        raise ValueError("alpha must be one of 0.25, 0.5, or 1.0")

    return init[..., None, None] * I


def _warm_start_residual(A: Tensor, X: Tensor, alpha: float, eps: float) -> Tensor:
    """Relative Frobenius residual for the inverse-power fixed point."""
    *_, n, _ = A.shape
    I = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)

    if alpha == 1.0:
        A_work = A + eps * I
        residual = A_work @ X - I
    elif alpha == 0.5:
        A_work = 0.5 * (A + A.transpose(-1, -2)) + eps * I
        residual = X @ A_work @ X - I
    elif alpha == 0.25:
        A_work = 0.5 * (A + A.transpose(-1, -2)) + eps * I
        X2 = X @ X
        X4 = X2 @ X2
        residual = A_work @ X4 - I
    else:
        raise ValueError("alpha must be one of 0.25, 0.5, or 1.0")

    return residual.norm(dim=(-2, -1)) / I.norm(dim=(-2, -1)).clamp_min(eps)


def _screen_warm_start_inverse_power(
    A: Tensor,
    warm_start: Tensor,
    alpha: float,
    eps: float,
    tol: float,
) -> Tensor:
    """Use warm-start elements only when they are finite and near the fixed point."""
    if alpha in (0.25, 0.5):
        warm_start = 0.5 * (warm_start + warm_start.transpose(-1, -2))

    cold_start = _cold_start_inverse_power(A, alpha, eps)
    residual = _warm_start_residual(A, warm_start, alpha, eps)
    finite = warm_start.isfinite().flatten(start_dim=-2).all(dim=-1)
    safe = finite & residual.isfinite() & (residual <= tol)
    return torch.where(safe[..., None, None], warm_start, cold_start)


def inverse_power_quarter(
    A: Tensor,
    alpha: float,
    num_iters: int = 10,
    eps: float = 1e-7,
    warm_start: Tensor | None = None,
    warm_start_tol: float | None = 0.5,
) -> Tensor:
    """Compute PSD A^{-alpha} for alpha in {0.25, 0.5, 1.0}.

    The covariance is trace-normalized before Newton-Schulz and then rescaled
    back by scale^{-alpha}. This keeps the iteration in a friendlier numerical
    range without changing the intended matrix power. If provided, warm_start
    should be an unnormalized previous estimate of A^{-alpha}. Warm starts are
    screened against the scaled covariance; unsafe batch elements fall back to
    the same cold initialization used by the underlying Newton-Schulz solver.
    """
    alpha = float(alpha)
    scale = (
        A.diagonal(dim1=-2, dim2=-1)
        .mean(dim=-1)
        .clamp_min(eps)
    )
    A_scaled = A / scale[..., None, None]
    warm_scaled = None
    if warm_start is not None:
        warm_scaled = warm_start * scale.pow(alpha)[..., None, None]
        if warm_start_tol is not None:
            tol = float(warm_start_tol)
            if alpha == 0.25:
                tol = min(tol, 0.25)
            warm_scaled = _screen_warm_start_inverse_power(
                A_scaled,
                warm_scaled,
                alpha,
                eps,
                tol,
            )

    if alpha == 0.25:
        out = inverse_fourth_root(
            A_scaled,
            num_iters_outer=num_iters,
            num_iters_inner=num_iters,
            eps=eps,
            warm_start=warm_scaled,
            warm_start_tol=None,
        )
    elif alpha == 0.5:
        out = inverse_sqrt(
            A_scaled,
            num_iters=num_iters,
            eps=eps,
            warm_start=warm_scaled,
            warm_start_tol=None,
        )
    elif alpha == 1.0:
        out = inverse(
            A_scaled,
            num_iters=num_iters,
            eps=eps,
            warm_start=warm_scaled,
            warm_start_tol=None,
        )
    else:
        raise ValueError("alpha must be one of 0.25, 0.5, or 1.0")
    return out / scale.pow(alpha)[..., None, None]


# =====================================================================
# Polar Decomposition / Orthogonalization
# =====================================================================

def orthogonalize(
    A: Tensor,
    num_iters: int = 5,
) -> Tensor:
    """
    Newton-Schulz iteration for the polar decomposition.

    For A = U @ S (polar decomposition), this converges to U,
    the closest orthogonal matrix to A.

        X_{k+1} = 0.5 * X_k @ (3I - X_k^T @ X_k)

    This is what Muon uses to orthogonalize the gradient.

    Args:
        A: (..., m, n) batch of matrices with m >= n
        num_iters: number of iterations (5-10 typically suffices)

    Returns:
        Approximate orthogonal factor U, shape (..., m, n) with U^T U ≈ I
    """
    *batch, m, n = A.shape
    assert m >= n, f"Requires m >= n, got {m} x {n}. Transpose first if needed."

    I = torch.eye(n, device=A.device, dtype=A.dtype)
    # expand to batch dims
    for _ in range(len(batch)):
        I = I.unsqueeze(0)
    I = I.expand(*batch, n, n)

    # Normalize by Frobenius norm for convergence
    norms = A.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    X = A / norms

    three_I = 3.0 * I

    for _ in range(num_iters):
        XtX = X.transpose(-2, -1) @ X  # (..., n, n)
        X = 0.5 * X @ (three_I - XtX)

    return X


def orthogonalize_quintic(
    A: Tensor,
    num_iters: int = 3,
) -> Tensor:
    """
    Higher-order (quintic) Newton-Schulz for polar decomposition.

    Uses the 5th-order variant from Nakamura et al. which converges faster
    per iteration at the cost of more matmuls per step:

        X_{k+1} = (1/16) * X_k @ (15I - B @ (10I - 3B))
        where B = X_k^T @ X_k

    Fewer iterations needed (typically 2-3 vs 5-10 for cubic).
    This is the variant used in the latest Muon implementations.

    Args:
        A: (..., m, n) batch of matrices with m >= n
        num_iters: number of iterations (2-3 typically suffices)

    Returns:
        Approximate orthogonal factor U
    """
    *batch, m, n = A.shape
    assert m >= n, f"Requires m >= n, got {m} x {n}. Transpose first if needed."

    I = torch.eye(n, device=A.device, dtype=A.dtype)
    for _ in range(len(batch)):
        I = I.unsqueeze(0)
    I = I.expand(*batch, n, n)

    norms = A.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    X = A / norms

    for _ in range(num_iters):
        B = X.transpose(-2, -1) @ X
        X = (1.0 / 16.0) * X @ (15.0 * I - B @ (10.0 * I - 3.0 * B))

    return X


# =====================================================================
# Matrix Sign Function: sign(A)
# =====================================================================

def sign(
    A: Tensor,
    num_iters: int = 15,
    eps: float = 1e-7,
    warm_start: Tensor | None = None,
) -> Tensor:
    """Regularized matrix sign for symmetric inputs.

    Uses the identity sign(A) = A @ (A^2)^{-1/2}, with the inverse square
    root computed by Newton-Schulz. The eps regularization means eigenvalues
    near zero are smoothly shrunk rather than mapped sharply to +/-1.

    Args:
        A: (..., n, n) batch of symmetric square matrices
        num_iters: number of inverse-square-root iterations
        eps: spectral regularization for A^2
        warm_start: accepted for API compatibility; ignored because it is a
            sign(A) estimate, not an (A^2)^{-1/2} estimate

    Returns:
        Approximate sign(A), same shape as A
    """
    del warm_start
    A_sym = 0.5 * (A + A.transpose(-1, -2))
    inv_abs = inverse_sqrt(A_sym @ A_sym, num_iters=num_iters, eps=eps)
    out = A_sym @ inv_abs
    return 0.5 * (out + out.transpose(-1, -2))


# =====================================================================
# Triangular Inverse (for PSGD-style triangular preconditioners)
# =====================================================================

def triangular_inverse(
    T: Tensor,
    num_iters: int = 10,
    eps: float = 1e-7,
    warm_start: Tensor | None = None,
) -> Tensor:
    """
    Newton-Schulz inversion specialized for triangular matrices.

    Uses the standard NS inversion iteration but with an initialization
    that exploits triangular structure: X_0 = diag(1/T_ii), which is
    already a decent approximation for diagonally dominant triangular matrices.

    This can replace torch.linalg.solve_triangular in PSGD's
    _calc_A_and_conjB where it computes X @ inv(Q).

    Args:
        T: (..., n, n) batch of (upper or lower) triangular matrices
        num_iters: number of iterations
        eps: regularization
        warm_start: previous T^{-1} estimate

    Returns:
        Approximate T^{-1}, same shape as T
    """
    *batch, n, _ = T.shape
    I = torch.eye(n, device=T.device, dtype=T.dtype).expand_as(T)

    T = T + eps * I

    if warm_start is not None:
        X = warm_start
    else:
        # Initialize with inverse of diagonal — much better than scalar * I
        # for triangular matrices since diag(T)^{-1} is already the right
        # structure and captures the dominant terms
        diag = T.diagonal(dim1=-2, dim2=-1).clamp(min=eps)  # (..., n)
        X = torch.zeros_like(T)
        X.diagonal(dim1=-2, dim2=-1).copy_(1.0 / diag)

    two_I = 2.0 * I

    for _ in range(num_iters):
        X = X @ (two_I - T @ X)

    return X


# =====================================================================
# Warm-startable wrappers for optimizer integration
# =====================================================================

class WarmNS:
    """
    Stateful wrapper that maintains the previous iterate for warm-starting.

    Usage in an optimizer:
        # In __init__ or state setup:
        state['ns_inv_sqrt'] = WarmNS(method='inverse_sqrt', num_iters=10)

        # In step():
        C_inv_sqrt = state['ns_inv_sqrt'](C, num_iters=2)  # warm: only 2 iters needed

    The first call uses the full num_iters from init. Subsequent calls
    use the (typically smaller) num_iters passed at call time, warm-starting
    from the previous result.
    """

    METHODS = {
        'inverse': inverse,
        'inverse_sqrt': inverse_sqrt,
        'inverse_fourth_root': inverse_fourth_root,
        'inverse_three_fourths_root': inverse_three_fourths_root,
        'orthogonalize': orthogonalize,
        'orthogonalize_quintic': orthogonalize_quintic,
        'triangular_inverse': triangular_inverse,
    }

    def __init__(
        self,
        method: str = 'inverse_sqrt',
        num_iters_cold: int = 10,
        eps: float = 1e-7,
    ):
        """
        Args:
            method: which NS variant to use
            num_iters_cold: iterations for the first (cold) call
            eps: regularization
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}', choose from {list(self.METHODS.keys())}")
        self.method = method
        self.fn = self.METHODS[method]
        self.num_iters_cold = num_iters_cold
        self.eps = eps
        self.prev: Tensor | None = None

    def __call__(self, A: Tensor, num_iters: int = 2) -> Tensor:
        """
        Compute the matrix function, warm-starting from previous result if available.

        Args:
            A: input matrix (batched ok)
            num_iters: iterations for this (warm) call

        Returns:
            Result of the NS iteration
        """
        if self.method in ('orthogonalize', 'orthogonalize_quintic'):
            # Orthogonalization doesn't use warm_start / eps the same way
            iters = num_iters if self.prev is not None else self.num_iters_cold
            result = self.fn(A, num_iters=iters)
        elif self.method == 'inverse_fourth_root':
            iters = num_iters if self.prev is not None else self.num_iters_cold
            result = self.fn(
                A,
                num_iters_outer=iters,
                num_iters_inner=iters,
                eps=self.eps,
                warm_start=self.prev,
            )
        elif self.method == 'inverse_three_fourths_root':
            # This composite helper does not use the single-tensor warm_start
            # protocol. It is still useful through WarmNS for a consistent
            # cold/warm iteration-count schedule.
            iters = num_iters if self.prev is not None else self.num_iters_cold
            result = self.fn(A, num_iters_outer=iters, num_iters_inner=iters, eps=self.eps)
        else:
            if self.prev is not None:
                result = self.fn(A, num_iters=num_iters, eps=self.eps, warm_start=self.prev)
            else:
                result = self.fn(A, num_iters=self.num_iters_cold, eps=self.eps)

        self.prev = result.detach().clone()
        return result

    def reset(self):
        """Clear the warm start state."""
        self.prev = None


# =====================================================================
# Convenience: batched versions with explicit batch dim
# =====================================================================

def batch_inverse(A: Tensor, **kwargs) -> Tensor:
    """Alias: A has shape (batch, n, n)."""
    return inverse(A, **kwargs)

def batch_inverse_sqrt(A: Tensor, **kwargs) -> Tensor:
    """Alias: A has shape (batch, n, n)."""
    return inverse_sqrt(A, **kwargs)

def batch_sqrt(A: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
    """Alias: A has shape (batch, n, n)."""
    return sqrt(A, **kwargs)

def batch_inverse_fourth_root(A: Tensor, **kwargs) -> Tensor:
    """Alias: A has shape (batch, n, n)."""
    return inverse_fourth_root(A, **kwargs)

def batch_orthogonalize(A: Tensor, **kwargs) -> Tensor:
    """Alias: A has shape (batch, m, n)."""
    return orthogonalize(A, **kwargs)
