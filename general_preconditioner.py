from __future__ import annotations

import math
from typing import Optional

import torch

from newton_schulz import inverse_power_quarter


def _resolve_compute_dtype(dtype: Optional[str | torch.dtype]) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key in ("none", "param", "default"):
        return None
    if key in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half", "torch.float16"):
        return torch.float16
    if key in ("fp32", "float32", "torch.float32"):
        return torch.float32
    raise ValueError(f"Unsupported compute dtype: {dtype}")


def _merge_dims(shape: tuple[int, ...]) -> tuple[int, int]:
    """Reshape a tensor shape to a balanced 2D matrix shape."""
    if len(shape) < 2:
        return (int(math.prod(shape)), 1) if len(shape) >= 1 else (1, 1)
    if len(shape) == 2:
        return tuple(shape)
    dims = list(shape)
    best_ratio = float("inf")
    best_split = 1
    for i in range(1, len(dims)):
        left = math.prod(dims[:i])
        right = math.prod(dims[i:])
        ratio = max(left, right) / min(left, right)
        if ratio < best_ratio:
            best_ratio = ratio
            best_split = i
    return math.prod(dims[:best_split]), math.prod(dims[best_split:])


def _linear_warmup_scheduler(step, alpha_end, alpha_start=0.0, warmup=None):
    if warmup is None:
        return alpha_end
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def _linear_hl_warmup_scheduler(step, beta_end, beta_start, warmup=None):
    if warmup is None:
        return beta_end

    def f(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t):
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end


def _preconditioner_diag_sq(P: torch.Tensor) -> torch.Tensor:
    """Diagonal of P @ P.T, equal to squared row norms for symmetric P."""
    return P.square().sum(dim=1)


class CovariancePowerState:
    """EMA covariance plus dense P = C^{-alpha} for one tensor side."""

    def __init__(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
        c_beta: float,
        alpha: float,
        ns_iters: int,
        ns_iters_cold: int,
        ns_eps: float,
        covariance_compute_dtype: Optional[str | torch.dtype],
        ns_compute_dtype: Optional[str | torch.dtype],
    ):
        self.n = n
        self.c_beta = float(c_beta)
        self.alpha = float(alpha)
        self.ns_iters = int(ns_iters)
        self.ns_iters_cold = int(ns_iters_cold)
        self.ns_eps = float(ns_eps)
        self.covariance_compute_dtype = _resolve_compute_dtype(covariance_compute_dtype)
        self.ns_compute_dtype = _resolve_compute_dtype(ns_compute_dtype) or torch.float32
        self.C = torch.zeros(n, n, device=device, dtype=dtype)
        self.P = torch.eye(n, device=device, dtype=dtype)
        self.c_step = 0
        self.precond_updates = 0

    def _build_C(self, x: torch.Tensor, side: str) -> torch.Tensor:
        if self.covariance_compute_dtype is not None:
            x = x.to(self.covariance_compute_dtype)
        if side == "left":
            return (x @ x.T) / max(1, x.shape[1])
        if side == "right":
            return (x.T @ x) / max(1, x.shape[0])
        raise ValueError(f"side must be 'left' or 'right', got {side}")

    def observe(self, x: torch.Tensor, side: str) -> None:
        outer = self._build_C(x.detach(), side)
        self.C.mul_(self.c_beta).add_(outer.to(self.C.dtype), alpha=1 - self.c_beta)
        self.c_step += 1

    def bias_corrected_C(self) -> torch.Tensor:
        bc = 1 - self.c_beta ** max(1, self.c_step)
        return self.C / bc

    def update_preconditioner(self, use_warm_start: bool) -> None:
        use_warm_start = bool(use_warm_start and self.precond_updates > 0)
        num_iters = self.ns_iters if use_warm_start else self.ns_iters_cold
        C = self.bias_corrected_C().detach().to(self.ns_compute_dtype)
        warm_start = None
        if use_warm_start:
            warm_start = self.P.detach().to(self.ns_compute_dtype)
        P = inverse_power_quarter(
            C,
            self.alpha,
            num_iters=num_iters,
            eps=self.ns_eps,
            warm_start=warm_start,
        )
        self.P.copy_(P.to(self.P.dtype))
        self.precond_updates += 1
        self.last_warm_start_used = use_warm_start

    def apply(
        self,
        x: torch.Tensor,
        side: str,
        compute_dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        out_dtype = x.dtype
        P = self.P
        if compute_dtype is not None:
            P = P.to(compute_dtype)
            x = x.to(compute_dtype)
        if side == "left":
            out = P @ x
        elif side == "right":
            out = x @ P
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side}")
        return out.to(out_dtype) if out.dtype != out_dtype else out

    def diag_sq(self) -> torch.Tensor:
        return _preconditioner_diag_sq(self.P)

    def diagnostics(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "ns_iters": float(self.ns_iters),
            "ns_iters_cold": float(self.ns_iters_cold),
            "c_step": float(self.c_step),
            "precond_updates": float(self.precond_updates),
            "warm_start_used": float(getattr(self, "last_warm_start_used", False)),
            "P_fro": self.P.norm().item(),
            "P_min": self.P.min().item(),
            "P_max": self.P.max().item(),
            "C_trace": self.bias_corrected_C().diagonal().sum().item(),
        }


class GeneralPrecond(torch.optim.Optimizer):
    """Direct dense covariance-power preconditioner.

    Tracks EMA gradient covariance per side, periodically computes
    P = C^{-alpha} with Newton-Schulz, and applies P_left @ m_hat @ P_right.
    This optimizer is independent from AATD: it directly estimates dense
    covariance-power preconditioners instead of fitting a low-rank-plus-diagonal
    parameterization. Optionally adds an AdEMAMix-style slow momentum term to
    the preconditioned update; ``slow_momentum_alpha=0`` avoids allocating it.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.9,
        slow_momentum_beta: float = 0.9999,
        slow_momentum_alpha: float = 0.0,
        slow_momentum_beta_warmup: Optional[int] = None,
        slow_momentum_alpha_warmup: Optional[int] = None,
        weight_decay: float = 0.01,
        alpha: float = 0.5,
        ns_iters: int = 5,
        ns_iters_cold: Optional[int] = None,
        ns_eps: float = 1e-7,
        precond_update_every: int = 5,
        c_beta: float = 0.99,
        clip_update_rms: float = 1.1,
        covariance_compute_dtype: Optional[str | torch.dtype] = None,
        ns_compute_dtype: Optional[str | torch.dtype] = "fp32",
        precond_compute_dtype: Optional[str | torch.dtype] = None,
        warm_start_after_steps: int = -1,
        kronecker_correction: str = "none",
        correction_beta2: float = 0.999,
        correction_eps: float = 1e-8,
        log_diagnostics_every: int = 100,
    ):
        ns_iters_cold = ns_iters if ns_iters_cold is None else ns_iters_cold
        if float(alpha) not in (0.25, 0.5, 1.0):
            raise ValueError("alpha must be one of 0.25, 0.5, or 1.0")
        if int(ns_iters) < 1:
            raise ValueError("ns_iters must be >= 1")
        if int(ns_iters_cold) < 1:
            raise ValueError("ns_iters_cold must be >= 1")
        if float(ns_eps) <= 0:
            raise ValueError("ns_eps must be > 0")
        if int(precond_update_every) < 1:
            raise ValueError("precond_update_every must be >= 1")
        if not 0 <= float(c_beta) < 1:
            raise ValueError("c_beta must be in [0, 1)")
        if float(clip_update_rms) <= 0:
            raise ValueError("clip_update_rms must be > 0")
        if int(warm_start_after_steps) < -1:
            raise ValueError("warm_start_after_steps must be >= -1")
        if kronecker_correction not in ("none", "post", "post_actual"):
            raise ValueError("kronecker_correction must be 'none', 'post', or 'post_actual'")
        if not 0 <= float(correction_beta2) < 1:
            raise ValueError("correction_beta2 must be in [0, 1)")
        if float(correction_eps) <= 0:
            raise ValueError("correction_eps must be > 0")
        if int(log_diagnostics_every) < 0:
            raise ValueError("log_diagnostics_every must be >= 0")
        if not 0 <= float(slow_momentum_beta) < 1:
            raise ValueError("slow_momentum_beta must be in [0, 1)")
        if float(slow_momentum_alpha) < 0:
            raise ValueError("slow_momentum_alpha must be >= 0")
        if slow_momentum_beta_warmup is not None and int(slow_momentum_beta_warmup) < 1:
            raise ValueError("slow_momentum_beta_warmup must be None or >= 1")
        if slow_momentum_alpha_warmup is not None and int(slow_momentum_alpha_warmup) < 1:
            raise ValueError("slow_momentum_alpha_warmup must be None or >= 1")

        covariance_compute_dtype = _resolve_compute_dtype(covariance_compute_dtype)
        ns_compute_dtype = _resolve_compute_dtype(ns_compute_dtype) or torch.float32
        precond_compute_dtype = _resolve_compute_dtype(precond_compute_dtype)
        defaults = dict(
            lr=lr,
            momentum=momentum,
            slow_momentum_beta=float(slow_momentum_beta),
            slow_momentum_alpha=float(slow_momentum_alpha),
            slow_momentum_beta_warmup=slow_momentum_beta_warmup,
            slow_momentum_alpha_warmup=slow_momentum_alpha_warmup,
            weight_decay=weight_decay,
            alpha=float(alpha),
            ns_iters=int(ns_iters),
            ns_iters_cold=int(ns_iters_cold),
            ns_eps=float(ns_eps),
            precond_update_every=int(precond_update_every),
            c_beta=float(c_beta),
            clip_update_rms=float(clip_update_rms),
            covariance_compute_dtype=covariance_compute_dtype,
            ns_compute_dtype=ns_compute_dtype,
            precond_compute_dtype=precond_compute_dtype,
            warm_start_after_steps=int(warm_start_after_steps),
            kronecker_correction=kronecker_correction,
            correction_beta2=float(correction_beta2),
            correction_eps=float(correction_eps),
            log_diagnostics_every=int(log_diagnostics_every),
        )
        super().__init__(params, defaults)
        self._optimizer_step = 0
        self.just_gathered_stats = False
        self.latest_stats: dict[str, dict[str, float]] = {}

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        merged_shape = _merge_dims(tuple(p.shape))
        state["merged_shape"] = merged_shape
        state["step"] = 0
        state["momentum"] = torch.zeros(merged_shape, device=p.device, dtype=p.dtype)
        if group["slow_momentum_alpha"] > 0.0:
            state["slow_momentum"] = torch.zeros(merged_shape, device=p.device, dtype=p.dtype)
        if group["kronecker_correction"] == "post":
            state["adam_v"] = torch.zeros(merged_shape, device=p.device, dtype=p.dtype)
            state["last_correction_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)
        elif group["kronecker_correction"] == "post_actual":
            state["post_v"] = torch.zeros(merged_shape, device=p.device, dtype=p.dtype)
            state["last_correction_rms"] = torch.zeros((), device=p.device, dtype=torch.float32)

        n_left, n_right = merged_shape

        def make_side(n: int) -> CovariancePowerState:
            return CovariancePowerState(
                n=n,
                device=p.device,
                dtype=p.dtype,
                c_beta=group["c_beta"],
                alpha=group["alpha"],
                ns_iters=group["ns_iters"],
                ns_iters_cold=group["ns_iters_cold"],
                ns_eps=group["ns_eps"],
                covariance_compute_dtype=group["covariance_compute_dtype"],
                ns_compute_dtype=group["ns_compute_dtype"],
            )

        state["left"] = make_side(n_left)
        state["right"] = make_side(n_right) if n_right > 1 else None

    @staticmethod
    def _v_kron(
        left: CovariancePowerState,
        right: Optional[CovariancePowerState],
        eps: float,
    ) -> torch.Tensor:
        left_diag = left.diag_sq()
        if right is None:
            return 1.0 / left_diag.unsqueeze(1).clamp(min=eps)
        right_diag = right.diag_sq()
        return 1.0 / (left_diag.unsqueeze(1) * right_diag.unsqueeze(0)).clamp(min=eps)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    def _gather_aggregate_diagnostics(self) -> dict[str, float]:
        side_count = 0
        c_steps = 0.0
        precond_updates = 0.0
        warm_start_used = 0.0
        p_fros: list[float] = []
        p_mins: list[float] = []
        p_maxs: list[float] = []
        c_traces: list[float] = []
        correction_rms: list[float] = []

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                for side_state in (state["left"], state["right"]):
                    if side_state is None:
                        continue
                    diag = side_state.diagnostics()
                    side_count += 1
                    c_steps += diag["c_step"]
                    precond_updates += diag["precond_updates"]
                    warm_start_used += diag["warm_start_used"]
                    p_fros.append(diag["P_fro"])
                    p_mins.append(diag["P_min"])
                    p_maxs.append(diag["P_max"])
                    c_traces.append(diag["C_trace"])
                if "last_correction_rms" in state:
                    correction_rms.append(float(state["last_correction_rms"].item()))

        if side_count == 0:
            return {}
        stats = {
            "side_count": float(side_count),
            "alpha": float(self.defaults["alpha"]),
            "ns_iters": float(self.defaults["ns_iters"]),
            "ns_iters_cold": float(self.defaults["ns_iters_cold"]),
            "warm_start_after_steps": float(self.defaults["warm_start_after_steps"]),
            "c_step_mean": c_steps / side_count,
            "precond_updates_sum": precond_updates,
            "warm_start_used_sides": warm_start_used,
            "P_fro_mean": self._mean(p_fros),
            "P_min": min(p_mins),
            "P_max": max(p_maxs),
            "C_trace_mean": self._mean(c_traces),
        }
        if correction_rms:
            stats["correction_rms_mean"] = self._mean(correction_rms)
        return stats

    @torch.no_grad()
    def step(self, closure=None):
        self.just_gathered_stats = False
        self.latest_stats = {}
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)

                state["step"] += 1
                g = p.grad.view(state["merged_shape"])
                m = state["momentum"]
                m.mul_(group["momentum"]).add_(g, alpha=1 - group["momentum"])
                bc = 1 - group["momentum"] ** state["step"]
                m_hat = m / bc
                if "slow_momentum" in state:
                    slow_beta = _linear_hl_warmup_scheduler(
                        state["step"],
                        beta_end=group["slow_momentum_beta"],
                        beta_start=group["momentum"],
                        warmup=group["slow_momentum_beta_warmup"],
                    )
                    slow_momentum = state["slow_momentum"]
                    slow_momentum.lerp_(g, 1 - slow_beta)
                    slow_alpha = _linear_warmup_scheduler(
                        state["step"],
                        alpha_end=group["slow_momentum_alpha"],
                        alpha_start=0.0,
                        warmup=group["slow_momentum_alpha_warmup"],
                    )
                    m_hat = m_hat + slow_alpha * slow_momentum

                left: CovariancePowerState = state["left"]
                right: Optional[CovariancePowerState] = state["right"]
                left.observe(g, side="left")
                if right is not None:
                    right.observe(g, side="right")

                if state["step"] % group["precond_update_every"] == 0:
                    use_warm_start = (
                        group["warm_start_after_steps"] >= 0
                        and state["step"] >= group["warm_start_after_steps"]
                    )
                    left.update_preconditioner(use_warm_start=use_warm_start)
                    if right is not None:
                        right.update_preconditioner(use_warm_start=use_warm_start)

                compute_dtype = group["precond_compute_dtype"]
                g_pre = left.apply(m_hat, side="left", compute_dtype=compute_dtype)
                if right is not None:
                    g_pre = right.apply(g_pre, side="right", compute_dtype=compute_dtype)

                if group["kronecker_correction"] != "none":
                    beta2 = group["correction_beta2"]
                    corr_eps = group["correction_eps"]
                    bc2 = 1 - beta2 ** state["step"]
                    if group["kronecker_correction"] == "post":
                        state["adam_v"].lerp_(g.square(), 1 - beta2)
                        adam_v = (state["adam_v"] / bc2).clamp_min(corr_eps)
                        v_kron = self._v_kron(left, right, corr_eps).to(adam_v.dtype)
                        log_correction = 0.5 * (
                            v_kron.clamp_min(corr_eps).log() - adam_v.log()
                        )
                    else:
                        state["post_v"].lerp_(g_pre.square(), 1 - beta2)
                        post_v = (state["post_v"] / bc2).clamp_min(corr_eps)
                        log_correction = -0.5 * post_v.log()
                    correction = log_correction.exp()
                    g_pre = g_pre * correction
                    state["last_correction_rms"].copy_(correction.float().square().mean().sqrt())

                rms = g_pre.square().mean().sqrt().clamp(min=group["clip_update_rms"])
                g_pre.mul_(group["clip_update_rms"] / rms)
                g_pre = g_pre.view(p.shape)

                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(g_pre, alpha=-group["lr"])

        self._optimizer_step += 1
        log_every = self.defaults["log_diagnostics_every"]
        if log_every and self._optimizer_step % log_every == 0:
            stats = self._gather_aggregate_diagnostics()
            if stats:
                self.latest_stats = {"general_precond": stats}
                self.just_gathered_stats = True

        return loss

    def diagnostics(self) -> dict:
        out = {}
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                entry = {
                    "shape": tuple(p.shape),
                    "merged": state["merged_shape"],
                    "left": state["left"].diagnostics(),
                }
                if state["right"] is not None:
                    entry["right"] = state["right"].diagnostics()
                if "last_correction_rms" in state:
                    entry["correction_rms"] = state["last_correction_rms"].item()
                out[id(p)] = entry
        return out
