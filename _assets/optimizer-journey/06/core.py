"""Research implementation of the calibrated, rate-limited CrossAdamW candidate.

See theory/06_CALIBRATED_CROSSADAMW_ZH.md for assumptions and limitations.
No claim of a universal speedup, fused execution, or large-model validation.
"""

import math
import torch
from torch.optim import Optimizer


def history_noise_ratio(beta: float, observations: int) -> float:
    """Sum of squared weights of a bias-corrected EMA of iid observations."""
    if observations < 1:
        return 0.0
    power = beta ** observations
    return (1.0 - beta) / (1.0 + beta) * (1.0 + power) / (1.0 - power)


def calibrated_residual_weight(beta: float, observations: int, retention: float) -> float:
    rho = history_noise_ratio(beta, observations)
    return max(0.0, retention * (1.0 - rho) / (1.0 + rho))


class CrossAdamW(Optimizer):
    """Two-state AdamW candidate with calibrated crossed residual energy.

    After ``warmup_steps`` AdamW-style initialization steps, use the previous
    variance for the parameter update. The new second-moment input is
    h**2 + c*(g-h)**2, h = bias-corrected previous momentum and
    c = retention*(1-rho)/(1+rho). The variance update is restricted to
    [old_v*exp(-2*tau), old_v*exp(2*tau)] and fixed positive bounds.

    ``tau = max_log_scale_change * lr / reference_lr``. This caps changes
    in the inverse-square-root preconditioner, not arbitrary gradient spikes
    or the complete parameter displacement. Warmup and changes in the external
    learning-rate schedule are explicitly outside that cap.

    ``statistic`` and ``lagged`` are ablation switches, not distinct guarantees.
    State tensors are float32 for low-precision parameters, float64 for float64.
    No sparse gradients, differentiable step, capturable step, or fused kernel.
    """

    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, retention=0.25, max_log_scale_change=0.02,
                 variance_bounds=(1e-10, 1e4), warmup_steps=20,
                 statistic="calibrated", lagged=True, reference_lr=3e-3):
        if lr < 0 or not math.isfinite(lr):
            raise ValueError("lr must be finite and nonnegative")
        if not (0 < reference_lr < math.inf):
            raise ValueError("reference_lr must be finite and positive")
        if not all(0 <= b < 1 for b in betas):
            raise ValueError("betas must be in [0, 1)")
        if not 0 <= retention <= 1:
            raise ValueError("retention must be in [0, 1]")
        if eps < 0 or weight_decay < 0 or not math.isfinite(eps + weight_decay):
            raise ValueError("eps and weight_decay must be finite and nonnegative")
        if max_log_scale_change <= 0 or math.isnan(max_log_scale_change):
            raise ValueError("max_log_scale_change must be positive, possibly inf for ablation")
        lo, hi = variance_bounds
        if not (0 < lo < hi < math.inf):
            raise ValueError("variance bounds must satisfy 0 < lower < upper < inf")
        if not isinstance(warmup_steps, int) or warmup_steps < 1:
            raise ValueError("warmup_steps must be a positive integer")
        if statistic not in ("calibrated", "uncalibrated", "raw", "mixture"):
            raise ValueError("unknown second-moment statistic")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        retention=retention, max_log_scale_change=max_log_scale_change,
                        variance_bounds=variance_bounds, warmup_steps=warmup_steps,
                        statistic=statistic, lagged=lagged,
                        reference_lr=reference_lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            lo, hi = group["variance_bounds"]
            tau = group["max_log_scale_change"] * lr / group["reference_lr"] if lr else 0.0
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse or param.is_complex():
                    raise RuntimeError("CrossAdamW supports dense real gradients only")
                state = self.state[param]
                if not state:
                    dtype = torch.float64 if param.dtype == torch.float64 else torch.float32
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=dtype)
                    # This stores the corrected, projected second moment.
                    state["variance"] = torch.full_like(param, lo, dtype=dtype)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["variance"]
                g = param.grad.to(dtype=m.dtype)
                warming = t <= group["warmup_steps"]
                if warming or group["statistic"] == "raw":
                    q = g.square()
                else:
                    h = m / (1.0 - beta1 ** (t - 1))
                    if group["statistic"] == "mixture":
                        q = (1.0 - group["retention"]) * h.square() + group["retention"] * g.square()
                    else:
                        c = (calibrated_residual_weight(beta1, t - 1, group["retention"])
                             if group["statistic"] == "calibrated" else group["retention"])
                        q = h.square() + c * (g - h).square()
                gamma = (1.0 - beta2) / (1.0 - beta2 ** t)
                proposal = ((1.0 - gamma) * v + gamma * q).clamp_(lo, hi)
                if not warming and math.isfinite(tau):
                    # Clamp multiplicatively without a persistent log-variance state.
                    lower = v * math.exp(-2.0 * min(tau, 350.0))
                    upper = v * math.exp(2.0 * min(tau, 350.0))
                    proposal = torch.maximum(lower, torch.minimum(upper, proposal))
                denominator = v.sqrt().add_(eps) if group["lagged"] and not warming else proposal.sqrt().add_(eps)
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                direction = (m / (1.0 - beta1 ** t)) / denominator
                param.mul_(1.0 - lr * group["weight_decay"])
                param.add_(direction.to(dtype=param.dtype), alpha=-lr)
                v.copy_(proposal)
        return loss
