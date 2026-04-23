import math
import torch

WEIGHTING_REGISTRY = {}

# Use the limit formula when |t - r| is below this threshold.
# With float32 (~7 significant digits), log(f(t)/f(r)) loses precision when
# |t - r| approaches 1e-7. 1e-3 provides a large safety margin while keeping
# the limit approximation error O(1e-3), well within training noise.
_NEAR_EQ_THRESHOLD = 1e-3


def register(name, scale=1.0):
    def decorator(fn):
        WEIGHTING_REGISTRY[name] = (fn, scale)
        return fn
    return decorator


def get_weighting_fn(name):
    if name not in WEIGHTING_REGISTRY:
        raise ValueError(f"Unknown loss weighting: '{name}'. Available: {list(WEIGHTING_REGISTRY.keys())}")
    return WEIGHTING_REGISTRY[name]  # returns (fn, scale)


@register("none", scale=1.0)
def no_weighting(t, r, lam):
    return 1.0


@register("vanilla_weighting", scale=1.0)
def vanilla_weighting(t, r, lam):
    """
    1/(t-r) * lam/sqrt(lam^2+1) * ln(f(t)/f(r))
    where f(x) = sqrt(lam^2+1)*c(x) + (lam^2+1)*x - 1
    and   c(x) = sqrt((1-x)^2 + (lam*x)^2)

    When |t-r| < threshold, uses the L'Hopital limit: (lam/sqrt(lam^2+1)) * f'(t)/f(t)
    where f'(x) = sqrt(lam^2+1)*((lam^2+1)*x-1)/c(x) + (lam^2+1)
    """
    lam2 = lam ** 2
    sl = math.sqrt(lam2 + 1)

    c_t = torch.sqrt((1 - t) ** 2 + (lam * t) ** 2)
    c_r = torch.sqrt((1 - r) ** 2 + (lam * r) ** 2)
    f_t = sl * c_t + (lam2 + 1) * t - 1
    f_r = sl * c_r + (lam2 + 1) * r - 1

    near_eq = torch.abs(t - r) < _NEAR_EQ_THRESHOLD

    # Stable denominator and ratio to avoid NaN in eager evaluation of both branches
    dt = t - r
    safe_dt = torch.where(near_eq, torch.ones_like(dt), dt)
    safe_ratio = torch.where(near_eq, torch.ones_like(f_t), f_t / f_r)
    full = (1.0 / safe_dt) * (lam / sl) * torch.log(safe_ratio)

    # L'Hopital limit when |t - r| < threshold
    fp_t = sl * ((lam2 + 1) * t - 1) / c_t + (lam2 + 1)
    limit = (lam / sl) * fp_t / f_t

    return torch.where(near_eq, limit, full)


@register("straight_weighting", scale=1.0)
def straight_weighting(t, r, lam):
    """
    1/(t-r) * lam/(lam-1) * ln((1+(lam-1)*t) / (1+(lam-1)*r))

    When |t-r| < threshold, uses the L'Hopital limit: lam / (1 + (lam-1)*t)
    When lam==1, the whole formula collapses to 1.0 (uniform weight).
    """
    if lam == 1.0:
        return torch.ones_like(t)

    near_eq = torch.abs(t - r) < _NEAR_EQ_THRESHOLD

    # Stable denominator and ratio to avoid NaN in eager evaluation of both branches
    dt = t - r
    safe_dt = torch.where(near_eq, torch.ones_like(dt), dt)
    num = 1 + (lam - 1) * t
    den = 1 + (lam - 1) * r
    safe_ratio = torch.where(near_eq, torch.ones_like(num), num / den)
    full = (1.0 / safe_dt) * (lam / (lam - 1)) * torch.log(safe_ratio)

    # L'Hopital limit when |t - r| < threshold
    limit = lam / (1 + (lam - 1) * t)

    return torch.where(near_eq, limit, full)


# ── Add custom weighting functions below ──────────────────────────────────────
# Each function receives per-sample timestep tensors t and r (shape [B]) and
# a scalar lam (Python float), and should return a per-sample weight tensor
# (shape [B]) or a scalar.
#
# Example:
# @register("my_weight", scale=2.0)
# def my_weight(t, r, lam):
#     return some_function_of(t, r, lam)
