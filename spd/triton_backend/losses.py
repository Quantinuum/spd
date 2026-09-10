"""GPU-resident terminal measurements and adjoints for basis/OSE losses."""

import math

import torch

from . import SparsePauliOp
from .gradient import SparsePauliGradientOp


def _validate_alpha(alpha):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")


def _basis_mask(spo, basis):
    half = spo.xz_array.shape[1] // 2
    if basis in ("0", "Z"):
        part = spo.xz_array[:, :half]
    elif basis in ("+", "X"):
        part = spo.xz_array[:, half:]
    else:
        raise NotImplementedError(f"Expectation value in basis {basis} not implemented.")
    return torch.all(part == 0, dim=1)


def operator_stabilizer_entropy(spo, alpha=1.0):
    _validate_alpha(alpha)
    squared = spo.c_array.square()
    normalization = squared.sum()
    probabilities = squared / torch.where(normalization > 0, normalization, 1)
    if alpha == 1:
        entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
    else:
        entropy = (probabilities.pow(alpha).sum() + 1e-12).log() / (1 - alpha)
    # Reporting a zero state has a defined zero entropy; its OSE gradient does not.
    return torch.where(normalization > 0, entropy, 0).item()


def _check_spo(spo):
    if not isinstance(spo, SparsePauliOp) or isinstance(spo, SparsePauliGradientOp):
        raise TypeError("Terminal gradient initialization requires a SparsePauliOp")


def init_gradient_from_basis_expectation(spo, basis="0"):
    _check_spo(spo)
    gradient = _basis_mask(spo, basis).to(spo.c_array.dtype)
    return SparsePauliGradientOp(spo.xz_array, spo.c_array, gradient, spo.num_qubits)


def init_gradient_from_ose(spo, alpha=1.0):
    _check_spo(spo)
    _validate_alpha(alpha)
    c = spo.c_array
    normalization = c.square().sum()
    if normalization.item() == 0:
        raise ValueError("OSE gradient is undefined for a zero-norm state")
    p = c.square() / normalization
    if alpha == 1:
        probability_grads = -((p + 1e-12).log() + p / (p + 1e-12))
    else:
        moment = p.pow(alpha).sum() + 1e-12
        probability_grads = alpha * torch.where(p > 0, p.pow(alpha - 1), 0) / ((1 - alpha) * moment)
    mean = (p * probability_grads).sum()
    gradient = 2 * c / normalization * (probability_grads - mean)
    return SparsePauliGradientOp(spo.xz_array, c, gradient, spo.num_qubits)


def init_gradient_spo(spo, *, loss_type="basis_expectation", basis="0",
                      target_spo=None, lambda_ose=0.0, alpha=1.0):
    """Initialize basis or restricted L2 loss plus optional lambda * OSE."""
    if loss_type == "l2_difference":
        if target_spo is None:
            raise ValueError("target_spo must be provided for l2_difference")
        from .algebra import init_gradient_from_l2_difference
        result = init_gradient_from_l2_difference(spo, target_spo)
    elif loss_type == "basis_expectation":
        result = init_gradient_from_basis_expectation(spo, basis)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    if not math.isfinite(lambda_ose):
        raise ValueError("lambda_ose must be finite")
    if lambda_ose != 0:
        # L2 restricted support excludes explicit zero-primal rows.
        ose = init_gradient_from_ose(result.to_spo(), alpha)
        result = SparsePauliGradientOp(result.xz_array, result.c_array,
                                      result.grad_c_array + lambda_ose * ose.grad_c_array,
                                      result.num_qubits)
    return result
