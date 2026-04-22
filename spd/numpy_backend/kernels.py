import time
import numpy as np
from .sparse_pauli import SparsePauliGradientOp, SparsePauliOp
from . import utils

PHASES = np.array([1.0+0j, -1j, -1.0+0j, 1j])

"""
Convention:
    We represent a Pauli string P as (x, z), where
    P = i^(x*z) X^x Z^z
    x,z are binary arrays of shape (N,) for N qubits.
    (x*z = sum_i x_i * z_i mod 4)

    So for example:
    I = (0,0)
    X = (1,0)
    Y = (1,1)
    Z = (0,1)
    The phase in Y is implicit in the formula.
"""


def set_precision(precision: str):
    utils.set_precision(precision)


def create_measurement_op(measurement_dict, padded_system_size):
    """
    Create a SparsePauliOp from a measurement dict.
    measurement_dict: dict
        key: tuple of int - qubit indices measured in Z basis
        val: real coefficient
    padded_system_size: int - total number of qubits (after padding)
    """
    spo = SparsePauliOp()
    for key, val in measurement_dict.items():
        x_array = np.zeros((1, padded_system_size), dtype=np.bool_)
        z_array = np.array([1 if i in key else 0 for i in range(padded_system_size)], dtype=bool).reshape(1, -1)
        xz_array = np.concatenate((x_array, z_array), axis=1)
        xz_array = utils.pack_bits_to_uint(xz_array.flatten())
        spo[tuple(xz_array)] = val

    return spo

def create_op(pauli_dict):
    spo = SparsePauliOp()
    for key, val in pauli_dict.items():
        xz = utils.pauli_str_to_uint(key)
        spo[tuple(xz)] = val

    return spo

def init_gradient_from_basis_expectation(spo, basis='0'):
    gradient_spo = SparsePauliGradientOp()
    N = len(next(iter(spo))) // 2
    if basis in ['0', 'Z']:
        for P, P_val in spo.items():
            xz_array = np.array(P)
            if np.all(xz_array[:N] == 0):
                g_val = 1.
            else:
                g_val = 0.

            gradient_spo[P] = (P_val, utils.as_real_scalar(g_val))
    elif basis in ['+', 'X']:
        for P, P_val in spo.items():
            xz_array = np.array(P)
            if np.all(xz_array[N:] == 0):
                g_val = 1.
            else:
                g_val = 0.

            gradient_spo[P] = (P_val, utils.as_real_scalar(g_val))
    else:
        raise ValueError("Unsupported basis: {}".format(basis))

    return gradient_spo

def init_gradient_from_ose(spo, alpha=1.0):
    gradient_spo = SparsePauliGradientOp()
    vals = np.fromiter(spo.values(), dtype=utils.get_real_dtype())
    probabilities = np.abs(vals) ** 2
    eps = utils.as_real_scalar(1e-12)

    if alpha == 1:
        grad_vals = np.array(
            [-2.0 * coeff * (np.log(coeff * coeff + eps) + 1.0) for coeff in vals],
            dtype=utils.get_real_dtype(),
        )
    else:
        denom = np.sum((probabilities + eps) ** alpha) + eps
        grad_vals = np.array(
            [
                2.0 * alpha * coeff * (coeff * coeff + eps) ** (alpha - 1.0)
                / ((1.0 - alpha) * denom)
                for coeff in vals
            ],
            dtype=utils.get_real_dtype(),
        )

    for (packed, coeff), grad in zip(spo.items(), grad_vals):
        gradient_spo[packed] = (coeff, grad)

    return gradient_spo

def init_gradient_from_l2_difference(spo, target_spo):
    gradient_spo = SparsePauliGradientOp()
    for packed in set(spo.keys()) | set(target_spo.keys()):
        coeff = spo.get(packed, 0.0)
        target_coeff = target_spo.get(packed, 0.0)
        gradient_spo[packed] = (coeff, 2.0 * (coeff - target_coeff))
    return gradient_spo

def init_gradient_spo(
    spo,
    *,
    loss_type='basis_expectation',
    basis='0',
    target_spo=None,
    lambda_ose=0.0,
    alpha=1.0,
):
    """Canonical gradient initializer for terminal losses on the NumPy backend."""
    if loss_type == 'basis_expectation':
        gradient_spo = init_gradient_from_basis_expectation(spo, basis=basis)
    elif loss_type == 'l2_difference':
        if target_spo is None:
            raise ValueError("target_spo must be provided when loss_type='l2_difference'.")
        gradient_spo = init_gradient_from_l2_difference(spo, target_spo)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    if lambda_ose != 0.0:
        gradient_spo = gradient_spo + lambda_ose * init_gradient_from_ose(spo, alpha=alpha)

    return gradient_spo

# ---------------------------------------------------------------------- #

def pauli_product_uint(xz1, c1, xz2, c2):
    """
    Multiply two Pauli strings in packed format using NumPy.
    Supports uint8/16/32/64 arrays and complex coefficients.

    """
    N = xz1.shape[0] // 2

    xz_new = xz1 ^ xz2  # XOR for new Pauli

    # population counts (same as JAX)
    pop = np.bitwise_count  # NumPy 2.1+ unified bit count

    count = ((2 * pop(xz1[:N] & xz2[N:]).astype(np.int32).sum() +
              pop(xz1[:N] & xz1[N:]).astype(np.int32).sum() +
              pop(xz2[:N] & xz2[N:]).astype(np.int32).sum() -
              pop(xz_new[:N] & xz_new[N:]).astype(np.int32).sum()) % 4)

    phase = (-1j) ** count
    c_new = c1 * c2 * phase
    return xz_new, c_new

def pauli_product_batched_second_uint(xz1, c1, xz2_array, c2_array):
    """
    Batched version of pauli_product_uint (NumPy).
    xz1: uint arrays of shape (nwords,)
    c1: complex scalar
    xz2_array: uint arrays of shape (M, nwords)
    c2_array: complex array of shape (M,)

    Returns:
        xz_new_array: uint arrays of shape (M, nwords)
        c_new_array: complex array of shape (M,)
    """
    N = xz2_array.shape[1] // 2
    xz_new_array = xz1 ^ xz2_array

    pop = np.bitwise_count
    count = (
        2 * pop(xz1[:N] & xz2_array[:, N:]).astype(np.int32).sum(axis=1)
        + pop(xz1[:N] & xz1[N:]).astype(np.int32).sum()
        + pop(xz2_array[:, :N] & xz2_array[:, N:]).astype(np.int32).sum(axis=1)
        - pop(xz_new_array[:, :N] & xz_new_array[:, N:]).astype(np.int32).sum(axis=1)
    ) % 4

    phase = np.take(PHASES, count)
    c_new_array = c1 * c2_array * phase
    return xz_new_array, c_new_array

def check_anticommute_uint(xz1, xz2):
    """
    Check if two Pauli strings in packed uint form anticommute.
    Returns 1 if anticommute, 0 if commute.
    """
    N = xz1.shape[0] // 2
    # population count of bitwise AND
    term1 = np.bitwise_count(xz1[:N] & xz2[N:]).astype(np.int32).sum()
    term2 = np.bitwise_count(xz1[N:] & xz2[:N]).astype(np.int32).sum()
    acq = (term1 - term2) % 2  # 0 = commute, 1 = anticommute
    return acq


def _make_step_info(num_str_truncated, truncated_l1_norm, truncated_l2_sq_norm):
    return {
        "num_str_truncated": int(num_str_truncated),
        "truncated_l1_norm": float(truncated_l1_norm),
        "truncated_l2_norm": float(np.sqrt(truncated_l2_sq_norm)),
    }

def conjugate_pauli_rot_forward(spo, xzk, theta, trunc_val, max_num_str=None):
    """
    [Support uint8, uint16, uint32, uint64]
    Conjugate a batch of Pauli strings in packed uint form by rotation R_k(theta):
    exp(i theta/2 * sigma_k) * sigma_j * exp(-i theta/2 * sigma_k)
    """
    new_spo_c = SparsePauliOp()
    new_spo_a = SparsePauliOp()
    # 1. Split the Op into C and AC parts
    for xz_key, c_val in spo.items():
        xz = np.array(xz_key)
        acq_val = check_anticommute_uint(xz, xzk)
        if acq_val == 0:
            # commute
            new_spo_c[xz_key] = new_spo_c.get(xz_key, 0) + c_val
        else:
            # anticommute
            new_spo_a[xz_key] = new_spo_a.get(xz_key, 0) + c_val

    # 2. construct the pairs of AC parts
    new_spo_a_pairs = {}
    for xz_key, c_val in new_spo_a.items():
        P = xz_key
        P_array = np.array(P)
        Q_array, c_phase = pauli_product_uint(xzk, 1., P_array, 1.)
        Q = tuple(Q_array)

        # We want to order the pairs in [\sigma, P, Q] s.t.
        # \sigma P = i Q, or equivalently P Q = i \sigma
        if np.isclose(c_phase, 1j):
            P_val, Q_val = new_spo_a_pairs.get((P, Q), (0, 0))
            new_spo_a_pairs[(P, Q)] = (P_val + c_val, Q_val + 0)
        elif np.isclose(c_phase, -1j):
            Q_val, P_val = new_spo_a_pairs.get((Q, P), (0, 0))
            new_spo_a_pairs[(Q, P)] = (Q_val + 0, P_val + c_val)
        else:
            raise ValueError("Unexpected phase in Pauli product: {}".format(c_phase))

    # 3. Apply the rotation to each AC pair
    for (P, Q), (c_P, c_Q) in new_spo_a_pairs.items():
        # _, c_phase = pauli_product_uint(xzk, 1., np.array(P), 1.)
        # plus_or_minus = np.sign(c_phase * 1j)  # ±1
        plus_or_minus = 1
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # Update P
        new_spo_c[P] = cos_theta * c_P + plus_or_minus * sin_theta * c_Q
        # Update Q
        new_spo_c[Q] = -plus_or_minus * sin_theta * c_P + cos_theta * c_Q

    num_str_truncated = 0
    truncated_l1_norm = 0.0
    truncated_l2_sq_norm = 0.0

    if max_num_str is not None and max_num_str < len(new_spo_c):
        vals = np.abs(np.array(list(new_spo_c.values())))
        vals.sort()
        additional_cutoff = vals[-max_num_str]
        trunc_val = max([additional_cutoff, trunc_val])

    # 4. Truncate small values
    for P in list(new_spo_c.keys()):
        coeff = new_spo_c[P]
        magnitude = float(np.abs(coeff))
        if magnitude < trunc_val:
            num_str_truncated += 1
            truncated_l1_norm += magnitude
            truncated_l2_sq_norm += magnitude * magnitude
            new_spo_c.pop(P)

    if max_num_str is not None and len(new_spo_c) > max_num_str:
        ranked_keys = sorted(new_spo_c, key=lambda key: np.abs(new_spo_c[key]), reverse=True)
        for key in ranked_keys[max_num_str:]:
            magnitude = float(np.abs(new_spo_c[key]))
            num_str_truncated += 1
            truncated_l1_norm += magnitude
            truncated_l2_sq_norm += magnitude * magnitude
            new_spo_c.pop(key)

    return (
        new_spo_c,
        len(new_spo_c),
        _make_step_info(num_str_truncated, truncated_l1_norm, truncated_l2_sq_norm),
    )
# ---------------------------------------------------------------------- #
def tuple_sum(a, b):
    return tuple(a[i] + b[i] for i in range(len(a)))

def zeros_like_tuple(t):
    return tuple(0 for _ in t)

def conjugate_pauli_rot_backward(spo_val_grad, xzk, theta, trunc_val, max_num_str=None):
    """
    [Support uint8, uint16, uint32, uint64]
    Conjugate a batch of Pauli strings in packed uint form by rotation R_k(theta):
    exp(-i theta/2 * sigma_k) * sigma_j * exp(i theta/2 * sigma_k)
    """
    new_spo_c = SparsePauliGradientOp()
    old_spo_a = SparsePauliGradientOp()
    # 1. Split the Op into C and AC parts
    for xz_key, vals in spo_val_grad.items():
        xz = np.array(xz_key)
        acq_val = check_anticommute_uint(xz, xzk)
        if acq_val == 0:
            new_spo_c[xz_key] = vals  # commute
        else:
            old_spo_a[xz_key] = vals  # anticommute

    # 2. construct the pairs of AC parts
    old_spo_a_pairs = {}
    for P, vals in old_spo_a.items():
        Q_array, c_phase = pauli_product_uint(xzk, 1., np.array(P), 1.)
        Q = tuple(Q_array)

        # We want to order the pairs in [\sigma, P, Q] s.t.
        # \sigma P = i Q, P Q = i \sigma
        if np.isclose(c_phase, 1j):
            P_vals, Q_vals = old_spo_a_pairs.get((P, Q), (zeros_like_tuple(vals), zeros_like_tuple(vals)))
            old_spo_a_pairs[(P, Q)] = (tuple_sum(P_vals, vals), Q_vals)
        elif np.isclose(c_phase, -1j):
            Q_vals, P_vals = old_spo_a_pairs.get((Q, P), (zeros_like_tuple(vals), zeros_like_tuple(vals)))
            old_spo_a_pairs[(Q, P)] = (Q_vals, tuple_sum(P_vals, vals))
        else:
            raise ValueError("Unexpected phase in Pauli product: {}".format(c_phase))

    # 2.5 Get gradient with respect to theta
    theta_grad = 0
    for (P, Q), (P_vals, Q_vals) in old_spo_a_pairs.items():
        P_val, P_grad = P_vals
        Q_val, Q_grad = Q_vals
        theta_grad += (-P_val * Q_grad + Q_val * P_grad)

    # 3. Apply the rotation channel-wise to each AC pair
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    pm = -1  # backward in time

    for (P, Q), (P_vals, Q_vals) in old_spo_a_pairs.items():
        rot_P_vals = tuple(cos_theta * P_vals[i] + pm * sin_theta * Q_vals[i] for i in range(len(P_vals)))
        rot_Q_vals = tuple(-pm * sin_theta * P_vals[i] + cos_theta * Q_vals[i] for i in range(len(Q_vals)))
        new_spo_c[P] = rot_P_vals
        new_spo_c[Q] = rot_Q_vals

    num_str_truncated = 0
    truncated_l1_norm = 0.0
    truncated_l2_sq_norm = 0.0

    if max_num_str is not None and max_num_str < len(new_spo_c):
        vals = np.abs(np.array([value_grad[0] for value_grad in new_spo_c.values()]))
        vals.sort()
        additional_cutoff = vals[-max_num_str]
        trunc_val = max([additional_cutoff, trunc_val])

    # 4. Truncate small values
    for P in list(new_spo_c.keys()):
        coeff = new_spo_c[P][0]
        magnitude = float(np.abs(coeff))
        if magnitude < trunc_val:
            num_str_truncated += 1
            truncated_l1_norm += magnitude
            truncated_l2_sq_norm += magnitude * magnitude
            new_spo_c.pop(P)

    if max_num_str is not None and len(new_spo_c) > max_num_str:
        ranked_keys = sorted(new_spo_c, key=lambda key: np.abs(new_spo_c[key][0]), reverse=True)
        for key in ranked_keys[max_num_str:]:
            magnitude = float(np.abs(new_spo_c[key][0]))
            num_str_truncated += 1
            truncated_l1_norm += magnitude
            truncated_l2_sq_norm += magnitude * magnitude
            new_spo_c.pop(key)

    return (
        new_spo_c,
        len(new_spo_c),
        theta_grad,
        _make_step_info(num_str_truncated, truncated_l1_norm, truncated_l2_sq_norm),
    )
# ---------------------------------------------------------------------- #


def conjugate_H_forward(spo, qubit):
    """
    Apply Hadamard gate on the specified qubit for a batch of packed (x,z) representations.

    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply H on.

    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), float (±1.0 per batch)

    """
    site = qubit // 32
    bit = qubit % 32
    bit_mask = np.uint32(1 << (31 - bit))

    new_spo = SparsePauliOp()
    for xz_key, coeff in spo.items():
        xz = np.array(xz_key, dtype=np.uint32, copy=True)
        n_words = xz.shape[0] // 2

        x_word = xz[site]
        z_word = xz[n_words + site]
        diff = (x_word & bit_mask) ^ (z_word & bit_mask)
        x_word ^= diff
        z_word ^= diff
        xz[site] = x_word
        xz[n_words + site] = z_word

        phase = -1.0 if ((x_word & bit_mask) and (z_word & bit_mask)) else 1.0
        new_spo[tuple(xz)] = phase * coeff

    return new_spo

def conjugate_S_forward(spo, qubit):
    """
    Apply S gate on the specified qubit for a batch of packed (x,z) representations.

    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply S on.

    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), complex (±1, ±i per batch)

    """
    site = qubit // 32
    bit = qubit % 32
    bit_mask = np.uint32(1 << (31 - bit))

    new_spo = SparsePauliOp()
    for xz_key, coeff in spo.items():
        xz = np.array(xz_key, dtype=np.uint32, copy=True)
        n_words = xz.shape[0] // 2

        x_word = xz[site]
        z_word = xz[n_words + site]
        x_bit = x_word & bit_mask
        z_word ^= x_bit
        xz[n_words + site] = z_word

        phase = -1.0 if (x_bit and (z_word & bit_mask)) else 1.0
        new_spo[tuple(xz)] = phase * coeff

    return new_spo

def conjugate_Sdg_forward(spo, qubit):
    """
    Apply Sdg gate on the specified qubit for a batch of packed (x,z) representations.

    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply S on.

    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), complex (±1, ±i per batch)

    """
    site = qubit // 32
    bit = qubit % 32
    bit_mask = np.uint32(1 << (31 - bit))

    new_spo = SparsePauliOp()
    for xz_key, coeff in spo.items():
        xz = np.array(xz_key, dtype=np.uint32, copy=True)
        n_words = xz.shape[0] // 2

        x_word = xz[site]
        z_word = xz[n_words + site]
        x_bit = x_word & bit_mask
        phase = -1.0 if (x_bit and (z_word & bit_mask)) else 1.0
        z_word ^= x_bit
        xz[n_words + site] = z_word

        new_spo[tuple(xz)] = phase * coeff

    return new_spo

def conjugate_CX_forward(spo, control_qubit, target_qubit):
    """
    Apply CX gate on the specified qubits for a batch of packed (x,z) representations.
    x_t <-- x_t XOR x_c
    z_c <-- z_c XOR z_t
    phase = (-1)^{x_c z_t (z_c \\oplus x_t)}

    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        control_qubit: int, control qubit index.
        target_qubit: int, target qubit index.

    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), complex (±1, ±i per batch)

    """
    control_site = control_qubit // 32
    control_bit = control_qubit % 32
    target_site = target_qubit // 32
    target_bit = target_qubit % 32

    c_bit_mask = np.uint32(1 << (31 - control_bit))
    t_bit_mask = np.uint32(1 << (31 - target_bit))

    new_spo = SparsePauliOp()
    for xz_key, coeff in spo.items():
        xz = np.array(xz_key, dtype=np.uint32, copy=True)
        n_words = xz.shape[0] // 2

        x_c_word = xz[control_site]
        x_t_word = xz[target_site]
        z_c_word = xz[n_words + control_site]
        z_t_word = xz[n_words + target_site]

        x_c_bit = int((x_c_word & c_bit_mask) != 0)
        z_c_bit = int((z_c_word & c_bit_mask) != 0)
        x_t_bit = int((x_t_word & t_bit_mask) != 0)
        z_t_bit = int((z_t_word & t_bit_mask) != 0)

        if x_c_bit:
            xz[target_site] = x_t_word ^ t_bit_mask
        if z_t_bit:
            xz[n_words + control_site] = z_c_word ^ c_bit_mask

        phase = -1.0 if (x_c_bit and z_t_bit and (z_c_bit == x_t_bit)) else 1.0
        new_spo[tuple(xz)] = phase * coeff

    return new_spo

def conjugate_CY_forward(spo, control_qubit, target_qubit):
    spo = conjugate_S_forward(spo, target_qubit)
    spo = conjugate_CX_forward(spo, control_qubit, target_qubit)
    spo = conjugate_Sdg_forward(spo, target_qubit)
    return spo

def conjugate_CZ_forward(spo, control_qubit, target_qubit):
    """
    Apply CZ gate on the specified qubits for a batch of packed (x,z) representations.
    z_c' = z_c XOR x_t
    z_t' = z_t XOR x_c
    phase = (-1)^( x_c * x_t * (z_c XOR z_t) )

    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        control_qubit: int, control qubit index.
        target_qubit: int, target qubit index.

    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), complex (±1, ±i per batch)
    """
    control_site = control_qubit // 32
    control_bit = control_qubit % 32
    target_site = target_qubit // 32
    target_bit = target_qubit % 32

    c_bit_mask = np.uint32(1 << (31 - control_bit))
    t_bit_mask = np.uint32(1 << (31 - target_bit))

    new_spo = SparsePauliOp()
    for xz_key, coeff in spo.items():
        xz = np.array(xz_key, dtype=np.uint32, copy=True)
        n_words = xz.shape[0] // 2

        x_c_word = xz[control_site]
        x_t_word = xz[target_site]
        z_c_word = xz[n_words + control_site]
        z_t_word = xz[n_words + target_site]

        x_c_bit = int((x_c_word & c_bit_mask) != 0)
        z_c_bit = int((z_c_word & c_bit_mask) != 0)
        x_t_bit = int((x_t_word & t_bit_mask) != 0)
        z_t_bit = int((z_t_word & t_bit_mask) != 0)

        if x_t_bit:
            xz[n_words + control_site] = z_c_word ^ c_bit_mask
        if x_c_bit:
            z_t_word = xz[n_words + target_site]
            xz[n_words + target_site] = z_t_word ^ t_bit_mask

        phase = -1.0 if (x_c_bit and x_t_bit and (z_c_bit ^ z_t_bit)) else 1.0
        new_spo[tuple(xz)] = phase * coeff

    return new_spo

def conjugate_X_forward(spo, qubit):
    """
    Apply X gate on the specified qubit for a batch of packed (x,z) representations.
    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply X on.
    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), float (1.0 per batch)
    """
    site = qubit // 32
    bit = qubit % 32
    bit_mask = np.uint32(1 << (31 - bit))

    new_spo = SparsePauliOp()
    for xz_key, coeff in spo.items():
        xz = np.array(xz_key, dtype=np.uint32, copy=True)
        n_words = xz.shape[0] // 2
        z_word = xz[n_words + site]
        phase = -1.0 if (z_word & bit_mask) else 1.0
        new_spo[tuple(xz)] = phase * coeff

    return new_spo

def conjugate_X_backward(spgo, qubit):
    """
    Apply X gate on the specified qubit to a SparsePauliGradientOp.
    """
    site = qubit // 32
    bit = qubit % 32
    bit_mask = np.uint32(1 << (31 - bit))

    new_spgo = SparsePauliGradientOp()
    for xz_key, value in spgo.items():
        xz = np.array(xz_key, dtype=np.uint32, copy=True)
        n_words = xz.shape[0] // 2
        z_word = xz[n_words + site]
        phase = -1.0 if (z_word & bit_mask) else 1.0
        new_spgo[tuple(xz)] = (phase * value[0], phase * value[1])

    return new_spgo

def conjugate_Y_forward(spo, qubit):
    """
    Apply Y gate on the specified qubit for a batch of packed (x,z) representations.
    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply Y on.
    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), complex (±1, ±i per batch)
    """
    site = qubit // 32
    bit = qubit % 32
    bit_mask = np.uint32(1 << (31 - bit))

    new_spo = SparsePauliOp()
    for xz_key, coeff in spo.items():
        xz = np.array(xz_key, dtype=np.uint32, copy=True)
        n_words = xz.shape[0] // 2

        x_bit = int((xz[site] & bit_mask) != 0)
        z_bit = int((xz[n_words + site] & bit_mask) != 0)
        phase = -1.0 if (x_bit ^ z_bit) else 1.0
        new_spo[tuple(xz)] = phase * coeff

    return new_spo

def conjugate_Z_forward(spo, qubit):
    """
    Apply Z gate on the specified qubit for a batch of packed (x,z) representations.
    Args:
        xz_in_packed_int32: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            Packed (x,z) representation for M samples.
        qubit: int, qubit index to apply Z on.
    Returns:
        (updated_xz, phase): tuple
            updated_xz: jnp.ndarray of shape (M, 2N), dtype=jnp.int32
            phase: jnp.ndarray of shape (M,), float (1.0 per batch)
    """
    site = qubit // 32
    bit = qubit % 32
    bit_mask = np.uint32(1 << (31 - bit))

    new_spo = SparsePauliOp()
    for xz_key, coeff in spo.items():
        xz = np.array(xz_key, dtype=np.uint32, copy=True)
        phase = -1.0 if (xz[site] & bit_mask) else 1.0
        new_spo[tuple(xz)] = phase * coeff

    return new_spo
