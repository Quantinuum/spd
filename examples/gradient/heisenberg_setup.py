import numpy as np
from pytket import Circuit


def _checkerboard_sign(*coords):
    return 1 if sum(coords) % 2 == 0 else -1


def _apply_staggered_rz_1d(circ, theta, system_size):
    for i in range(system_size):
        circ.Rz(_checkerboard_sign(i) * theta, i)


def _apply_staggered_rz_2d(circ, theta, system_size_x, system_size_y):
    for x in range(system_size_x):
        for y in range(system_size_y):
            circ.Rz(_checkerboard_sign(x, y) * theta, x * system_size_y + y)


def _apply_staggered_rz_3d(circ, theta, system_size_x, system_size_y, system_size_z):
    plane_size = system_size_y * system_size_z
    for x in range(system_size_x):
        for y in range(system_size_y):
            for z in range(system_size_z):
                idx = x * plane_size + y * system_size_z + z
                circ.Rz(_checkerboard_sign(x, y, z) * theta, idx)


def _apply_ring_rotation(circ, gate_name, theta, system_size):
    for i in range(system_size):
        getattr(circ, gate_name)(theta, i, (i + 1) % system_size)


def _apply_grid_rotation_2d(circ, gate_name, theta, system_size_x, system_size_y):
    for x in range(system_size_x):
        for y in range(system_size_y):
            i = x * system_size_y + y
            jx = ((x + 1) % system_size_x) * system_size_y + y
            jy = x * system_size_y + (y + 1) % system_size_y
            getattr(circ, gate_name)(theta, i, jx)
            getattr(circ, gate_name)(theta, i, jy)


def _apply_grid_rotation_3d(circ, gate_name, theta, system_size_x, system_size_y, system_size_z):
    plane_size = system_size_y * system_size_z
    for x in range(system_size_x):
        for y in range(system_size_y):
            for z in range(system_size_z):
                i = x * plane_size + y * system_size_z + z
                jx = ((x + 1) % system_size_x) * plane_size + y * system_size_z + z
                jy = x * plane_size + ((y + 1) % system_size_y) * system_size_z + z
                jz = x * plane_size + y * system_size_z + (z + 1) % system_size_z
                getattr(circ, gate_name)(theta, i, jx)
                getattr(circ, gate_name)(theta, i, jy)
                getattr(circ, gate_name)(theta, i, jz)


def _measure_all(circ, system_size):
    for i in range(system_size):
        circ.Measure(i, i)


def gen_1d_stagger_signs(system_size):
    return [_checkerboard_sign(i) for i in range(system_size)]


def gen_2d_stagger_signs(system_size_x, system_size_y):
    return [
        _checkerboard_sign(x, y)
        for x in range(system_size_x)
        for y in range(system_size_y)
    ]


def gen_3d_stagger_signs(system_size_x, system_size_y, system_size_z):
    return [
        _checkerboard_sign(x, y, z)
        for x in range(system_size_x)
        for y in range(system_size_y)
        for z in range(system_size_z)
    ]


def gen_afh_grad_multiplicities(num_layers, spatial_dim):
    if spatial_dim not in (1, 2, 3):
        raise ValueError(f"Unsupported spatial_dim={spatial_dim}. Expected 1, 2, or 3.")

    return [mult for _ in range(num_layers) for mult in (spatial_dim, spatial_dim, spatial_dim, 1)]


def combine_afh_parameter_grads(grads, system_size, stagger_signs, grad_multiplicities):
    expected_num_grads = sum(grad_multiplicities) * system_size
    if len(grads) != expected_num_grads:
        raise ValueError(
            "Expected {} sitewise gradients from multiplicities {} on {} sites, got {}.".format(
                expected_num_grads,
                grad_multiplicities,
                system_size,
                len(grads),
            )
        )

    combined = []
    stagger_signs = np.asarray(stagger_signs)
    if stagger_signs.shape != (system_size,):
        raise ValueError(
            f"Expected stagger_signs to have shape ({system_size},), got {stagger_signs.shape}."
        )

    start = 0
    for i, multiplicity in enumerate(grad_multiplicities):
        stop = start + multiplicity * system_size
        window = np.asarray(grads[start:stop])
        if i % 4 == 3:
            combined.append(np.dot(window.reshape(multiplicity, system_size).sum(axis=0), stagger_signs) * np.pi)
        else:
            combined.append(window.sum() * np.pi)
        start = stop

    return np.asarray(combined)


def gen_1d_AFH_ansatz_circuit(
    thetas,
    system_size: int = 12,
) -> Circuit:
    circ = Circuit(system_size, system_size)

    assert len(thetas) % 4 == 0, "The length of thetas should be a multiple of 4."
    num_layers = len(thetas) // 4

    for i in range(1, system_size, 2):
        circ.X(i)

    for depth in range(num_layers):
        _apply_ring_rotation(circ, "XXPhase", thetas[4 * depth], system_size)
        circ.add_barrier(list(range(system_size)))
        _apply_ring_rotation(circ, "YYPhase", thetas[4 * depth + 1], system_size)
        circ.add_barrier(list(range(system_size)))
        _apply_ring_rotation(circ, "ZZPhase", thetas[4 * depth + 2], system_size)
        circ.add_barrier(list(range(system_size)))
        _apply_staggered_rz_1d(circ, thetas[4 * depth + 3], system_size)
        circ.add_barrier(list(range(system_size)))

    _measure_all(circ, system_size)
    return circ


def gen_2d_AFH_ansatz_circuit(
    thetas,
    system_size_x: int = 4,
    system_size_y: int = 4,
) -> Circuit:
    system_size = system_size_x * system_size_y
    circ = Circuit(system_size, system_size)

    assert len(thetas) % 4 == 0, "The length of thetas should be a multiple of 4."
    num_layers = len(thetas) // 4

    for x in range(system_size_x):
        for y in range(system_size_y):
            if (x + y) % 2 == 1:
                circ.X(x * system_size_y + y)

    for depth in range(num_layers):
        _apply_grid_rotation_2d(circ, "XXPhase", thetas[4 * depth], system_size_x, system_size_y)
        circ.add_barrier(list(range(system_size)))
        _apply_grid_rotation_2d(circ, "YYPhase", thetas[4 * depth + 1], system_size_x, system_size_y)
        circ.add_barrier(list(range(system_size)))
        _apply_grid_rotation_2d(circ, "ZZPhase", thetas[4 * depth + 2], system_size_x, system_size_y)
        circ.add_barrier(list(range(system_size)))
        _apply_staggered_rz_2d(circ, thetas[4 * depth + 3], system_size_x, system_size_y)
        circ.add_barrier(list(range(system_size)))

    _measure_all(circ, system_size)
    return circ


def gen_3d_AFH_ansatz_circuit(
    thetas,
    system_size_x: int = 3,
    system_size_y: int = 3,
    system_size_z: int = 3,
) -> Circuit:
    system_size = system_size_x * system_size_y * system_size_z
    circ = Circuit(system_size, system_size)
    plane_size = system_size_y * system_size_z

    assert len(thetas) % 4 == 0, "The length of thetas should be a multiple of 4."
    num_layers = len(thetas) // 4

    for x in range(system_size_x):
        for y in range(system_size_y):
            for z in range(system_size_z):
                if (x + y + z) % 2 == 1:
                    circ.X(x * plane_size + y * system_size_z + z)

    for depth in range(num_layers):
        _apply_grid_rotation_3d(
            circ, "XXPhase", thetas[4 * depth], system_size_x, system_size_y, system_size_z
        )
        circ.add_barrier(list(range(system_size)))
        _apply_grid_rotation_3d(
            circ, "YYPhase", thetas[4 * depth + 1], system_size_x, system_size_y, system_size_z
        )
        circ.add_barrier(list(range(system_size)))
        _apply_grid_rotation_3d(
            circ, "ZZPhase", thetas[4 * depth + 2], system_size_x, system_size_y, system_size_z
        )
        circ.add_barrier(list(range(system_size)))
        _apply_staggered_rz_3d(
            circ, thetas[4 * depth + 3], system_size_x, system_size_y, system_size_z
        )
        circ.add_barrier(list(range(system_size)))

    _measure_all(circ, system_size)
    return circ


def gen_1d_Hamiltonian_dict(system_size, full=True):
    ham_dict = {}
    bonds = [(0, 1)] if not full else [(i, (i + 1) % system_size) for i in range(system_size)]

    for i, j in bonds:
        for pauli in ["X", "Y", "Z"]:
            pauli_str = ["I"] * system_size
            pauli_str[i] = pauli
            pauli_str[j] = pauli
            ham_dict["".join(pauli_str)] = 1.0

    return ham_dict


def gen_2d_Hamiltonian_dict(system_size_x, system_size_y, full=True):
    system_size = system_size_x * system_size_y
    ham_dict = {}

    if not full:
        bonds = [
            (0, system_size_y),
            (0, 1),
        ]
    else:
        bonds = []
        for x in range(system_size_x):
            for y in range(system_size_y):
                i = x * system_size_y + y
                bonds.append((i, ((x + 1) % system_size_x) * system_size_y + y))
                bonds.append((i, x * system_size_y + (y + 1) % system_size_y))

    for i, j in bonds:
        for pauli in ["X", "Y", "Z"]:
            pauli_str = ["I"] * system_size
            pauli_str[i] = pauli
            pauli_str[j] = pauli
            ham_dict["".join(pauli_str)] = 1.0

    return ham_dict


def gen_3d_Hamiltonian_dict(system_size_x, system_size_y, system_size_z, full=True):
    system_size = system_size_x * system_size_y * system_size_z
    plane_size = system_size_y * system_size_z
    ham_dict = {}

    if not full:
        bonds = [
            (0, plane_size),
            (0, system_size_z),
            (0, 1),
        ]
    else:
        bonds = []
        for x in range(system_size_x):
            for y in range(system_size_y):
                for z in range(system_size_z):
                    i = x * plane_size + y * system_size_z + z
                    bonds.append((i, ((x + 1) % system_size_x) * plane_size + y * system_size_z + z))
                    bonds.append((i, x * plane_size + ((y + 1) % system_size_y) * system_size_z + z))
                    bonds.append((i, x * plane_size + y * system_size_z + (z + 1) % system_size_z))

    for i, j in bonds:
        for pauli in ["X", "Y", "Z"]:
            pauli_str = ["I"] * system_size
            pauli_str[i] = pauli
            pauli_str[j] = pauli
            ham_dict["".join(pauli_str)] = 1.0

    return ham_dict
