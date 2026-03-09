import numpy as np
import pytest

# backend fixture is provided by tests/conftest.py

# Expected results copied from legacy test/test_clifford.py
expected_products = {
    ('IXYZIIII', ('H', 0)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('H', 1)): ('IZYZIIII', 1.0),
    ('IXYZIIII', ('H', 2)): ('IXYZIIII', -1.0),
    ('IXYZIIII', ('H', 3)): ('IXYXIIII', 1.0),
    ('IXYZIIII', ('H', 4)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('H', 5)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('H', 6)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('H', 7)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('S', 0)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('S', 1)): ('IYYZIIII', -1.0),
    ('IXYZIIII', ('S', 2)): ('IXXZIIII', 1.0),
    ('IXYZIIII', ('S', 3)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('S', 4)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('S', 5)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('S', 6)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('S', 7)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('Sdg', 0)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('Sdg', 1)): ('IYYZIIII', 1.0),
    ('IXYZIIII', ('Sdg', 2)): ('IXXZIIII', -1.0),
    ('IXYZIIII', ('Sdg', 3)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('Sdg', 4)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('Sdg', 5)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('Sdg', 6)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('Sdg', 7)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('X', 0)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('X', 1)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('X', 2)): ('IXYZIIII', -1.0),
    ('IXYZIIII', ('X', 3)): ('IXYZIIII', -1.0),
    ('IXYZIIII', ('Y', 0)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('Y', 1)): ('IXYZIIII', -1.0),
    ('IXYZIIII', ('Y', 2)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('Y', 3)): ('IXYZIIII', -1.0),
    ('IXYZIIII', ('Z', 0)): ('IXYZIIII', 1.0),
    ('IXYZIIII', ('Z', 1)): ('IXYZIIII', -1.0),
    ('IXYZIIII', ('Z', 2)): ('IXYZIIII', -1.0),
    ('IXYZIIII', ('Z', 3)): ('IXYZIIII', 1.0),
}

expected_products_two_sites = {
    ('IXYZIXYZ', ('CX', 0, 4)): ('IXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 0, 1)): ('IXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 0, 2)): ('ZXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 0, 3)): ('ZXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 1, 0)): ('XXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 1, 5)): ('IXYZIIYZ', 1.0),
    ('IXYZIXYZ', ('CX', 1, 2)): ('IYZZIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 1, 3)): ('IYYYIXYZ', -1.0),
    ('IXYZIXYZ', ('CX', 2, 0)): ('XXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 2, 1)): ('IIYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 2, 6)): ('IXXZIXZZ', -1.0),
    ('IXYZIXYZ', ('CX', 2, 3)): ('IXXYIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 3, 0)): ('IXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 3, 1)): ('IXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 3, 2)): ('IXYIIXYZ', 1.0),
    ('IXYZIXYZ', ('CX', 3, 7)): ('IXYIIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 0, 4)): ('IXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 0, 1)): ('ZXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 0, 2)): ('ZXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 0, 3)): ('IXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 1, 0)): ('ZXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 1, 5)): ('IYYZIYYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 1, 2)): ('IYXZIXYZ', -1.0),
    ('IXYZIXYZ', ('CZ', 1, 3)): ('IXYIIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 2, 0)): ('ZXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 2, 1)): ('IYXZIXYZ', -1.0),
    ('IXYZIXYZ', ('CZ', 2, 6)): ('IXXZIXXZ', 1.0),
    ('IXYZIXYZ', ('CZ', 2, 3)): ('IXYIIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 3, 0)): ('IXYZIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 3, 1)): ('IXYIIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 3, 2)): ('IXYIIXYZ', 1.0),
    ('IXYZIXYZ', ('CZ', 3, 7)): ('IXYZIXYZ', 1.0),
}



def _apply_one_site(module, clifford, spo, qubit):
    if clifford == "H":
        return module.conjugated_pauli_batched_uint32_H(spo, qubit)
    if clifford == "S":
        return module.conjugated_pauli_batched_uint32_S(spo, qubit)
    if clifford == "Sdg":
        return module.conjugated_pauli_batched_uint32_Sdg(spo, qubit)
    if clifford == "X":
        return module.conjugated_pauli_batched_uint32_X(spo, qubit)
    if clifford == "Y":
        return module.conjugated_pauli_batched_uint32_Y(spo, qubit)
    if clifford == "Z":
        return module.conjugated_pauli_batched_uint32_Z(spo, qubit)
    raise ValueError(f"Unknown Clifford gate: {clifford}")


def _apply_two_sites(module, clifford, spo, control_qubit, target_qubit):
    if clifford == "CX":
        return module.conjugated_pauli_batched_uint32_CX(spo, control_qubit, target_qubit)
    if clifford == "CZ":
        return module.conjugated_pauli_batched_uint32_CZ(spo, control_qubit, target_qubit)
    raise ValueError(f"Unknown Clifford gate: {clifford}")


def _extract_single_term(name, spo):
    if name == "jax":
        return np.asarray(spo.xz_array[0]), np.asarray(spo.c_array[0])

    key, value = next(iter(spo.items()))
    return np.asarray(key), np.asarray(value)


def _run_or_xfail(backend_name, fn):
    try:
        return fn()
    except NotImplementedError:
        pytest.xfail(f"{backend_name} backend clifford path not implemented yet")



def test_apply_one_site(backend):
    backend_name, module = backend
    for (input_str, (clifford, qubit)), (expected_str, expected_phase) in expected_products.items():
        spo = module.create_op({input_str: 1.0})

        out_spo = _run_or_xfail(
            backend_name,
            lambda: _apply_one_site(module, clifford, spo, qubit),
        )

        xz_out, phase_out = _extract_single_term(backend_name, out_spo)
        xz_exp = np.asarray(module.utils.pauli_str_to_uint32(expected_str))

        assert np.array_equal(xz_out, xz_exp), (
            f"Failed for {input_str} with {clifford} on qubit {qubit}"
        )
        assert np.isclose(phase_out, expected_phase), (
            f"Phase mismatch for {input_str} with {clifford} on qubit {qubit}"
        )



def test_apply_two_sites(backend):
    backend_name, module = backend
    for (input_str, (clifford, control_qubit, target_qubit)), (
        expected_str,
        expected_phase,
    ) in expected_products_two_sites.items():
        spo = module.create_op({input_str: 1.0})

        out_spo = _run_or_xfail(
            backend_name,
            lambda: _apply_two_sites(module, clifford, spo, control_qubit, target_qubit),
        )

        xz_out, phase_out = _extract_single_term(backend_name, out_spo)
        xz_exp = np.asarray(module.utils.pauli_str_to_uint32(expected_str))

        assert np.array_equal(xz_out, xz_exp), (
            f"Failed for {input_str} with {clifford} on qubits {control_qubit}, {target_qubit}"
        )
        assert np.isclose(phase_out, expected_phase), (
            f"Phase mismatch for {input_str} with {clifford} on qubits {control_qubit}, {target_qubit}"
        )
