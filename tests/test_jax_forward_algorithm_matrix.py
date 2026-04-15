import numpy as np
import pytest
from pytket.circuit import Circuit

import spd
from spd import jax_backend, numpy_backend
from spd.backend_adapter import BackendAdapter
from spd.circuit_ir import PauliRotation
from tests.helpers import shift_case, to_term_dict


JAX_FORWARD_ALGORITHMS = ["stack_sort_merge", "search_update_merge"]


def _configure_rotation_backends():
    for module in (jax_backend, numpy_backend):
        module.utils.set_packbit(32)
    jax_backend.set_precision("single")


ROTATION_CASES = [
    {
        "P": "XIIIIIII",
        "sigma": "ZIIIIIII",
        "theta": np.pi / 3,
        "a_p": 1.0,
        "expected": {
            "XIIIIIII": float(np.cos(np.pi / 3)),
            "YIIIIIII": float(-np.sin(np.pi / 3)),
        },
    },
    {
        "P": "XIYIIIII",
        "sigma": "ZZIIIIII",
        "theta": np.pi / 4,
        "a_p": 1.0,
        "expected": {
            "XIYIIIII": float(np.cos(np.pi / 4)),
            "YZYIIIII": float(-np.sin(np.pi / 4)),
        },
    },
]

ROTATION_CASES = [case for case_data in ROTATION_CASES for case in shift_case(case_data)]


@pytest.mark.parametrize("jax_forward_algorithm", JAX_FORWARD_ALGORITHMS, indirect=True)
@pytest.mark.parametrize("case", ROTATION_CASES)
def test_jax_forward_algorithms_match_numpy_on_rotation_examples(jax_forward_algorithm, case):
    _configure_rotation_backends()

    spo_jax = jax_backend.create_op({case["P"]: case["a_p"]})
    spo_numpy = numpy_backend.create_op({case["P"]: case["a_p"]})
    sigma_jax = np.asarray(jax_backend.utils.pauli_str_to_uint32(case["sigma"]))
    sigma_numpy = np.asarray(numpy_backend.utils.pauli_str_to_uint32(case["sigma"]))

    spo_jax_out, _ = jax_backend.conjugate_pauli_rot_forward(
        spo_jax,
        sigma_jax,
        case["theta"],
        trunc_val=1e-12,
        max_num_str=1000,
    )
    spo_numpy_out, _ = numpy_backend.conjugate_pauli_rot_forward(
        spo_numpy,
        sigma_numpy,
        case["theta"],
        trunc_val=1e-12,
        max_num_str=1000,
    )

    actual_terms = to_term_dict("jax", jax_backend, spo_jax_out, n_qubits=8)
    expected_terms = to_term_dict("numpy", numpy_backend, spo_numpy_out, n_qubits=8)
    assert actual_terms == pytest.approx(expected_terms, abs=1e-6)


@pytest.mark.parametrize("jax_forward_algorithm", JAX_FORWARD_ALGORITHMS, indirect=True)
def test_jax_forward_algorithms_match_numpy_through_backend_adapter(jax_forward_algorithm):
    _configure_rotation_backends()

    jax_adapter = BackendAdapter.from_name("jax", packbit=32)
    numpy_adapter = BackendAdapter.from_name("numpy", packbit=32)

    spo_jax = jax_backend.create_op({"XIII": 1.0, "ZIII": 0.5})
    spo_numpy = numpy_backend.create_op({"XIII": 1.0, "ZIII": 0.5})
    operation = PauliRotation(
        gate_name="OpType.Rz",
        pauli="ZIII" + "I" * 28,
        theta=np.pi / 3,
    )

    spo_jax_out, _ = jax_adapter.apply_forward(
        spo_jax,
        operation,
        trunc_val=1e-12,
        max_num_str=1000,
    )
    spo_numpy_out, _ = numpy_adapter.apply_forward(
        spo_numpy,
        operation,
        trunc_val=1e-12,
        max_num_str=1000,
    )

    actual_terms = to_term_dict("jax", jax_backend, spo_jax_out, n_qubits=4)
    expected_terms = to_term_dict("numpy", numpy_backend, spo_numpy_out, n_qubits=4)
    assert actual_terms == pytest.approx(expected_terms, abs=1e-6)


@pytest.mark.parametrize("jax_forward_algorithm", JAX_FORWARD_ALGORITHMS, indirect=True)
def test_jax_forward_algorithms_match_numpy_on_multistep_runner_flow(jax_forward_algorithm):
    _configure_rotation_backends()

    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Rz(0.125, 1)
    circ.XXPhase(0.2, 0, 1)

    exp_jax, final_spo_jax = spd.run_pytket_circuit(
        circ,
        [0, 1],
        1e-12,
        1000,
        backend_name="jax",
    )
    exp_numpy, final_spo_numpy = spd.run_pytket_circuit(
        circ,
        [0, 1],
        1e-12,
        1000,
        backend_name="numpy",
    )

    assert float(np.asarray(exp_jax)) == pytest.approx(float(np.asarray(exp_numpy)), abs=1e-6)

    actual_terms = to_term_dict("jax", jax_backend, final_spo_jax, n_qubits=2)
    expected_terms = to_term_dict("numpy", numpy_backend, final_spo_numpy, n_qubits=2)
    assert actual_terms == pytest.approx(expected_terms, abs=1e-6)
