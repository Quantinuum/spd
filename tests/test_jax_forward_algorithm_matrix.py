import numpy as np
import pytest
from pytket.circuit import Circuit

import spd
from spd import jax_backend, numpy_backend
from spd.jax_backend.algorithms import stack_sort_merge
from spd.backend_adapter import BackendAdapter
from spd.circuit_ir import PauliRotation
from tests.helpers import assert_info_consistent, make_initial_spo, shift_case, to_term_dict


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

    spo_jax_out, _, step_info_jax = jax_backend.conjugate_pauli_rot_forward(
        spo_jax,
        sigma_jax,
        case["theta"],
        trunc_val=1e-12,
        max_num_str=1000,
    )
    spo_numpy_out, _, step_info_numpy = numpy_backend.conjugate_pauli_rot_forward(
        spo_numpy,
        sigma_numpy,
        case["theta"],
        trunc_val=1e-12,
        max_num_str=1000,
    )

    actual_terms = to_term_dict("jax", jax_backend, spo_jax_out, n_qubits=8)
    expected_terms = to_term_dict("numpy", numpy_backend, spo_numpy_out, n_qubits=8)
    assert actual_terms == pytest.approx(expected_terms, abs=1e-6)
    assert step_info_jax["num_str_truncated"] >= 0
    assert step_info_numpy["num_str_truncated"] >= 0


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

    spo_jax_out, _, _, step_info_jax = jax_adapter.apply_forward(
        spo_jax,
        operation,
        trunc_val=1e-12,
        max_num_str=1000,
    )
    spo_numpy_out, _, _, step_info_numpy = numpy_adapter.apply_forward(
        spo_numpy,
        operation,
        trunc_val=1e-12,
        max_num_str=1000,
    )

    actual_terms = to_term_dict("jax", jax_backend, spo_jax_out, n_qubits=4)
    expected_terms = to_term_dict("numpy", numpy_backend, spo_numpy_out, n_qubits=4)
    assert actual_terms == pytest.approx(expected_terms, abs=1e-6)
    assert step_info_jax["num_str_truncated"] >= 0
    assert step_info_numpy["num_str_truncated"] >= 0


def test_stack_sort_merge_does_not_keep_below_threshold_terms_live():
    _configure_rotation_backends()
    jax_backend.set_algorithm("stack_sort_merge")

    spo = jax_backend.create_op({"XIII": 1.0})
    sigma = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))

    spo_out, num_string, step_info = jax_backend.conjugate_pauli_rot_forward(
        spo,
        sigma,
        np.pi / 4,
        trunc_val=0.8,
        max_num_str=1000,
    )

    assert num_string == 0
    assert to_term_dict("jax", jax_backend, spo_out, n_qubits=4) == {}
    assert float(np.asarray(spo_out.get_expectation_value(basis="X"))) == pytest.approx(0.0)
    assert step_info["num_str_truncated"] == 2
    assert step_info["truncated_l1_norm"] == pytest.approx(np.sqrt(2), abs=1e-6)
    assert step_info["truncated_l2_norm"] == pytest.approx(1.0, abs=1e-6)


def test_stack_sort_merge_soft_cutoff_forward_keeps_old_tail_behavior():
    _configure_rotation_backends()

    spo = jax_backend.create_op({"XIII": 1.0})
    sigma = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))

    spo_out, num_string, step_info = stack_sort_merge.forward_step_soft_cutoff(
        spo,
        sigma,
        np.pi / 4,
        trunc_val=0.8,
        max_num_str=1000,
    )

    assert num_string == 1
    assert len(to_term_dict("jax", jax_backend, spo_out, n_qubits=4)) == 1
    assert step_info["num_str_truncated"] == 1
    assert step_info["truncated_l1_norm"] == pytest.approx(np.sqrt(0.5), abs=1e-6)
    assert step_info["truncated_l2_norm"] == pytest.approx(np.sqrt(0.5), abs=1e-6)


def test_search_update_merge_max_num_str_cap_keeps_largest_terms_like_stack_sort():
    _configure_rotation_backends()
    sigma = np.asarray(jax_backend.utils.pauli_str_to_uint32("ZIII"))
    outputs = {}

    for algorithm in JAX_FORWARD_ALGORITHMS:
        jax_backend.set_algorithm(algorithm)
        spo = jax_backend.create_op({"XIII": 1.0, "ZIII": 0.5})
        spo_out, num_string, step_info = jax_backend.conjugate_pauli_rot_forward(
            spo,
            sigma,
            np.pi / 3,
            trunc_val=1e-12,
            max_num_str=2,
        )
        outputs[algorithm] = (
            to_term_dict("jax", jax_backend, spo_out, n_qubits=4),
            num_string,
            step_info,
        )

    assert outputs["search_update_merge"][0] == pytest.approx(
        outputs["stack_sort_merge"][0],
        abs=1e-6,
    )
    assert outputs["search_update_merge"][1] == outputs["stack_sort_merge"][1]
    assert outputs["search_update_merge"][2] == pytest.approx(
        outputs["stack_sort_merge"][2],
        abs=1e-6,
    )


@pytest.mark.parametrize("jax_forward_algorithm", JAX_FORWARD_ALGORITHMS, indirect=True)
def test_jax_forward_algorithms_match_numpy_on_multistep_runner_flow(jax_forward_algorithm):
    _configure_rotation_backends()

    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Rz(0.125, 1)
    circ.XXPhase(0.2, 0, 1)

    final_spo_jax, info_jax = spd.evolve(make_initial_spo("jax", [0, 1], circ.n_qubits), circ, 1e-12, 1000)
    final_spo_numpy, info_numpy = spd.evolve(
        make_initial_spo("numpy", [0, 1], circ.n_qubits),
        circ,
        1e-12,
        1000,
    )
    exp_jax = final_spo_jax.get_expectation_value()
    exp_numpy = final_spo_numpy.get_expectation_value()

    assert float(np.asarray(exp_jax)) == pytest.approx(float(np.asarray(exp_numpy)), abs=1e-6)

    actual_terms = to_term_dict("jax", jax_backend, final_spo_jax, n_qubits=2)
    expected_terms = to_term_dict("numpy", numpy_backend, final_spo_numpy, n_qubits=2)
    assert actual_terms == pytest.approx(expected_terms, abs=1e-6)
    assert_info_consistent(info_jax, expected_steps=3)
    assert_info_consistent(info_numpy, expected_steps=3)
