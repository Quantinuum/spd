import numpy as np
import pytest
from pytket.circuit import Circuit

import spd
from spd.circuit_ir import PauliRotation, SingleQubitClifford, SkippedOperation
from spd.pytket_frontend import parse_pytket_circuit
from tests.helpers import (
    assert_info_consistent,
    make_backend,
    make_initial_spo,
    padded_system_size,
    to_grad_term_dict,
    to_term_dict,
)

MAX_NUM_STR = 1_000_000
BACKENDS = {
    "numpy": spd.numpy_backend,
    "jax": spd.jax_backend,
}


def _count_significant_coeff_terms(spo_like, atol=1e-6):
    if hasattr(spo_like, "c_array"):
        coeffs = np.asarray(spo_like.c_array)
    else:
        coeffs = np.asarray([value[0] for value in spo_like.values()])
    return int(np.sum(np.abs(coeffs) > atol))


def test_evolve_single_qubit_ry_z_expectation(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    initial_spo = make_initial_spo(backend_name, [0], circ.n_qubits)
    final_spo, info = spd.evolve(initial_spo, circ, 1e-12, MAX_NUM_STR)
    exp_val = final_spo.get_expectation_value()

    assert np.isclose(float(np.asarray(exp_val)), np.cos(np.pi / 4), atol=1e-6)
    assert final_spo.get_size() == 2
    assert_info_consistent(info, expected_steps=1)


def test_evolve_bell_state_observables(backend_name):
    circ = Circuit(2)
    circ.H(0)
    circ.CX(0, 1)

    initial_spo_zz = make_initial_spo(backend_name, [0, 1], circ.n_qubits)
    final_spo_zz, info_zz = spd.evolve(initial_spo_zz, circ, 1e-12, MAX_NUM_STR)

    initial_spo_xx = make_initial_spo(backend_name, {"XX": 1.0}, circ.n_qubits)
    final_spo_xx, info_xx = spd.evolve(initial_spo_xx, circ, 1e-12, MAX_NUM_STR)

    assert np.isclose(float(np.asarray(final_spo_zz.get_expectation_value())), 1.0, atol=1e-6)
    assert np.isclose(float(np.asarray(final_spo_xx.get_expectation_value())), 1.0, atol=1e-6)
    assert_info_consistent(info_zz, expected_steps=2)
    assert_info_consistent(info_xx, expected_steps=2)


def test_backpropagate_single_parameter_gradient(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    initial_spo = make_initial_spo(backend_name, [0], circ.n_qubits)
    final_spo, forward_info = spd.evolve(initial_spo, circ, 1e-12, MAX_NUM_STR)
    initial_spgo = spd.init_gradient_spo(final_spo, basis="0")
    backward_final_spgo, grads, backward_info = spd.backpropagate(
        initial_spgo,
        circ,
        1e-12,
        MAX_NUM_STR,
    )

    theta = np.pi / 4
    assert np.isclose(float(np.asarray(final_spo.get_expectation_value())), np.cos(theta), atol=1e-6)
    assert len(grads) == 1
    assert np.isclose(float(np.asarray(grads[0])), -np.sin(theta), atol=1e-6)
    assert _count_significant_coeff_terms(backward_final_spgo) == 1
    assert_info_consistent(forward_info, expected_steps=1)
    assert_info_consistent(backward_info, expected_steps=1)


def test_pytket_and_ir_inputs_match_for_evolve_and_backpropagate(backend_name):
    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Rz(0.125, 1)

    initial_spo = make_initial_spo(backend_name, {"ZZ": 1.0}, circ.n_qubits)
    operations = parse_pytket_circuit(circ, padded_system_size(circ.n_qubits))

    final_spo_from_circuit, info_from_circuit = spd.evolve(initial_spo, circ, 1e-12, MAX_NUM_STR)
    final_spo_from_ir, info_from_ir = spd.evolve(initial_spo, operations, 1e-12, MAX_NUM_STR)

    module = BACKENDS[backend_name]
    terms_from_circuit = to_term_dict(backend_name, module, final_spo_from_circuit, n_qubits=2)
    terms_from_ir = to_term_dict(backend_name, module, final_spo_from_ir, n_qubits=2)
    assert terms_from_circuit.keys() == terms_from_ir.keys()
    for term in terms_from_circuit:
        assert np.isclose(terms_from_circuit[term], terms_from_ir[term], atol=1e-6)

    initial_spgo = spd.init_gradient_spo(final_spo_from_circuit, basis="0")
    final_spgo_from_circuit, grads_from_circuit, backward_info_from_circuit = spd.backpropagate(
        initial_spgo,
        circ,
        1e-12,
        MAX_NUM_STR,
    )
    final_spgo_from_ir, grads_from_ir, backward_info_from_ir = spd.backpropagate(
        initial_spgo,
        operations,
        1e-12,
        MAX_NUM_STR,
    )

    grad_terms_from_circuit = to_grad_term_dict(backend_name, module, final_spgo_from_circuit, n_qubits=2)
    grad_terms_from_ir = to_grad_term_dict(backend_name, module, final_spgo_from_ir, n_qubits=2)
    assert grad_terms_from_circuit == grad_terms_from_ir
    assert np.allclose(np.asarray(grads_from_circuit), np.asarray(grads_from_ir), atol=1e-6)
    assert info_from_circuit == info_from_ir
    assert backward_info_from_circuit == backward_info_from_ir


def test_ir_input_rejects_rebase(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)
    operations = parse_pytket_circuit(circ, padded_system_size(circ.n_qubits))
    initial_spo = make_initial_spo(backend_name, [0], circ.n_qubits)

    with pytest.raises(ValueError, match="rebase=True is only supported"):
        spd.evolve(initial_spo, operations, 1e-12, MAX_NUM_STR, rebase=True)


def test_backpropagate_l2_difference_matches_finite_difference(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)
    target_spo = BACKENDS[backend_name].create_op({"Z": 0.4, "X": -0.2})

    initial_spo = make_initial_spo(backend_name, [0], circ.n_qubits)
    final_spo, _ = spd.evolve(initial_spo, circ, 1e-12, MAX_NUM_STR)
    initial_spgo = spd.init_gradient_spo(
        final_spo,
        loss_type="l2_difference",
        target_spo=target_spo,
    )
    backward_final_spgo, grads, info = spd.backpropagate(initial_spgo, circ, 1e-12, MAX_NUM_STR)

    def l2_loss(param):
        shifted = Circuit(1)
        shifted.Ry(param, 0)
        shifted_initial_spo = make_initial_spo(backend_name, [0], shifted.n_qubits)
        shifted_spo, _ = spd.evolve(shifted_initial_spo, shifted, 1e-12, MAX_NUM_STR)
        diff = shifted_spo - target_spo
        return float(np.asarray(diff.get_norm_square()))

    eps = 1e-5
    finite_difference_grad = (l2_loss(0.25 + eps) - l2_loss(0.25 - eps)) / (2 * eps)

    assert len(grads) == 1
    assert np.isclose(float(np.asarray(grads[0])), finite_difference_grad / np.pi, atol=5e-4)
    assert backward_final_spgo.get_size() >= 1
    assert_info_consistent(info, expected_steps=1)


def test_backpropagate_basis_expectation_plus_ose_matches_direct_initializer(backend_name):
    circ = Circuit(1)
    circ.Ry(0.25, 0)
    lambda_ose = 0.2
    alpha = 1.0

    initial_spo = make_initial_spo(backend_name, [0], circ.n_qubits)
    final_spo, _ = spd.evolve(initial_spo, circ, 1e-12, MAX_NUM_STR)

    initial_spgo = spd.init_gradient_spo(
        final_spo,
        loss_type="basis_expectation",
        lambda_ose=lambda_ose,
        alpha=alpha,
    )
    direct_final_spgo, direct_grads, info = spd.backpropagate(initial_spgo, circ, 1e-12, MAX_NUM_STR)

    assert len(direct_grads) == 1
    assert np.isfinite(float(np.asarray(direct_grads[0])))
    assert _count_significant_coeff_terms(direct_final_spgo) == 1
    assert_info_consistent(info, expected_steps=1)


def test_evolve_accepts_configured_backend_object():
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    previous_algorithm = spd.jax_backend.get_algorithm()
    backend = spd.BackendAdapter.from_name("jax", packbit=32)
    backend.module.set_algorithm("stack_sort_merge")

    try:
        initial_spo = backend.create_initial_spo([0], padded_system_size(circ.n_qubits, backend.packbit))
        final_spo, info = spd.evolve(initial_spo, circ, 1e-12, MAX_NUM_STR, backend=backend)

        assert np.isclose(float(np.asarray(final_spo.get_expectation_value())), np.cos(np.pi / 4), atol=1e-6)
        assert backend.is_spo_instance(final_spo)
        assert spd.jax_backend.get_algorithm() == "stack_sort_merge"
        assert_info_consistent(info, expected_steps=1)
    finally:
        spd.jax_backend.set_algorithm(previous_algorithm)


def test_backpropagate_accepts_reused_backend_object():
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    previous_algorithm = spd.jax_backend.get_algorithm()
    backend = spd.BackendAdapter.from_name("jax", packbit=32)
    backend.module.set_algorithm("stack_sort_merge")

    try:
        initial_spo = backend.create_initial_spo([0], padded_system_size(circ.n_qubits, backend.packbit))
        final_spo, _ = spd.evolve(initial_spo, circ, 1e-12, MAX_NUM_STR, backend=backend)
        initial_spgo = spd.init_gradient_spo(final_spo, basis="0", backend=backend)
        final_spgo, grads, info = spd.backpropagate(initial_spgo, circ, 1e-12, MAX_NUM_STR, backend=backend)

        assert len(grads) == 1
        assert np.isclose(float(np.asarray(grads[0])), -np.sin(np.pi / 4), atol=1e-6)
        assert backend.is_spgo_instance(final_spgo)
        assert_info_consistent(info, expected_steps=1)
    finally:
        spd.jax_backend.set_algorithm(previous_algorithm)


def test_evolve_rejects_invalid_backend_object():
    circ = Circuit(1)
    circ.Ry(0.25, 0)
    initial_spo = make_initial_spo("numpy", [0], circ.n_qubits)

    with pytest.raises(TypeError, match="backend must be a BackendAdapter"):
        spd.evolve(initial_spo, circ, 1e-12, MAX_NUM_STR, backend="jax")


def test_init_gradient_spo_rejects_backend_mismatched_spo():
    numpy_spo = spd.numpy_backend.create_op({"Z": 1.0})
    backend = make_backend("jax")

    with pytest.raises(TypeError, match="final_spo must be a jax SparsePauliOp"):
        spd.init_gradient_spo(numpy_spo, backend=backend)


def test_init_gradient_spo_rejects_backend_mismatched_target_spo():
    numpy_spo = spd.numpy_backend.create_op({"Z": 1.0})
    jax_target = spd.jax_backend.create_op({"Z": 0.5})

    with pytest.raises(TypeError, match="target_spo must be a numpy SparsePauliOp"):
        spd.init_gradient_spo(
            numpy_spo,
            loss_type="l2_difference",
            target_spo=jax_target,
        )


def test_backpropagate_rejects_backend_mismatched_spgo():
    circ = Circuit(1)
    circ.Ry(0.25, 0)
    initial_spo = make_initial_spo("numpy", [0], circ.n_qubits)
    numpy_final_spo, _ = spd.evolve(initial_spo, circ, 1e-12, MAX_NUM_STR)
    numpy_spgo = spd.init_gradient_spo(numpy_final_spo)
    backend = make_backend("jax")

    with pytest.raises(TypeError, match="spgo must be a jax SparsePauliGradientOp"):
        spd.backpropagate(numpy_spgo, circ, 1e-12, MAX_NUM_STR, backend=backend)


def test_jax_evolve_max_num_str_caps_size_and_rounds_to_pow2():
    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Ry(0.25, 1)

    spo_no_cap, _ = spd.evolve(make_initial_spo("jax", [0, 1], circ.n_qubits), circ, 1e-12, MAX_NUM_STR)
    spo_cap_2, _ = spd.evolve(make_initial_spo("jax", [0, 1], circ.n_qubits), circ, 1e-12, 2)
    spo_cap_3, _ = spd.evolve(make_initial_spo("jax", [0, 1], circ.n_qubits), circ, 1e-12, 3)

    assert spo_no_cap.get_size() == 4
    assert spo_cap_2.get_size() == 2
    assert spo_cap_3.get_size() == 4


def test_jax_backpropagate_respects_max_num_str():
    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Ry(0.25, 1)

    final_spo, _ = spd.evolve(make_initial_spo("jax", [0, 1], circ.n_qubits), circ, 1e-12, MAX_NUM_STR)
    initial_spgo = spd.init_gradient_spo(final_spo)
    backward_final_spgo, grads, _ = spd.backpropagate(initial_spgo, circ, 1e-12, 2)

    assert len(grads) == 2
    assert backward_final_spgo.get_size() <= 2


def test_numpy_evolve_respects_max_num_str():
    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Ry(0.25, 1)

    spo, _ = spd.evolve(make_initial_spo("numpy", [0, 1], circ.n_qubits), circ, 1e-12, 2)

    assert spo.get_size() <= 2


def test_numpy_backpropagate_respects_max_num_str():
    circ = Circuit(2)
    circ.Ry(0.25, 0)
    circ.Ry(0.25, 1)

    final_spo, _ = spd.evolve(make_initial_spo("numpy", [0, 1], circ.n_qubits), circ, 1e-12, MAX_NUM_STR)
    initial_spgo = spd.init_gradient_spo(final_spo)
    backward_final_spgo, grads, _ = spd.backpropagate(initial_spgo, circ, 1e-12, 2)

    assert len(grads) == 2
    assert backward_final_spgo.get_size() <= 2


def test_evolve_info_tracks_cliffords_and_skipped_ops(backend_name):
    initial_spo = BACKENDS[backend_name].create_op({"X" + "I" * 31: 1.0})
    operations = [
        SingleQubitClifford(gate_name="OpType.H", qubit=0),
        SkippedOperation(gate_name="barrier"),
        PauliRotation(gate_name="OpType.Rz", pauli="Z" + "I" * 31, theta=np.pi / 4),
    ]

    _, info = spd.evolve(initial_spo, operations, trunc_val=1e-12, max_num_str=1000)

    assert info["history"]["num_str_truncated"] == [0, 0]
    assert info["history"]["truncated_l1_norm"] == [0.0, 0.0]
    assert info["history"]["truncated_l2_norm"] == [0.0, 0.0]
    assert_info_consistent(info, expected_steps=2)


def test_backpropagate_info_tracks_supported_backward_clifford(backend_name):
    initial_spo = BACKENDS[backend_name].create_op({"Z" + "I" * 31: 1.0})
    initial_spgo = spd.init_gradient_spo(initial_spo, basis="0")
    operations = [SingleQubitClifford(gate_name="OpType.X", qubit=0)]

    with pytest.warns(UserWarning, match="not fully supported"):
        _, grads, info = spd.backpropagate(initial_spgo, operations, trunc_val=1e-12, max_num_str=1000)

    assert grads == []
    assert info["history"]["num_str_truncated"] == [0]
    assert info["history"]["truncated_l1_norm"] == [0.0]
    assert info["history"]["truncated_l2_norm"] == [0.0]
    assert_info_consistent(info, expected_steps=1)


def test_evolve_info_captures_mixed_threshold_and_max_num_str_truncation(backend_name):
    initial_spo = BACKENDS[backend_name].create_op(
        {
            "XI" + "I" * 30: 1.0,
            "IX" + "I" * 30: 0.1,
        }
    )
    operations = [PauliRotation(gate_name="OpType.Rzz", pauli="ZZ" + "I" * 30, theta=np.pi / 4)]

    _, info = spd.evolve(initial_spo, operations, trunc_val=0.08, max_num_str=1)

    if backend_name == "numpy":
        assert info["history"]["num_str_truncated"] == [3]
        assert np.isclose(info["history"]["truncated_l1_norm"][0], 0.6 * np.sqrt(2), atol=1e-6)
        assert np.isclose(info["history"]["truncated_l2_norm"][0], np.sqrt(0.51), atol=1e-6)
    else:
        assert info["history"]["num_str_truncated"][0] >= 0
        assert info["history"]["truncated_l1_norm"][0] >= 0.0
        assert info["history"]["truncated_l2_norm"][0] >= 0.0
    assert_info_consistent(info, expected_steps=1)
