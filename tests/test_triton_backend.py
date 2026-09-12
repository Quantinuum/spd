"""Coefficient-level checks against the independent NumPy implementation."""

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")

from spd import numpy_backend as reference
from spd.triton_backend import create_op, conjugate_pauli_rotation, evolve_step
from spd.circuit_ir import PauliRotation, SingleQubitClifford


def as_dict(spo):
    keys, vals = spo.to_host()
    result = {tuple(k): float(v) for k, v in zip(keys, vals)}
    assert len(result) == len(keys), "GPU compaction emitted duplicate keys"
    return result


@pytest.mark.parametrize("precision,atol", [("double", 2e-13), ("single", 2e-6)])
@pytest.mark.parametrize("qubits", [3, 33, 65, 121])
def test_random_rotations(qubits, precision, atol):
    rng = np.random.default_rng(725)
    reference.utils.set_packbit(32)
    reference.set_precision(precision)
    strings = ["".join(rng.choice(list("IXYZ"), qubits)) for _ in range(173)]
    observable = dict(zip(strings, rng.normal(size=len(strings))))
    cpu = reference.create_op(observable)
    gpu = create_op(observable, precision=precision)
    for _ in range(5):
        gate = "".join(rng.choice(list("IXYZ"), qubits))
        angle = rng.uniform(-2, 2)
        cpu, *_ = reference.conjugate_pauli_rot_forward(
            cpu, reference.utils.pauli_str_to_uint(gate), angle, 0.03, max_num_str=2000)
        gpu = conjugate_pauli_rotation(gpu, gate, angle, 0.03, max_num_str=2000)
        result = as_dict(gpu)
        assert result.keys() == cpu.keys()
        np.testing.assert_allclose(list(result.values()), [cpu[k] for k in result], atol=atol, rtol=atol)


def test_all_two_qubit_pairs_and_size_cap():
    reference.utils.set_packbit(32)
    reference.set_precision("double")
    observable = {"".join(p): (i + 1) / 17 for i, p in enumerate(itertools.product("IXYZ", repeat=2))}
    for gate in observable:
        cpu, *_ = reference.conjugate_pauli_rot_forward(
            reference.create_op(observable), reference.utils.pauli_str_to_uint(gate), 0.37, 0.01, 9)
        original = create_op(observable)
        gpu = conjugate_pauli_rotation(original, gate, 0.37, 0.01, 9)
        result = as_dict(gpu)
        assert result.keys() == cpu.keys()
        np.testing.assert_allclose(list(result.values()), [cpu[k] for k in result], atol=1e-14)
        assert original.get_size() == 16


def test_cutoff_cancellation_empty_and_identity():
    # Test the JAX strict boundary explicitly (NumPy keeps equality).
    assert conjugate_pauli_rotation(create_op({"Z": 0.5}), "I", 0., 0.5).get_size() == 0
    state = conjugate_pauli_rotation(create_op({"Y": 1., "Z": 1.}), "X", np.pi / 4, 1e-14)
    assert state.get_size() == 1
    assert state.get_norm_square() == pytest.approx(2.)
    empty = create_op({}, num_qubits=65)
    assert conjugate_pauli_rotation(empty, "X", .1).get_size() == 0
    assert empty.get_expectation_value() == 0.
    assert empty.get_norm_square() == 0.


def test_reverse_order_and_unsupported_operations():
    state = create_op({"Z": 1.})
    operations = [PauliRotation("Rz", "Z", .3), PauliRotation("Rx", "X", .2)]
    expected = conjugate_pauli_rotation(conjugate_pauli_rotation(state, "X", .2), "Z", .3)
    result = evolve_step(state, operations)
    assert as_dict(result) == as_dict(expected)
    with pytest.raises(NotImplementedError):
        evolve_step(state, [object()])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        create_op({"A": 1.})
    with pytest.raises(ValueError):
        create_op({"X": 1j})
    with pytest.raises(ValueError):
        conjugate_pauli_rotation(create_op({"Z": 1.}), "X", .1, -1.)
