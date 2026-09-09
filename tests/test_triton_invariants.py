"""Independent physics checks and adversarial GPU storage tests."""

import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")

from spd import numpy_backend as reference
from spd.triton_backend import create_op, conjugate_pauli_rotation, evolve_step
from spd.circuit_ir import SkippedOperation
from tests.test_triton_backend import as_dict
from tests.test_rotation import ROTATION_CASES


def labels_from_keys(keys, qubits):
    half = keys.shape[1] // 2
    return ["".join("IXZY"[((int(row[q // 32]) >> (31 - q % 32)) & 1)
                          + 2 * ((int(row[half + q // 32]) >> (31 - q % 32)) & 1)]
                    for q in range(qubits)) for row in keys]


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("case", ROTATION_CASES)
def test_shared_rotation_examples(case, precision):
    result = conjugate_pauli_rotation(create_op({case["P"]: case["a_p"]}, precision=precision),
                                      case["sigma"], case["theta"], 1e-12)
    keys, coefficients = result.to_host()
    actual = dict(zip(labels_from_keys(keys, 8), coefficients))
    assert len(actual) == result.get_size()
    assert actual.keys() == case["expected"].keys()
    np.testing.assert_allclose(list(actual.values()), [case["expected"][p] for p in actual],
                               atol=2e-7, rtol=2e-7)


def matrix(pauli):
    single = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
              "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    result = np.ones((1, 1), dtype=complex)
    for p in pauli:
        result = np.kron(result, single[p])
    return result


@pytest.mark.parametrize("precision,atol", [("single", 2e-6), ("double", 2e-13)])
@pytest.mark.parametrize("cutoff", [0., .15])
def test_dense_matrix_oracle_all_three_qubit_generators(precision, atol, cutoff):
    # No SPD packing, phase, or rotation routine is used to form this oracle.
    labels = ["".join(p) for p in itertools.product("IXYZ", repeat=3)]
    basis = np.array([matrix(p) for p in labels])
    rng = np.random.default_rng(417)
    selected = rng.choice(64, 13, replace=False)
    coeff = rng.normal(size=13).astype(np.float32 if precision == "single" else np.float64)
    original = create_op(dict(zip((labels[i] for i in selected), coeff)), precision=precision)
    observable = np.einsum("i,ijk->jk", coeff, basis[selected])
    for gate, generator in zip(labels, basis):
        theta = rng.uniform(-2 * np.pi, 2 * np.pi)
        u = math.cos(theta / 2) * np.eye(8) - 1j * math.sin(theta / 2) * generator
        expected = u.conj().T @ observable @ u
        if cutoff:
            values = np.einsum("aij,ji->a", basis, expected).real / 8
            expected = np.einsum("i,ijk->jk", np.where(abs(values) > cutoff, values, 0), basis)
        actual = conjugate_pauli_rotation(original, gate, theta, cutoff)
        keys, values = actual.to_host()
        assert len(as_dict(actual)) == len(values)
        reconstructed = sum((c * matrix(p) for p, c in zip(labels_from_keys(keys, 3), values)),
                            np.zeros((8, 8), dtype=complex))
        np.testing.assert_allclose(reconstructed, expected, atol=atol, rtol=atol)


@pytest.mark.parametrize("count", [127, 128, 129, 255, 256, 257, 8193])
def test_compaction_boundaries_and_inverse(count):
    # Unique X/I strings, with some commuting rows and some absent partners.
    observable = {"".join("X" if (i >> q) & 1 else "I" for q in range(14)): .1 + i / count
                  for i in range(count)}
    original = create_op(observable)
    before = as_dict(original)
    forward = conjugate_pauli_rotation(original, "Z" + "I" * 13, .37)
    assert forward.get_size() == count + count // 2
    assert len(as_dict(forward)) == forward.get_size()
    assert forward.get_norm_square() == pytest.approx(original.get_norm_square(), rel=2e-14)
    restored = conjugate_pauli_rotation(forward, "Z" + "I" * 13, -.37, 1e-12)
    actual = as_dict(restored)
    assert actual.keys() == before.keys()
    np.testing.assert_allclose(list(actual.values()), [before[k] for k in actual], atol=2e-14)
    assert as_dict(original) == before


def test_deliberate_hash_collision_chain():
    # Construct 129 distinct keys mapping to bucket zero of the initial
    # 512-slot table. This deliberately stresses probing, not just random input.
    rng = np.random.default_rng(921)
    rows = []
    while len(rows) < 129:
        keys = rng.integers(0, 2**16, size=(131072, 2), dtype=np.uint32) << 16
        h = np.full(len(keys), 0x9e3779b9, dtype=np.uint32)
        for w in range(2):
            h = (h ^ keys[:, w]) * np.uint32(0x85ebca6b)
            h ^= h >> 13
        rows.extend(keys[(h & 511) == 0])
    keys = np.unique(np.array(rows), axis=0)[:129]
    assert len(keys) == 129
    labels = labels_from_keys(keys, 16)
    observable = dict(zip(labels, rng.normal(size=129)))
    reference.utils.set_packbit(32)
    reference.set_precision("double")
    cpu = reference.create_op(observable)
    gpu = create_op(observable)
    # Repeat from the same state: scheduling may change the insertion order.
    for _ in range(4):
        expected, *_ = reference.conjugate_pauli_rot_forward(
            cpu, reference.utils.pauli_str_to_uint("YZ" * 8), .43, .01)
        actual = as_dict(conjugate_pauli_rotation(gpu, "YZ" * 8, .43, .01))
        assert actual.keys() == expected.keys()
        np.testing.assert_allclose(list(actual.values()), [expected[k] for k in actual], atol=1e-13)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_creation_measurements_and_padding(precision):
    state = create_op({"II": .2, "XI": .3, "IX": -.4, "XX": .5,
                       "ZI": .7, "IZ": .1, "ZZ": -.9, "YI": 2.}, precision=precision)
    for basis in ("X", "+"):
        assert state.get_expectation_value(basis) == pytest.approx(.6, abs=2e-7)
    for basis in ("Z", "0"):
        assert state.get_expectation_value(basis) == pytest.approx(.1, abs=2e-7)
    assert state.get_norm_square() == pytest.approx(5.85, abs=1e-6)
    padded = create_op({"X": 1., "XII": 2.}, num_qubits=65, precision=precision)
    keys, values = padded.to_host()
    assert keys.shape == (1, 6)
    assert keys.dtype == np.uint32
    assert values.dtype == (np.float32 if precision == "single" else np.float64)
    assert labels_from_keys(keys, 65) == ["X" + "I" * 64]
    assert values[0] == 3.
    assert evolve_step(padded, [SkippedOperation("Barrier")]) is padded


def test_cutoff_after_pair_merge_and_topk_ties():
    # Both contributions to Z are individually below .1 but sum above it.
    merged = conjugate_pauli_rotation(create_op({"Y": -.11, "Z": .11}), "X", np.pi / 4, .1)
    keys, values = merged.to_host()
    assert labels_from_keys(keys, 1) == ["Z"]
    np.testing.assert_allclose(values, [.11 * np.sqrt(2)], atol=1e-14)
    capped = conjugate_pauli_rotation(create_op({"II": 1., "XI": -1., "YI": 1., "ZI": -1.}),
                                      "II", 0., max_num_str=2)
    assert len(as_dict(capped)) == 2
    assert np.all(np.abs(capped.to_host()[1]) == 1.)


@pytest.mark.parametrize("precision,dtype", [("single", np.float32), ("double", np.float64)])
def test_adjacent_representable_values_at_cutoff(precision, dtype):
    cutoff = dtype(.5)
    below = np.nextafter(cutoff, dtype(0))
    above = np.nextafter(cutoff, dtype(1))
    state = create_op({"II": below, "XI": cutoff, "ZI": above, "YI": -above}, precision=precision)
    result = conjugate_pauli_rotation(state, "II", 0., float(cutoff))
    keys, values = result.to_host()
    assert set(labels_from_keys(keys, 2)) == {"ZI", "YI"}
    assert np.all(abs(values) == above)


@pytest.mark.parametrize("kwargs", [{"theta": np.nan}, {"theta": np.inf},
    {"trunc_val": np.nan}, {"trunc_val": np.inf}, {"max_num_str": 0},
    {"max_num_str": -1}, {"max_num_str": 1.5}])
def test_invalid_rotation_arguments(kwargs):
    args = {"theta": .1, **kwargs}
    with pytest.raises(ValueError):
        conjugate_pauli_rotation(create_op({"Z": 1.}), "X", **args)
