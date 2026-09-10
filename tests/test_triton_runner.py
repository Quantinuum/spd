"""Public Triton workflow, terminal losses and TFI parameter gradients."""
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")

import spd
from spd import triton_backend as gpu, jax_backend as jax
from spd.ansatz import tfi_2d_hva
from pytket.circuit import Circuit
from tests.test_triton_invariants import matrix


def arrays(state):
    if isinstance(state, gpu.SparsePauliOp):
        return state.to_host()
    out = (np.asarray(state.xz_array), np.asarray(state.c_array))
    return out + (np.asarray(state.grad_c_array),) if hasattr(state, "grad_c_array") else out


def term_dict(state):
    data = arrays(state)
    return {tuple(k): tuple(float(v[i]) for v in data[1:])
            for i, k in enumerate(data[0]) if any(v[i] != 0 for v in data[1:])}


def assert_states(a, b, tol=1e-11):
    a, b = term_dict(a), term_dict(b)
    # Roundoff-only terms are not semantic support differences.
    for key in a.keys() | b.keys():
        zeros = (0.,) * len(a.get(key, b.get(key)))
        np.testing.assert_allclose(a.get(key, zeros), b.get(key, zeros), atol=tol, rtol=tol)


@pytest.mark.parametrize("precision,tol", [("single", 2e-5), ("double", 1e-11)])
@pytest.mark.parametrize("alpha", [0.5, 1., 2.])
@pytest.mark.parametrize("basis", ["0", "Z", "+", "X"])
def test_terminal_losses_match_reference(precision, tol, alpha, basis):
    rng = np.random.default_rng(742)
    data = dict(zip(map("".join, itertools.product("IXYZ", repeat=2)), rng.normal(size=16)))
    data["XX"] = 0.
    jax.set_precision(precision); jax.utils.set_packbit(32)
    a, b = gpu.create_op(data, precision=precision), jax.create_op(data)
    assert a.get_OSE(alpha) == pytest.approx(float(b.get_OSE(alpha)), abs=tol)
    for initializer, kwargs in [("init_gradient_from_basis_expectation", {"basis": basis}),
                                ("init_gradient_from_ose", {"alpha": alpha}),
                                ("init_gradient_spo", {"basis": basis, "alpha": alpha, "lambda_ose": .2})]:
        actual = getattr(gpu, initializer)(a, **kwargs)
        expected = getattr(jax.kernels, initializer)(b, **kwargs)
        assert_states(actual, expected, tol)
        assert actual.c_array.data_ptr() == a.c_array.data_ptr()


@pytest.mark.parametrize("alpha", [0.5, 1., 2.])
def test_ose_coefficient_finite_difference(alpha):
    c = np.array([.23, -.51, .77, -.31])
    labels = ["I", "X", "Y", "Z"]
    state = gpu.create_op(dict(zip(labels, c)), precision="double")
    grad = gpu.init_gradient_from_ose(state, alpha).to_host()[2]
    def entropy(values):
        p = values**2 / np.sum(values**2)
        return -np.sum(p * np.log(p + 1e-12)) if alpha == 1 else np.log(np.sum(p**alpha) + 1e-12) / (1-alpha)
    fd = []
    for i in range(4):
        delta = np.eye(4)[i] * 1e-6
        fd.append((entropy(c + delta) - entropy(c - delta)) / 2e-6)
    np.testing.assert_allclose(grad, fd, atol=1e-9)


def tfi_hamiltonian(size):
    labels = []
    for q in [1, size]:
        p = ["I"] * size**2; p[0] = p[q] = "Z"; labels.append("".join(p))
    return {**dict.fromkeys(labels, -1.), "X" + "I" * (size**2-1): -3.1}


def evaluate(name, params, size, alpha, regularizer, cutoff=0., cap=65536):
    backend = spd.BackendAdapter.from_name(name, precision="double")
    ansatz = tfi_2d_hva(params, system_size_x=size, system_size_y=size)
    initial = spd.create_spo(tfi_hamiltonian(size), backend=backend)
    final, info = spd.evolve(initial, ansatz.circuit, cutoff, cap, backend=backend, progress=False)
    terminal = spd.init_gradient_spo(final, basis="+", lambda_ose=regularizer, alpha=alpha, backend=backend)
    back, grad, backward_info = spd.backpropagate(terminal, ansatz.circuit, cutoff, cap, backend=backend, progress=False)
    return float(final.get_expectation_value("+")), float(final.get_OSE(alpha)), ansatz.parameter_gradients(grad), info, backward_info


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("alpha", [1., 2.])
@pytest.mark.parametrize("regularizer", [0., .13])
@pytest.mark.parametrize("algorithm", ["search_update_merge", "stack_sort_merge"])
def test_tfi_objective_and_gradients_against_jax(size, alpha, regularizer, algorithm):
    jax.set_algorithm(algorithm)
    params = np.array([.137, -.219])
    a = evaluate("triton", params, size, alpha, regularizer, cutoff=1e-7)
    b = evaluate("jax", params, size, alpha, regularizer, cutoff=1e-7)
    np.testing.assert_allclose(a[:2], b[:2], atol=1e-11)
    np.testing.assert_allclose(a[2], b[2], atol=1e-10)
    for actual, expected in zip(a[3:], b[3:]):
        for field in actual:
            if field == "history":
                for k in actual[field]:
                    np.testing.assert_allclose(actual[field][k], expected[field][k], atol=1e-10)
            else:
                assert actual[field] == pytest.approx(expected[field], abs=1e-10)


@pytest.mark.parametrize("alpha", [1., 2.])
def test_tfi_dense_objective_parameter_finite_difference(alpha):
    params = np.array([.137, -.219, .073, .111])
    energy, entropy, gradient, *_ = evaluate("triton", params, 2, alpha, .13)
    labels = list(map("".join, itertools.product("IXYZ", repeat=4)))
    matrices = np.array([matrix(p) for p in labels])
    h = sum(c * matrix(p) for p, c in tfi_hamiltonian(2).items())
    def dense_objective(theta):
        from pytket.circuit import OpType
        circuit = tfi_2d_hva(theta, system_size_x=2, system_size_y=2).circuit
        unitary_circuit = Circuit(4)
        for command in circuit.get_commands():
            if command.op.type not in (OpType.Measure, OpType.Barrier):
                unitary_circuit.add_gate(command.op, command.qubits)
        u = unitary_circuit.get_unitary()
        evolved = u.conj().T @ h @ u
        plus = np.ones(16) / 4
        e = np.vdot(plus, evolved @ plus).real
        coefficients = np.einsum("aij,ji->a", matrices, evolved).real / 16
        probabilities = coefficients**2 / np.sum(coefficients**2)
        ose = -np.sum(probabilities*np.log(probabilities+1e-12)) if alpha == 1 else np.log(np.sum(probabilities**alpha)+1e-12)/(1-alpha)
        return e + .13 * ose
    assert energy + .13*entropy == pytest.approx(dense_objective(params), abs=1e-11)
    fd = [(dense_objective(params+d)-dense_objective(params-d))/2e-6 for d in np.eye(4)*1e-6]
    np.testing.assert_allclose(gradient, fd, atol=2e-8)


def test_empty_zero_and_invalid_terminal_losses():
    for state in [gpu.create_op({}, num_qubits=3), gpu.create_op({"III": 0.})]:
        for alpha in [.5, 1., 2.]:
            assert state.get_OSE(alpha) == 0
            with pytest.raises(ValueError, match="zero-norm"):
                gpu.init_gradient_from_ose(state, alpha)
        final, info = spd.evolve(state, Circuit(3).H(0), 0., 8, progress=True)
        assert final.get_norm_square() == 0
        assert info['num_steps_tracked'] == 1
        spd.init_gradient_spo(state)
    for alpha in [0., -1., np.nan, np.inf]:
        with pytest.raises(ValueError, match="alpha"):
            gpu.create_op({"Z": 1.}).get_OSE(alpha)
    with pytest.raises(ValueError, match="target_spo"):
        spd.init_gradient_spo(gpu.create_op({"Z": 1.}), loss_type="l2_difference")


def test_runner_inference_does_not_import_jax():
    code = '''
import sys
import spd
from spd.circuit_ir import CircuitIR, PauliRotation
state = spd.create_spo({"Z": 1.}, backend_name="triton", precision="double")
circuit = CircuitIR(1, (PauliRotation("Rx", "X", .2),))
state, _ = spd.evolve(state, circuit, 0., 8, progress=False)
grad = spd.init_gradient_spo(state, lambda_ose=.1)
_, values, _ = spd.backpropagate(grad, circuit, 0., 8, progress=False)
assert len(values) == 1
assert 'jax' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_progress_disabled_skips_reporting_reductions(monkeypatch):
    state = spd.create_spo({"Z": 1.}, backend_name="triton")
    def unexpected(*args, **kwargs):
        raise AssertionError("Reporting reduction executed with progress=False")
    monkeypatch.setattr(gpu.SparsePauliOp, 'get_norm_square', unexpected)
    monkeypatch.setattr(gpu.SparsePauliOp, 'get_OSE', unexpected)
    final, info = spd.evolve(state, Circuit(1).Rx(.2, 0), 0., 8, progress=False)
    assert final.get_size() == 2
    assert info['num_steps_tracked'] == 1


def test_capped_mixed_circuit_and_ir_round_trip(tmp_path, monkeypatch):
    from spd.pytket_frontend import parse_pytket_circuit
    import pickle
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(662)
    data = dict(zip(map("".join, itertools.product("IXYZ", repeat=3)), rng.normal(size=64)))
    circ = Circuit(3).H(0).S(1).Sdg(2).X(0).Y(1).Z(2).CX(0, 1).CY(1, 2).CZ(2, 0).Rx(.137, 0).Ry(-.291, 1).Rz(.083, 2)
    jax.set_algorithm("search_update_merge")
    results = []
    for name in ["jax", "triton"]:
        backend = spd.BackendAdapter.from_name(name, precision="double")
        initial = spd.create_spo(data, backend=backend)
        final, info = spd.evolve(initial, circ, .03, 23, backend=backend, progress=False, save_strings=True)
        terminal = spd.init_gradient_spo(final, basis="X", alpha=2., lambda_ose=.1)
        back, grads, back_info = spd.backpropagate(terminal, circ, .03, 23, progress=False, save_strings=True)
        results.append((final, back, grads, info, back_info))
        if name == "triton":
            assert final.get_size() <= 32  # public cap rounds 23 to 32
            ir = parse_pytket_circuit(circ, 32)
            for circuit_input in [ir, list(ir.operations)]:
                other, other_info = spd.evolve(initial, circuit_input, .03, 23, progress=False)
                assert_states(other, final)
                assert other_info == info
            for path, expected in [("strings_0.03.pickle", final), ("grad_strings_0.03.pickle", back)]:
                with open(path, "rb") as handle:
                    assert_states(pickle.load(handle), expected)
    a, b = results
    assert_states(a[0], b[0]); assert_states(a[1], b[1])
    np.testing.assert_allclose(a[2], b[2], atol=1e-11)
    for direction, (info_a, info_b) in enumerate(zip(a[3:], b[3:])):
        for k in info_a["history"]:
            va, vb = np.array(info_a["history"][k]), np.array(info_b["history"][k])
            if direction == 1 and k == "num_str_truncated":
                # The first inverse rotation cancels a generated primal row.
                # JAX can leave a roundoff-sized residual where Triton gets 0.
                # Only that gate may differ by one discarded nonzero row;
                # discarded norms and all state/gradient values must still match.
                np.testing.assert_array_equal(np.delete(va, 9), np.delete(vb, 9))
                assert abs(va[9] - vb[9]) <= 1
            else:
                np.testing.assert_allclose(va, vb, atol=1e-11)


def test_public_creation_width_precision_and_types():
    adapter = spd.BackendAdapter.from_name("triton", precision="double")
    # A second adapter must not alter explicit construction precision of the first.
    spd.BackendAdapter.from_name("triton", precision="single")
    state = spd.create_spo({"Z": 1.}, system_size=65, backend=adapter)
    assert state.num_qubits == 96
    assert state.c_array.dtype == torch.float64
    assert spd.create_spo({}, system_size=65, backend=adapter).get_size() == 0
    gradient = spd.init_gradient_spo(state)
    with pytest.raises(TypeError):
        spd.evolve(gradient, Circuit(1), 0., 8, progress=False)
    with pytest.raises(TypeError):
        spd.backpropagate(state, Circuit(1), 0., 8, progress=False)


@pytest.mark.parametrize("method,niter", [("eval_only", 0), ("adam", 2), ("lbfgs", 1), ("basinhopping", 1)])
def test_tfi_example_outputs_and_cpu_optimizer(tmp_path, method, niter):
    script = Path(__file__).resolve().parents[1] / "examples/gradient/run_tfi_gs.py"
    # Check loaded JAX backends without initializing any new backend in the test.
    code = """
import runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
runpy.run_path(sys.argv.pop(1), run_name='__main__')
if 'jax' in sys.modules:
    from jax._src import xla_bridge
    assert not any(key in xla_bridge._backends for key in ('cuda', 'gpu')), xla_bridge._backends
"""
    command = [sys.executable, "-c", code, str(script), "2", "2", str(niter),
               "--linear-system-size", "2", "--method", method, "--backend", "triton",
               "--trunc-val", "0", "--max-num-str", "1000", "--lambda-ose", ".13", "--alpha", "2"]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run_dir, = tmp_path.iterdir()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata['backend'] == 'triton'
    assert metadata['algorithm'] == 'native_hash'
    assert metadata['max_num_str'] == 1000
    assert metadata['effective_max_num_str'] == 1024
    assert not metadata['progress']
    for name in ['final_params.txt', 'evals.csv', 'history.csv', 'params_history.csv', 'final.json']:
        assert (run_dir / name).is_file()
    assert np.isfinite(np.loadtxt(run_dir / 'final_params.txt')).all()
