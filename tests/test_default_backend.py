"""Automatic selection without requiring a GPU in the test environment."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

import spd
from spd import run_circuit


@pytest.mark.parametrize("torch_present,cuda_build,cuda_available,triton_present,expected", [
    (False, False, False, False, "numpy"),
    (True, False, False, True, "numpy"),
    (True, True, False, True, "numpy"),
    (True, True, True, False, "numpy"),
    (True, True, True, True, "triton"),
])
def test_creation_selects_available_backend(monkeypatch, torch_present, cuda_build,
                                            cuda_available, triton_present, expected):
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.8" if cuda_build else None),
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    )
    monkeypatch.setitem(sys.modules, "torch", torch if torch_present else None)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace() if triton_present else None)
    calls = []
    sentinel = SimpleNamespace()
    sentinel.set_active_qubits = lambda system_size, active_qubits: sentinel

    def make_backend(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(packbit=32, create_initial_spo=lambda data, size: sentinel)

    monkeypatch.setattr(run_circuit, "_make_backend", make_backend)
    assert spd.create_spo({"Z": 1.0}, precision="double") is sentinel
    assert calls == [(expected, {"packbit": 32, "precision": "double"})]


def test_explicit_selection_and_state_inference_do_not_probe_cuda(monkeypatch):
    def unexpected_probe():
        pytest.fail("Explicit selection or state inference probed CUDA")

    monkeypatch.setattr(run_circuit, "_default_backend_name", unexpected_probe)
    backend = spd.BackendAdapter.from_name("numpy", precision="double")
    for state in (spd.create_spo({"Z": 1.0}, backend_name="numpy"),
                  spd.create_spo({"Z": 1.0}, backend=backend)):
        circuit = spd.CircuitIR(1, ())
        final, _ = spd.evolve(state, circuit, 0., 8, progress=False)
        terminal = spd.init_gradient_spo(final)
        backward, _, _ = spd.backpropagate(terminal, circuit, 0., 8, progress=False)
        assert backend.is_spo_instance(final)
        assert backend.is_spgo_instance(backward)


def test_cpu_default_pipeline_does_not_import_jax_or_triton():
    subprocess.run([sys.executable, "-c", """
import sys
sys.modules['torch'] = None
import spd
state = spd.create_spo({'Z': 1.})
assert isinstance(state, spd.numpy_backend.SparsePauliOp)
from spd.circuit_ir import CircuitIR, PauliRotation
circuit = CircuitIR(1, (PauliRotation('Rx', 'X', .2),))
final, _ = spd.evolve(state, circuit, 0., 8, progress=False)
terminal = spd.init_gradient_spo(final)
spd.backpropagate(terminal, circuit, 0., 8, progress=False)
assert 'jax' not in sys.modules
assert 'triton' not in sys.modules
"""], check=True)


def test_default_creates_gpu_state_when_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if torch.version.cuda is None or not torch.cuda.is_available():
        pytest.skip("requires NVIDIA CUDA GPU")
    state = spd.create_spo({"Z": 1.0})
    assert type(state).__module__ == "spd.triton_backend"
    assert state.c_array.is_cuda
