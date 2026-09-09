# from . import pytorch_backend
from . import run_circuit
from .run_circuit import backpropagate
from .run_circuit import backpropagate_noise_analysis
from .backend_adapter import BackendAdapter
from .run_circuit import create_spo
from .run_circuit import evolve
from .run_circuit import init_gradient_spo
from .circuit_ir import CircuitIR
from .variational_circuit import VariationalCircuit


def __getattr__(name):
    # Importing the native Triton backend must not initialize JAX's GPU allocator.
    if name in ("jax_backend", "numpy_backend"):
        from importlib import import_module
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "jax_backend", "numpy_backend", "run_circuit", "backpropagate",
    "backpropagate_noise_analysis", "BackendAdapter", "create_spo", "evolve",
    "init_gradient_spo", "CircuitIR", "VariationalCircuit",
]
