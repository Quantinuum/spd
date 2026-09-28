from . import run_circuit
from .run_circuit import backpropagate
from .run_circuit import backpropagate_noise_analysis
from .backend_adapter import BackendAdapter
from .run_circuit import create_spo
from .run_circuit import evolve
from .run_circuit import init_gradient_spo
from .circuit_ir import CircuitIR, CreateZero, ResetZero, Discard
from .variational_circuit import VariationalCircuit
from .run_circuit import close_channel_checkpoints


def __getattr__(name):
    # Importing the native Triton backend must not initialize JAX's GPU allocator.
    if name in ("jax_backend", "numpy_backend"):
        from importlib import import_module
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in ("parse_pytket_circuit", "export_pytket_circuit"):
        from . import pytket_frontend
        value = getattr(pytket_frontend, name)
        globals()[name] = value
        return value
    if name in ("render_guppy_source", "write_guppy_source"):
        from . import guppy_emitter
        value = getattr(guppy_emitter, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "jax_backend", "numpy_backend", "run_circuit", "backpropagate",
    "backpropagate_noise_analysis", "BackendAdapter", "create_spo", "evolve",
    "close_channel_checkpoints", "init_gradient_spo", "CircuitIR", "VariationalCircuit", "CreateZero", "ResetZero", "Discard",
    "parse_pytket_circuit", "export_pytket_circuit",
    "render_guppy_source", "write_guppy_source",
]
