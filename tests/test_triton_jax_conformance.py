"""Compare every retained coefficient after multiple lattice timesteps."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")

from spd import jax_backend
from spd.backend_adapter import BackendAdapter
from spd.circuit_ir import PauliRotation
from spd.triton_backend import create_op, evolve_step


@pytest.mark.parametrize("algorithm", ["stack_sort_merge", "search_update_merge"])
@pytest.mark.parametrize("precision,rtol,atol", [("single", 2e-4, 2e-6), ("double", 2e-11, 2e-13)])
def test_lattice_coefficients(algorithm, precision, rtol, atol):
    previous = jax_backend.get_algorithm()
    jax_backend.set_algorithm(algorithm)
    try:
        n = 3
        z = []
        xx = []
        for q in range(n * n):
            pauli = ["I"] * (n * n)
            pauli[q] = "Z"
            z.append(PauliRotation("Rz", "".join(pauli), -2 * .04 * 3.044382))
            for neighbor in (q + 1 if q % n < n - 1 else None,
                             q + n if q // n < n - 1 else None):
                if neighbor is not None:
                    pauli = ["I"] * (n * n)
                    pauli[q] = pauli[neighbor] = "X"
                    xx.append(PauliRotation("XXPhase", "".join(pauli), -2 * .04))
        operations = z + xx
        observable = {"IIIIZIIII": 1.}
        adapter = BackendAdapter.from_name("jax", precision=precision)
        old = adapter.create_initial_spo(observable)
        new = create_op(observable, precision=precision)
        for _ in range(4):
            for op in reversed(operations):
                old, *_ = adapter.apply_forward(old, op, 2**-18, 10**9)
            new = evolve_step(new, operations, 2**-18, 10**9)
            keys, vals = new.to_host()
            actual = {tuple(k): v for k, v in zip(keys, vals)}
            assert len(actual) == len(vals)
            expected = {tuple(k): v for k, v in zip(np.asarray(old.xz_array), np.asarray(old.c_array))
                        if abs(v) > 2**-18}
            assert actual.keys() == expected.keys()
            np.testing.assert_allclose(list(actual.values()), [expected[k] for k in actual],
                                       rtol=rtol, atol=atol)
    finally:
        jax_backend.set_algorithm(previous)
