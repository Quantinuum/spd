import numpy as np

import spd
from spd import jax_backend
from spd.backend_adapter import BackendAdapter
from pytket.circuit import Circuit

MAX_NUM_STR = 1_000_000


def test_jax_default_precision_is_float32():
    jax_backend.utils.set_packbit(32)
    jax_backend.set_precision("single")

    spo = jax_backend.create_op({"XIII": 1.0})
    spgo = jax_backend.create_gradient_spo(spo, basis="Z")

    assert spo.c_array.dtype == np.dtype(np.float32)
    assert spgo.c_array.dtype == np.dtype(np.float32)
    assert spgo.grad_c_array.dtype == np.dtype(np.float32)
    assert not np.iscomplexobj(np.asarray(spo.c_array))
    assert not np.iscomplexobj(np.asarray(spgo.grad_c_array))


def test_jax_double_precision_is_float64_across_forward_and_backward():
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    adapter = BackendAdapter.from_name("jax", packbit=32, precision="double")
    assert adapter.module.get_precision() == "double"

    exp_val, final_spo = spd.run_pytket_circuit(
        circ,
        [0],
        1e-12,
        MAX_NUM_STR,
        backend_name="jax",
        precision="double",
    )
    grads, spgo = spd.run_pytket_circuit_backward(
        circ,
        final_spo,
        1e-12,
        MAX_NUM_STR,
        backend_name="jax",
        precision="double",
    )

    assert np.asarray(final_spo.c_array).dtype == np.dtype(np.float64)
    assert np.asarray(spgo.c_array).dtype == np.dtype(np.float64)
    assert np.asarray(spgo.grad_c_array).dtype == np.dtype(np.float64)
    assert not np.iscomplexobj(np.asarray(final_spo.c_array))
    assert not np.iscomplexobj(np.asarray(spgo.grad_c_array))
    assert np.asarray(exp_val).dtype == np.dtype(np.float64)
    assert np.asarray(grads[0]).dtype == np.dtype(np.float64)


def test_jax_invalid_precision_is_rejected():
    try:
        BackendAdapter.from_name("jax", packbit=32, precision="half")
    except ValueError as exc:
        assert "precision" in str(exc)
    else:
        raise AssertionError("Expected invalid precision to raise ValueError")
