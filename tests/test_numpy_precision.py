import numpy as np

import spd
from spd import numpy_backend
from spd.backend_adapter import BackendAdapter
from pytket.circuit import Circuit

MAX_NUM_STR = 1_000_000


def test_numpy_default_precision_is_float32():
    numpy_backend.utils.set_packbit(32)
    numpy_backend.set_precision("single")

    spo = numpy_backend.create_op({"XIII": 1.0})
    spgo = numpy_backend.init_gradient_spo(
        spo,
        loss_type="basis_expectation",
        basis="Z",
    )

    coeff = np.asarray(next(iter(spo.values())))
    grad_coeff, grad_val = next(iter(spgo.values()))

    assert coeff.dtype == np.dtype(np.float32)
    assert np.asarray(grad_coeff).dtype == np.dtype(np.float32)
    assert np.asarray(grad_val).dtype == np.dtype(np.float32)
    assert not np.iscomplexobj(coeff)
    assert not np.iscomplexobj(np.asarray(grad_coeff))
    assert not np.iscomplexobj(np.asarray(grad_val))


def test_numpy_double_precision_is_float64_across_forward_and_backward():
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    adapter = BackendAdapter.from_name("numpy", packbit=32, precision="double")
    assert adapter.module.utils.get_precision() == "double"

    exp_val, final_spo = spd.run_pytket_circuit(
        circ,
        [0],
        1e-12,
        MAX_NUM_STR,
        backend_name="numpy",
        precision="double",
    )
    grads, spgo = spd.run_pytket_circuit_backward(
        circ,
        final_spo,
        1e-12,
        MAX_NUM_STR,
        backend_name="numpy",
        precision="double",
    )

    final_coeff = np.asarray(next(iter(final_spo.values())))
    spgo_coeff, spgo_grad = next(iter(spgo.values()))

    assert final_coeff.dtype == np.dtype(np.float64)
    assert np.asarray(spgo_coeff).dtype == np.dtype(np.float64)
    assert np.asarray(spgo_grad).dtype == np.dtype(np.float64)
    assert np.asarray(exp_val).dtype == np.dtype(np.float64)
    assert np.asarray(grads[0]).dtype == np.dtype(np.float64)


def test_numpy_rejects_complex_coefficients():
    numpy_backend.utils.set_packbit(32)
    numpy_backend.set_precision("single")

    try:
        numpy_backend.create_op({"ZZ": -0.25j})
    except ValueError as exc:
        assert "real-valued" in str(exc)
    else:
        raise AssertionError("Expected complex coefficients to be rejected")
