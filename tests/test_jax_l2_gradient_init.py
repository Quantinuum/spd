import numpy as np

from spd import jax_backend


def _rows_to_coeffs(spgo):
    return {
        tuple(np.asarray(xz_row).tolist()): (float(coeff), float(grad))
        for xz_row, coeff, grad in zip(
            np.asarray(spgo.xz_array),
            np.asarray(spgo.c_array),
            np.asarray(spgo.grad_c_array),
        )
    }


def test_init_gradient_from_l2_difference_uses_current_support_only():
    current = jax_backend.create_op({"X": 1.0})
    target = jax_backend.create_op({"X": 2.0, "Y": 3.0})

    gradient = jax_backend.init_gradient_from_l2_difference(current, target)

    assert gradient.get_size() == 1
    rows = _rows_to_coeffs(gradient)
    current_row = tuple(np.asarray(current.xz_array[0]).tolist())
    assert rows == {current_row: (1.0, -2.0)}


def test_init_gradient_from_l2_difference_union_keeps_union_support():
    current = jax_backend.create_op({"X": 1.0})
    target = jax_backend.create_op({"X": 2.0, "Y": 3.0})

    gradient = jax_backend.init_gradient_from_l2_difference_union(current, target)

    assert gradient.get_size() == 2
    rows = _rows_to_coeffs(gradient)
    x_row = tuple(np.asarray(jax_backend.create_op({"X": 1.0}).xz_array[0]).tolist())
    y_row = tuple(np.asarray(jax_backend.create_op({"Y": 1.0}).xz_array[0]).tolist())
    assert rows[x_row] == (1.0, -2.0)
    assert rows[y_row] == (0.0, -6.0)
