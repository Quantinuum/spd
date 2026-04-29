import math

import numpy as np

from examples.time_evolution.common import (
    ansatz_parameter_count,
    compress_tfi_gradients,
    expected_raw_gradient_count,
    initial_ansatz_parameters,
)


def test_expected_raw_gradient_count_second_order():
    assert expected_raw_gradient_count(system_size=2, num_steps=2, trotter_order=2) == 14


def test_ansatz_parameter_count_second_order():
    assert ansatz_parameter_count(2, trotter_order=2) == 5


def test_initial_ansatz_parameters_second_order():
    params = initial_ansatz_parameters(num_steps=2, total_time=0.2, trotter_order=2)
    assert params.shape == (5,)
    assert np.isclose(params[0], params[-1])


def test_compress_tfi_gradients_second_order():
    raw_grads = np.asarray(
        [
            1.0,
            2.0,
            10.0,
            20.0,
            30.0,
            40.0,
            4.0,
            6.0,
            5.0,
            5.0,
            5.0,
            5.0,
            7.0,
            9.0,
        ]
    )

    compressed = compress_tfi_gradients(
        raw_grads,
        system_size=2,
        num_steps=2,
        trotter_order=2,
    )

    expected = np.asarray(
        [
            3.0 * math.pi,
            100.0 * math.pi,
            10.0 * math.pi,
            20.0 * math.pi,
            16.0 * math.pi,
        ]
    )
    np.testing.assert_allclose(compressed, expected)
