import numpy as np
import pytest

from tests.helpers import assert_phase_close, shift_product_case, u32


@pytest.mark.parametrize(
    "p1,p2,expected_p3,expected_phase",
    [
        *shift_product_case(("XIIIIIII", "ZIIIIIII", "YIIIIIII", -1j)),
        *shift_product_case(("ZIIIIIII", "XIIIIIII", "YIIIIIII", 1j)),
        *shift_product_case(("YIIIIIII", "XIIIIIII", "ZIIIIIII", -1j)),
        *shift_product_case(("XIIIIIII", "YIIIIIII", "ZIIIIIII", 1j)),
        *shift_product_case(("YIIIIIII", "ZIIIIIII", "XIIIIIII", 1j)),
        *shift_product_case(("ZIIIIIII", "YIIIIIII", "XIIIIIII", -1j)),
        *shift_product_case(("XIIIIIII", "XIIIIIII", "IIIIIIII", 1.0)),
        *shift_product_case(("YIIIIIII", "YIIIIIII", "IIIIIIII", 1.0)),
        *shift_product_case(("ZIIIIIII", "ZIIIIIII", "IIIIIIII", 1.0)),
        *shift_product_case(("XXIIIIII", "ZZIIIIII", "YYIIIIII", -1.0)),
        *shift_product_case(("ZZIIIIII", "XXIIIIII", "YYIIIIII", -1.0)),
        *shift_product_case(("XIYIIIII", "ZZIIIIII", "YZYIIIII", -1j)),
    ],
)
def test_pauli_product_uint_known_cases(backend, p1, p2, expected_p3, expected_phase):
    backend_name, module = backend

    xz1 = u32(module, p1)
    xz2 = u32(module, p2)

    xz3, c3 = module.pauli_product_uint(xz1, 1.0, xz2, 1.0)
    xz3 = np.asarray(xz3)

    got_p3 = module.utils.uint32_to_pauli_str(xz3, 32)[:8]
    assert got_p3 == expected_p3, (
        f"{backend_name}: {p1} * {p2} produced {got_p3}, expected {expected_p3}"
    )
    assert_phase_close(complex(c3), complex(expected_phase))


@pytest.mark.parametrize("c1", [1.0, -0.3, 0.5 + 0.2j])
def test_pauli_product_batched_second_matches_scalar(backend, c1):
    backend_name, module = backend

    sigma = u32(module, "ZZIIIIII")
    paulis = [
        "XIIIIIII",
        "YIIIIIII",
        "ZIIIIIII",
        "XIYIIIII",
        "IIXXIIII",
    ]
    xz2_array = np.stack([u32(module, p) for p in paulis], axis=0)
    c2_array = np.array([1.0, -2.0, 0.7, 1.0 + 0.5j, -0.25j], dtype=np.complex64)

    xz_batch, c_batch = module.pauli_product_batched_second_uint(
        sigma,
        c1,
        xz2_array,
        c2_array,
    )

    xz_batch = np.asarray(xz_batch)
    c_batch = np.asarray(c_batch)

    assert xz_batch.shape == xz2_array.shape
    assert c_batch.shape == c2_array.shape

    for i in range(len(paulis)):
        xz_ref, c_ref = module.pauli_product_uint(sigma, c1, xz2_array[i], c2_array[i])
        xz_ref = np.asarray(xz_ref)
        assert np.array_equal(xz_batch[i], xz_ref), (
            f"{backend_name}: xz mismatch at row {i}"
        )
        assert_phase_close(c_batch[i], c_ref)
