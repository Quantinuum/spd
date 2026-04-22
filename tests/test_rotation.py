import numpy as np
import pytest

from tests.helpers import anticommutes, shift_case, to_term_dict


ROTATION_CASES = [
    # U_sigma(theta) = exp(-i theta sigma / 2)
    # Commute: U_sigma^\dagger P U_sigma = P
    {
        "P": "ZIIIIIII",
        "sigma": "ZIIIIIII",
        "theta": np.pi / 5,
        "a_p": 0.7,
        "expected": {
            "ZIIIIIII": 0.7,
        },
    },
    # Commute on two anti sites (overall parity even).
    {
        "P": "XXIIIIII",
        "sigma": "ZZIIIIII",
        "theta": np.pi / 7,
        "a_p": -1.2,
        "expected": {
            "XXIIIIII": -1.2,
        },
    },
    {
        "P": "YZIIIIII",
        "sigma": "ZYIIIIII",
        "theta": np.pi / 13,
        "a_p": -1.2,
        "expected": {
            "YZIIIIII": -1.2,
        },
    },
    # Anti-commute with sigma P = iQ, here Q=YIIIIIII:
    # U_sigma^\dagger P U_sigma = cos(theta) P + i sin(theta) sigma P
    #                           = cos(theta) P - sin(theta) Q
    {
        "P": "XIIIIIII",
        "sigma": "ZIIIIIII",
        "theta": np.pi / 3,
        "a_p": 1.0,
        "expected": {
            "XIIIIIII": float(np.cos(np.pi / 3)),
            "YIIIIIII": float(-np.sin(np.pi / 3)),
        },
    },
    # Anti-commute with sigma P = iQ, here Q=YZYIIIII:
    # U_sigma^\dagger P U_sigma = cos(theta) P - sin(theta) Q
    {
        "P": "XIYIIIII",
        "sigma": "ZZIIIIII",
        "theta": np.pi / 4,
        "a_p": 1.0,
        "expected": {
            "XIYIIIII": float(np.cos(np.pi / 4)),
            "YZYIIIII": float(-np.sin(np.pi / 4)),
        },
    },
]

# Test each case under all cyclic shifts.
ROTATION_CASES = [case for data_dict in ROTATION_CASES for case in shift_case(data_dict)]


@pytest.mark.parametrize("case", ROTATION_CASES)
def test_rotation_examples_8q(backend, case):
    backend_name, module = backend
    p_u = np.asarray(module.utils.pauli_str_to_uint32(case["P"]))
    sigma_u = np.asarray(module.utils.pauli_str_to_uint32(case["sigma"]))
    is_ac = anticommutes(p_u, sigma_u)
    if is_ac:
        q_u, phase = module.pauli_product_uint(sigma_u, 1.0, p_u, 1.0)
        q_u = np.asarray(q_u)
        q_str = module.utils.uint32_to_pauli_str(np.asarray(q_u), 32)[:8]
        assert np.isclose(complex(phase), 1j), (
            f"Anti-commuting examples are chosen with sigma*P = iQ. Got phase={phase} "
            f"for P={case['P']}, sigma={case['sigma']}, Q={q_str}"
        )

    spo_in = module.create_op({case["P"]: case["a_p"]})

    spo_out, _, _ = module.conjugate_pauli_rot_forward(
        spo_in,
        sigma_u,
        case["theta"],
        trunc_val=1e-12,
        max_num_str=1000,
    )

    terms_out = to_term_dict(backend_name, module, spo_out)

    assert set(terms_out.keys()) == set(case["expected"].keys())
    for pstr, coeff_exp in case["expected"].items():
        assert np.isclose(terms_out[pstr], coeff_exp, atol=1e-6), (
            f"Mismatch for backend={backend_name}, P={case['P']}, sigma={case['sigma']}, "
            f"theta={case['theta']}, term={pstr}: got {terms_out[pstr]}, expected {coeff_exp}"
        )
