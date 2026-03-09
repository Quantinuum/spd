import numpy as np


def u32(module, pstr):
    return np.asarray(module.utils.pauli_str_to_uint32(pstr))


def assert_phase_close(actual, expected, atol=1e-6):
    assert np.isclose(np.real(actual), np.real(expected), atol=atol)
    assert np.isclose(np.imag(actual), np.imag(expected), atol=atol)


def to_term_dict(backend_name, module, spo, n_qubits=8):
    terms = {}

    if backend_name == "jax":
        xz_rows = np.asarray(spo.xz_array)
        c_vals = np.asarray(spo.c_array)
        for xz, c in zip(xz_rows, c_vals):
            pstr = module.utils.uint32_to_pauli_str(np.asarray(xz), 32)[:n_qubits]
            terms[pstr] = terms.get(pstr, 0.0) + float(np.real(c))
        return terms

    for xz_key, c in spo.items():
        pstr = module.utils.uint32_to_pauli_str(np.asarray(xz_key), 32)[:n_qubits]
        terms[pstr] = terms.get(pstr, 0.0) + float(np.real(c))
    return terms


def anticommutes(p_u, sigma_u):
    n_words = p_u.shape[0] // 2
    term1 = np.bitwise_count(p_u[:n_words] & sigma_u[n_words:]).astype(np.int32).sum()
    term2 = np.bitwise_count(p_u[n_words:] & sigma_u[:n_words]).astype(np.int32).sum()
    return (term1 - term2) % 2


def shift_case(data_dict):
    num_sites = len(data_dict["P"])
    all_new_cases = []
    for shift in range(num_sites):
        new_case = {
            "P": data_dict["P"][shift:] + data_dict["P"][:shift],
            "sigma": data_dict["sigma"][shift:] + data_dict["sigma"][:shift],
            "theta": data_dict["theta"],
            "a_p": data_dict["a_p"],
            "expected": {k[shift:] + k[:shift]: v for k, v in data_dict["expected"].items()},
        }
        all_new_cases.append(new_case)

    return all_new_cases


def shift_product_case(case_tuple):
    p1, p2, p3, phase = case_tuple
    permutations = []
    for idx in range(len(p1)):
        p1_shifted = p1[idx:] + p1[:idx]
        p2_shifted = p2[idx:] + p2[:idx]
        p3_shifted = p3[idx:] + p3[:idx]
        permutations.append((p1_shifted, p2_shifted, p3_shifted, phase))

    return permutations
