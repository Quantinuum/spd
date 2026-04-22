import numpy as np

from spd.backend_adapter import BackendAdapter


def u32(module, pstr):
    return np.asarray(module.utils.pauli_str_to_uint32(pstr))


def assert_phase_close(actual, expected, atol=1e-6):
    assert np.isclose(np.real(actual), np.real(expected), atol=atol)
    assert np.isclose(np.imag(actual), np.imag(expected), atol=atol)


def assert_step_info_close(actual, expected, atol=1e-6):
    assert actual["num_str_truncated"] == expected["num_str_truncated"]
    assert np.isclose(actual["truncated_l1_norm"], expected["truncated_l1_norm"], atol=atol)
    assert np.isclose(actual["truncated_l2_norm"], expected["truncated_l2_norm"], atol=atol)


def assert_info_consistent(info, expected_steps=None, atol=1e-6):
    history = info["history"]
    lengths = {len(history[key]) for key in history}
    assert len(lengths) == 1
    num_steps = lengths.pop()
    if expected_steps is not None:
        assert num_steps == expected_steps
    assert info["num_steps_tracked"] == num_steps
    assert info["sum_num_str_truncated"] == sum(history["num_str_truncated"])
    assert np.isclose(
        info["sum_truncated_l1_norm"],
        sum(history["truncated_l1_norm"]),
        atol=atol,
    )
    assert np.isclose(
        info["sum_truncated_l2_norm"],
        sum(history["truncated_l2_norm"]),
        atol=atol,
    )
    assert np.isclose(
        info["total_truncated_l2_norm"],
        np.sqrt(sum(value * value for value in history["truncated_l2_norm"])),
        atol=atol,
    )


def to_term_dict(backend_name, module, spo, n_qubits=8):
    terms = {}

    if backend_name == "jax":
        xz_rows = np.asarray(spo.xz_array)
        c_vals = np.asarray(spo.c_array)
        for xz, c in zip(xz_rows, c_vals):
            if np.isclose(np.real(c), 0.0):
                continue
            pstr = module.utils.uint32_to_pauli_str(np.asarray(xz), 32)[:n_qubits]
            terms[pstr] = terms.get(pstr, 0.0) + float(np.real(c))
        return terms

    for xz_key, c in spo.items():
        pstr = module.utils.uint32_to_pauli_str(np.asarray(xz_key), 32)[:n_qubits]
        terms[pstr] = terms.get(pstr, 0.0) + float(np.real(c))
    return terms


def to_grad_term_dict(backend_name, module, spgo, n_qubits=8):
    terms = {}

    if backend_name == "jax":
        xz_rows = np.asarray(spgo.xz_array)
        c_vals = np.asarray(spgo.c_array)
        grad_vals = np.asarray(spgo.grad_c_array)
        for xz, c, grad in zip(xz_rows, c_vals, grad_vals):
            if np.isclose(np.real(c), 0.0) and np.isclose(np.real(grad), 0.0):
                continue
            pstr = module.utils.uint32_to_pauli_str(np.asarray(xz), 32)[:n_qubits]
            terms[pstr] = (
                terms.get(pstr, (0.0, 0.0))[0] + float(np.real(c)),
                terms.get(pstr, (0.0, 0.0))[1] + float(np.real(grad)),
            )
        return terms

    for xz_key, value_grad in spgo.items():
        coeff, grad = value_grad
        pstr = module.utils.uint32_to_pauli_str(np.asarray(xz_key), 32)[:n_qubits]
        terms[pstr] = (
            terms.get(pstr, (0.0, 0.0))[0] + float(np.real(coeff)),
            terms.get(pstr, (0.0, 0.0))[1] + float(np.real(grad)),
        )
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


def padded_system_size(num_qubits, packbit=32):
    return packbit * ((num_qubits + packbit - 1) // packbit)


def make_backend(backend_name, precision="single"):
    return BackendAdapter.from_name(backend_name, packbit=32, precision=precision)


def make_initial_spo(backend_name, measure_qubits_data, num_qubits, precision="single"):
    backend = make_backend(backend_name, precision=precision)
    return backend.create_initial_spo(
        measure_qubits_data,
        padded_system_size(num_qubits, backend.packbit),
    )
