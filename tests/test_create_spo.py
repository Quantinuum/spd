import numpy as np
import pytest

import spd
from tests.helpers import make_backend, to_term_dict


BACKENDS = {
    "numpy": spd.numpy_backend,
    "jax": spd.jax_backend,
}


def test_create_spo_builds_z_basis_measurement_from_qubit_list(backend_name):
    spo = spd.create_spo([0, 2], system_size=3, backend_name=backend_name)

    terms = to_term_dict(backend_name, BACKENDS[backend_name], spo, n_qubits=3)
    assert terms == {"ZIZ": 1.0}


def test_create_spo_builds_general_pauli_operator_from_string_key_dict(backend_name):
    spo = spd.create_spo({"XIZ": 0.5, "ZZI": -1.0}, backend_name=backend_name)

    terms = to_term_dict(backend_name, BACKENDS[backend_name], spo, n_qubits=3)
    assert terms == {"XIZ": 0.5, "ZZI": -1.0}


def test_create_spo_requires_system_size_for_qubit_list():
    with pytest.raises(ValueError, match="system_size is required"):
        spd.create_spo([0, 2])


def test_create_spo_accepts_configured_backend_object():
    backend = make_backend("jax")
    spo = spd.create_spo({"Z": 1.0}, backend=backend)

    assert backend.is_spo_instance(spo)


def test_create_spo_rejects_invalid_backend_object():
    with pytest.raises(TypeError, match="backend must be a BackendAdapter"):
        spd.create_spo({"Z": 1.0}, backend="jax")


def test_create_spo_rejects_tuple_key_measurement_dict():
    with pytest.raises(ValueError, match="Tuple-key measurement dicts are no longer supported"):
        spd.create_spo({(0, 2): 1.0}, system_size=3)


def test_backend_create_initial_spo_rejects_tuple_key_measurement_dict():
    backend = make_backend("numpy")

    with pytest.raises(ValueError, match="Tuple-key measurement dicts are no longer supported"):
        backend.create_initial_spo({(0, 2): 1.0}, padded_system_size=32)
