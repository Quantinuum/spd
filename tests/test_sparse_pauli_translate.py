import numpy as np
import pytest

from spd import jax_backend, numpy_backend
from tests.helpers import to_term_dict


BACKENDS = {
    "numpy": numpy_backend,
    "jax": jax_backend,
}


def _right_rotate(pauli_str: str, x: int) -> str:
    x_mod = x % len(pauli_str)
    if x_mod == 0:
        return pauli_str
    return pauli_str[-x_mod:] + pauli_str[:-x_mod]


def _full_pauli_str(backend_name, module, spo):
    if backend_name == "jax":
        return module.utils.uint_to_pauli_str(np.asarray(spo.xz_array[0]), 32)
    packed = next(iter(spo.keys()))
    return module.utils.uint_to_pauli_str(np.asarray(packed), 32)


@pytest.mark.parametrize("backend_name,module", [("numpy", numpy_backend), ("jax", jax_backend)])
def test_sparse_pauli_translate_shifts_physical_sites_only(backend_name, module):
    module.utils.set_packbit(32)
    if backend_name == "jax":
        module.set_precision("single")

    physical = "XYZIIZ"
    spo = module.create_op({physical: 1.5})

    translated = spo.translate(2, system_size=len(physical))

    assert isinstance(translated, module.SparsePauliOp)
    assert to_term_dict(backend_name, module, translated, n_qubits=len(physical)) == {
        _right_rotate(physical, 2): 1.5
    }
    assert _full_pauli_str(backend_name, module, translated) == _right_rotate(physical, 2) + "I" * (32 - len(physical))
    assert _full_pauli_str(backend_name, module, spo) == physical + "I" * (32 - len(physical))


@pytest.mark.parametrize("backend_name,module", [("numpy", numpy_backend), ("jax", jax_backend)])
def test_sparse_pauli_translate_wraparound_and_negative_shifts(backend_name, module):
    module.utils.set_packbit(32)
    if backend_name == "jax":
        module.set_precision("single")

    physical = "XZIYI"
    spo = module.create_op({physical: 2.0})

    wrapped = spo.translate(physical.__len__() + 2, system_size=len(physical))
    negative = spo.translate(-1, system_size=len(physical))
    zero_shift = spo.translate(0, system_size=len(physical))
    unchanged = spo.translate(len(physical), system_size=len(physical))

    assert to_term_dict(backend_name, module, wrapped, n_qubits=len(physical)) == {
        _right_rotate(physical, 2): 2.0
    }
    assert to_term_dict(backend_name, module, negative, n_qubits=len(physical)) == {
        _right_rotate(physical, -1): 2.0
    }
    assert to_term_dict(backend_name, module, zero_shift, n_qubits=len(physical)) == {
        physical: 2.0
    }
    assert to_term_dict(backend_name, module, unchanged, n_qubits=len(physical)) == {
        physical: 2.0
    }


@pytest.mark.parametrize("backend_name,module", [("numpy", numpy_backend), ("jax", jax_backend)])
def test_sparse_pauli_translate_preserves_coefficients_for_multiple_terms(backend_name, module):
    module.utils.set_packbit(32)
    if backend_name == "jax":
        module.set_precision("single")

    physical_terms = {
        "XYZIII": 1.0,
        "IYZXII": -0.5,
        "ZZXYYI": 0.25,
    }
    spo = module.create_op(physical_terms)

    translated = spo.translate(3, system_size=6)

    expected = {_right_rotate(term, 3): coeff for term, coeff in physical_terms.items()}
    assert to_term_dict(backend_name, module, translated, n_qubits=6) == pytest.approx(expected, abs=1e-6)
    assert to_term_dict(backend_name, module, spo, n_qubits=6) == pytest.approx(physical_terms, abs=1e-6)


def test_sparse_pauli_translate_backend_parity():
    physical_terms = {
        "XYZZII": 1.25,
        "IYXIIZ": -0.75,
        "ZZIIII": 0.5,
    }

    numpy_backend.utils.set_packbit(32)
    jax_backend.utils.set_packbit(32)
    jax_backend.set_precision("single")

    spo_numpy = numpy_backend.create_op(physical_terms)
    spo_jax = jax_backend.create_op(physical_terms)

    translated_numpy = spo_numpy.translate(4, system_size=6)
    translated_jax = spo_jax.translate(4, system_size=6)

    assert to_term_dict("numpy", numpy_backend, translated_numpy, n_qubits=6) == pytest.approx(
        to_term_dict("jax", jax_backend, translated_jax, n_qubits=6),
        abs=1e-6,
    )


@pytest.mark.parametrize("backend_name,module", [("numpy", numpy_backend), ("jax", jax_backend)])
def test_sparse_pauli_translate_rejects_invalid_system_size(backend_name, module):
    module.utils.set_packbit(32)
    if backend_name == "jax":
        module.set_precision("single")

    spo = module.create_op({"XYZ": 1.0})

    with pytest.raises(ValueError, match="at least 1"):
        spo.translate(1, system_size=0)

    with pytest.raises(ValueError, match="exceeds the represented site capacity"):
        spo.translate(1, system_size=33)
