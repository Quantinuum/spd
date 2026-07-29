import numpy as np
import pytest

from spd import jax_backend, numpy_backend


@pytest.mark.parametrize("backend_name,module", [("numpy", numpy_backend), ("jax", jax_backend)])
def test_sparse_pauli_op_string_rendering_real_coefficients(backend_name, module):
    module.utils.set_packbit(32)
    if backend_name == "jax":
        module.set_precision("single")

    spo = module.create_op({
        "XIZY": 1.5,
        "ZZ": -0.25,
        "IIII": 2.0,
    })

    rendered = str(spo)

    assert "SparsePauliOp[" in rendered
    assert "XIZY" in rendered
    assert "ZZ" in rendered
    assert "IIII" in rendered
    assert "=> 1.5" in rendered
    assert "=> -0.25" in rendered
    assert "=> 2.0" in rendered


@pytest.mark.parametrize("backend_name,module", [("numpy", numpy_backend), ("jax", jax_backend)])
@pytest.mark.parametrize(
    "packbit,pauli_str",
    [
        (8, "XIIIIIIIXIIIIIII"),
        (32, "X" + "I" * 31 + "X" + "I" * 31),
    ],
)
def test_sparse_pauli_op_string_rendering_multiple_words(
    backend_name,
    module,
    packbit,
    pauli_str,
):
    module.utils.set_packbit(packbit)
    spo = module.create_op({pauli_str: 1.0})

    assert str(spo) == f"SparsePauliOp[\n  {pauli_str} => 1.0\n]"


@pytest.mark.parametrize("backend_name,module", [("numpy", numpy_backend), ("jax", jax_backend)])
def test_sparse_pauli_gradient_op_string_rendering_multiple_words(backend_name, module):
    module.utils.set_packbit(32)
    pauli_str = "X" + "I" * 31 + "X" + "I" * 31
    spo = module.create_op({pauli_str: 1.0})

    if backend_name == "jax":
        spgo = module.SparsePauliGradientOp(
            spo.xz_array,
            spo.c_array,
            np.asarray([0.5]),
        )
    else:
        spgo = module.SparsePauliGradientOp()
        packed = next(iter(spo))
        spgo[packed] = (1.0, 0.5)

    assert str(spgo) == (
        "SparsePauliGradientOp[\n"
        f"  {pauli_str} => coeff=1.0, grad=0.5\n"
        "]"
    )


@pytest.mark.parametrize("module", [numpy_backend, jax_backend])
@pytest.mark.parametrize(
    "packbit,pauli_str",
    [
        (8, "XIIIIIIIXIIIIIII"),
        (32, "X" + "I" * 31 + "X" + "I" * 31),
    ],
)
def test_pauli_string_pack_unpack_round_trip_multiple_words(module, packbit, pauli_str):
    module.utils.set_packbit(packbit)
    packed = module.utils.pauli_str_to_uint(pauli_str)

    assert module.utils.uint_to_pauli_str(packed, len(pauli_str)) == pauli_str


def test_jax_sparse_pauli_op_rejects_complex_coefficients():
    jax_backend.utils.set_packbit(32)
    jax_backend.set_precision("single")

    with pytest.raises(ValueError, match="real-valued"):
        jax_backend.create_op({
            "ZZ": -0.25j,
        })


def test_sparse_pauli_op_weight_distribution_and_counts(backend):
    _, module = backend
    spo = module.create_op({
        "IIII": 1.0,
        "XIII": 2.0,
        "YZZI": 3.0,
        "XYZX": 4.0,
    })

    distribution = spo.get_pauli_weight_distribution()
    counts = spo.get_pauli_weight_counts()

    assert distribution == {
        0: 1.0,
        1: 4.0,
        3: 9.0,
        4: 16.0,
    }
    assert counts == {
        0: 1,
        1: 1,
        3: 1,
        4: 1,
    }
    assert spo.get_pauli_weight_count() == counts
    assert spo.get_Pauli_weight_distribution() == distribution
