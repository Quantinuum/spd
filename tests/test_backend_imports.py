"""Backend selection must not initialize an unused GPU allocator."""

import subprocess
import sys

import pytest


def test_native_import_does_not_load_jax():
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    subprocess.run([
        sys.executable, "-c",
        "import sys; import spd.triton_backend; assert 'jax' not in sys.modules",
    ], check=True)


def test_public_backend_attributes():
    import spd
    from spd import jax_backend, numpy_backend

    assert spd.jax_backend is jax_backend
    assert spd.numpy_backend is numpy_backend
    assert callable(spd.evolve)
    assert callable(spd.create_spo)
    assert "jax_backend" in spd.__all__
    with pytest.raises(AttributeError):
        spd.not_a_backend
