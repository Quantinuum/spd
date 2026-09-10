"""Backend selection must not initialize an unused GPU allocator."""

import subprocess
import sys
import os

import pytest


@pytest.mark.parametrize("config_name,config", [
    (None, None),
    ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False"),
    ("PYTORCH_CUDA_ALLOC_CONF", "backend:native"),
    ("PYTORCH_ALLOC_CONF", "expandable_segments:False"),
])
def test_triton_allocator_default_respects_environment(config_name, config):
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    env = os.environ.copy()
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(name, None)
    if config_name is not None:
        env[config_name] = config
    expected = ["expandable_segments:True"] if config_name is None else []
    subprocess.run([sys.executable, "-c", f"""
import torch
calls = []
torch.cuda.memory._set_allocator_settings = calls.append
import spd
assert calls == []
import spd.triton_backend
assert calls == {expected!r}
assert not torch.cuda.is_initialized()
"""], env=env, check=True)


@pytest.mark.parametrize("initialize_first", [False, True])
def test_triton_default_uses_expandable_segments(initialize_first):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    env = os.environ.copy()
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(name, None)
    subprocess.run([sys.executable, "-c", f"""
import torch
if {initialize_first!r}:
    existing = torch.empty(1024, device='cuda')
import spd.triton_backend
if not {initialize_first!r}:
    assert not torch.cuda.is_initialized()
state = torch.empty(4 * 1024 * 1024, device='cuda')
assert any(segment['is_expandable'] for segment in torch.cuda.memory_snapshot())
"""], env=env, check=True)


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
