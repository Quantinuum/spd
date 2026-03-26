import pytest

from spd import jax_backend, numpy_backend


BACKENDS = {
    "jax": jax_backend,
    "numpy": numpy_backend,
}


@pytest.fixture(params=["numpy", "jax"], ids=["numpy", "jax"])
def backend(request):
    module = BACKENDS[request.param]
    module.utils.set_packbit(32)
    return request.param, module


@pytest.fixture(params=["numpy", "jax"], ids=["numpy", "jax"])
def backend_name(request):
    return request.param
