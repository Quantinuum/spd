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
    module.set_precision("single")
    if request.param == "jax":
        module.set_algorithm("search_update_merge")
    return request.param, module


@pytest.fixture(params=["numpy", "jax"], ids=["numpy", "jax"])
def backend_name(request):
    return request.param


@pytest.fixture
def jax_algorithm(request):
    algorithm = getattr(request, "param", "search_update_merge")
    previous_algorithm = jax_backend.get_algorithm()
    jax_backend.set_algorithm(algorithm)
    try:
        yield algorithm
    finally:
        jax_backend.set_algorithm(previous_algorithm)


@pytest.fixture
def jax_forward_algorithm(request):
    algorithm = getattr(request, "param", "search_update_merge")
    previous_algorithm = jax_backend.get_algorithm()
    jax_backend.set_algorithm(algorithm)
    try:
        yield algorithm
    finally:
        jax_backend.set_algorithm(previous_algorithm)


@pytest.fixture(params=["numpy", "jax", "triton"])
def all_backend(request):
    if request.param == "triton":
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("requires NVIDIA GPU")
        from spd import triton_backend
        module = triton_backend
    else:
        module = BACKENDS[request.param]
    module.utils.set_packbit(32)
    module.set_precision("single")
    if request.param == "jax":
        module.set_algorithm("search_update_merge")
    return request.param, module
