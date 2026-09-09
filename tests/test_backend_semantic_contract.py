"""Boundary cases defining the Triton parity work's numerical contract.

Run against references now; extend the reference-only cases to Triton as each
capability lands. One strict xfail records a pre-existing JAX diagnostics bug.
"""

import numpy as np
import pytest

from spd import jax_backend, numpy_backend


REFERENCES = ["numpy", "jax-stack", "jax-search"]


@pytest.fixture
def engine(request):
    name = request.param
    if name == "triton":
        torch = pytest.importorskip("torch")
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("requires NVIDIA GPU")
        from spd import triton_backend
        yield name, triton_backend
        return
    module = numpy_backend if name == "numpy" else jax_backend
    previous_precision = module.utils.get_precision()
    previous_algorithm = jax_backend.get_algorithm()
    module.utils.set_packbit(32)
    module.set_precision("double")
    if name.startswith("jax"):
        module.set_algorithm("stack_sort_merge" if name == "jax-stack" else "search_update_merge")
    try:
        yield name, module
    finally:
        module.set_precision(previous_precision)
        jax_backend.set_algorithm(previous_algorithm)


def forward(engine, terms, cutoff, cap):
    name, module = engine
    return module.conjugate_pauli_rot_forward(module.create_op(terms),
                                               module.utils.pauli_str_to_uint("I"), 0., cutoff, cap)


def coefficients(engine, state):
    if engine[0] == "numpy":
        return np.array(list(state.values()))
    if engine[0] == "triton":
        return state.to_host()[1]
    values = np.asarray(state.c_array)
    return values[values != 0]  # Ignore reference storage padding.


@pytest.mark.parametrize("engine", REFERENCES + ["triton"], indirect=True)
def test_forward_cutoff_equality_policy(engine):
    state, count, _ = forward(engine, {"X": .5}, .5, 8)
    expected = 1 if engine[0] == "numpy" else 0
    assert count == expected
    assert len(coefficients(engine, state)) == expected


@pytest.mark.parametrize("engine", REFERENCES + ["triton"], indirect=True)
def test_direct_size_cap_is_exact_even_when_not_power_of_two(engine):
    state, count, _ = forward(engine, {"I": .8, "X": .6, "Y": .4, "Z": .2}, 0., 3)
    assert count == 3
    np.testing.assert_allclose(sorted(coefficients(engine, state)), [.4, .6, .8])


@pytest.mark.parametrize("engine", REFERENCES, indirect=True)
def test_public_runner_cap_policy_differs_from_direct_kernel(engine):
    import spd
    from spd.circuit_ir import CircuitIR, PauliRotation

    name, module = engine
    adapter = spd.BackendAdapter.from_name("numpy" if name == "numpy" else "jax", precision="double")
    initial = module.create_op({"I": .8, "X": .6, "Y": .4, "Z": .2})
    circuit = CircuitIR(32, (PauliRotation("identity", "I" * 32, 0.),))
    final, _ = spd.evolve(initial, circuit, 0., max_num_str=3, backend=adapter)
    # Future Triton runner uses the JAX policy approved by the user.
    assert len(coefficients(engine, final)) == (3 if name == "numpy" else 4)


@pytest.mark.parametrize("engine", REFERENCES + ["triton"], indirect=True)
def test_size_cap_ties_require_magnitudes_not_specific_keys(engine):
    state, count, _ = forward(engine, {"I": 1., "X": -1., "Y": 1., "Z": -1.}, 0., 2)
    assert count == 2
    np.testing.assert_array_equal(abs(coefficients(engine, state)), [1., 1.])


@pytest.mark.parametrize("engine", [
    "jax-search", "triton",
    pytest.param("jax-stack", marks=pytest.mark.xfail(strict=True, reason=
        "Existing stack_sort_merge removes cutoff-equal rows but omits their discarded norms")),
], indirect=True)
def test_discarded_norms_include_cutoff_equal_coefficients(engine):
    _, count, info = forward(engine, {"X": .5}, .5, 8)
    assert count == 0
    assert info == pytest.approx({"num_str_truncated": 1, "truncated_l1_norm": .5,
                                 "truncated_l2_norm": .5})


@pytest.mark.parametrize("engine", REFERENCES + ["triton"], indirect=True)
def test_cap_diagnostics_count_each_removed_nonzero_term_once(engine):
    _, count, info = forward(engine, {"I": .8, "X": .6, "Z": .2}, 0., 2)
    assert count == 2
    assert info == pytest.approx({"num_str_truncated": 1, "truncated_l1_norm": .2,
                                 "truncated_l2_norm": .2})


@pytest.mark.parametrize("engine", REFERENCES + ["triton"], indirect=True)
@pytest.mark.parametrize("cutoff,expected_count", [(0., 2), (.01, 0)])
def test_backward_gradient_only_support_follows_coefficient_cutoff(engine, cutoff, expected_count):
    name, module = engine
    key = np.asarray(module.utils.pauli_str_to_uint("X"))
    if name == "numpy":
        initial = module.SparsePauliGradientOp()
        initial[tuple(key)] = (0., 1.)
    else:
        initial = module.SparsePauliGradientOp(key[None, :], np.array([0.]), np.array([1.]))
    state, count, grad, _ = module.conjugate_pauli_rot_backward(
        initial, module.utils.pauli_str_to_uint("Z"), .37, cutoff, 8)
    assert count == expected_count
    assert grad == 0.
    if name == "numpy":
        gradients = np.array([g for c, g in state.values()])
    elif name == "triton":
        gradients = state.to_host()[2]
    else:
        gradients = np.asarray(state.grad_c_array)
        gradients = gradients[gradients != 0]
    if expected_count:
        np.testing.assert_allclose(sorted(gradients), sorted([np.cos(.37), np.sin(.37)]), atol=1e-13)
    else:
        assert len(gradients) == 0


@pytest.mark.parametrize("engine", REFERENCES, indirect=True)
def test_l2_initializer_support_and_union_are_distinct(engine):
    name, module = engine
    initial = module.create_op({"X": 1.})
    target = module.create_op({"Z": 2.})
    restricted = module.init_gradient_from_l2_difference(initial, target)
    union = module.init_gradient_from_l2_difference_union(initial, target)

    def pairs(state):
        if name == "numpy":
            return sorted(state.values())
        return sorted((c, g) for c, g in zip(np.asarray(state.c_array), np.asarray(state.grad_c_array))
                      if c != 0 or g != 0)

    assert pairs(restricted) == [(1., 2.)]
    assert pairs(union) == [(0., -4.), (1., 2.)]
