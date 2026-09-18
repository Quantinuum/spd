"""Resource bounds and reference calculations for the small example."""
import numpy as np
import pytest
from spd.ansatz import binary_mera_parameter_shape
from examples.gradient import run_1d_tfi_hva_tmera as example


@pytest.mark.parametrize('cap',[0,-1,True,1.5])
def test_invalid_term_cap(cap):
    with pytest.raises(ValueError,match='positive integer'):
        example.spd_value_gradient(np.zeros(binary_mera_parameter_shape(4)),4,4,.9,max_num_str=cap)


def test_finite_ring_ground_energy_formula_and_large_system():
    assert example.tfi_ground_energy(16,0.) == -16.
    assert example.tfi_ground_energy(16,1.) == pytest.approx(
        -2 / np.sin(np.pi / 32)
    )
    assert np.isfinite(example.tfi_ground_energy(2**18,.9))


@pytest.mark.parametrize('chi,physical_bottom', [(2, True), (4, True), (4, False)])
@pytest.mark.parametrize('rounds', [1, 2])
def test_local_cell_is_not_the_full_energy_density(chi, physical_bottom, rounds):
    """Spatially shared tensors do not give this finite MERA a small unit cell.

    Test every shared parameter, including those in coarser layers. Averaging
    all cells is exact; multiplying the first cell by their count is not.
    """
    n, g = 8, 1.1
    cell_size = 2 * (chi.bit_length() - 1)
    params = np.random.default_rng(7).normal(
        0, .15, binary_mera_parameter_shape(
            n, rounds, chi=chi, physical_bottom=physical_bottom,
        ),
    )
    state, tangent = example.exact_state_and_tangents(
        params, n, chi, physical_bottom=physical_bottom,
    )
    energy, gradient = example.exact_value_gradient(
        params, n, chi, g, physical_bottom=physical_bottom,
    )
    cell_values, cell_gradients = [], []
    for start in range(0, n, cell_size):
        h_state = -sum(
            example.pauli_action(state, (i, (i + 1) % n), 'Z')
            + g * example.pauli_action(state, (i,), 'X')
            for i in range(start, start + cell_size)
        ) / cell_size
        cell_values.append(np.vdot(state, h_state).real)
        cell_gradients.append(2 * (tangent.conj() @ h_state).real)
    assert abs(cell_values[0] - energy / n) > 1e-5
    assert np.max(np.abs(cell_gradients[0] - gradient / n)) > 1e-3
    np.testing.assert_allclose(np.mean(cell_values), energy / n, atol=1e-13)
    np.testing.assert_allclose(np.mean(cell_gradients, axis=0), gradient / n, atol=1e-13)


@pytest.mark.parametrize('chi', [2, 4])
def test_spd_local_cell_gradient_matches_dense_but_not_full_hamiltonian(chi):
    """Verify the failed unit-cell reduction using the actual SPD backward pass."""
    import spd
    from spd.ansatz import tfi_binary_mera_channels

    n, g = 8, 1.1
    cell_size = 2 * (chi.bit_length() - 1)
    params = np.random.default_rng(7).normal(
        0, .15, binary_mera_parameter_shape(n, chi=chi),
    )
    ansatz = tfi_binary_mera_channels(params, n, chi=chi)
    backend = spd.BackendAdapter.from_name('numpy', precision='double')
    # Include the bond crossing the cell's right edge, as in the proposal.
    terms = {}
    for i in range(cell_size):
        zz, x = ['I'] * n, ['I'] * n
        zz[i] = zz[(i + 1) % n] = 'Z'
        x[i] = 'X'
        terms[''.join(zz)] = -1.
        terms[''.join(x)] = -g
    observable = spd.create_spo(terms, backend=backend)
    final, forward = spd.evolve(
        observable, ansatz.circuit, 0., 4**n, backend=backend, progress=False,
    )
    _, gate_gradients, backward = spd.backpropagate(
        spd.init_gradient_spo(final, backend=backend), ansatz.circuit,
        0., 4**n, backend=backend, progress=False,
    )
    local_value = float(final.get_expectation_value()) / cell_size
    local_gradient = ansatz.parameter_gradients(gate_gradients) / cell_size
    state, tangent = example.exact_state_and_tangents(params, n, chi)
    h_state = -sum(
        example.pauli_action(state, (i, (i + 1) % n), 'Z')
        + g * example.pauli_action(state, (i,), 'X')
        for i in range(cell_size)
    ) / cell_size
    np.testing.assert_allclose(local_value, np.vdot(state, h_state).real, atol=1e-11)
    np.testing.assert_allclose(local_gradient, 2 * (tangent.conj() @ h_state).real, atol=1e-10)
    value, gradient, diagnostics = example.spd_value_gradient(params, n, chi, g)
    expected_value, expected_gradient = example.exact_value_gradient(params, n, chi, g)
    np.testing.assert_allclose(value, expected_value, atol=1e-11)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-10)
    assert abs(local_value - value / n) > 1e-5
    assert np.max(np.abs(local_gradient - gradient / n)) > 1e-3
    assert sum(forward['history']['num_str_truncated']) == 0
    assert sum(backward['history']['num_str_truncated']) == 0
    assert diagnostics['discarded_terms_sum'] == 0
    assert diagnostics['backward_discarded_terms_sum'] == 0
