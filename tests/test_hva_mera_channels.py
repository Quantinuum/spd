"""Ansatz/channel integration using the static-channel API merged into main.

Projection and checkpoint internals are covered by the channel infrastructure tests.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import spd

from spd.ansatz import (
    binary_mera_parameter_shape, binary_mera_qubit_initializations,
    tfi_binary_mera, tfi_binary_mera_channels,
)
from examples.gradient.run_1d_tfi_hva_tmera import exact_value_gradient, tfi_terms


@pytest.mark.parametrize('chi', [2,4,8])
def test_adapter_metadata_order_units_and_activity(chi):
    from spd.circuit_ir import PauliRotation, CreateZero
    from spd.pytket_frontend import parse_pytket_circuit
    params = np.random.default_rng(5).normal(0,.1,binary_mera_parameter_shape(8,2,chi=chi))
    reference = tfi_binary_mera(params,8,2,chi=chi)
    channel = tfi_binary_mera_channels(params,8,2,chi=chi)
    assert channel.circuit.initial_active_qubits == []
    assert channel.circuit.final_active_qubits == list(range(8))
    assert sorted(op.qubit for op in channel.circuit.operations if isinstance(op,CreateZero)) == list(range(8))
    lowered = parse_pytket_circuit(reference.circuit)
    assert tuple(op for op in channel.circuit.operations if not isinstance(op,CreateZero)) == lowered.operations
    np.testing.assert_array_equal(channel.gate_parameter_indices,reference.gate_parameter_indices)
    np.testing.assert_allclose(channel.gate_parameter_factors,np.pi*reference.gate_parameter_factors)
    raw = np.arange(sum(isinstance(op,PauliRotation) for op in lowered.operations))*.13
    np.testing.assert_allclose(channel.parameter_gradients(raw),reference.parameter_gradients(raw))


def test_chi4_channel_energy_gradient_and_actual_initialization_widths():
    from spd.circuit_ir import CreateZero
    n, steps = 4, 2
    params = np.random.default_rng(7).normal(0,.05,binary_mera_parameter_shape(n,steps))
    vc = tfi_binary_mera_channels(params,n,steps)
    observable = spd.create_spo(tfi_terms(n,.9),backend_name='numpy',precision='double')
    final,info = spd.evolve(observable,vc.circuit,0.,4**n,progress=False)
    value = float(final.get_expectation_value())
    assert final.active_qubits == []
    # Grouped execution still reports one width per original IR operation.
    assert len(info["active_widths"]) == len(vc.circuit.operations) + 1
    starts = {indices[0] for _,indices in binary_mera_qubit_initializations(n,steps)}
    initialization_widths = [info['active_widths'][0]]
    for step,op in enumerate(reversed(vc.circuit.operations),1):
        if isinstance(op,CreateZero) and op.qubit in starts:
            initialization_widths.append(info['active_widths'][step])
    assert initialization_widths == [4,2,0]
    _,raw,_ = spd.backpropagate(spd.init_gradient_spo(final),vc.circuit,0.,4**n,progress=False)
    expected,gradient = exact_value_gradient(params,n,4,.9)
    assert value == pytest.approx(expected,abs=1e-11)
    np.testing.assert_allclose(vc.parameter_gradients(raw),gradient,atol=1e-10)
    assert sum(info['history']['num_str_truncated']) == 0


@pytest.mark.parametrize('backend', ['numpy', 'triton'])
def test_example_rejects_unavailable_capability_before_optimization(backend,monkeypatch,capsys):
    from examples.gradient import run_1d_tfi_hva_tmera as example
    def require():
        raise NotImplementedError(f'test backend {backend} lacks channels')
    monkeypatch.setattr(spd.BackendAdapter,'from_name',staticmethod(
        lambda *args,**kwargs: SimpleNamespace(require_channel_support=require)))
    monkeypatch.setattr(sys,'argv',['run_1d_tfi_hva_tmera','8','4','30','--backend',backend])
    def unexpected(*args,**kwargs):
        pytest.fail('Optimization started before capability validation')
    monkeypatch.setattr(example,'exact_value_gradient',unexpected)
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2
    assert f'test backend {backend} lacks channels' in capsys.readouterr().err


def test_channel_example_uses_native_numpy_backend():
    from examples.gradient.run_1d_tfi_hva_tmera import spd_value_gradient
    params = np.random.default_rng(19).normal(0,.1,binary_mera_parameter_shape(4))
    value, gradient, diagnostics = spd_value_gradient(params,4,4,.9,channels=True)
    expected, expected_gradient = exact_value_gradient(params,4,4,.9)
    assert diagnostics['backend'] == 'numpy'
    assert diagnostics['qubit_initialization_widths'] == [4,2,0]
    assert value == pytest.approx(expected,abs=1e-11)
    np.testing.assert_allclose(gradient,expected_gradient,atol=1e-10)
