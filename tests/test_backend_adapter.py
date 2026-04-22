import numpy as np
import pytest

from spd.backend_adapter import BackendAdapter
from spd.circuit_ir import PauliRotation, SingleQubitClifford
from tests.helpers import assert_step_info_close, to_term_dict


def test_backend_adapter_applies_rotation_and_clifford(backend):
    backend_name, module = backend
    adapter = BackendAdapter.from_name(backend_name, packbit=32)

    spo = module.create_op({"XIII": 1.0})
    spo, _, _, step_info = adapter.apply_forward(
        spo,
        PauliRotation(gate_name="OpType.Rz", pauli="ZIII" + "I" * 28, theta=np.pi / 3),
        trunc_val=1e-12,
        max_num_str=1000,
    )
    terms = to_term_dict(backend_name, module, spo, n_qubits=4)
    assert np.isclose(terms["XIII"], 0.5, atol=1e-6)
    assert np.isclose(terms["YIII"], -np.sqrt(3) / 2, atol=1e-6)
    assert_step_info_close(step_info, {"num_str_truncated": 0, "truncated_l1_norm": 0.0, "truncated_l2_norm": 0.0})

    spo = module.create_op({"XIII": 1.0})
    spo, num_string, _, step_info = adapter.apply_forward(
        spo,
        SingleQubitClifford(gate_name="OpType.H", qubit=0),
        trunc_val=1e-12,
        max_num_str=1000,
    )
    terms = to_term_dict(backend_name, module, spo, n_qubits=4)
    assert terms == {"ZIII": 1.0}
    assert num_string == spo.get_size()
    assert_step_info_close(step_info, {"num_str_truncated": 0, "truncated_l1_norm": 0.0, "truncated_l2_norm": 0.0})


@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_backend_adapter_backward_clifford_returns_three_tuple(backend_name):
    adapter = BackendAdapter.from_name(backend_name, packbit=32)
    spo = adapter.module.create_op({"ZIII": 1.0})
    spgo = adapter.init_gradient_spo(spo, basis="0")

    with pytest.warns(UserWarning, match="not fully supported"):
        next_state, num_string, grad_i, step_info = adapter.apply_backward(
            spgo,
            SingleQubitClifford(gate_name="OpType.X", qubit=0),
            trunc_val=1e-12,
            max_num_str=1000,
    )

    assert adapter.is_spgo_instance(next_state)
    assert num_string == next_state.get_size()
    assert grad_i is None
    assert_step_info_close(step_info, {"num_str_truncated": 0, "truncated_l1_norm": 0.0, "truncated_l2_norm": 0.0})
