"""Simple forward/backward example using the public SPD state APIs."""

import pickle
from pathlib import Path

import spd


if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent / "simple_test_circuit.pkl"
    with file_path.open("rb") as handle:
        circ = pickle.load(handle)

    backend = spd.BackendAdapter.from_name("jax", packbit=32, precision="single")
    backend.module.set_algorithm("stack_sort_merge")

    trunc_val = 3e-5
    max_num_str = int(1e6)

    initial_spo = spd.create_spo([0, 1], system_size=circ.n_qubits, backend=backend)
    final_spo, forward_info = spd.evolve(
        initial_spo,
        circ,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        backend=backend,
    )
    exp_val = final_spo.get_expectation_value(basis="0")

    initial_spgo = spd.init_gradient_spo(final_spo, basis="0", backend=backend)
    final_spgo, grads, backward_info = spd.backpropagate(
        initial_spgo,
        circ,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        backend=backend,
    )

    print("circuit file:", file_path.name)
    print("algorithm:", backend.module.get_algorithm())
    print("trunc_val:", trunc_val)
    print("expectation value:", exp_val)
    print("forward SPO size:", final_spo.get_size())
    print("forward truncated strings:", forward_info["sum_num_str_truncated"])
    print("backward SPGO size:", final_spgo.get_size())
    print("number of gradients:", len(grads))
    print("backward truncated strings:", backward_info["sum_num_str_truncated"])
