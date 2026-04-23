"""Run a simple `pytket` circuit with a reusable configured backend."""

from pytket.circuit import Circuit

import spd

if __name__ == "__main__":
    circ = Circuit(1)
    circ.Ry(0.25, 0)

    backend = spd.BackendAdapter.from_name("jax", packbit=32, precision="single")
    backend.module.set_algorithm("stack_sort_merge")
    initial_spo = backend.create_initial_spo({"Z": 1.0})

    final_spo, forward_info = spd.evolve(initial_spo, circ, 1e-12, int(1e6), backend=backend)
    exp_val = final_spo.get_expectation_value()

    initial_spgo = spd.init_gradient_spo(
        final_spo,
        basis="0",
        backend=backend,
    )
    final_spgo, grads, backward_info = spd.backpropagate(
        initial_spgo,
        circ,
        1e-12,
        int(1e6),
        backend=backend,
    )

    print("backend:", backend.name)
    print("algorithm:", backend.module.get_algorithm())
    print("expectation value:", exp_val)
    print("forward SPO size:", final_spo.get_size())
    print("forward truncated strings:", forward_info["sum_num_str_truncated"])
    print("gradients:", grads)
    print("backward SPGO size:", final_spgo.get_size())
    print("backward truncated strings:", backward_info["sum_num_str_truncated"])
