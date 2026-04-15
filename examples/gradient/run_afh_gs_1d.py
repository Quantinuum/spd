import numpy as np
import scipy.optimize
import spd
import sys

import heisenberg_setup


# np.random.seed(0)


if __name__ == "__main__":
    backend_name = "jax"
    precision = "double"
    basis = "0"

    import spd.jax_backend as jax_backend
    # jax_backend.set_algorithm("search_update_merge")
    jax_backend.set_algorithm("stack_sort_merge")

    num_layers = int(sys.argv[1])
    niter = int(sys.argv[2])

    number_of_parameters = 4 * num_layers
    # system_size = 2 * num_layers + 2
    system_size = num_layers * 3 * 2 + 2
    full_H = False
    factor = system_size if full_H else 1

    base_filename = f"L_{system_size}_layers_{num_layers}_np_{number_of_parameters}"
    random_thetas = (np.random.rand(number_of_parameters) - 0.5) * 0.1
    stagger_signs = heisenberg_setup.gen_1d_stagger_signs(system_size)
    grad_multiplicities = heisenberg_setup.gen_afh_grad_multiplicities(
        num_layers, spatial_dim=1
    )

    ham_dict = heisenberg_setup.gen_1d_Hamiltonian_dict(system_size, full=full_H)
    trunc_val = 1e-3
    max_num_str = 1e6

    print(random_thetas)
    print(f"\n Truncation Value: {trunc_val} | max num str: {max_num_str}")

    history = []

    def get_f_g(thetas):
        circ = heisenberg_setup.gen_1d_AFH_ansatz_circuit(thetas, system_size)
        exp_val, final_spo = spd.run_pytket_circuit(
            circ,
            ham_dict,
            trunc_val,
            max_num_str=max_num_str,
            precision=precision,
            basis=basis,
            backend_name=backend_name,
        )
        raw_grads, _ = spd.run_pytket_circuit_backward(
            circ,
            final_spo,
            trunc_val,
            max_num_str=max_num_str,
            precision=precision,
            basis=basis,
            backend_name=backend_name,
        )
        grads = heisenberg_setup.combine_afh_parameter_grads(
            raw_grads, system_size, stagger_signs, grad_multiplicities
        )
        exp_val /= factor
        grads /= factor
        history.append(exp_val)
        print("step = ", len(history) - 1, "num_param", number_of_parameters)
        print(f"cost: {exp_val}, ||theta||: {np.linalg.norm(thetas)}, ||grad||: {np.linalg.norm(grads)}")
        return exp_val, grads

    initial_cost, initial_grads = get_f_g(random_thetas)
    print("\n Expectation Value:", initial_cost)
    print("\n SPD Computed Gradients:", initial_grads)

    if niter == 0:
        raise SystemExit(0)

    result = scipy.optimize.minimize(
        get_f_g,
        random_thetas,
        method="L-BFGS-B",
        jac=True,
        options={"disp": True, "gtol": 1e-6, "maxiter": niter},
    )
    print(result)
    print("params: ", result.x)
    np.savetxt(base_filename + "_params.txt", result.x)
    np.savetxt(base_filename + "_history.txt", history)
