import numpy as np
import scipy.optimize
import spd
import sys

import heisenberg_setup


# np.random.seed(43)


if __name__ == "__main__":
    precision = "double"
    basis = "0"
    backend = spd.BackendAdapter.from_name("jax", packbit=32, precision=precision)
    backend.module.set_algorithm("stack_sort_merge")

    num_layers = int(sys.argv[1])
    niter = int(sys.argv[2])

    number_of_parameters = 4 * num_layers
    # The light cone spread out by +2 in one spatial direction per gate type.
    # There are XX, YY, ZZ gates.
    # So per layer it increases + 6
    system_size_x = num_layers * 6 + 2
    system_size_y = system_size_x
    system_size = system_size_x * system_size_y
    full_H = False
    factor = system_size if full_H else 1

    base_filename = (
        f"Lx_{system_size_x}_Ly_{system_size_y}_layers_{num_layers}_np_{number_of_parameters}"
    )
    random_thetas = (np.random.rand(number_of_parameters) - 0.5) * 0.1
    # random_thetas[:4] += np.array([-7.519e-02, -3.286e-02, -1.151e-02,  2.213e-01])

    stagger_signs = heisenberg_setup.gen_2d_stagger_signs(system_size_x, system_size_y)
    grad_multiplicities = heisenberg_setup.gen_afh_grad_multiplicities(
        num_layers, spatial_dim=2
    )

    ham_dict = heisenberg_setup.gen_2d_Hamiltonian_dict(
        system_size_x, system_size_y, full=full_H
    )
    E_weight = np.sqrt(np.sum(np.array(list(ham_dict.values())) ** 2))
    trunc_val = 1e-4
    max_num_str = 1e6
    lambda_ose = 1e-2

    print(random_thetas)
    print(f"\n Truncation Value: {trunc_val} | max num str: {max_num_str}")

    history = []

    def get_f_g(thetas):
        circ = heisenberg_setup.gen_2d_AFH_ansatz_circuit(
            thetas, system_size_x, system_size_y
        )
        initial_spo = spd.create_spo(ham_dict, backend=backend)
        final_spo, forward_info = spd.evolve(
            initial_spo,
            circ,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
        )
        E_err_estimate = forward_info["total_truncated_l2_norm"] * E_weight
        exp_val = final_spo.get_expectation_value(basis=basis)
        initial_spgo = spd.init_gradient_spo(
            final_spo,
            basis=basis,
            lambda_ose=lambda_ose,
            backend=backend,
        )
        _, raw_grads, backward_info = spd.backpropagate(
            initial_spgo,
            circ,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
        )
        grads = heisenberg_setup.combine_afh_parameter_grads(
            raw_grads, system_size, stagger_signs, grad_multiplicities
        )
        OSE = final_spo.get_OSE()
        exp_val /= factor
        E_err_estimate /= factor
        grads /= factor
        cost = exp_val + lambda_ose * OSE
        history.append(cost)

        print("step = ", len(history) - 1, "num_param", number_of_parameters)
        print(f"cost: {cost}, <E>: {exp_val} ± {E_err_estimate}, OSE: {OSE}")
        print(f"||theta||: {np.linalg.norm(thetas)}, ||grad||: {np.linalg.norm(grads)}")
        return cost, grads

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
