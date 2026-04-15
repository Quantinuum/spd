import numpy as np
import scipy.optimize
import spd
import sys

import heisenberg_setup


# np.random.seed(43)


if __name__ == "__main__":
    backend_name = "jax"
    precision = "double"
    basis = "0"

    num_layers = int(sys.argv[1])
    niter = int(sys.argv[2])

    number_of_parameters = 4 * num_layers
    system_size_x = 4
    # system_size_x = num_layers * 3 * 2 + 2
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
            lambda_ose=lambda_ose,
        )
        grads = heisenberg_setup.combine_afh_parameter_grads(
            raw_grads, system_size, stagger_signs, grad_multiplicities
        )
        OSE = final_spo.get_OSE()
        cost = exp_val + lambda_ose * OSE

        exp_val /= factor
        grads /= factor
        history.append(exp_val)

        print("step = ", len(history) - 1, "num_param", number_of_parameters)
        print(f"E = {exp_val/4}, OSE = {OSE}, cost = {cost}")
        print(f"||theta||: {np.linalg.norm(thetas)}, ||grad||: {np.linalg.norm(grads)}")
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
