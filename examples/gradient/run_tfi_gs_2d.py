import numpy as np
import scipy.optimize
import spd
import sys

import tfi_setup


def combine_grads(grads, number_of_parameters, system_size):
    combined = []
    for i in range(number_of_parameters // 2):
        combined.append(np.array(grads[3 * i * system_size:(3 * i + 2) * system_size]).sum() * np.pi)
        combined.append(np.array(grads[(3 * i + 2) * system_size:(3 * i + 3) * system_size]).sum() * np.pi)
    return np.array(combined)


if __name__ == "__main__":
    precision = "double"
    backend = spd.BackendAdapter.from_name("jax", packbit=32, precision=precision)
    backend.module.set_algorithm("stack_sort_merge")

    method = "basinhopping"
    number_of_parameters = int(sys.argv[1])
    system_size_x = number_of_parameters + 3
    system_size_y = system_size_x
    system_size = system_size_x * system_size_y
    basis = "+"
    g = 3.1
    niter = int(sys.argv[2])

    random_thetas = (np.random.rand(number_of_parameters) - 0.5) * 0.1
    ham_dict = tfi_setup.gen_2d_Hamiltonian_dict(system_size_x, system_size_y, g=g)

    trunc_val = 1e-6
    max_num_str = int(1e6)
    print("\n Truncation Value:", trunc_val)

    history = []

    def get_f_g(thetas):
        circ = tfi_setup.gen_2d_TFI_ansatz_circuit(thetas, system_size_x, system_size_y)
        initial_spo = spd.create_spo(ham_dict, backend=backend)
        final_spo = spd.evolve(
            initial_spo,
            circ,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
        )
        exp_val = final_spo.get_expectation_value(basis=basis)
        initial_spgo = spd.init_gradient_spo(final_spo, basis=basis, backend=backend)
        _, raw_grads = spd.backpropagate(
            initial_spgo,
            circ,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
        )
        grads = combine_grads(raw_grads, number_of_parameters, system_size)

        lambda_reg = 0.0
        cost = exp_val + lambda_reg * np.sum(thetas ** 2)
        print(f"cost: {cost}, <E>: {exp_val}, Reg: {0.03 * np.sum(thetas**2)}")
        print(f"||theta||: {np.linalg.norm(thetas)}, ||grad||: {np.linalg.norm(grads)}")
        history.append(cost)
        return exp_val, grads

    if method == "adam":
        import optax

        optimizer = optax.adam(learning_rate=1e-2)
        opt_state = optimizer.init(random_thetas)
        thetas = random_thetas.copy()
        for _ in range(500):
            print("====" * 30)
            print(f"Thetas[{_}]: ", thetas)
            print("====" * 30)
            _, grad = get_f_g(thetas)
            updates, opt_state = optimizer.update(grad, opt_state)
            thetas = optax.apply_updates(thetas, updates)
        raise SystemExit(0)

    if method == "basinhopping":
        minimizer_kwargs = {"method": "L-BFGS-B", "jac": True}
        from scipy.optimize import basinhopping

        ret = basinhopping(
            get_f_g,
            random_thetas,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
        )
        print(ret)
        np.savetxt(f"2D_bh_{system_size}_np_{number_of_parameters}_g_{g}_params.txt", ret.x)
        np.savetxt(f"2D_bh_{system_size}_np_{number_of_parameters}_g_{g}_history.txt", history)
        import pickle

        pickle.dump(ret, open(f"2D_bh_L_{system_size}_np_{number_of_parameters}_g_{g}_result.pkl", "wb"))
        raise SystemExit(0)

    result = scipy.optimize.minimize(
        get_f_g,
        random_thetas,
        method="L-BFGS-B",
        jac=True,
        options={"disp": True, "gtol": 1e-5, "maxiter": 100},
    )
    print(result)
