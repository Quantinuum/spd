import numpy as np
import scipy.optimize
import spd
import sys

import tfi_setup


np.random.seed(0)


def combine_grads(grads, number_of_parameters, system_size):
    combined = []
    for i in range(number_of_parameters):
        combined.append(np.array(grads[i * system_size:(i + 1) * system_size]).sum() * np.pi)
    return np.array(combined)


if __name__ == "__main__":
    precision = "double"
    backend = spd.BackendAdapter.from_name("jax", packbit=32, precision=precision)
    backend.module.set_algorithm("stack_sort_merge")

    method = ""
    number_of_parameters = int(sys.argv[1])
    num_layers = int(number_of_parameters // 2)
    system_size = 2 * num_layers + 3

    g = float(sys.argv[2])
    basis = sys.argv[3]
    niter = int(sys.argv[4])
    assert basis in ["0", "+"]

    base_filename = f"L_{system_size}_np_{number_of_parameters}_g_{g}_basis_{basis}"
    random_thetas = (np.random.rand(number_of_parameters) - 0.5) * 0.1
    print(random_thetas)

    full_H = False
    ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=g, full=full_H)
    factor = system_size if full_H else 1

    trunc_val = 1e-14
    max_num_str = int(1e5)
    print(f"\n Truncation Value: {trunc_val} | max num str: {max_num_str}")

    history = []

    def get_f_g(thetas):
        circ = tfi_setup.gen_1d_TFI_ansatz_circuit(thetas, system_size, basis)
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
        exp_val /= factor
        grads /= factor

        lambda_reg = 0.0
        cost = exp_val + lambda_reg * np.sum(thetas ** 2)
        print("step = ", len(history), "num_param", number_of_parameters)
        print(f"cost: {cost}, <E>: {exp_val}, Reg: {0.03 * np.sum(thetas**2)}")
        print(f"||theta||: {np.linalg.norm(thetas)}, ||grad||: {np.linalg.norm(grads)}")
        history.append(cost)
        return exp_val, grads

    initial_cost, initial_grads = get_f_g(random_thetas)
    print("\n Expectation Value:", initial_cost)
    print("\n SPD Computed Gradients:", initial_grads)

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
        np.savetxt(base_filename + "_params.txt", ret.x)
        np.savetxt(base_filename + "_history.txt", history)
        import pickle

        pickle.dump(ret, open(base_filename + "_result.pkl", "wb"))
        raise SystemExit(0)

    result = scipy.optimize.minimize(
        get_f_g,
        random_thetas,
        method="L-BFGS-B",
        jac=True,
        options={"disp": True, "gtol": 1e-6, "maxiter": 1000},
    )
    print(result)
    print("params: ", result.x)
    print(history)
