import numpy as np
import scipy.optimize
import spd
import sys

import run_utils
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

    method = "lbfgsb"
    number_of_parameters = int(sys.argv[1])
    num_layers = int(number_of_parameters // 2)
    system_size = 2 * num_layers + 3
    lambda_ose = 0.0

    g = float(sys.argv[2])
    basis = sys.argv[3]
    niter = int(sys.argv[4])
    assert basis in ["0", "+"]

    random_thetas = (np.random.rand(number_of_parameters) - 0.5) * 0.1
    print(random_thetas)

    full_H = False
    ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=g, full=full_H)
    factor = system_size if full_H else 1

    trunc_val = 1e-14
    max_num_str = int(1e5)
    print(f"\n Truncation Value: {trunc_val} | max num str: {max_num_str}")

    run_name = run_utils.format_run_name(
        model="tfi",
        dim=1,
        size=system_size,
        num_params=number_of_parameters,
        g=g,
        method=method,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        lambda_ose=lambda_ose,
    )
    run_dir = run_utils.make_run_dir(run_name)
    optimizer_options = {"disp": True, "gtol": 1e-6, "maxiter": 1000}
    metadata = {
        "model": "tfi",
        "dim": 1,
        "system_size": system_size,
        "num_layers": num_layers,
        "num_params": number_of_parameters,
        "g": g,
        "basis": basis,
        "full_H": full_H,
        "trunc_val": trunc_val,
        "max_num_str": max_num_str,
        "lambda_ose": lambda_ose,
        "backend": backend.name,
        "precision": precision,
        "packbit": backend.packbit,
        "algorithm": "stack_sort_merge",
        "method": method,
        "optimizer_options": optimizer_options,
        "seed": 0,
        "script": __file__,
        "argv": sys.argv[1:],
    }

    evals = []
    history = []
    params_history = []
    last_eval = {}

    def get_f_g(thetas):
        circ = tfi_setup.gen_1d_TFI_ansatz_circuit(thetas, system_size, basis)
        initial_spo = spd.create_spo(ham_dict, backend=backend)
        final_spo, forward_info = spd.evolve(
            initial_spo,
            circ,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
        )

        E_err_estimate = forward_info["total_truncated_l2_norm"]
        OSE = final_spo.get_OSE()

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
        grads = combine_grads(raw_grads, number_of_parameters, system_size)
        exp_val /= factor
        E_err_estimate /= factor
        grads /= factor

        lambda_reg = 0.0
        cost = exp_val + lambda_ose * OSE + lambda_reg * np.sum(thetas ** 2)
        print("eval = ", len(evals), "num_param", number_of_parameters)
        print(f"cost: {cost}, <E>: {exp_val} ± {E_err_estimate}, OSE: {OSE}")
        print(f"||theta||: {np.linalg.norm(thetas)}, ||grad||: {np.linalg.norm(grads)}")
        run_utils.record_eval(
            evals,
            last_eval,
            thetas,
            cost=cost,
            energy=exp_val,
            energy_error=E_err_estimate,
            ose=OSE,
            grad_norm=np.linalg.norm(grads),
            lambda_ose=lambda_ose,
        )
        return cost, grads

    def log_step(thetas):
        # SciPy callback reports accepted parameters, while get_f_g is also called
        # during line search. We attach the latest matching eval if available.
        run_utils.record_step(history, params_history, last_eval, thetas)

    initial_cost, initial_grads = get_f_g(random_thetas)
    log_step(random_thetas)
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
            callback=lambda x, f, accept: log_step(x) if accept else None,
        )
        print(ret)
        final = run_utils.make_final_summary(ret, evals, ret.x)
        run_utils.save_run_outputs(
            run_dir,
            metadata=metadata,
            final=final,
            evals=evals,
            history=history,
            params_history=params_history,
            initial_params=random_thetas,
            final_params=ret.x,
            result=ret,
        )
        raise SystemExit(0)

    result = scipy.optimize.minimize(
        get_f_g,
        random_thetas,
        method="L-BFGS-B",
        jac=True,
        callback=log_step,
        options=optimizer_options,
    )
    print(result)
    print("params: ", result.x)
    print(history)
    final = run_utils.make_final_summary(result, evals, result.x)
    run_utils.save_run_outputs(
        run_dir,
        metadata=metadata,
        final=final,
        evals=evals,
        history=history,
        params_history=params_history,
        initial_params=random_thetas,
        final_params=result.x,
        result=result,
    )
