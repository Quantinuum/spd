import numpy as np
import optax
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

    method = "adam"
    run_method = "adam_lbfgsb"
    number_of_parameters = int(sys.argv[1])
    num_layers = int(number_of_parameters // 3)
    system_size = 2 * num_layers + 3
    print(f"========      system size = {system_size}     =======")

    g = float(sys.argv[2])
    niter = int(sys.argv[3])

    basis = "+"
    trunc_val = 1e-8
    max_num_str = int(1e6)
    print(
        f"\n Truncation Value: {trunc_val} | max num str: {max_num_str} | "
        f"precision: {precision} | basis: {basis}"
    )

    random_thetas = (np.random.rand(number_of_parameters) - 0.5) * 0.1

    ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=g, full=False)
    run_name = run_utils.format_run_name(
        model="tfi_symm_breaking",
        dim=1,
        size=system_size,
        num_params=number_of_parameters,
        g=g,
        method=run_method,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        lambda_ose="schedule",
    )
    run_dir = run_utils.make_run_dir(run_name)
    initial_params = random_thetas.copy()
    optimizer_options = {
        "adam_learning_rate": 3e-2,
        "adam_steps": 100,
        "lbfgsb": {"disp": True, "gtol": 1e-4, "maxiter": 1000},
    }
    metadata = {
        "model": "tfi_symm_breaking",
        "dim": 1,
        "system_size": system_size,
        "num_layers": num_layers,
        "num_params": number_of_parameters,
        "g": g,
        "basis": basis,
        "trunc_val": trunc_val,
        "max_num_str": max_num_str,
        "lambda_ose": "1e-1 until eval 100, then 0",
        "backend": backend.name,
        "precision": precision,
        "packbit": backend.packbit,
        "algorithm": "stack_sort_merge",
        "method": run_method,
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
        lambda_ose = 1e-1 if len(evals) < 100 else 0.0
        print("lambda_ose: ", lambda_ose)

        circ = tfi_setup.gen_1d_TFI_symm_breaking_ansatz_circuit(thetas, system_size)
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
        optimizer = optax.adam(learning_rate=optimizer_options["adam_learning_rate"])
        opt_state = optimizer.init(random_thetas)
        thetas = random_thetas.copy()
        for _ in range(optimizer_options["adam_steps"]):
            print("====" * 30)
            print(f"Thetas[{_}]: ", thetas)
            print("====" * 30)
            _, grad = get_f_g(thetas)
            updates, opt_state = optimizer.update(grad, opt_state)
            thetas = optax.apply_updates(thetas, updates)
            log_step(thetas)

        random_thetas = thetas

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
            initial_params=initial_params,
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
        options=optimizer_options["lbfgsb"],
    )
    print(result)
    print("params: ", result.x)
    final = run_utils.make_final_summary(result, evals, result.x)
    run_utils.save_run_outputs(
        run_dir,
        metadata=metadata,
        final=final,
        evals=evals,
        history=history,
        params_history=params_history,
        initial_params=initial_params,
        final_params=result.x,
        result=result,
    )
