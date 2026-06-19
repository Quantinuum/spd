import argparse

import numpy as np
import scipy.optimize
import spd

import run_utils
import tfi_setup


def combine_grads(grads, number_of_parameters, system_size):
    combined = []
    for i in range(number_of_parameters):
        combined.append(np.array(grads[i * system_size:(i + 1) * system_size]).sum() * np.pi)
    return np.array(combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("number_of_parameters", type=int)
    parser.add_argument("g", type=float)
    parser.add_argument("basis", choices=("0", "+"))
    parser.add_argument("niter", type=int)
    parser.add_argument("--system-size", type=int, default=None)
    run_utils.add_common_args(
        parser,
        trunc_val=1e-14,
        max_num_str=int(1e5),
        lambda_ose=0.0,
    )
    args = parser.parse_args()

    precision = "double"
    backend = spd.BackendAdapter.from_name("jax", packbit=32, precision=precision)
    algorithm = args.algorithm
    backend.module.set_algorithm(algorithm)

    method = args.method
    number_of_parameters = args.number_of_parameters
    num_layers = int(number_of_parameters // 2)
    system_size = args.system_size
    if system_size is None:
        system_size = 2 * num_layers + 3
    elif system_size <= 0:
        raise ValueError("system_size must be positive.")
    lambda_ose = args.lambda_ose

    g = args.g
    basis = args.basis
    niter = args.niter

    init_params_path = args.init_params_path
    init_mode = run_utils.infer_init_mode(init_params_path)
    run_utils.validate_method(method, niter)
    np.random.seed(args.seed)
    initial_thetas, init_metadata = run_utils.init_thetas(
        num_params=number_of_parameters,
        init_mode=init_mode,
        init_params_path=init_params_path,
        random_scale=args.random_scale,
    )
    print(initial_thetas)

    full_H = False
    ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=g, full=full_H)
    factor = system_size if full_H else 1

    trunc_val = args.trunc_val
    max_num_str = args.max_num_str
    print(f"\n Truncation Value: {trunc_val} | max num str: {max_num_str}")
    memory_estimate = run_utils.print_jax_memory_estimate(
        system_size,
        max_num_str,
        packbit=backend.packbit,
        precision=precision,
    )

    run_name = run_utils.format_run_name(
        model="tfi",
        dim=1,
        size=system_size,
        num_params=number_of_parameters,
        g=g,
        method=method,
        init_mode=init_mode,
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
        "algorithm": algorithm,
        "memory_estimate": memory_estimate,
        "method": method,
        "optimizer_options": optimizer_options,
        "init": init_metadata,
        "seed": args.seed,
        "script": __file__,
        "argv": vars(args),
    }
    run_utils.init_run_outputs(
        run_dir,
        metadata=metadata,
        initial_params=initial_thetas,
    )

    evals = []
    history = []
    params_history = []
    last_eval = {}
    start_time = run_utils.start_timer()

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
            run_dir=run_dir,
            start_time=start_time,
        )
        return cost, grads

    def log_step(thetas):
        # SciPy callback reports accepted parameters, while get_f_g is also called
        # during line search. We attach the latest matching eval if available.
        run_utils.record_step(
            history,
            params_history,
            last_eval,
            thetas,
            run_dir=run_dir,
            start_time=start_time,
        )

    initial_cost, initial_grads = get_f_g(initial_thetas)
    log_step(initial_thetas)
    print("\n Expectation Value:", initial_cost)
    print("\n SPD Computed Gradients:", initial_grads)

    if method == "eval_only":
        final = run_utils.make_final_summary(None, evals, initial_thetas)
        run_utils.save_run_outputs(
            run_dir,
            metadata=metadata,
            final=final,
            evals=evals,
            history=history,
            params_history=params_history,
            initial_params=initial_thetas,
            final_params=initial_thetas,
        )
        raise SystemExit(0)

    elif method == "adam":
        import optax

        optimizer = optax.adam(learning_rate=1e-2)
        opt_state = optimizer.init(initial_thetas)
        thetas = initial_thetas.copy()
        for _ in range(500):
            print("====" * 30)
            print(f"Thetas[{_}]: ", thetas)
            print("====" * 30)
            _, grad = get_f_g(thetas)
            updates, opt_state = optimizer.update(grad, opt_state)
            thetas = optax.apply_updates(thetas, updates)
            log_step(thetas)

        final = run_utils.make_final_summary(None, evals, thetas)
        run_utils.save_run_outputs(
            run_dir,
            metadata=metadata,
            final=final,
            evals=evals,
            history=history,
            params_history=params_history,
            initial_params=initial_thetas,
            final_params=thetas,
        )
        raise SystemExit(0)

    elif method == "basinhopping":
        minimizer_kwargs = {"method": "L-BFGS-B", "jac": True}
        from scipy.optimize import basinhopping

        ret = basinhopping(
            get_f_g,
            initial_thetas,
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
            initial_params=initial_thetas,
            final_params=ret.x,
            result=ret,
        )
        raise SystemExit(0)

    elif method == "lbfgs":
        result = scipy.optimize.minimize(
            get_f_g,
            initial_thetas,
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
            initial_params=initial_thetas,
            final_params=result.x,
            result=result,
        )
