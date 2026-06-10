import argparse

import numpy as np
import scipy.optimize
import spd

import heisenberg_setup
import run_utils


np.random.seed(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_layers", type=int)
    parser.add_argument("niter", type=int)
    parser.add_argument("--linear-system-size", type=int, default=None)
    run_utils.add_common_args(
        parser,
        method="lbfgs",
        methods=("eval_only", "lbfgs"),
        trunc_val=1e-3,
        max_num_str=int(3e6),
        lambda_ose=0.0,
    )
    args = parser.parse_args()

    precision = "double"
    basis = "0"

    backend = spd.BackendAdapter.from_name("jax", packbit=32, precision=precision)
    backend.module.set_algorithm("stack_sort_merge")

    num_layers = args.num_layers
    niter = args.niter

    number_of_parameters = 4 * num_layers
    # The light cone spread out by +2 in one spatial direction per gate type.
    # There are XX, YY, ZZ gates.
    # So per layer it increases + 6
    system_size_x = args.linear_system_size
    if system_size_x is None:
        system_size_x = num_layers * 6 + 2
    elif system_size_x <= 0:
        raise ValueError("linear_system_size must be positive.")
    system_size_y = system_size_x
    system_size_z = system_size_x
    system_size = system_size_x * system_size_y * system_size_z
    full_H = False
    factor = system_size if full_H else 1

    method = args.method
    init_params_path = args.init_params_path
    init_mode = run_utils.infer_init_mode(init_params_path)
    run_utils.validate_method(method, niter)
    initial_thetas, init_metadata = run_utils.init_thetas(
        num_params=number_of_parameters,
        init_mode=init_mode,
        init_params_path=init_params_path,
        random_scale=args.random_scale,
    )
    stagger_signs = heisenberg_setup.gen_3d_stagger_signs(
        system_size_x, system_size_y, system_size_z
    )
    grad_multiplicities = heisenberg_setup.gen_afh_grad_multiplicities(
        num_layers, spatial_dim=3
    )

    ham_dict = heisenberg_setup.gen_3d_Hamiltonian_dict(
        system_size_x, system_size_y, system_size_z, full=full_H
    )
    trunc_val = args.trunc_val
    max_num_str = args.max_num_str
    lambda_ose = args.lambda_ose

    print(initial_thetas)
    print(f"\n Truncation Value: {trunc_val} | max num str: {max_num_str}")

    run_name = run_utils.format_run_name(
        model="afh",
        dim=3,
        size=(system_size_x, system_size_y, system_size_z),
        num_params=number_of_parameters,
        method=method,
        init_mode=init_mode,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        lambda_ose=lambda_ose,
    )
    run_dir = run_utils.make_run_dir(run_name)
    optimizer_options = {"disp": True, "gtol": 1e-6, "maxiter": niter}
    metadata = {
        "model": "afh",
        "dim": 3,
        "system_size": system_size,
        "system_size_x": system_size_x,
        "system_size_y": system_size_y,
        "system_size_z": system_size_z,
        "num_layers": num_layers,
        "num_params": number_of_parameters,
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
        "init": init_metadata,
        "seed": 0,
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

    def get_f_g(thetas):
        circ = heisenberg_setup.gen_3d_AFH_ansatz_circuit(
            thetas, system_size_x, system_size_y, system_size_z
        )
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
        grads = heisenberg_setup.combine_afh_parameter_grads(
            raw_grads, system_size, stagger_signs, grad_multiplicities
        )
        exp_val /= factor
        E_err_estimate /= factor
        grads /= factor
        cost = exp_val + lambda_ose * OSE
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
        )
        return cost, grads

    def log_step(thetas):
        # SciPy callback reports accepted parameters, while get_f_g is also called
        # during line search. We attach the latest matching eval if available.
        run_utils.record_step(history, params_history, last_eval, thetas, run_dir=run_dir)

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
    else:
        raise ValueError(f"Unsupported method={method}.")
