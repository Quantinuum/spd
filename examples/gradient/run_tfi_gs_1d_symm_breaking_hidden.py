import argparse
import csv

import jax.numpy as jnp
import numpy as np
import optax
import scipy.optimize
import spd

import run_utils
import tfi_setup


def combine_grads(grads, number_of_parameters, system_size):
    combined = []
    for i in range(number_of_parameters):
        combined.append(np.array(grads[i * system_size:(i + 1) * system_size]).sum() * np.pi)
    return np.array(combined)


def format_value(value):
    text = f"{value:.12g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p")


def hidden_sizes(final_spo, *, size_kind):
    xz_array = final_spo.xz_array
    half_words = xz_array.shape[1] // 2
    x_words = xz_array[:, :half_words]
    z_words = xz_array[:, half_words:]

    if size_kind == "yz":
        words = z_words
    elif size_kind == "pauli":
        words = x_words | z_words
    else:
        raise ValueError("size_kind must be 'yz' or 'pauli'.")

    return jnp.sum(jnp.asarray(np.bitwise_count(np.asarray(words))), axis=1)


def hidden_weights(final_spo, *, size_kind, weight_kind, power, gamma, cutoff):
    sizes = hidden_sizes(final_spo, size_kind=size_kind).astype(final_spo.c_array.dtype)
    excess = jnp.maximum(sizes - cutoff, 0.0)

    if weight_kind == "power":
        if power == 0:
            return jnp.where(excess > 0.0, 1.0, 0.0).astype(final_spo.c_array.dtype)
        return jnp.where(excess > 0.0, excess ** power, 0.0)

    if weight_kind == "exp":
        return jnp.where(excess > 0.0, jnp.exp(gamma * excess) - 1.0, 0.0)

    raise ValueError("weight_kind must be 'power' or 'exp'.")


def hidden_cost_and_spgo(
    final_spo,
    *,
    backend,
    lambda_hidden,
    size_kind,
    weight_kind,
    power,
    gamma,
    cutoff,
):
    weights = hidden_weights(
        final_spo,
        size_kind=size_kind,
        weight_kind=weight_kind,
        power=power,
        gamma=gamma,
        cutoff=cutoff,
    )
    hidden_cost = jnp.sum(weights * final_spo.c_array ** 2)
    grad_c_array = 2.0 * lambda_hidden * weights * final_spo.c_array
    hidden_spgo = backend.module.SparsePauliGradientOp(
        final_spo.xz_array,
        final_spo.c_array,
        grad_c_array,
        lexsorted=final_spo.lexsorted,
    )
    return hidden_cost, hidden_spgo


def init_hidden_eval_log(run_dir):
    with open(run_dir / "hidden_evals.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval",
                "hidden_cost",
                "scaled_hidden_cost",
                "lambda_hidden",
            ],
        )
        writer.writeheader()


def append_hidden_eval(run_dir, eval_index, hidden_cost, lambda_hidden):
    with open(run_dir / "hidden_evals.csv", "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval",
                "hidden_cost",
                "scaled_hidden_cost",
                "lambda_hidden",
            ],
        )
        writer.writerow(
            {
                "eval": eval_index,
                "hidden_cost": float(hidden_cost),
                "scaled_hidden_cost": float(lambda_hidden * hidden_cost),
                "lambda_hidden": float(lambda_hidden),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("number_of_parameters", type=int)
    parser.add_argument("g", type=float)
    parser.add_argument("niter", type=int)
    parser.add_argument("--system-size", type=int, default=None)
    parser.add_argument(
        "--free-fermion",
        dest="free_fermion",
        action="store_true",
        default=True,
        help="Zero every third initialized theta so Rz angles start at zero.",
    )
    parser.add_argument(
        "--no-free-fermion",
        dest="free_fermion",
        action="store_false",
        help="Keep all initialized theta values, including Rz angles.",
    )
    parser.add_argument("--lambda-hidden", type=float, default=0.0)
    parser.add_argument("--hidden-size-kind", choices=("yz", "pauli"), default="yz")
    parser.add_argument("--hidden-weight-kind", choices=("power", "exp"), default="power")
    parser.add_argument("--hidden-power", type=float, default=0.0)
    parser.add_argument("--hidden-gamma", type=float, default=1.0)
    parser.add_argument("--hidden-cutoff", type=float, default=0.0)
    run_utils.add_common_args(
        parser,
        trunc_val=1e-8,
        max_num_str=int(1e6),
        lambda_ose=0.0,
    )
    args = parser.parse_args()

    precision = "double"
    backend = spd.BackendAdapter.from_name("jax", packbit=32, precision=precision)
    algorithm = args.algorithm
    backend.module.set_algorithm(algorithm)

    method = args.method
    number_of_parameters = args.number_of_parameters
    num_layers = int(number_of_parameters // 3)
    system_size = args.system_size
    if system_size is None:
        system_size = 2 * num_layers + 3
    elif system_size <= 0:
        raise ValueError("system_size must be positive.")
    print(f"========      system size = {system_size}     =======")

    g = args.g
    niter = args.niter

    basis = "+"
    trunc_val = args.trunc_val
    max_num_str = args.max_num_str
    lambda_ose = args.lambda_ose
    alpha = args.alpha
    lambda_hidden = args.lambda_hidden
    print(
        f"\n Truncation Value: {trunc_val} | max num str: {max_num_str} | "
        f"precision: {precision} | basis: {basis}"
    )
    print(
        "Hidden penalty: "
        f"lambda={lambda_hidden}, size={args.hidden_size_kind}, "
        f"weight={args.hidden_weight_kind}, power={args.hidden_power}, "
        f"gamma={args.hidden_gamma}, cutoff={args.hidden_cutoff}"
    )
    memory_estimate = run_utils.print_jax_memory_estimate(
        system_size,
        max_num_str,
        packbit=backend.packbit,
        precision=precision,
    )

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
    initial_thetas = np.array(initial_thetas)
    if args.free_fermion:
        initial_thetas[2::3] = 0.0

    ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=g, full=False)
    run_name = run_utils.format_run_name(
        model="tfi_symm_breaking_hidden",
        dim=1,
        size=system_size,
        num_params=number_of_parameters,
        g=g,
        method=method,
        init_mode=init_mode,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        lambda_ose=lambda_ose,
        alpha=alpha,
    )
    run_name += (
        f"_lh{format_value(lambda_hidden)}"
        f"_hs{args.hidden_size_kind}"
        f"_hw{args.hidden_weight_kind}"
        f"_hp{format_value(args.hidden_power)}"
        f"_hg{format_value(args.hidden_gamma)}"
        f"_hc{format_value(args.hidden_cutoff)}"
    )
    if not args.free_fermion:
        run_name += "_ff0"
    run_dir = run_utils.make_run_dir(run_name)
    print("free_fermion:", args.free_fermion)
    print(initial_thetas)
    optimizer_options = {
        "adam_learning_rate": 3e-2,
        "adam_steps": 100,
        "lbfgs": {"disp": True, "gtol": 1e-4, "maxiter": 1000},
        "basinhopping": {"niter": niter, "minimizer_kwargs": {"method": "L-BFGS-B", "jac": True}},
    }
    metadata = {
        "model": "tfi_symm_breaking_hidden",
        "dim": 1,
        "system_size": system_size,
        "num_layers": num_layers,
        "num_params": number_of_parameters,
        "g": g,
        "basis": basis,
        "trunc_val": trunc_val,
        "max_num_str": max_num_str,
        "lambda_ose": lambda_ose,
        "alpha": alpha,
        "free_fermion": args.free_fermion,
        "lambda_hidden": lambda_hidden,
        "hidden_size_kind": args.hidden_size_kind,
        "hidden_weight_kind": args.hidden_weight_kind,
        "hidden_power": args.hidden_power,
        "hidden_gamma": args.hidden_gamma,
        "hidden_cutoff": args.hidden_cutoff,
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
    init_hidden_eval_log(run_dir)
    evals = []
    history = []
    params_history = []
    hidden_evals = []
    last_eval = {}
    start_time = run_utils.start_timer()

    def get_f_g(thetas):
        circ = tfi_setup.gen_1d_TFI_symm_breaking_ansatz_circuit(thetas, system_size)
        initial_spo = spd.create_spo(ham_dict, backend=backend)
        final_spo, forward_info = spd.evolve(
            initial_spo,
            circ,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
        )
        energy_error = forward_info["total_truncated_l2_norm"]
        ose = final_spo.get_OSE(alpha=alpha)
        energy = final_spo.get_expectation_value(basis=basis)
        hidden_cost, hidden_spgo = hidden_cost_and_spgo(
            final_spo,
            backend=backend,
            lambda_hidden=lambda_hidden,
            size_kind=args.hidden_size_kind,
            weight_kind=args.hidden_weight_kind,
            power=args.hidden_power,
            gamma=args.hidden_gamma,
            cutoff=args.hidden_cutoff,
        )
        initial_spgo = spd.init_gradient_spo(
            final_spo,
            basis=basis,
            lambda_ose=lambda_ose,
            alpha=alpha,
            backend=backend,
        )
        if lambda_hidden != 0.0:
            initial_spgo = initial_spgo + hidden_spgo
        _, raw_grads, backward_info = spd.backpropagate(
            initial_spgo,
            circ,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
        )
        grads = combine_grads(raw_grads, number_of_parameters, system_size)

        cost = energy + lambda_ose * ose + lambda_hidden * hidden_cost
        print("eval = ", len(evals), "num_param", number_of_parameters)
        print(
            f"cost: {cost}, <E>: {energy} ± {energy_error}, "
            f"OSE: {ose}, hidden: {hidden_cost}"
        )
        print(f"||theta||: {np.linalg.norm(thetas)}, ||grad||: {np.linalg.norm(grads)}")
        eval_index = len(evals)
        run_utils.record_eval(
            evals,
            last_eval,
            thetas,
            cost=cost,
            energy=energy,
            energy_error=energy_error,
            ose=ose,
            grad_norm=np.linalg.norm(grads),
            lambda_ose=lambda_ose,
            run_dir=run_dir,
            start_time=start_time,
        )
        hidden_evals.append(float(hidden_cost))
        append_hidden_eval(run_dir, eval_index, hidden_cost, lambda_hidden)
        return cost, grads

    def log_step(thetas):
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
    print("\n Initial cost:", initial_cost)
    print("\n SPD Computed Gradients:", initial_grads)

    if method == "eval_only":
        final = run_utils.make_final_summary(None, evals, initial_thetas)
        final["final_hidden_cost"] = hidden_evals[-1] if hidden_evals else None
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
        optimizer = optax.adam(learning_rate=optimizer_options["adam_learning_rate"])
        opt_state = optimizer.init(initial_thetas)
        thetas = initial_thetas.copy()
        for _ in range(optimizer_options["adam_steps"]):
            print("====" * 30)
            print(f"Thetas[{_}]: ", thetas)
            print("====" * 30)
            _, grad = get_f_g(thetas)
            updates, opt_state = optimizer.update(grad, opt_state)
            thetas = optax.apply_updates(thetas, updates)
            log_step(thetas)

        final = run_utils.make_final_summary(None, evals, thetas)
        final["final_hidden_cost"] = hidden_evals[-1] if hidden_evals else None
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
        from scipy.optimize import basinhopping

        ret = basinhopping(
            get_f_g,
            initial_thetas,
            minimizer_kwargs=optimizer_options["basinhopping"]["minimizer_kwargs"],
            niter=niter,
            callback=lambda x, f, accept: log_step(x) if accept else None,
        )
        print(ret)
        final = run_utils.make_final_summary(ret, evals, ret.x)
        final["final_hidden_cost"] = hidden_evals[-1] if hidden_evals else None
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
            options=optimizer_options["lbfgs"],
        )
        print(result)
        print("params: ", result.x)
        final = run_utils.make_final_summary(result, evals, result.x)
        final["final_hidden_cost"] = hidden_evals[-1] if hidden_evals else None
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
