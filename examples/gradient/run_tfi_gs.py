"""Optimize a periodic 1D, 2D, or 3D TFI variational circuit."""

import argparse

import numpy as np
import scipy.optimize

import spd
from spd.ansatz import tfi_1d_hva, tfi_2d_hva, tfi_3d_hva

import run_utils


def make_tfi_ansatz(params, dimension, linear_system_size, basis="+"):
    if dimension == 1:
        return tfi_1d_hva(params, system_size=linear_system_size, basis=basis)
    if basis != "+":
        raise ValueError("basis='0' is only supported for the 1D TFI ansatz.")
    if dimension == 2:
        return tfi_2d_hva(
            params,
            system_size_x=linear_system_size,
            system_size_y=linear_system_size,
        )
    if dimension == 3:
        return tfi_3d_hva(
            params,
            system_size_x=linear_system_size,
            system_size_y=linear_system_size,
            system_size_z=linear_system_size,
        )
    raise ValueError("dimension must be 1, 2, or 3.")


def make_local_tfi_hamiltonian(dimension, linear_system_size, g):
    """Return the local TFI energy terms used by the archived examples."""
    system_size = linear_system_size**dimension
    hamiltonian = {}
    for axis in range(dimension):
        paulis = ["I"] * system_size
        paulis[0] = "Z"
        paulis[linear_system_size**axis] = "Z"
        hamiltonian["".join(paulis)] = -1.0

    paulis = ["I"] * system_size
    paulis[0] = "X"
    hamiltonian["".join(paulis)] = -g
    return hamiltonian


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dimension", type=int, choices=(1, 2, 3))
    parser.add_argument("number_of_parameters", type=int)
    parser.add_argument("niter", type=int)
    parser.add_argument("--g", type=float, default=3.1)
    parser.add_argument("--basis", choices=("0", "+"), default="+")
    parser.add_argument("--linear-system-size", type=int, default=None)
    run_utils.add_common_args(
        parser,
        trunc_val=None,
        max_num_str=None,
        lambda_ose=0.0,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_utils.validate_method(args.method, args.niter)

    linear_system_size = args.linear_system_size
    if linear_system_size is None:
        linear_system_size = args.number_of_parameters + 2
    if linear_system_size < 2 or linear_system_size % 2:
        raise ValueError("linear_system_size must be even and at least 2.")
    if args.number_of_parameters <= 0 or args.number_of_parameters % 2:
        raise ValueError("number_of_parameters must be positive and even.")
    if args.dimension != 1 and args.basis != "+":
        raise ValueError("basis='0' is only supported for dimension 1.")

    precision = "double"
    backend = spd.BackendAdapter.from_name("jax", packbit=32, precision=precision)
    backend.module.set_algorithm(args.algorithm)

    trunc_val = args.trunc_val
    if trunc_val is None:
        trunc_val = 1e-14 if args.dimension == 1 else 1e-6
    max_num_str = args.max_num_str
    if max_num_str is None:
        max_num_str = int(1e5) if args.dimension == 1 else int(1e6)

    np.random.seed(args.seed)
    init_mode = run_utils.infer_init_mode(args.init_params_path)
    initial_thetas, init_metadata = run_utils.init_thetas(
        num_params=args.number_of_parameters,
        init_mode=init_mode,
        init_params_path=args.init_params_path,
        random_scale=args.random_scale,
    )
    print(initial_thetas)

    system_size = linear_system_size**args.dimension
    hamiltonian = make_local_tfi_hamiltonian(
        args.dimension,
        linear_system_size,
        args.g,
    )
    print(f"\nTruncation value: {trunc_val} | max num str: {max_num_str}")
    memory_estimate = run_utils.print_jax_memory_estimate(
        system_size,
        max_num_str,
        packbit=backend.packbit,
        precision=precision,
    )

    lattice_size = (linear_system_size,) * args.dimension
    run_name = run_utils.format_run_name(
        model="tfi",
        dim=args.dimension,
        size=linear_system_size if args.dimension == 1 else lattice_size,
        num_params=args.number_of_parameters,
        g=args.g,
        method=args.method,
        init_mode=init_mode,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        lambda_ose=args.lambda_ose,
        alpha=args.alpha,
    )
    run_dir = run_utils.make_run_dir(run_name)

    if args.method == "basinhopping":
        optimizer_options = {
            "niter": args.niter,
            "minimizer_kwargs": {"method": "L-BFGS-B", "jac": True},
        }
    elif args.method == "lbfgs":
        optimizer_options = {"disp": True, "gtol": 1e-6, "maxiter": args.niter}
    elif args.method == "adam":
        optimizer_options = {"learning_rate": 1e-2, "niter": args.niter}
    else:
        optimizer_options = {}

    metadata = {
        "model": "tfi",
        "dim": args.dimension,
        "system_size": system_size,
        "linear_system_size": linear_system_size,
        "num_layers": args.number_of_parameters // 2,
        "num_params": args.number_of_parameters,
        "g": args.g,
        "basis": args.basis,
        "hamiltonian": "local_energy",
        "trunc_val": trunc_val,
        "max_num_str": max_num_str,
        "lambda_ose": args.lambda_ose,
        "alpha": args.alpha,
        "backend": backend.name,
        "precision": precision,
        "packbit": backend.packbit,
        "algorithm": args.algorithm,
        "memory_estimate": memory_estimate,
        "method": args.method,
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
        ansatz = make_tfi_ansatz(
            thetas,
            args.dimension,
            linear_system_size,
            args.basis,
        )
        initial_spo = spd.create_spo(hamiltonian, backend=backend)
        final_spo, forward_info = spd.evolve(
            initial_spo,
            ansatz.circuit,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
        )
        energy_error = forward_info["total_truncated_l2_norm"]
        ose = final_spo.get_OSE(alpha=args.alpha)
        energy = final_spo.get_expectation_value(basis=args.basis)
        initial_spgo = spd.init_gradient_spo(
            final_spo,
            basis=args.basis,
            lambda_ose=args.lambda_ose,
            alpha=args.alpha,
            backend=backend,
        )
        _, gate_gradients, _ = spd.backpropagate(
            initial_spgo,
            ansatz.circuit,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
        )
        gradients = ansatz.parameter_gradients(gate_gradients)
        cost = energy + args.lambda_ose * ose

        print("eval =", len(evals), "num_param =", args.number_of_parameters)
        print(f"cost: {cost}, <E>: {energy} ± {energy_error}, OSE: {ose}")
        print(
            f"||theta||: {np.linalg.norm(thetas)}, "
            f"||grad||: {np.linalg.norm(gradients)}"
        )
        run_utils.record_eval(
            evals,
            last_eval,
            thetas,
            cost=cost,
            energy=energy,
            energy_error=energy_error,
            ose=ose,
            grad_norm=np.linalg.norm(gradients),
            lambda_ose=args.lambda_ose,
            run_dir=run_dir,
            start_time=start_time,
        )
        return cost, gradients

    def log_step(thetas):
        run_utils.record_step(
            history,
            params_history,
            last_eval,
            thetas,
            run_dir=run_dir,
            start_time=start_time,
        )

    initial_cost, initial_gradients = get_f_g(initial_thetas)
    log_step(initial_thetas)
    print("\nExpectation value:", initial_cost)
    print("\nSPD parameter gradients:", initial_gradients)

    result = None
    final_thetas = initial_thetas
    if args.method == "adam":
        import optax

        optimizer = optax.adam(learning_rate=optimizer_options["learning_rate"])
        opt_state = optimizer.init(initial_thetas)
        final_thetas = initial_thetas.copy()
        for step in range(args.niter):
            print(f"\nStep {step}: {final_thetas}")
            _, gradients = get_f_g(final_thetas)
            updates, opt_state = optimizer.update(gradients, opt_state)
            final_thetas = optax.apply_updates(final_thetas, updates)
            log_step(final_thetas)
    elif args.method == "basinhopping":
        result = scipy.optimize.basinhopping(
            get_f_g,
            initial_thetas,
            callback=lambda x, f, accept: log_step(x) if accept else None,
            **optimizer_options,
        )
        final_thetas = result.x
    elif args.method == "lbfgs":
        result = scipy.optimize.minimize(
            get_f_g,
            initial_thetas,
            method="L-BFGS-B",
            jac=True,
            callback=log_step,
            options=optimizer_options,
        )
        final_thetas = result.x
        print(result)

    final = run_utils.make_final_summary(result, evals, final_thetas)
    run_utils.save_run_outputs(
        run_dir,
        metadata=metadata,
        final=final,
        evals=evals,
        history=history,
        params_history=params_history,
        initial_params=initial_thetas,
        final_params=final_thetas,
        result=result,
    )


if __name__ == "__main__":
    main()
