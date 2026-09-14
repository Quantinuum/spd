"""Optimize a periodic 1D, 2D, or 3D AFH variational circuit."""

import argparse

import numpy as np
import scipy.optimize

import spd
from spd.ansatz import afh_1d_hva, afh_2d_hva, afh_3d_hva

import run_utils


def make_afh_ansatz(params, dimension, linear_system_size):
    if dimension == 1:
        return afh_1d_hva(params, system_size=linear_system_size)
    if dimension == 2:
        return afh_2d_hva(
            params,
            system_size_x=linear_system_size,
            system_size_y=linear_system_size,
        )
    if dimension == 3:
        return afh_3d_hva(
            params,
            system_size_x=linear_system_size,
            system_size_y=linear_system_size,
            system_size_z=linear_system_size,
        )
    raise ValueError("dimension must be 1, 2, or 3.")


def make_local_afh_hamiltonian(dimension, linear_system_size):
    """Return the local AFH energy terms used by the archived examples."""
    system_size = linear_system_size**dimension
    hamiltonian = {}
    for axis in range(dimension):
        neighbor = linear_system_size**axis
        for pauli in "XYZ":
            paulis = ["I"] * system_size
            paulis[0] = pauli
            paulis[neighbor] = pauli
            hamiltonian["".join(paulis)] = 1.0
    return hamiltonian


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "triton"), default="jax")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print per-gate norm/OSE (adds GPU reductions and readback).",
    )
    parser.add_argument("dimension", type=int, choices=(1, 2, 3))
    parser.add_argument("num_layers", type=int)
    parser.add_argument("niter", type=int)
    parser.add_argument("--linear-system-size", type=int, default=None)
    run_utils.add_common_args(
        parser,
        method="lbfgs",
        methods=("eval_only", "lbfgs"),
        trunc_val=None,
        max_num_str=None,
        lambda_ose=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_utils.validate_method(args.method, args.niter)
    if args.num_layers <= 0:
        raise ValueError("num_layers must be positive.")

    linear_system_size = args.linear_system_size
    if linear_system_size is None:
        linear_system_size = 6 * args.num_layers + 2
    if linear_system_size < 2 or linear_system_size % 2:
        raise ValueError("linear_system_size must be even and at least 2.")

    defaults = {
        1: {"trunc_val": 1e-3, "max_num_str": int(1e6), "lambda_ose": 0.0},
        2: {"trunc_val": 1e-4, "max_num_str": int(1e6), "lambda_ose": 1e-2},
        3: {"trunc_val": 1e-3, "max_num_str": int(3e6), "lambda_ose": 0.0},
    }[args.dimension]
    trunc_val = defaults["trunc_val"] if args.trunc_val is None else args.trunc_val
    max_num_str = (
        defaults["max_num_str"] if args.max_num_str is None else args.max_num_str
    )
    lambda_ose = (
        defaults["lambda_ose"] if args.lambda_ose is None else args.lambda_ose
    )

    precision = "double"
    basis = "0"
    backend = spd.BackendAdapter.from_name(
        args.backend,
        packbit=32,
        precision=precision,
    )
    if args.backend == "jax":
        backend.module.set_algorithm(args.algorithm)
    else:
        print("Triton uses native hash kernels; --algorithm applies only to JAX.")

    number_of_parameters = 4 * args.num_layers
    np.random.seed(args.seed)
    init_mode = run_utils.infer_init_mode(args.init_params_path)
    initial_thetas, init_metadata = run_utils.init_thetas(
        num_params=number_of_parameters,
        init_mode=init_mode,
        init_params_path=args.init_params_path,
        random_scale=args.random_scale,
    )
    print(initial_thetas)

    system_size = linear_system_size**args.dimension
    hamiltonian = make_local_afh_hamiltonian(
        args.dimension,
        linear_system_size,
    )
    print(f"\nTruncation value: {trunc_val} | max num str: {max_num_str}")
    effective_max_num_str = 1 << (int(max_num_str) - 1).bit_length()
    memory_estimate = run_utils.print_backend_memory_estimate(
        args.backend,
        system_size,
        max_num_str,
        packbit=backend.packbit,
        precision=precision,
    )

    lattice_size = (linear_system_size,) * args.dimension
    run_name = run_utils.format_run_name(
        model="afh",
        dim=args.dimension,
        size=linear_system_size if args.dimension == 1 else lattice_size,
        num_params=number_of_parameters,
        method=args.method,
        init_mode=init_mode,
        trunc_val=trunc_val,
        max_num_str=max_num_str,
        lambda_ose=lambda_ose,
        alpha=args.alpha,
    )
    run_dir = run_utils.make_run_dir(f"{run_name}_{args.backend}")
    optimizer_options = {"disp": True, "gtol": 1e-6, "maxiter": args.niter}
    metadata = {
        "model": "afh",
        "dim": args.dimension,
        "system_size": system_size,
        "linear_system_size": linear_system_size,
        "num_layers": args.num_layers,
        "num_params": number_of_parameters,
        "basis": basis,
        "hamiltonian": "local_energy",
        "trunc_val": trunc_val,
        "max_num_str": max_num_str,
        "effective_max_num_str": effective_max_num_str,
        "lambda_ose": lambda_ose,
        "alpha": args.alpha,
        "backend": backend.name,
        "precision": precision,
        "packbit": backend.packbit,
        "algorithm": args.algorithm if args.backend == "jax" else "native_hash",
        "progress": args.progress,
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
        ansatz = make_afh_ansatz(
            thetas,
            args.dimension,
            linear_system_size,
        )
        initial_spo = spd.create_spo(hamiltonian, backend=backend)
        final_spo, forward_info = spd.evolve(
            initial_spo,
            ansatz.circuit,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
            progress=args.progress,
            pruning="light-cone-barrier",
        )
        energy_error = forward_info["total_truncated_l2_norm"]
        ose = final_spo.get_OSE(alpha=args.alpha)
        energy = final_spo.get_expectation_value(basis=basis)
        initial_spgo = spd.init_gradient_spo(
            final_spo,
            basis=basis,
            lambda_ose=lambda_ose,
            alpha=args.alpha,
            backend=backend,
        )
        _, gate_gradients, _ = spd.backpropagate(
            initial_spgo,
            ansatz.circuit,
            trunc_val,
            max_num_str=max_num_str,
            backend=backend,
            progress=args.progress,
        )
        gradients = ansatz.parameter_gradients(gate_gradients)
        cost = energy + lambda_ose * ose

        print("eval =", len(evals), "num_param =", number_of_parameters)
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
            lambda_ose=lambda_ose,
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
    if args.method == "lbfgs":
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
