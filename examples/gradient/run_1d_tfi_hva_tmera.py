"""Optimize the periodic 1D TFI model with the binary HVA-TMERA ansatz.

Example:
    python examples/gradient/run_1d_tfi_hva_tmera.py 16 4 100 --rounds 2 --backend triton

The positional arguments are SYSTEM_SIZE, CHI, and the L-BFGS iteration limit.
The exact ground-state reference is the finite-ring free-fermion result, so its
cost is linear in SYSTEM_SIZE and does not restrict the example to small rings.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

# Direct script execution puts examples/gradient, not the repository root, at
# sys.path[0]. Prefer this checkout over an editable install from another one.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import spd
from spd.ansatz import (
    binary_mera_qubit_initializations,
    binary_mera_parameter_shape,
    tfi_binary_mera,
    tfi_binary_mera_channels,
)

try:
    from . import run_utils
except ImportError:  # Direct execution from the repository root.
    import run_utils


def tfi_ground_energy(system_size, g):
    r"""Exact finite-ring ground energy of ``-sum ZZ - g sum X``.

    For even N, the ground state is in the even fermion-parity sector. Its
    Jordan-Wigner fermions have antiperiodic momenta

        k_m = (2m + 1) pi / N,  m = 0, ..., N-1,

    and

        E0 = -sum_m sqrt(1 + g**2 - 2 g cos(k_m)).

    This is the finite-size sum, not the thermodynamic-limit integral. See
    P. Pfeuty, Annals of Physics 57, 79-90 (1970),
    https://doi.org/10.1016/0003-4916(70)90270-8.
    """
    if (isinstance(system_size, bool) or not isinstance(system_size, (int, np.integer))
            or system_size < 4 or system_size % 2):
        raise ValueError("system_size must be an even integer at least 4.")
    if not np.isfinite(g):
        raise ValueError("g must be finite.")
    momenta = (2 * np.arange(int(system_size)) + 1) * np.pi / int(system_size)
    dispersion = np.sqrt(1 + float(g)**2 - 2 * float(g) * np.cos(momenta))
    return -float(np.sum(dispersion))


def tfi_terms(system_size, g):
    """Full periodic ``-sum ZZ - g sum X`` Hamiltonian."""
    terms = {}
    for i in range(system_size):
        zz = ["I"] * system_size
        zz[i] = zz[(i + 1) % system_size] = "Z"
        terms["".join(zz)] = -1.0
        x = ["I"] * system_size
        x[i] = "X"
        terms["".join(x)] = -float(g)
    return terms


def _parameter_rounds(params, system_size, chi, physical_bottom=True):
    values = np.asarray(params, dtype=float)
    per_round = binary_mera_parameter_shape(
        system_size, 1, chi=chi, physical_bottom=physical_bottom,
    )[0]
    if values.ndim != 1 or values.size % per_round:
        raise ValueError(f"params must be flat with a multiple of {per_round} entries.")
    return values, values.size // per_round


def _execution_backend(name, channels=True, *, algorithm=None):
    backend = spd.BackendAdapter.from_name(name, precision="double")
    if name == "jax" and algorithm is not None:
        backend.module.set_algorithm(algorithm)
    if channels:
        require = getattr(backend, "require_channel_support", None)
        if require is None:
            raise ImportError("This backend does not expose static quantum channels.")
        require()
    return backend


def spd_value_gradient(params, system_size, chi, g, *, channels=True, cutoff=0.0,
                       backend_name="numpy", backend=None, max_num_str=None,
                       physical_bottom=True, progress=False):
    """Return the SPD energy, parameter gradient, and run diagnostics."""
    params, rounds = _parameter_rounds(params, system_size, chi, physical_bottom)
    builder = tfi_binary_mera_channels if channels else tfi_binary_mera
    ansatz = builder(
        params, system_size, rounds, chi=chi, physical_bottom=physical_bottom,
    )
    if backend is None:
        backend = _execution_backend(backend_name, channels)
    if max_num_str is None:
        max_num_str = min(4**system_size, 1_000_000)
    if (isinstance(max_num_str, bool)
            or not isinstance(max_num_str, (int, np.integer))
            or max_num_str < 1):
        raise ValueError("max_num_str must be a positive integer.")

    # Shared tensors do not give the finite MERA a two-/four-site unit cell.
    # Keep all translated terms; see docs/hva_tmera.md for the gradient check.
    observable = spd.create_spo(tfi_terms(system_size, g), backend=backend)
    if backend.name == "triton":
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    final, forward_info = spd.evolve(
        observable, ansatz.circuit, cutoff, max_num_str, backend=backend,
        progress=progress,
    )
    energy = float(final.get_expectation_value(basis="0"))
    gradient_spo = spd.init_gradient_spo(final, basis="0", backend=backend)
    _, gate_gradients, backward_info = spd.backpropagate(
        gradient_spo, ansatz.circuit, cutoff, max_num_str, backend=backend,
        progress=progress,
    )
    if backend.name == "triton":
        torch.cuda.synchronize()

    diagnostics = {
        "backend": backend.name,
        "algorithm": backend.module.get_algorithm() if backend.name == "jax" else None,
        "term_cap": int(max_num_str),
        "seconds_forward_backward": time.perf_counter() - started,
        "final_pauli_terms": final.get_size(),
        "energy_error": float(forward_info["total_truncated_l2_norm"]),
        "discarded_terms_sum": int(sum(forward_info["history"]["num_str_truncated"])),
        "backward_discarded_terms_sum": int(
            sum(backward_info["history"]["num_str_truncated"])
        ),
    }
    if backend.name == "triton":
        diagnostics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    if channels:
        from spd.circuit_ir import CreateZero
        starts = {
            indices[0]
            for _, indices in binary_mera_qubit_initializations(
                system_size, rounds, chi=chi, physical_bottom=physical_bottom,
            )
        }
        widths = [forward_info["active_widths"][0]]
        for step, operation in enumerate(reversed(ansatz.circuit.operations), 1):
            if isinstance(operation, CreateZero) and operation.qubit in starts:
                widths.append(forward_info["active_widths"][step])
        diagnostics["qubit_initialization_widths"] = widths
    return energy, ansatz.parameter_gradients(gate_gradients), diagnostics


# Independent dense references used only by the small-system tests. The CLI
# below never constructs a statevector or dense Hamiltonian.
def pauli_action(state, sites, axis):
    n = state.shape[-1].bit_length() - 1
    indices = np.arange(2**n)
    flip, sign = 0, np.ones(2**n)
    for site in sites:
        bit = 1 << (n - 1 - site)
        if axis in ("X", "Y"):
            flip ^= bit
        if axis in ("Z", "Y"):
            sign *= 1 - 2 * ((indices & bit) != 0)
    phase = (-1j)**len(sites) if axis == "Y" else 1
    return phase * sign * state[..., indices ^ flip]


def exact_state_and_tangents(params, system_size, chi, *, physical_bottom=True):
    """Independent dense ansatz reference for small regression tests."""
    params, rounds = _parameter_rounds(params, system_size, chi, physical_bottom)
    q = int(chi).bit_length() - 1
    state = np.zeros(2**system_size, complex)
    state[0] = 1
    tangent = np.zeros((params.size, 2**system_size), complex)

    def rotate(sites, axis, angle, index=None):
        nonlocal state, tangent
        c, sn = np.cos(np.pi * angle / 2), np.sin(np.pi * angle / 2)
        state = c * state - 1j * sn * pauli_action(state, sites, axis)
        tangent = c * tangent - 1j * sn * pauli_action(tangent, sites, axis)
        if index is not None:
            tangent[index] += -0.5j * np.pi * pauli_action(state, sites, axis)

    for i in range(system_size - q, system_size):
        rotate((i,), "Y", 0.5)
    stride, cursor = system_size // 2, 0
    minimum_stride = 1 if physical_bottom else 2
    while stride >= minimum_stride:
        width, parent_width = min(q, stride), min(q, 2 * stride)
        sites = [
            tuple(range(end - width, end))
            for end in range(stride, system_size + 1, stride)
        ]
        isometries = [sites[j] + sites[j + 1] for j in range(0, len(sites), 2)]
        for j, outputs in enumerate(isometries):
            end = (j + 1) * 2 * stride
            retained = set(range(end - parent_width, end))
            for i in outputs:
                if i not in retained:
                    rotate((i,), "Y", 0.5)
        roles = [isometries]
        if len(sites) > 2:
            roles.append([
                sites[j] + sites[(j + 1) % len(sites)]
                for j in range(1, len(sites), 2)
            ])
        for tensors in roles:
            block_width = len(tensors[0])
            count = 3 if block_width == 2 else 3 * block_width - 1
            for _ in range(rounds):
                base = cursor
                if block_width == 2:
                    for outputs in tensors:
                        rotate(outputs, "Z", params[base], base)
                    for position in range(2):
                        for outputs in tensors:
                            rotate((outputs[position],), "X",
                                   params[base + 1 + position], base + 1 + position)
                else:
                    half = block_width // 2
                    for bond in range(half):
                        for outputs in tensors:
                            j = 2 * bond
                            rotate(outputs[j:j + 2], "Z",
                                   params[base + bond], base + bond)
                    xbase = base + half
                    for position in range(block_width):
                        for outputs in tensors:
                            rotate((outputs[position],), "X",
                                   params[xbase + position], xbase + position)
                    zzbase = xbase + block_width
                    for bond in range(half - 1):
                        for outputs in tensors:
                            j = 2 * bond + 1
                            rotate(outputs[j:j + 2], "Z",
                                   params[zzbase + bond], zzbase + bond)
                    xbase = zzbase + half - 1
                    for position in range(block_width):
                        for outputs in tensors:
                            rotate((outputs[position],), "X",
                                   params[xbase + position], xbase + position)
                cursor += count
        stride //= 2
    if cursor != params.size:
        raise AssertionError((cursor, params.size))
    return state, tangent


def hamiltonian_action(state, system_size, g):
    return -sum(
        pauli_action(state, (i, (i + 1) % system_size), "Z")
        + g * pauli_action(state, (i,), "X")
        for i in range(system_size)
    )


def exact_value_gradient(params, system_size, chi, g, *, physical_bottom=True):
    state, tangent = exact_state_and_tangents(
        params, system_size, chi, physical_bottom=physical_bottom,
    )
    h_state = hamiltonian_action(state, system_size, g)
    value = float(np.vdot(state, h_state).real)
    gradient = (2 * (tangent.conj() @ h_state).real).reshape(params.shape)
    return value, gradient


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system_size", type=int)
    parser.add_argument("chi", type=int, choices=(2, 4))
    parser.add_argument("niter", type=int)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--g", type=float, default=1.1)
    parser.add_argument("--backend", choices=("numpy", "jax", "triton"), default="numpy")
    parser.add_argument("--algorithm", choices=("stack_sort_merge", "search_update_merge"),
                        default="stack_sort_merge", help="JAX rotation algorithm.")
    parser.add_argument("--method", choices=("eval_only", "lbfgs"), default="lbfgs")
    parser.add_argument("--no-physical-bottom", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--init-params-path", default=None)
    parser.add_argument("--random-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trunc-val", type=float, default=1e-8)
    parser.add_argument("--max-num-str", type=int, default=1_000_000)
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    run_utils.validate_method(args.method, args.niter)
    if args.system_size < 4 or args.system_size & (args.system_size - 1):
        raise ValueError("system_size must be a power of two, at least 4.")
    if args.rounds < 1:
        raise ValueError("rounds must be positive.")
    if args.chi == 2 and args.no_physical_bottom:
        raise ValueError("chi=2 requires the physical bottom layer.")
    if not np.isfinite(args.g) or not np.isfinite(args.trunc_val) or args.trunc_val < 0:
        raise ValueError("g must be finite and trunc_val must be finite and nonnegative.")
    if args.max_num_str < 1:
        raise ValueError("max_num_str must be positive.")

    physical_bottom = not args.no_physical_bottom
    shape = binary_mera_parameter_shape(
        args.system_size, args.rounds, chi=args.chi,
        physical_bottom=physical_bottom,
    )
    np.random.seed(args.seed)
    init_mode = run_utils.infer_init_mode(args.init_params_path)
    initial_params, init_metadata = run_utils.init_thetas(
        num_params=shape[0], init_mode=init_mode,
        init_params_path=args.init_params_path,
        random_scale=args.random_scale,
    )
    try:
        backend = _execution_backend(args.backend, channels=True, algorithm=args.algorithm)
    except (ImportError, NotImplementedError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    ground_energy = tfi_ground_energy(args.system_size, args.g)
    if args.backend == "triton":
        memory_estimate = run_utils.print_backend_memory_estimate(
            args.backend, args.system_size, args.max_num_str,
            packbit=backend.packbit, precision="double",
        )
    else:
        memory_estimate = run_utils.estimate_jax_memory_usage(
            args.system_size, args.max_num_str,
            packbit=backend.packbit, precision="double",
        )
        memory_estimate["scope"] = "NumPy live-array estimate"
    run_name = run_utils.format_run_name(
        model="tfi_hva_tmera", dim=1, size=args.system_size,
        num_params=shape[0], g=args.g, method=args.method,
        init_mode=init_mode, trunc_val=args.trunc_val,
        max_num_str=args.max_num_str, lambda_ose=0.0, alpha=1.0,
    )
    run_name += f"_chi{args.chi}_r{args.rounds}_{args.backend}"
    run_dir = run_utils.make_run_dir(run_name)
    metadata = {
        "model": "periodic_1d_tfi_hva_tmera",
        "system_size": args.system_size,
        "chi": args.chi,
        "brickwork_rounds": args.rounds,
        "physical_bottom": physical_bottom,
        "num_params": shape[0],
        "g": args.g,
        "hamiltonian": "-sum_i Z_i Z_(i+1) - g sum_i X_i",
        "exact_ground_energy": ground_energy,
        "exact_reference": "finite-ring Jordan-Wigner momentum sum",
        "trunc_val": args.trunc_val,
        "max_num_str": args.max_num_str,
        "backend": backend.name,
        "algorithm": backend.module.get_algorithm() if backend.name == "jax" else None,
        "precision": "double",
        "memory_estimate": memory_estimate,
        "method": args.method,
        "optimizer_options": {"gtol": 1e-6, "maxiter": args.niter},
        "init": init_metadata,
        "seed": args.seed,
        "script": __file__,
        "argv": vars(args),
    }
    run_utils.init_run_outputs(run_dir, metadata=metadata, initial_params=initial_params)

    evals, history, params_history, last_eval = [], [], [], {}
    started = run_utils.start_timer()

    def objective(params):
        energy, gradient, diagnostics = spd_value_gradient(
            params, args.system_size, args.chi, args.g,
            channels=True, cutoff=args.trunc_val, backend=backend,
            max_num_str=args.max_num_str, physical_bottom=physical_bottom,
            progress=args.progress,
        )
        print(
            f"eval={len(evals)} energy={energy:.12g} "
            f"E-E0={energy-ground_energy:.6g} "
            f"|grad|={np.linalg.norm(gradient):.6g} "
            f"terms={diagnostics['final_pauli_terms']}"
        )
        run_utils.record_eval(
            evals, last_eval, params, cost=energy, energy=energy,
            energy_error=diagnostics["energy_error"], ose=0.0,
            grad_norm=np.linalg.norm(gradient), lambda_ose=0.0,
            run_dir=run_dir, start_time=started,
        )
        return energy, gradient

    def log_step(params):
        run_utils.record_step(
            history, params_history, last_eval, params,
            run_dir=run_dir, start_time=started,
        )

    initial_energy, initial_gradient = objective(initial_params)
    log_step(initial_params)
    print(f"Exact ground energy: {ground_energy:.12g}")
    print(f"Initial energy: {initial_energy:.12g}")
    print(f"Initial gradient norm: {np.linalg.norm(initial_gradient):.6g}")

    result = None
    final_params = initial_params
    if args.method == "lbfgs":
        result = minimize(
            objective, initial_params, method="L-BFGS-B", jac=True,
            callback=log_step,
            options={"disp": True, "gtol": 1e-6, "maxiter": args.niter},
        )
        final_params = result.x
        print(result)

    final = run_utils.make_final_summary(result, evals, final_params)
    final["exact_ground_energy"] = ground_energy
    final["energy_above_ground"] = final["final_energy"] - ground_energy
    run_utils.save_run_outputs(
        run_dir, metadata=metadata, final=final, evals=evals,
        history=history, params_history=params_history,
        initial_params=initial_params, final_params=final_params, result=result,
    )


if __name__ == "__main__":
    main()
