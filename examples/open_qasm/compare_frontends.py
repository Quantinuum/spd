"""Compare SPD's built-in OpenQASM frontend with the `pytket` import path.

This script is intended as an inspection example rather than a quick-start demo.
It runs the same OpenQASM circuit through both frontends, compares the resulting
expectation values, and reports the largest coefficient mismatches in the final
sparse-Pauli operators.
"""

import argparse
from pathlib import Path

import numpy as np
import spd
from pytket.qasm import circuit_from_qasm


BACKENDS = {
    'numpy': spd.numpy_backend,
    'jax': spd.jax_backend,
}


def to_term_dict(backend_name, module, spo, n_qubits):
    terms = {}
    if backend_name == 'jax':
        xz_rows = np.asarray(spo.xz_array)
        c_vals = np.asarray(spo.c_array)
        for xz, c in zip(xz_rows, c_vals):
            if np.isclose(np.real(c), 0.0):
                continue
            pstr = module.utils.uint32_to_pauli_str(np.asarray(xz), 32)[:n_qubits]
            terms[pstr] = terms.get(pstr, 0.0) + float(np.real(c))
        return terms

    for xz_key, c in spo.items():
        pstr = module.utils.uint32_to_pauli_str(np.asarray(xz_key), 32)[:n_qubits]
        terms[pstr] = terms.get(pstr, 0.0) + float(np.real(c))
    return terms


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the built-in OpenQASM frontend against the pytket import "
            "path on the same circuit."
        )
    )
    default_path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "open_qasm"
        / "periodic_small_8q.qasm"
    )
    parser.add_argument('--path', default=str(default_path))
    parser.add_argument('--backend', choices=['numpy', 'jax'], default='numpy')
    parser.add_argument('--precision', choices=['single', 'double'], default='double')
    parser.add_argument('--trunc-val', type=float, default=1e-4)
    parser.add_argument('--max-num-str', type=int, default=100000)
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args()

    backend_name = args.backend
    module = BACKENDS[backend_name]
    pytket_circ = circuit_from_qasm(args.path)
    measurement = list(range(pytket_circ.n_qubits))

    openqasm_exp_val, openqasm_final_spo = spd.run_openqasm_file(
        args.path,
        measurement,
        trunc_val=args.trunc_val,
        max_num_str=args.max_num_str,
        backend_name=backend_name,
        precision=args.precision,
    )
    pytket_exp_val, pytket_final_spo = spd.run_pytket_circuit(
        pytket_circ,
        measurement,
        trunc_val=args.trunc_val,
        max_num_str=args.max_num_str,
        backend_name=backend_name,
        precision=args.precision,
    )

    openqasm_terms = to_term_dict(backend_name, module, openqasm_final_spo, n_qubits=pytket_circ.n_qubits)
    pytket_terms = to_term_dict(backend_name, module, pytket_final_spo, n_qubits=pytket_circ.n_qubits)

    all_terms = sorted(set(openqasm_terms) | set(pytket_terms))
    diffs = []
    for term in all_terms:
        openqasm_coeff = openqasm_terms.get(term, 0.0)
        pytket_coeff = pytket_terms.get(term, 0.0)
        abs_diff = abs(openqasm_coeff - pytket_coeff)
        diffs.append((abs_diff, term, openqasm_coeff, pytket_coeff))
    diffs.sort(reverse=True)

    print('Comparison target:', args.path)
    print('Backend:', backend_name)
    print('Precision:', args.precision)
    print('trunc_val:', args.trunc_val)
    print('max_num_str:', args.max_num_str)
    print('n_qubits:', pytket_circ.n_qubits)
    print('openqasm expectation:', float(np.asarray(openqasm_exp_val)))
    print('pytket expectation:', float(np.asarray(pytket_exp_val)))
    print('abs expectation diff:', abs(float(np.asarray(openqasm_exp_val)) - float(np.asarray(pytket_exp_val))))
    print('openqasm final size:', openqasm_final_spo.get_size())
    print('pytket final size:', pytket_final_spo.get_size())
    print('openqasm-only terms:', len(set(openqasm_terms) - set(pytket_terms)))
    print('pytket-only terms:', len(set(pytket_terms) - set(openqasm_terms)))
    print('shared terms:', len(set(openqasm_terms) & set(pytket_terms)))
    print()
    print(f'Top {args.top_k} term mismatches by absolute coefficient difference:')
    for abs_diff, term, openqasm_coeff, pytket_coeff in diffs[: args.top_k]:
        print(
            f'{term:>12}  openqasm={openqasm_coeff:+.12e}  pytket={pytket_coeff:+.12e}  abs_diff={abs_diff:.12e}'
        )


if __name__ == '__main__':
    main()
