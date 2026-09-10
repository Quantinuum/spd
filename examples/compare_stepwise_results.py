"""Validate and compare two locally generated stepwise benchmark checkpoints."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def compare(reference, candidate, rtol=1e-10, atol=1e-12):
    steps = min(len(reference["times"]), len(candidate["times"]))
    if steps == 0:
        raise ValueError("Both checkpoints must contain completed steps")
    np.testing.assert_array_equal(reference["num_paulis"][:steps], candidate["num_paulis"][:steps])
    for name, count in (("all_results", steps + 1), ("norms", steps)):
        np.testing.assert_allclose(reference[name][:count], candidate[name][:count], rtol=rtol, atol=atol)
    return {
        "compared_steps": steps,
        "reference_completed_steps": len(reference["times"]),
        "candidate_completed_steps": len(candidate["times"]),
        "max_expectation_abs_error": float(np.max(np.abs(
            np.asarray(reference["all_results"][:steps + 1]) - candidate["all_results"][:steps + 1]))),
        "max_norm_abs_error": float(np.max(np.abs(
            np.asarray(reference["norms"][:steps]) - candidate["norms"][:steps]))),
        "rows": [{
            "step": i + 1,
            "num_paulis": reference["num_paulis"][i],
            "reference_seconds": reference["times"][i],
            "candidate_seconds": candidate["times"][i],
            "speedup": reference["times"][i] / candidate["times"][i],
            "reference_throughput": reference["avg_speeds"][i - 1] if i else None,
            "candidate_throughput": candidate["avg_speeds"][i - 1] if i else None,
        } for i in range(steps)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    with args.reference.open("rb") as handle:
        reference = pickle.load(handle)
    with args.candidate.open("rb") as handle:
        candidate = pickle.load(handle)
    report = compare(reference, candidate)
    print("step       strings     reference s    candidate s    speedup")
    for row in report["rows"]:
        print(f'{row["step"]:4d} {row["num_paulis"]:13,d} '
              f'{row["reference_seconds"]:15.6f} {row["candidate_seconds"]:14.6f} '
              f'{row["speedup"]:9.2f}x')
    print(f'Counts match through step {report["compared_steps"]}; '
          f'max expectation error {report["max_expectation_abs_error"]:.3g}; '
          f'max norm error {report["max_norm_abs_error"]:.3g}')
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
