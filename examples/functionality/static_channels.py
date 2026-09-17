"""Run from the repository root: python -m examples.functionality.static_channels."""
import numpy as np

from spd import (CircuitIR, CreateZero, ResetZero, VariationalCircuit,
                 create_spo, evolve, init_gradient_spo, backpropagate)
from spd.circuit_ir import PauliRotation, TwoQubitClifford, SkippedOperation


def creation_example():
    # Fixed indices, three creation sublayers; no MERA ansatz is constructed.
    operations = []
    for indices, pairs in [([3, 7], [(3, 7)]),
                           ([1, 5], [(1, 3), (5, 7)]),
                           ([0, 2, 4, 6], [(0, 1), (2, 3), (4, 5), (6, 7)])]:
        operations.extend(CreateZero(q) for q in indices)
        for a, b in pairs:
            p = ['I'] * 8
            p[a] = 'Y'
            operations.append(PauliRotation('Ry', ''.join(p), .3))
            operations.append(TwoQubitClifford('OpType.CX', a, b))
        operations.append(SkippedOperation('OpType.Barrier'))
    circuit = CircuitIR(8, operations)
    observable = create_spo({'ZIIIIIII': 1.}, backend_name='numpy', precision='double')
    result, info = evolve(observable, circuit, 0., 10000, progress=False)
    widths = info['active_widths']
    print('Creation widths:', [w for i, w in enumerate(widths) if i == 0 or w != widths[i-1]])
    print('Scalar expectation:', result.get_expectation_value(), 'active indices:', result.active_qubits)
    _, gradients, _ = backpropagate(init_gradient_spo(result), circuit, 0., 10000, progress=False)
    print('Rotation gradients:', np.asarray(gradients))


def reset_example(parameter=.4):
    # Prepare a correlated pair, replace one state, then rotate the replacement.
    # Both rotations share the parameter, with different radian-angle factors.
    circuit = CircuitIR(2, [PauliRotation('Ry', 'YI', parameter),
                            TwoQubitClifford('OpType.CX', 0, 1), ResetZero(0),
                            PauliRotation('Ry', 'YI', 2 * parameter)])
    variational = VariationalCircuit(circuit, [0, 0], (1,), [1., 2.])
    observable = create_spo({'ZZ': 1.}, backend_name='numpy', precision='double')
    result, _ = evolve(observable, circuit, 0., 1000, progress=False)
    _, gradients, _ = backpropagate(init_gradient_spo(result), circuit, 0., 1000, progress=False)
    value = result.get_expectation_value()
    gradient = variational.parameter_gradients(gradients)[0]
    exact = np.cos(parameter) * np.cos(2 * parameter)
    exact_gradient = -np.sin(parameter) * np.cos(2 * parameter) - 2*np.cos(parameter)*np.sin(2*parameter)
    np.testing.assert_allclose([value, gradient], [exact, exact_gradient], atol=1e-12)
    print('Reset preparation expectation / shared-parameter gradient:', value, gradient)


if __name__ == '__main__':
    creation_example()
    reset_example()
