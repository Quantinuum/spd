"""Check sparse-index membership independently of the rotation arithmetic."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires NVIDIA GPU')

from spd.triton_backend import create_op, _pack
from spd.triton_backend.kernels import build_anticommuting_index


@pytest.mark.parametrize('gate_sites', [{}, {0: 'Z'}, {32: 'X'}, {64: 'Y'},
                                      {31: 'X', 32: 'X'}, {0: 'Y', 64: 'Z', 120: 'X'}])
def test_index_contains_only_anticommuting_original_rows(gate_sites):
    rng = np.random.default_rng(1937)
    labels = [''.join(row) for row in rng.choice(list('IXYZ'), (257, 121))]
    gate_chars = ['I'] * 121
    for q, char in gate_sites.items():
        gate_chars[q] = char
    gate = ''.join(gate_chars)
    # Pauli letters anticommute exactly when both are nonidentity and distinct.
    expected = [i for i, p in enumerate(labels)
                if sum(p[q] != 'I' and p[q] != g for q, g in gate_sites.items()) % 2]
    state = create_op(dict.fromkeys(labels, 1.), precision='double')
    packed_gate = torch.as_tensor(_pack(gate, 121), device='cuda')
    table = torch.full((1024,), -1, dtype=torch.int32, device='cuda')
    build_anticommuting_index[(3,)](state.xz_array, table, packed_gate, 257, 1023, 8, 128)
    indices = table.cpu().numpy()
    assert sorted(indices[indices >= 0].tolist()) == expected
    assert np.all((indices == -1) | ((indices >= 0) & (indices < 257)))


@pytest.mark.parametrize('active_rows', [[], [0, 128, 256], list(range(257))])
def test_index_inactive_lanes_and_partial_block(active_rows):
    # A Z on qubit 120 sees exactly the selected rows. Encode uniqueness on
    # different qubits so no input rows merge during construction.
    labels = []
    for i in range(257):
        p = ['I'] * 121
        for q in range(9):
            p[q] = 'X' if (i >> q) & 1 else 'I'
        p[120] = 'X' if i in active_rows else 'I'
        labels.append(''.join(p))
    state = create_op(dict.fromkeys(labels, 1.), precision='double')
    gate = torch.as_tensor(_pack('I' * 120 + 'Z', 121), device='cuda')
    table = torch.full((1024,), -1, dtype=torch.int32, device='cuda')
    build_anticommuting_index[(3,)](state.xz_array, table, gate, 257, 1023, 8, 128)
    indices = table.cpu().numpy()
    assert sorted(indices[indices >= 0].tolist()) == active_rows
    assert np.all(indices >= -1)
