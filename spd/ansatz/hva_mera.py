"""TFI HVA tensors and a periodic binary MERA with explicit bond dimension.

See docs/hva_tmera.md for conventions. All optimizer angles use half turns.

Reading order
-------------
1. ``binary_mera_sites`` assigns fixed qubit indices to sites at one scale.
2. ``binary_mera_layers`` uses those sites to describe every W and U placement.
3. ``_selected_layers`` includes or omits the optional physical bottom layer.
4. ``_build_network`` walks the description coarse to fine and emits gates.
5. ``tfi_binary_mera_channels`` inserts CreateZero before new qubits are used.

One preparation layer expands parent sites, then applies disentanglers between
neighboring child sites::

    scale s+1:       [ parent 0 ]       [ parent 1 ]
                         | W0                | W1
    scale s:         [c0] [c1] ---- U ---- [c2] [c3]
                         ^                         |
                         +------ periodic U ------+

``BinaryMERALayer.scale`` is the child scale ``s`` in this picture. Geometry is
constructed first; ``_Builder`` only translates that geometry into rotations,
barriers, and VariationalCircuit parameter metadata.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from pytket import Circuit, OpType

from ..circuit_ir import CircuitIR, CreateZero
from ..pytket_frontend import parse_pytket_circuit
from ..variational_circuit import VariationalCircuit


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@dataclass(frozen=True)
class HVATensor:
    """One isometry (W) or disentangler (U), using fixed qubit indices.

    ``outputs`` is the ordered qubit chain seen by the HVA block. A W also
    partitions it into existing ``retained`` inputs and fresh ``ancillas``.
    A U has no retained/ancilla partition. A square bottom W adds no qubits.
    """
    outputs: tuple[int, ...]
    retained: tuple[int, ...] = ()
    ancillas: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("outputs", "retained", "ancillas"):
            values = tuple(getattr(self, name))
            if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in values):
                raise ValueError("Tensor indices must be nonnegative integers.")
            if len(set(values)) != len(values):
                raise ValueError("Tensor indices must be unique.")
            object.__setattr__(self, name, values)
        if len(self.outputs) < 2:
            raise ValueError("A tensor needs at least two outputs.")
        if self.retained or self.ancillas:
            if (not self.retained
                    or set(self.retained) & set(self.ancillas)
                    or set(self.retained) | set(self.ancillas) != set(self.outputs)):
                raise ValueError("Retained inputs and ancillas must partition outputs.")


@dataclass(frozen=True)
class BinaryMERALayer:
    """All tensor placements for one child scale.

    Preparation applies every W in ``isometries`` in parallel, then every U in
    ``disentanglers`` in parallel. Parameters are shared within each tuple.
    """
    scale: int
    isometries: tuple[HVATensor, ...]
    disentanglers: tuple[HVATensor, ...]

    @property
    def ancillas(self):
        return tuple(i for tensor in self.isometries for i in tensor.ancillas)



# ---- Fixed-index MERA geometry -------------------------------------------

def _dimensions(system_size, chi):
    n = _positive_int(system_size, "system_size")
    if n < 2 or n & (n - 1):
        raise ValueError("system_size must be a power of two, at least 2.")
    chi = _positive_int(chi, "chi")
    if chi not in (2, 4, 8):
        raise ValueError("chi must be 2, 4, or 8.")
    q = chi.bit_length() - 1
    if q > n:
        raise ValueError("The top site's qubit count log2(chi) cannot exceed system_size.")
    return n, q


def binary_mera_sites(system_size, scale, *, chi=4):
    """Ordered fixed indices for each site at scale s (s=0 is physical).

    A site's dimension is min(chi, 2**(2**s)). Each block retains its
    rightmost min(log2(chi), 2**s) physical indices. This includes the top.
    """
    n, q = _dimensions(system_size, chi)
    if (isinstance(scale, bool) or not isinstance(scale, Integral)
            or not 0 <= scale <= n.bit_length() - 1):
        raise ValueError("scale must be an integer between 0 and log2(system_size).")
    stride = 2**int(scale)
    width = min(q, stride)
    return tuple(tuple(range(end - width, end)) for end in range(stride, n + 1, stride))


def binary_mera_layers(system_size, *, chi=4):
    """Layers in preparation order, from coarse to fine.

    At chi=4 the bottom W is square (two retained qubits, no ancillas).
    At chi=8 the next W maps three retained qubits to four outputs.
    Once each site has q = log2(chi) qubits, W maps q inputs to 2q
    outputs and U acts on 2q qubits.
    """
    n, _ = _dimensions(system_size, chi)
    layers = []
    for s in reversed(range(n.bit_length() - 1)):
        sites = binary_mera_sites(n, s, chi=chi)
        parents = binary_mera_sites(n, s + 1, chi=chi)
        w = []
        for j, parent in enumerate(parents):
            outputs = sites[2*j] + sites[2*j+1]
            ancillas = tuple(i for i in outputs if i not in parent)
            w.append(HVATensor(outputs, parent, ancillas))
        # Chosen top termination: one W on the two child sites, with no U.
        u = (() if len(sites) == 2 else tuple(
            HVATensor(sites[j] + sites[(j + 1) % len(sites)])
            for j in range(1, len(sites), 2)))
        layers.append(BinaryMERALayer(s, tuple(w), u))
    return tuple(layers)


def _selected_layers(system_size, chi, physical_bottom):
    """Return coarse-to-fine layers, optionally dropping physical scale 0."""
    layers = binary_mera_layers(system_size, chi=chi)
    if not isinstance(physical_bottom, (bool, np.bool_)):
        raise ValueError("physical_bottom must be boolean.")
    if not physical_bottom:
        if chi == 2:
            raise ValueError("physical_bottom=False is not defined for chi=2.")
        layers = layers[:-1]
    return layers



# ---- Parameter layout ----------------------------------------------------

def _brickwork_parameter_count(width):
    """Independent angles in one shared brickwork round for W or U."""
    if width == 2:
        return 3  # ZZ(0,1), X(0), X(1)
    if width < 4 or width % 2:
        raise ValueError("Brickwork tensors need an even width of at least two.")
    return 3 * width - 1  # even ZZ + X + open-chain odd ZZ + X


def binary_mera_parameter_shape(system_size, brickwork_rounds=1, *, chi=4,
                                physical_bottom=True):
    """Shape of the flat parameter array, with spatial sharing within each layer.

    Layout is coarse to fine, W then U at each scale. Within each W or U round,
    blocks of four or more qubits store even ZZ, X, open-chain odd ZZ, X.
    A two-qubit block stores ZZ, X0, X1.
    """
    rounds = _positive_int(brickwork_rounds, "brickwork_rounds")
    layers = _selected_layers(system_size, chi, physical_bottom)
    per_round = sum(_brickwork_parameter_count(len(tensors[0].outputs))
                    for layer in layers
                    for tensors in (layer.isometries, layer.disentanglers)
                    if tensors)
    return (rounds * per_round,)


def _params(params, shape):
    values = np.asarray(params, dtype=float)
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError(f"params must be finite with shape {shape}.")
    return values



# ---- Gate emission -------------------------------------------------------

class _Builder:
    """Emit one circuit while preserving the geometry parameter sharing."""
    def __init__(self, n, params):
        self.circuit = Circuit(n)
        self.params = params
        self.metadata = {}

    def rotation(self, name, qubits, index, factor=1., fixed=0.):
        # Tags survive pytket's topological reordering of disjoint commands.
        tag = f"hva_{len(self.metadata)}"
        self.metadata[tag] = (index, factor)
        angle = fixed if index == -1 else self.params.flat[index] * factor
        getattr(self.circuit, name)(float(angle), *qubits, opgroup=tag)

    def barrier(self):
        self.circuit.add_barrier(list(range(self.circuit.n_qubits)))

    def prepare_zero_inputs(self, indices):
        for i in indices:
            self.rotation("Ry", (i,), -1, fixed=.5)
        self.barrier()

    def add_tensors(self, tensors, parameter_indices):
        """Apply spatially shared brickwork rounds to disjoint tensors."""
        width = len(tensors[0].outputs)
        expected = _brickwork_parameter_count(width)
        if parameter_indices.ndim != 2 or parameter_indices.shape[1] != expected:
            raise ValueError(f"Expected brickwork parameter rows of width {expected}.")
        for indices in parameter_indices:
            cursor = 0
            if width == 2:
                for tensor in tensors:
                    self.rotation("ZZPhase", tensor.outputs, int(indices[cursor]))
                cursor += 1
                self.barrier()
                for tensor in tensors:
                    for position in range(2):
                        self.rotation("Rx", (tensor.outputs[position],),
                                      int(indices[cursor + position]))
                self.barrier()
                continue
            for tensor in tensors:
                for bond in range(width // 2):
                    j = 2 * bond
                    self.rotation("ZZPhase", tensor.outputs[j:j+2],
                                  int(indices[cursor + bond]))
            cursor += width // 2
            self.barrier()
            for tensor in tensors:
                for position in range(width):
                    self.rotation("Rx", (tensor.outputs[position],),
                                  int(indices[cursor + position]))
            cursor += width
            self.barrier()
            for tensor in tensors:
                for bond in range(width // 2 - 1):
                    j = 2 * bond + 1
                    pair = tensor.outputs[j:j+2]
                    self.rotation("ZZPhase", pair, int(indices[cursor + bond]))
            cursor += width // 2 - 1
            self.barrier()
            for tensor in tensors:
                for position in range(width):
                    self.rotation("Rx", (tensor.outputs[position],),
                                  int(indices[cursor + position]))
            self.barrier()

    def build(self):
        metadata = [self.metadata[c.opgroup] for c in self.circuit.get_commands()
                    if c.op.type != OpType.Barrier]
        indices, factors = zip(*metadata)
        return VariationalCircuit(self.circuit, indices, self.params.shape, factors)



# ---- Public block and network builders ----------------------------------

def tfi_hva_tensor_layer(params, tensors, system_size):
    """Build disjoint spatially shared tensors with synchronized brickwork.

    Parameters are independent within one tensor and shared between tensor
    instances in this layer. Four-qubit rounds have 11 parameters ordered as
    even ZZ, X, open-chain odd ZZ, X. Two-qubit rounds have three.
    An isometry includes fixed
    Ry(1/2) on its zero ancillas; ordinary unitary tensors have no preparation.
    All tensors in this call must have the same retained and ancilla positions within their output chains.
    """
    n = _positive_int(system_size, "system_size")
    tensors = tuple(tensors)
    if not tensors or not all(isinstance(t, HVATensor) for t in tensors):
        raise ValueError("Supply at least one HVATensor.")
    outputs = [i for t in tensors for i in t.outputs]
    if len(set(outputs)) != len(outputs) or max(outputs) >= n:
        raise ValueError("Tensor outputs must be disjoint and within system_size.")
    input_layouts = {(len(t.outputs), tuple(t.outputs.index(i) for i in t.retained),
              tuple(t.outputs.index(i) for i in t.ancillas)) for t in tensors}
    if len(input_layouts) != 1:
        raise ValueError(
            "Spatially shared tensors must have identical retained and ancilla positions."
        )
    values = np.asarray(params, dtype=float)
    width = len(tensors[0].outputs)
    count = _brickwork_parameter_count(width)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError(f"params must have shape (rounds, {count}).")
    values = _params(values, (values.shape[0], count))
    builder = _Builder(n, values)
    ancillas = tuple(i for t in tensors for i in t.ancillas)
    if ancillas:
        builder.prepare_zero_inputs(ancillas)
    builder.add_tensors(tensors, np.arange(values.size).reshape(values.shape))
    return builder.build()


def _build_network(params, system_size, brickwork_rounds, chi, physical_bottom):
    """Emit the full-register circuit and record later CreateZero positions."""
    rounds = _positive_int(brickwork_rounds, "brickwork_rounds")
    layers = _selected_layers(system_size, chi, physical_bottom)
    values = _params(params, binary_mera_parameter_shape(
        system_size, rounds, chi=chi, physical_bottom=physical_bottom))
    builder = _Builder(int(system_size), values)
    top_scale = int(system_size).bit_length() - 1
    top = binary_mera_sites(system_size, top_scale, chi=chi)[0]
    # The top site is the only input present before coarse-to-fine expansion.
    initializations = [(0, top)]
    builder.prepare_zero_inputs(top)
    cursor = 0
    for layer in layers:
        # A W expands retained parent inputs with these newly available qubits.
        if layer.ancillas:
            initializations.append((len(builder.circuit.get_commands()), layer.ancillas))
            builder.prepare_zero_inputs(layer.ancillas)
        # Emit the complete spatial W sublayer, then the complete U sublayer.
        for tensors in (layer.isometries, layer.disentanglers):
            if not tensors:
                continue
            count = _brickwork_parameter_count(len(tensors[0].outputs))
            size = rounds * count
            indices = np.arange(cursor, cursor + size).reshape(rounds, count)
            builder.add_tensors(tensors, indices)
            cursor += size
    assert cursor == values.size
    return builder.build(), tuple(initializations)


def tfi_binary_mera(params, system_size=8, brickwork_rounds=1, *, chi=4,
                    physical_bottom=True):
    """Full-register zero-input reference; chi is the maximum bond dimension."""
    return _build_network(params, system_size, brickwork_rounds, chi, physical_bottom)[0]


def binary_mera_qubit_initializations(system_size, brickwork_rounds=1, *, chi=4,
                                    physical_bottom=True):
    """Return (command offset, qubit indices) for each group of new zero inputs.

    This includes the top input and the ancillas added by each layer. An offset
    is the position just before their Ry preparation in the reference pytket
    command list, including barriers, before CreateZero operations are inserted.
    Square bottom isometries add no qubits and therefore have no entry.
    """
    shape = binary_mera_parameter_shape(
        system_size, brickwork_rounds, chi=chi, physical_bottom=physical_bottom)
    return _build_network(np.zeros(shape), system_size, brickwork_rounds, chi,
                    physical_bottom)[1]


def tfi_binary_mera_channels(params, system_size=8, brickwork_rounds=1, *, chi=4,
                             physical_bottom=True):
    """Build CircuitIR with CreateZero before each new qubit is prepared.

    Uses the static-channel API from main. All input indices are created
    explicitly. Optimizer parameters remain half turns;
    direct-IR gate factors include pi because IR rotation angles are radians.
    Execute with a channel-capable backend (NumPy, JAX, or native Triton);
    the runner owns contraction grouping and checkpoints. No reset/discard
    operations are needed for this preparation network.
    """
    reference, initializations = _build_network(
        params, system_size, brickwork_rounds, chi, physical_bottom)
    ir = parse_pytket_circuit(reference.circuit, int(system_size))
    insertions = dict(initializations)
    operations = []
    for offset, op in enumerate(ir.operations):
        operations.extend(CreateZero(i) for i in insertions.get(offset, ()))
        operations.append(op)
    return VariationalCircuit(
        CircuitIR(int(system_size), tuple(operations)),
        reference.gate_parameter_indices,
        reference.parameter_shape,
        np.pi * reference.gate_parameter_factors,
    )


def binary_mera_causal_cone(system_size, physical_sites, *, chi=4,
                            physical_bottom=True):
    """Conservative causal-cone support after each coarse-graining layer.

    Returns sorted fixed indices, starting with the requested physical support.
    This is structural analysis only; it never removes SPO columns.
    """
    layers = _selected_layers(system_size, chi, physical_bottom)
    support = set(physical_sites)
    if any(isinstance(i, bool) or not isinstance(i, Integral)
           or i < 0 or i >= system_size for i in support):
        raise ValueError("physical_sites must contain valid integer indices.")
    result = [tuple(sorted(support))]
    for layer in reversed(layers):
        for tensor in layer.disentanglers:
            if support.intersection(tensor.outputs):
                support.update(tensor.outputs)
        support = {i for tensor in layer.isometries
                   if support.intersection(tensor.outputs) for i in tensor.retained}
        result.append(tuple(sorted(support)))
    return tuple(result)
