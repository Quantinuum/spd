"""Private persistent storage within one immutable-input forward sequence.

Dead keys remain indexed until compaction, but
have zero coefficients immediately. Pair owners update both coefficients; new
keys are indexed only after the rotation kernel completes.
"""
import math

import torch
import triton
import triton.language as tl

from .kernels import _mix, _popc, clifford


@triton.jit
def _cas(ptr, value, active):
    return tl.inline_asm_elementwise(
        """{ .reg .pred p; setp.ne.u32 p, $3, 0; mov.u32 $0, -1;
             @p atom.global.cas.b32 $0, [$1], -1, $2; }""",
        constraints="=r,l,r,r", args=[ptr, value, active.to(tl.int32)],
        dtype=tl.int32, is_pure=False, pack=1)


@triton.jit(do_not_specialize=['start', 'end', 'mask'])
def _insert(keys, table, start, end, mask, W: tl.constexpr, B: tl.constexpr):
    i = start + tl.program_id(0) * B + tl.arange(0, B)
    pending = i < end
    h = tl.full((B,), 0x9e3779b9, tl.uint32)
    for w in tl.static_range(W):
        h = _mix(h, tl.load(keys + i.to(tl.int64) * W + w, pending, 0).to(tl.uint32))
    slot = h & mask
    while tl.sum(pending.to(tl.int32), 0) > 0:
        old = _cas(table + slot, i, pending)
        pending &= old != -1
        slot = (slot + pending.to(tl.uint32)) & mask


@triton.jit(do_not_specialize=['n', 'mask'])
def _update(keys, coeff, table, gate, scalars, counts, n, mask,
            W: tl.constexpr, B: tl.constexpr,
            stats=None, DIAGNOSTICS: tl.constexpr = False):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < n
    parity = tl.full((B,), 0, tl.int32)
    phase = tl.full((B,), 0, tl.int32)
    for w in tl.static_range(W // 2):
        x = tl.load(keys + i.to(tl.int64) * W + w, valid, 0).to(tl.uint32)
        z = tl.load(keys + i.to(tl.int64) * W + W // 2 + w, valid, 0).to(tl.uint32)
        gx = tl.load(gate + w).to(tl.uint32)
        gz = tl.load(gate + W // 2 + w).to(tl.uint32)
        parity += _popc(x & gz) + _popc(z & gx)
        phase += (2 * _popc(gx & z) + _popc(gx & gz) + _popc(x & z)
                  - _popc((x ^ gx) & (z ^ gz)))
    anti = valid & ((parity & 1) != 0)
    h = tl.full((B,), 0x9e3779b9, tl.uint32)
    for w in tl.static_range(W):
        v = tl.load(keys + i.to(tl.int64) * W + w, anti, 0).to(tl.uint32)
        g = tl.load(gate + w).to(tl.uint32)
        h = _mix(h, v ^ g)
    slot = h & mask
    pending = anti
    partner = tl.full((B,), -1, tl.int32)
    while tl.sum(pending.to(tl.int32), 0) > 0:
        idx = tl.load(table + slot, pending, -1)
        equal = pending & (idx >= 0)
        for w in tl.static_range(W):
            v = tl.load(keys + i.to(tl.int64) * W + w, pending, 0).to(tl.uint32)
            g = tl.load(gate + w).to(tl.uint32)
            other = tl.load(keys + idx.to(tl.int64) * W + w, pending & (idx >= 0), 0).to(tl.uint32)
            equal &= other == (v ^ g)
        partner = tl.where(equal, idx, partner)
        pending &= (idx >= 0) & ~equal
        slot = (slot + pending.to(tl.uint32)) & mask
    # Ownership depends only on stable keys/indices, never concurrently written
    # coefficients. Even a dead owner must update its live partner.
    owner = valid & ((partner < 0) | (i < partner))
    paired_owner = owner & anti & (partner >= 0)
    c = tl.load(coeff + i, owner, 0)
    paired = tl.load(coeff + partner, paired_owner, 0)
    cos = tl.load(scalars)
    sin = tl.load(scalars + 1)
    cutoff = tl.load(scalars + 2)
    sign = tl.where((phase & 3) == 1, 1, -1)
    own = tl.where(anti, c * cos - paired * sin * sign, c)
    # Match the original partner row's arithmetic order (phase sign is opposite).
    other = paired * cos - c * sin * (-sign)
    keep_own = owner & (tl.abs(own) > cutoff)
    keep_other = owner & anti & (tl.abs(other) > cutoff)
    if DIAGNOSTICS:
        removed_own = owner & ~keep_own & (own != 0)
        removed_other = owner & anti & ~keep_other & (other != 0)
        a = tl.where(removed_own, tl.abs(own), 0).to(tl.float64)
        b = tl.where(removed_other, tl.abs(other), 0).to(tl.float64)
        tl.store(stats + tl.program_id(0) * 3,
                 tl.sum(removed_own.to(tl.int32) + removed_other.to(tl.int32), 0))
        tl.store(stats + tl.program_id(0) * 3 + 1, tl.sum(a + b, 0))
        tl.store(stats + tl.program_id(0) * 3 + 2, tl.sum(a * a + b * b, 0))
    tl.store(coeff + i, tl.where(keep_own, own, 0), owner)
    tl.store(coeff + partner, tl.where(keep_other, other, 0), paired_owner)
    create = keep_other & (partner < 0)
    offsets = tl.cumsum(create.to(tl.int32)) - create.to(tl.int32)
    base = tl.atomic_add(counts, tl.sum(create.to(tl.int32), 0), sem='relaxed')
    dest = n + base + offsets
    for w in tl.static_range(W):
        v = tl.load(keys + i.to(tl.int64) * W + w, create, 0)
        g = tl.load(gate + w).to(tl.uint32)
        tl.store(keys + dest.to(tl.int64) * W + w, v ^ g, create)
    tl.store(coeff + dest, other, create)
    delta = (keep_own.to(tl.int32) - (owner & (c != 0)).to(tl.int32)
             + keep_other.to(tl.int32) - (paired_owner & (paired != 0)).to(tl.int32))
    tl.atomic_add(counts + 1, tl.sum(delta, 0), sem='relaxed')


@triton.jit(do_not_specialize=['n'])
def _compact(keys, coeff, out_keys, out_coeff, count, n,
             W: tl.constexpr, B: tl.constexpr,
             grad=None, out_grad=None, GRADIENT: tl.constexpr = False):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c = tl.load(coeff + i, i < n, 0)
    keep = (i < n) & (c != 0)
    if GRADIENT:
        g = tl.load(grad + i, i < n, 0)
        keep |= (i < n) & (g != 0)
    offsets = tl.cumsum(keep.to(tl.int32)) - keep.to(tl.int32)
    base = tl.atomic_add(count, tl.sum(keep.to(tl.int32), 0), sem='relaxed')
    dest = base + offsets
    for w in tl.static_range(W):
        v = tl.load(keys + i.to(tl.int64) * W + w, keep, 0)
        tl.store(out_keys + dest.to(tl.int64) * W + w, v, keep)
    tl.store(out_coeff + dest, c, keep)
    if GRADIENT:
        tl.store(out_grad + dest, g, keep)


class _Storage:
    """Packed storage owned by an SPO/SPGO, with optional lazy execution buffers.

    Wrapped/exposed tensors may have aliases. Only an owned copy is mutated by
    kernels; this keeps constructors and zero-copy gradient views safe.
    """
    @classmethod
    def wrap(cls, keys, coeff, nq, grad=None, *, pruned=False, live=None):
        self = cls.__new__(cls)
        self.keys, self.coeff, self.grad = keys, coeff, grad
        self.n = self.capacity = len(coeff)
        self.width, self.nq = keys.shape[1], nq
        self.device, self.dtype = coeff.device, coeff.dtype
        self.pruned, self.live = pruned, live
        self.owned = False
        self.dead_fraction = .1
        self.table = self.counts = None
        self.table_size = 0
        self.rebuilds = self.compactions = self.growths = 0
        return self

    def __init__(self, state, dead_fraction):
        source = state._storage
        self.n = source.n
        self.width, self.nq = source.width, source.nq
        self.device, self.dtype = source.device, source.dtype
        self.pruned = source.pruned
        self.dead_fraction = dead_fraction
        self.capacity = max(16, 2 * self.n)
        self.keys = torch.empty((self.capacity, self.width), dtype=torch.int32, device=self.device)
        self.coeff = torch.empty(self.capacity, dtype=self.dtype, device=self.device)
        self.keys[:self.n].copy_(source.keys[:self.n])
        self.coeff[:self.n].copy_(source.coeff[:self.n])
        self.grad = None
        if source.grad is not None:
            self.grad = torch.empty(self.capacity, dtype=self.dtype, device=self.device)
            self.grad[:self.n].copy_(source.grad[:self.n])
            self.live = int(torch.count_nonzero(
                (source.coeff[:self.n] != 0) | (source.grad[:self.n] != 0)).item())
        else:
            self.live = int(torch.count_nonzero(source.coeff[:self.n]).item())
        self.owned = True
        self.counts = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.rebuilds = self.compactions = self.growths = 0
        if source.owned and source.table is not None:
            self.table_size = source.table_size
            self.table = source.table.clone()
        else:
            self._rebuild()

    def size(self):
        return self.live if self.pruned else self.n

    def materialize(self):
        """Compact only at an explicit interoperability/public return boundary."""
        if not self.owned:
            # Another wrapper may have exposed/modified these shared keys.
            self.table = None
            self.table_size = 0
        if self.pruned and self.n != self.live:
            self.compact(final=True)
        elif self.capacity != self.n:
            self.keys = self.keys[:self.n].clone()
            self.coeff = self.coeff[:self.n].clone()
            if self.grad is not None:
                self.grad = self.grad[:self.n].clone()
            self.capacity = self.n
            self.owned = True
        # Index contents stay valid when rows are merely copied at the same IDs.

    def export(self):
        """Expose compact tensors; subsequent mutation must protect their aliases."""
        with torch.cuda.device(self.device):
            self.materialize()
        self.owned = False
        self.table = None  # Exposed keys can be modified by an external caller.
        self.table_size = 0
        self.pruned = False  # Explicit tensor rows, including user-written zeros.
        return self.keys[:self.n], self.coeff[:self.n], (None if self.grad is None else self.grad[:self.n])

    def _rebuild(self):
        self.table_size = 1 << (max(4, 4 * self.n) - 1).bit_length()
        self.table = torch.full((self.table_size,), -1, dtype=torch.int32, device=self.device)
        if self.n:
            _insert[((self.n + 127) // 128,)](self.keys, self.table, 0, self.n,
                                             self.table_size - 1, self.width, 128)
        self.rebuilds += 1

    def _reserve(self):
        needed = 2 * self.n
        if needed <= self.capacity:
            return
        self.capacity = max(needed, int(self.capacity * 1.25))
        keys = torch.empty((self.capacity, self.width), dtype=torch.int32, device=self.device)
        coeff = torch.empty(self.capacity, dtype=self.dtype, device=self.device)
        keys[:self.n].copy_(self.keys[:self.n])
        coeff[:self.n].copy_(self.coeff[:self.n])
        if self.grad is not None:
            grad = torch.empty(self.capacity, dtype=self.dtype, device=self.device)
            grad[:self.n].copy_(self.grad[:self.n])
            self.grad = grad
        self.keys, self.coeff = keys, coeff
        self.growths += 1

    def compact(self, cap=None, final=False, totals=None):
        size = self.live if cap is None else min(self.live, cap)
        capacity = size if final else max(16, 2 * size)
        keys = torch.empty((capacity, self.width), dtype=torch.int32, device=self.device)
        coeff = torch.empty(capacity, dtype=self.dtype, device=self.device)
        grad = torch.empty_like(coeff) if self.grad is not None else None
        if size:
            if size < self.live:
                selected = torch.topk(self.coeff[:self.n].abs(), size, sorted=False).indices
                if totals is not None:
                    # Sum removed terms directly: subtracting retained norms
                    # would lose small discarded contributions to cancellation.
                    removed = torch.ones(self.n, dtype=torch.bool, device=self.device)
                    removed[selected] = False
                    magnitudes = torch.where(removed, self.coeff[:self.n].abs(), 0).to(torch.float64)
                    totals += torch.stack([(magnitudes != 0).sum().to(torch.float64),
                                           magnitudes.sum(), magnitudes.square().sum()])
                keys[:size].copy_(self.keys[selected])
                coeff[:size].copy_(self.coeff[selected])
                if grad is not None:
                    grad[:size].copy_(self.grad[selected])
            else:
                if self.counts is None:
                    self.counts = torch.zeros(2, dtype=torch.int32, device=self.device)
                self.counts.zero_()
                _compact[((self.n + 127) // 128,)](self.keys, self.coeff, keys, coeff,
                                                  self.counts, self.n, self.width, 128,
                                                  grad=self.grad, out_grad=grad, GRADIENT=grad is not None)
        self.keys, self.coeff, self.grad = keys, coeff, grad
        self.n = self.live = size
        self.capacity = capacity
        self.compactions += 1
        self.owned = True
        self.pruned = True
        self.table = None
        self.table_size = 0
        if not final:
            self._rebuild()

    def apply_clifford(self, operation):
        """Map each private row exactly, then rebuild the changed key index."""
        import operator
        from ..circuit_ir import TwoQubitClifford

        codes = {"H": 0, "S": 1, "Sdg": 2, "X": 3, "Y": 4,
                 "Z": 5, "CX": 6, "CZ": 7, "CY": 8}
        name = operation.gate_name.removeprefix("OpType.")
        two = isinstance(operation, TwoQubitClifford)
        if name not in codes or two != name.startswith("C"):
            raise ValueError(f"Unsupported Clifford operation: {operation}")
        q = operator.index(operation.control_qubit if two else operation.qubit)
        r = operator.index(operation.target_qubit) if two else q
        if not 0 <= q < self.nq or not 0 <= r < self.nq or (two and q == r):
            raise ValueError("Clifford requires distinct qubits within system size")
        if self.n:
            # The kernel reads/writes only its own row. No inter-row dependence,
            # and all control bits are loaded before any packed words change.
            clifford[((self.n + 127) // 128,)](
                self.keys, self.coeff, self.keys, self.coeff,
                self.n, q, r, self.width, codes[name], 128)
            self._rebuild()

    def apply(self, gate, params, cap, *, diagnostics=False):
        if not self.n:
            return
        if self.n >= 2**30:
            raise ValueError('Persistent prototype requires fewer than 2**30 slots')
        self._reserve()
        if self.table is None:
            self._rebuild()
        if self.counts is None:
            self.counts = torch.zeros(2, dtype=torch.int32, device=self.device)
        # Accommodate the worst-case new rows before mutation, so insertion can
        # never fill the table or fail partway through a gate.
        if 2 * self.n > self.table_size // 2:
            self.table_size = 1 << (max(4, 4 * self.n) - 1).bit_length()
            self.table = torch.full((self.table_size,), -1, dtype=torch.int32, device=self.device)
            _insert[((self.n + 127) // 128,)](self.keys, self.table, 0, self.n,
                                             self.table_size - 1, self.width, 128)
            self.rebuilds += 1
        blocks = (self.n + 127) // 128
        stats = torch.empty((blocks, 3), dtype=torch.float64, device=self.device) if diagnostics else None
        self.counts.zero_()
        _update[((self.n + 127) // 128,)](self.keys, self.coeff, self.table, gate, params,
                                         self.counts, self.n, self.table_size - 1,
                                         self.width, 128, stats=stats, DIAGNOSTICS=diagnostics,
                                         enable_fp_fusion=False)
        added, delta = self.counts.cpu().tolist()
        before = self.n
        self.n += added
        self.live += delta
        self.pruned = True
        totals = stats.sum(dim=0) if diagnostics else None
        if cap is not None and self.live > cap:
            self.compact(cap, totals=totals)
        elif self.n - self.live > max(16, self.dead_fraction * self.n):
            self.compact()
        elif added:
            _insert[((added + 127) // 128,)](self.keys, self.table, before, self.n,
                                            self.table_size - 1, self.width, 128)
        if diagnostics:
            values = totals.tolist()
            return {"num_str_truncated": int(values[0]), "truncated_l1_norm": values[1],
                    "truncated_l2_norm": math.sqrt(values[2])}


def evolve_step_persistent(state, operations, trunc_val=0., max_num_str=None,
                           *, dead_fraction=.1, stats=None):
    """Copy once, mutate privately across gates, materialize once."""
    from . import SparsePauliOp, _gate_data
    from ..circuit_ir import (PauliRotation, SkippedOperation,
                              SingleQubitClifford, TwoQubitClifford)
    operations = tuple(operations)
    if not math.isfinite(trunc_val) or trunc_val < 0:
        raise ValueError('trunc_val must be finite and nonnegative')
    if not 0 < dead_fraction < 1:
        raise ValueError('dead_fraction must be between zero and one')
    if max_num_str is not None and (not isinstance(max_num_str, int) or max_num_str < 1):
        raise ValueError('max_num_str must be a positive integer or None')
    for op in operations:
        if not isinstance(op, (PauliRotation, SkippedOperation,
                               SingleQubitClifford, TwoQubitClifford)):
            raise NotImplementedError(f'Unsupported persistent operation: {op}')
        if isinstance(op, PauliRotation) and not math.isfinite(op.theta):
            raise ValueError('theta must be finite')
    if all(isinstance(op, SkippedOperation) for op in operations):
        if stats is not None:
            stats.update(rebuilds=0, compactions=0, growths=0)
        return state
    result = state.copy()
    result._storage.dead_fraction = dead_fraction
    for op in reversed(operations):
        result.apply_in_place(op, trunc_val, max_num_str, diagnostics=False)
    result.compact()
    if stats is not None:
        storage = result._storage
        stats.update(rebuilds=storage.rebuilds, compactions=storage.compactions,
                     growths=storage.growths)
    return result
