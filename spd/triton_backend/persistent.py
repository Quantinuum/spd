"""Experimental persistent storage within one immutable-input forward timestep.

Not selected by the public API. Dead keys remain indexed until compaction, but
have zero coefficients immediately. Pair owners update both coefficients; new
keys are indexed only after the rotation kernel completes.
"""
import math

import torch
import triton
import triton.language as tl

from .kernels import _mix, _popc


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
            W: tl.constexpr, B: tl.constexpr):
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
             W: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c = tl.load(coeff + i, i < n, 0)
    keep = (i < n) & (c != 0)
    offsets = tl.cumsum(keep.to(tl.int32)) - keep.to(tl.int32)
    base = tl.atomic_add(count, tl.sum(keep.to(tl.int32), 0), sem='relaxed')
    dest = base + offsets
    for w in tl.static_range(W):
        v = tl.load(keys + i.to(tl.int64) * W + w, keep, 0)
        tl.store(out_keys + dest.to(tl.int64) * W + w, v, keep)
    tl.store(out_coeff + dest, c, keep)


class _Storage:
    def __init__(self, state, dead_fraction):
        self.n = state.get_size()
        self.live = int(torch.count_nonzero(state.c_array).item())
        self.width = state.xz_array.shape[1]
        self.nq = state.num_qubits
        self.device = state.c_array.device
        self.dtype = state.c_array.dtype
        self.dead_fraction = dead_fraction
        self.capacity = max(16, 2 * self.n)
        self.keys = torch.empty((self.capacity, self.width), dtype=torch.int32, device=self.device)
        self.coeff = torch.empty(self.capacity, dtype=self.dtype, device=self.device)
        self.keys[:self.n].copy_(state.xz_array)
        self.coeff[:self.n].copy_(state.c_array)
        self.counts = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.rebuilds = self.compactions = self.growths = 0
        self._rebuild()

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
        self.keys, self.coeff = keys, coeff
        self.growths += 1

    def compact(self, cap=None, final=False):
        size = self.live if cap is None else min(self.live, cap)
        capacity = size if final else max(16, 2 * size)
        keys = torch.empty((capacity, self.width), dtype=torch.int32, device=self.device)
        coeff = torch.empty(capacity, dtype=self.dtype, device=self.device)
        if size:
            if size < self.live:
                selected = torch.topk(self.coeff[:self.n].abs(), size, sorted=False).indices
                keys[:size].copy_(self.keys[selected])
                coeff[:size].copy_(self.coeff[selected])
            else:
                self.counts.zero_()
                _compact[((self.n + 127) // 128,)](self.keys, self.coeff, keys, coeff,
                                                  self.counts, self.n, self.width, 128)
        self.keys, self.coeff = keys, coeff
        self.n = self.live = size
        self.capacity = capacity
        self.compactions += 1
        if not final:
            self._rebuild()

    def apply(self, gate, params, cap):
        if not self.n:
            return
        if self.n >= 2**30:
            raise ValueError('Persistent prototype requires fewer than 2**30 slots')
        self._reserve()
        # Accommodate the worst-case new rows before mutation, so insertion can
        # never fill the table or fail partway through a gate.
        if 2 * self.n > self.table_size // 2:
            self.table_size = 1 << (max(4, 4 * self.n) - 1).bit_length()
            self.table = torch.full((self.table_size,), -1, dtype=torch.int32, device=self.device)
            _insert[((self.n + 127) // 128,)](self.keys, self.table, 0, self.n,
                                             self.table_size - 1, self.width, 128)
            self.rebuilds += 1
        self.counts.zero_()
        _update[((self.n + 127) // 128,)](self.keys, self.coeff, self.table, gate, params,
                                         self.counts, self.n, self.table_size - 1,
                                         self.width, 128, enable_fp_fusion=False)
        added, delta = self.counts.cpu().tolist()
        before = self.n
        self.n += added
        self.live += delta
        if cap is not None and self.live > cap:
            self.compact(cap)
        elif self.n - self.live > max(16, self.dead_fraction * self.n):
            self.compact()
        elif added:
            _insert[((added + 127) // 128,)](self.keys, self.table, before, self.n,
                                            self.table_size - 1, self.width, 128)


def evolve_step_persistent(state, operations, trunc_val=0., max_num_str=None,
                           *, dead_fraction=.1, stats=None):
    """Prototype: copy once, mutate privately across gates, materialize once."""
    from . import SparsePauliOp, _gate_data
    from ..circuit_ir import PauliRotation, SkippedOperation
    operations = tuple(operations)
    if not math.isfinite(trunc_val) or trunc_val < 0:
        raise ValueError('trunc_val must be finite and nonnegative')
    if not 0 < dead_fraction < 1:
        raise ValueError('dead_fraction must be between zero and one')
    if max_num_str is not None and (not isinstance(max_num_str, int) or max_num_str < 1):
        raise ValueError('max_num_str must be a positive integer or None')
    for op in operations:
        if not isinstance(op, (PauliRotation, SkippedOperation)):
            raise NotImplementedError('Persistent prototype supports Pauli rotations only')
        if isinstance(op, PauliRotation) and not math.isfinite(op.theta):
            raise ValueError('theta must be finite')
    if not any(isinstance(op, PauliRotation) for op in operations):
        if stats is not None:
            stats.update(rebuilds=0, compactions=0, growths=0)
        return state
    with torch.cuda.device(state.c_array.device):
        storage = _Storage(state, dead_fraction)
        for op in reversed(operations):
            if isinstance(op, PauliRotation):
                gate, params = _gate_data(op.pauli, float(op.theta), float(trunc_val),
                                          state.num_qubits, state.c_array.dtype, state.c_array.device)
                storage.apply(gate, params, max_num_str)
        storage.compact(final=True)
        if stats is not None:
            stats.update(rebuilds=storage.rebuilds, compactions=storage.compactions,
                         growths=storage.growths)
        return SparsePauliOp(storage.keys, storage.coeff, state.num_qubits)
