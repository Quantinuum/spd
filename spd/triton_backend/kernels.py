"""Native GPU hash lookup and fused Pauli rotation/compaction kernels."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _popc(v):
    return libdevice.popc(v.to(tl.int32))


@triton.jit
def _mix(h, v):
    h = (h ^ v) * 0x85ebca6b
    h = h ^ (h >> 13)
    return h


@triton.jit(do_not_specialize=["n", "table_mask"])
def build_index(keys, table, n, table_mask, W: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    live = i < n
    h = tl.full((B,), 0x9e3779b9, tl.uint32)
    for w in tl.static_range(W):
        h = _mix(h, tl.load(keys + i.to(tl.int64) * W + w, live, 0).to(tl.uint32))
    slot = h & table_mask
    pending = live
    while tl.sum(pending.to(tl.int32), 0) > 0:
        # Inactive lanes CAS -2 -> -2, which never changes an occupied slot.
        old = tl.atomic_cas(table + slot, tl.where(pending, -1, -2),
                            tl.where(pending, i, -2), sem="relaxed")
        pending = pending & (old != -1)
        slot = (slot + pending.to(tl.uint32)) & table_mask


@triton.jit(do_not_specialize=["n", "table_mask"])
def rotate(keys, coeff, table, gate, scalars, out_keys, out_coeff, count,
           n, table_mask, W: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    live = i < n
    h = tl.full((B,), 0x9e3779b9, tl.uint32)
    parity = tl.full((B,), 0, tl.int32)
    phase = tl.full((B,), 0, tl.int32)
    for w in tl.static_range(W // 2):
        x = tl.load(keys + i.to(tl.int64) * W + w, live, 0).to(tl.uint32)
        z = tl.load(keys + i.to(tl.int64) * W + W // 2 + w, live, 0).to(tl.uint32)
        gx = tl.load(gate + w).to(tl.uint32)
        gz = tl.load(gate + W // 2 + w).to(tl.uint32)
        parity += _popc(x & gz) + _popc(z & gx)
        phase += (2 * _popc(gx & z) + _popc(gx & gz)
                  + _popc(x & z) - _popc((x ^ gx) & (z ^ gz)))
    anti = live & ((parity & 1) != 0)
    for w in tl.static_range(W):
        v = tl.load(keys + i.to(tl.int64) * W + w, live, 0).to(tl.uint32)
        g = tl.load(gate + w).to(tl.uint32)
        h = _mix(h, v ^ g)
    slot = h & table_mask
    pending = anti
    partner = tl.full((B,), -1, tl.int32)
    while tl.sum(pending.to(tl.int32), 0) > 0:
        idx = tl.load(table + slot, pending, -1)
        equal = pending & (idx >= 0)
        for w in tl.static_range(W):
            v = tl.load(keys + i.to(tl.int64) * W + w, live, 0).to(tl.uint32)
            g = tl.load(gate + w).to(tl.uint32)
            other = tl.load(keys + idx.to(tl.int64) * W + w, pending & (idx >= 0), 0).to(tl.uint32)
            equal = equal & (other == (v ^ g))
        partner = tl.where(equal, idx, partner)
        pending = pending & (idx >= 0) & ~equal
        slot = (slot + pending.to(tl.uint32)) & table_mask
    c = tl.load(coeff + i, live, 0)
    paired = tl.load(coeff + partner, partner >= 0, 0)
    cos = tl.load(scalars)
    sin = tl.load(scalars + 1)
    cutoff = tl.load(scalars + 2)
    sign = tl.where((phase & 3) == 1, 1, -1)
    own = tl.where(anti, c * cos - paired * sin * sign, c)
    new = c * sin * sign
    keep_own = live & (tl.abs(own) > cutoff)
    keep_new = anti & (partner < 0) & (tl.abs(new) > cutoff)
    sizes = keep_own.to(tl.int32) + keep_new.to(tl.int32)
    offset = tl.cumsum(sizes) - sizes
    base = tl.atomic_add(count, tl.sum(sizes, 0), sem="relaxed")
    dest = base + offset
    for w in tl.static_range(W):
        v = tl.load(keys + i.to(tl.int64) * W + w, live, 0)
        g = tl.load(gate + w)
        tl.store(out_keys + dest.to(tl.int64) * W + w, v, keep_own)
        tl.store(out_keys + (dest.to(tl.int64) + keep_own.to(tl.int32)) * W + w, v ^ g, keep_new)
    tl.store(out_coeff + dest, own, keep_own)
    tl.store(out_coeff + dest + keep_own.to(tl.int32), new, keep_new)
