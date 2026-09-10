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
def build_anticommuting_index(keys, table, gate, n, table_mask,
                              W: tl.constexpr, B: tl.constexpr):
    """Index original row IDs only when they can be queried by forward rotate.

    P and P XOR G have identical commutation parity with G, so every needed
    partner is included. The full index remains available to other callers.
    """
    i = tl.program_id(0) * B + tl.arange(0, B)
    live = i < n
    parity = tl.full((B,), 0, tl.int32)
    for w in tl.static_range(W // 2):
        gx = tl.load(gate + w).to(tl.uint32)
        gz = tl.load(gate + W // 2 + w).to(tl.uint32)
        x = tl.load(keys + i.to(tl.int64) * W + w, live, 0).to(tl.uint32)
        z = tl.load(keys + i.to(tl.int64) * W + W // 2 + w, live, 0).to(tl.uint32)
        parity += _popc(x & gz) + _popc(z & gx)
    pending = live & ((parity & 1) != 0)
    h = tl.full((B,), 0x9e3779b9, tl.uint32)
    for w in tl.static_range(W):
        h = _mix(h, tl.load(keys + i.to(tl.int64) * W + w, pending, 0).to(tl.uint32))
    slot = h & table_mask
    while tl.sum(pending.to(tl.int32), 0) > 0:
        # tl.atomic_cas has no mask. Predicate the instruction itself so inactive
        # lanes do not generate atomic traffic (or contend on a dummy address).
        old = tl.inline_asm_elementwise(
            """{ .reg .pred p;
                 setp.ne.u32 p, $3, 0;
                 mov.u32 $0, -1;
                 @p atom.global.cas.b32 $0, [$1], -1, $2;
            }""",
            constraints="=r,l,r,r", args=[table + slot, i, pending.to(tl.int32)],
            dtype=tl.int32, is_pure=False, pack=1,
        )
        pending = pending & (old != -1)
        slot = (slot + pending.to(tl.uint32)) & table_mask


@triton.jit(do_not_specialize=["n", "table_mask"])
def rotate(keys, coeff, table, gate, scalars, out_keys, out_coeff, count,
           n, table_mask, W: tl.constexpr, B: tl.constexpr,
           grad=None, out_grad=None, stats=None, BACKWARD: tl.constexpr = False,
           DIAGNOSTICS: tl.constexpr = False):
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
    if BACKWARD:
        adj = tl.load(grad + i, live, 0)
        adj_pair = tl.load(grad + partner, partner >= 0, 0)
        # Each pair contributes once, independent of hash insertion order.
        theta_grad = tl.where(anti & (partner > i),
                              sign * (c * adj_pair - paired * adj), 0)
        tl.store(stats + tl.program_id(0) * 4 + 3,
                 tl.sum(theta_grad.to(tl.float64), 0))
        sin = -sin
        own_grad = tl.where(anti, adj * cos - adj_pair * sin * sign, adj)
        new_grad = adj * sin * sign
    own = tl.where(anti, c * cos - paired * sin * sign, c)
    new = c * sin * sign
    missing = anti & (partner < 0)
    if BACKWARD:
        keep_own = live & ((own != 0) | (own_grad != 0)) & (tl.abs(own) >= cutoff)
        keep_new = missing & ((new != 0) | (new_grad != 0)) & (tl.abs(new) >= cutoff)
    else:
        keep_own = live & (tl.abs(own) > cutoff)
        keep_new = missing & (tl.abs(new) > cutoff)
    if DIAGNOSTICS:
        removed_own = live & ~keep_own & (own != 0)
        removed_new = missing & ~keep_new & (new != 0)
        a = tl.where(removed_own, tl.abs(own), 0).to(tl.float64)
        b = tl.where(removed_new, tl.abs(new), 0).to(tl.float64)
        tl.store(stats + tl.program_id(0) * 4, tl.sum(removed_own.to(tl.int32) + removed_new.to(tl.int32), 0))
        tl.store(stats + tl.program_id(0) * 4 + 1, tl.sum(a + b, 0))
        tl.store(stats + tl.program_id(0) * 4 + 2, tl.sum(a * a + b * b, 0))
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

    if BACKWARD:
        tl.store(out_grad + dest, own_grad, keep_own)
        tl.store(out_grad + dest + keep_own.to(tl.int32), new_grad, keep_new)


@triton.jit(do_not_specialize=["n", "q", "r"])
def clifford(keys, coeff, out_keys, out_coeff, n, q, r,
             W: tl.constexpr, OP: tl.constexpr, B: tl.constexpr,
             grad=None, out_grad=None, GRADIENT: tl.constexpr = False):
    i = tl.program_id(0) * B + tl.arange(0, B)
    live = i < n
    qw, qb = q // 32, 31 - q % 32
    rw, rb = r // 32, 31 - r % 32
    qmask = tl.full((), 1, tl.uint32) << qb
    rmask = tl.full((), 1, tl.uint32) << rb
    x = (tl.load(keys + i.to(tl.int64) * W + qw, live, 0).to(tl.uint32) >> qb) & 1
    z = (tl.load(keys + i.to(tl.int64) * W + W // 2 + qw, live, 0).to(tl.uint32) >> qb) & 1
    xf = tl.full((B,), 0, tl.uint32)
    zf = tl.full((B,), 0, tl.uint32)
    xtf = tl.full((B,), 0, tl.uint32)
    ztf = tl.full((B,), 0, tl.uint32)
    if OP == 0:  # H
        xf = x ^ z
        zf = xf
        phase = x & z
    elif OP == 1:  # S: U^dagger P U
        zf = x
        phase = x & (z ^ 1)
    elif OP == 2:  # Sdg
        zf = x
        phase = x & z
    elif OP == 3:  # X
        phase = z
    elif OP == 4:  # Y
        phase = x ^ z
    elif OP == 5:  # Z
        phase = x
    else:
        xt = (tl.load(keys + i.to(tl.int64) * W + rw, live, 0).to(tl.uint32) >> rb) & 1
        zt = (tl.load(keys + i.to(tl.int64) * W + W // 2 + rw, live, 0).to(tl.uint32) >> rb) & 1
        if OP == 6:  # CX
            xtf = x
            zf = zt
            phase = x & zt & (1 ^ z ^ xt)
        elif OP == 7:  # CZ
            zf = xt
            ztf = x
            phase = x & xt & (z ^ zt)
        else:  # CY = Sdg/CX/S conjugation, fused into one bit map.
            xtf = x
            ztf = x
            zf = zt ^ xt
            phase = (xt & (zt ^ 1)) ^ (x & (zt ^ xt) & (1 ^ z ^ xt)) ^ ((xt ^ x) & (zt ^ xt))
    for w in tl.static_range(W):
        v = tl.load(keys + i.to(tl.int64) * W + w, live, 0).to(tl.uint32)
        v ^= tl.where(w == qw, xf * qmask, 0)
        v ^= tl.where(w == W // 2 + qw, zf * qmask, 0)
        v ^= tl.where(w == rw, xtf * rmask, 0)
        v ^= tl.where(w == W // 2 + rw, ztf * rmask, 0)
        tl.store(out_keys + i.to(tl.int64) * W + w, v, live)
    c = tl.load(coeff + i, live, 0)
    tl.store(out_coeff + i, tl.where(phase != 0, -c, c), live)
    if GRADIENT:
        g = tl.load(grad + i, live, 0)
        tl.store(out_grad + i, tl.where(phase != 0, -g, g), live)
