"""Isolated persistent-storage experiment kernels; not public dispatch."""
import triton
import triton.language as tl
from .persistent import _cas
from .kernels import _mix, _popc


@triton.jit
def key_offset(i, w, W: tl.constexpr, stride, SOA: tl.constexpr):
    if SOA:
        return (i.to(tl.int64) * 0 + w) * stride + i.to(tl.int64)
    return i.to(tl.int64) * W + w


@triton.jit(do_not_specialize=['start', 'end', 'mask', 'STRIDE'])
def insert_variant(keys, table, start, end, mask, W: tl.constexpr, B: tl.constexpr, STRIDE=0, SOA: tl.constexpr=False, fingerprints=None, FP: tl.constexpr=False):
    i = start + tl.program_id(0) * B + tl.arange(0, B)
    pending = i < end
    h = tl.full((B,), 0x9e3779b9, tl.uint32)
    for w in tl.static_range(W):
        h = _mix(h, tl.load(keys + key_offset(i, w, W, STRIDE, SOA), pending, 0).to(tl.uint32))
    if FP:
        tl.store(fingerprints + i, h, pending)
    slot = h & mask
    while tl.sum(pending.to(tl.int32), 0) > 0:
        old = _cas(table + slot, i, pending)
        pending &= old != -1
        slot = (slot + pending.to(tl.uint32)) & mask


@triton.jit(do_not_specialize=['n', 'mask', 'STRIDE', 'active_n', 'epoch'])
def update_variant(keys, coeff, table, gate, scalars, counts, n, mask,
            W: tl.constexpr, B: tl.constexpr, STRIDE=0, SOA: tl.constexpr=False,
            fingerprints=None, FP: tl.constexpr=False, info=None, LOCAL: tl.constexpr=0,
            active=None, active_n=0, ACTIVE: tl.constexpr=False,
            grad=None, gradient_stats=None, BACKWARD: tl.constexpr=False,
            state=None, epoch=0, DEVICE: tl.constexpr=False,
            sites=None, LOCAL_ARITY: tl.constexpr=0):
    j = tl.program_id(0) * B + tl.arange(0, B)
    if DEVICE:
        n = tl.load(state)
        running = (tl.load(state + 2) < 0) & (n <= active_n)
    else:
        running = True
    if ACTIVE:
        i = tl.load(active + j, j < active_n, n)
    else:
        i = j
    valid = (i < n) & running
    parity = tl.full((B,), 0, tl.int32)
    phase = tl.full((B,), 0, tl.int32)
    if LOCAL_ARITY:
        for s in tl.static_range(LOCAL_ARITY):
            word = tl.load(sites + 4 * s)
            bit = tl.load(sites + 4 * s + 1)
            gx = tl.load(sites + 4 * s + 2)
            gz = tl.load(sites + 4 * s + 3)
            x = ((tl.load(keys + key_offset(i, word, W, STRIDE, SOA), valid, 0).to(tl.uint32) >> bit) & 1).to(tl.int32)
            z = ((tl.load(keys + key_offset(i, W // 2 + word, W, STRIDE, SOA), valid, 0).to(tl.uint32) >> bit) & 1).to(tl.int32)
            parity += x * gz + z * gx
            phase += 2 * gx * z + gx * gz + x * z - ((x ^ gx) * (z ^ gz))
    elif LOCAL == 0:
        for w in tl.static_range(W // 2):
            x = tl.load(keys + key_offset(i, w, W, STRIDE, SOA), valid, 0).to(tl.uint32)
            z = tl.load(keys + key_offset(i, W // 2 + w, W, STRIDE, SOA), valid, 0).to(tl.uint32)
            gx = tl.load(gate + w).to(tl.uint32)
            gz = tl.load(gate + W // 2 + w).to(tl.uint32)
            parity += _popc(x & gz) + _popc(z & gx)
            phase += (2 * _popc(gx & z) + _popc(gx & gz) + _popc(x & z)
                      - _popc((x ^ gx) & (z ^ gz)))
    else:
        qw = tl.load(info)
        qb = tl.load(info + 1)
        xq = (tl.load(keys + key_offset(i, qw, W, STRIDE, SOA), valid, 0).to(tl.uint32) >> qb) & 1
        zq = (tl.load(keys + key_offset(i, W // 2 + qw, W, STRIDE, SOA), valid, 0).to(tl.uint32) >> qb) & 1
        if LOCAL == 1:
            parity = xq.to(tl.int32)
            phase = (2 * zq - 1).to(tl.int32)
        else:
            rw = tl.load(info + 2)
            rb = tl.load(info + 3)
            xr = (tl.load(keys + key_offset(i, rw, W, STRIDE, SOA), valid, 0).to(tl.uint32) >> rb) & 1
            zr = (tl.load(keys + key_offset(i, W // 2 + rw, W, STRIDE, SOA), valid, 0).to(tl.uint32) >> rb) & 1
            parity = (zq + zr).to(tl.int32)
            phase = (zq * (1 + 2 * xq) + zr * (1 + 2 * xr)).to(tl.int32)
    anti = valid & ((parity & 1) != 0)
    h = tl.full((B,), 0x9e3779b9, tl.uint32)
    for w in tl.static_range(W):
        v = tl.load(keys + key_offset(i, w, W, STRIDE, SOA), anti, 0).to(tl.uint32)
        g = tl.load(gate + w).to(tl.uint32)
        h = _mix(h, v ^ g)
    slot = h & mask
    pending = anti
    partner = tl.full((B,), -1, tl.int32)
    while tl.sum(pending.to(tl.int32), 0) > 0:
        idx = tl.load(table + slot, pending, -1)
        equal = pending & (idx >= 0)
        if FP:
            fingerprint = tl.load(fingerprints + idx, equal, 0).to(tl.uint32)
            equal &= fingerprint == h
        candidate = equal
        for w in tl.static_range(W):
            v = tl.load(keys + key_offset(i, w, W, STRIDE, SOA), pending, 0).to(tl.uint32)
            g = tl.load(gate + w).to(tl.uint32)
            other = tl.load(keys + key_offset(idx, w, W, STRIDE, SOA), candidate, 0).to(tl.uint32)
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
    if BACKWARD:
        adj = tl.load(grad + i, owner, 0)
        adj_pair = tl.load(grad + partner, paired_owner, 0)
        theta_grad = tl.where(paired_owner, sign * (c * adj_pair - paired * adj), 0)
        tl.store(gradient_stats + tl.program_id(0), tl.sum(theta_grad.to(tl.float64), 0))
        sin = -sin
        own_grad = tl.where(anti, adj * cos - adj_pair * sin * sign, adj)
        other_grad = adj_pair * cos - adj * sin * (-sign)
    own = tl.where(anti, c * cos - paired * sin * sign, c)
    # Match the original partner row's arithmetic order (phase sign is opposite).
    other = paired * cos - c * sin * (-sign)
    if BACKWARD:
        keep_own = owner & ((own != 0) | (own_grad != 0)) & (tl.abs(own) >= cutoff)
        keep_other = owner & anti & ((other != 0) | (other_grad != 0)) & (tl.abs(other) >= cutoff)
    else:
        keep_own = owner & (tl.abs(own) > cutoff)
        keep_other = owner & anti & (tl.abs(other) > cutoff)
    tl.store(coeff + i, tl.where(keep_own, own, 0), owner)
    tl.store(coeff + partner, tl.where(keep_other, other, 0), paired_owner)
    create = keep_other & (partner < 0)
    offsets = tl.cumsum(create.to(tl.int32)) - create.to(tl.int32)
    base = tl.atomic_add(counts, tl.sum(create.to(tl.int32), 0), sem='relaxed')
    dest = n + base + offsets
    for w in tl.static_range(W):
        v = tl.load(keys + key_offset(i, w, W, STRIDE, SOA), create, 0)
        g = tl.load(gate + w).to(tl.uint32)
        tl.store(keys + key_offset(dest, w, W, STRIDE, SOA), v ^ g, create)
    tl.store(coeff + dest, other, create)
    if BACKWARD:
        tl.store(grad + i, tl.where(keep_own, own_grad, 0), owner)
        tl.store(grad + partner, tl.where(keep_other, other_grad, 0), paired_owner)
        tl.store(grad + dest, other_grad, create)
    if BACKWARD:
        before_own = owner & ((c != 0) | (adj != 0))
        before_other = paired_owner & ((paired != 0) | (adj_pair != 0))
    else:
        before_own = owner & (c != 0)
        before_other = paired_owner & (paired != 0)
    delta = (keep_own.to(tl.int32) - before_own.to(tl.int32)
             + keep_other.to(tl.int32) - before_other.to(tl.int32))
    tl.atomic_add(counts + 1, tl.sum(delta, 0), sem='relaxed')


@triton.jit(do_not_specialize=['n', 'STRIDE', 'OUT_STRIDE'])
def compact_variant(keys, coeff, out_keys, out_coeff, count, n,
             W: tl.constexpr, B: tl.constexpr, STRIDE=0, SOA: tl.constexpr=False, OUT_STRIDE=0, OUT_SOA: tl.constexpr=False):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c = tl.load(coeff + i, i < n, 0)
    keep = (i < n) & (c != 0)
    offsets = tl.cumsum(keep.to(tl.int32)) - keep.to(tl.int32)
    base = tl.atomic_add(count, tl.sum(keep.to(tl.int32), 0), sem='relaxed')
    dest = base + offsets
    for w in tl.static_range(W):
        v = tl.load(keys + key_offset(i, w, W, STRIDE, SOA), keep, 0)
        tl.store(out_keys + key_offset(dest, w, W, OUT_STRIDE, OUT_SOA), v, keep)
    tl.store(out_coeff + dest, c, keep)



@triton.jit(do_not_specialize=['start', 'n', 'stride'])
def make_planes(keys, planes, start, n, stride, W: tl.constexpr):
    block = start // 32 + tl.program_id(0)
    w = tl.program_id(1)
    rows = block * 32 + tl.arange(0, 32)
    bits = tl.arange(0, 32)
    values = tl.load(keys + rows.to(tl.int64) * W + w, rows < n, 0).to(tl.uint32)
    matrix = ((values[None, :] >> bits[:, None]) & 1) << tl.arange(0, 32)[None, :]
    words = tl.sum(matrix.to(tl.uint32), 1)
    tl.store(planes + (w * 32 + bits).to(tl.int64) * stride + block, words)


@triton.jit(do_not_specialize=['n', 'stride'])
def select_rows(keys, planes, selected, count, info, n, stride,
                W: tl.constexpr, B: tl.constexpr, KIND: tl.constexpr,
                INVERTED: tl.constexpr, sites=None, ARITY: tl.constexpr=0):
    i = tl.program_id(0) * B + tl.arange(0, B)
    if ARITY:
        bit = tl.full((B,), 0, tl.int32)
        for s in tl.static_range(ARITY):
            qw = tl.load(sites + 4 * s)
            qb = tl.load(sites + 4 * s + 1)
            gx = tl.load(sites + 4 * s + 2)
            gz = tl.load(sites + 4 * s + 3)
            if INVERTED:
                xp = qw * 32 + qb
                zp = (W // 2 + qw) * 32 + qb
                x = (tl.load(planes + xp.to(tl.int64) * stride + i // 32, (i < n) & (gz != 0), 0).to(tl.uint32) >> (i % 32)) & 1
                z = (tl.load(planes + zp.to(tl.int64) * stride + i // 32, (i < n) & (gx != 0), 0).to(tl.uint32) >> (i % 32)) & 1
            else:
                x = (tl.load(keys + i.to(tl.int64) * W + qw, (i < n) & (gz != 0), 0).to(tl.uint32) >> qb) & 1
                z = (tl.load(keys + i.to(tl.int64) * W + W // 2 + qw, (i < n) & (gx != 0), 0).to(tl.uint32) >> qb) & 1
            bit ^= (x * gz) ^ (z * gx)
    else:
        qw = tl.load(info)
        qb = tl.load(info + 1)
        if INVERTED:
            plane = qw * 32 + qb
            if KIND == 2:
                plane += (W // 2) * 32
            word = tl.load(planes + plane.to(tl.int64) * stride + i // 32, i < n, 0).to(tl.uint32)
            bit = (word >> (i % 32)) & 1
        else:
            word = qw if KIND == 1 else W // 2 + qw
            bit = (tl.load(keys + i.to(tl.int64) * W + word, i < n, 0).to(tl.uint32) >> qb) & 1
        if KIND == 2:
            rw = tl.load(info + 2)
            rb = tl.load(info + 3)
            if INVERTED:
                plane = (W // 2 + rw) * 32 + rb
                word = tl.load(planes + plane.to(tl.int64) * stride + i // 32, i < n, 0).to(tl.uint32)
                bit ^= (word >> (i % 32)) & 1
            else:
                bit ^= (tl.load(keys + i.to(tl.int64) * W + W // 2 + rw, i < n, 0).to(tl.uint32) >> rb) & 1
    keep = (i < n) & (bit != 0)
    size = tl.sum(keep.to(tl.int32), 0)
    if size > 0:
        base = tl.atomic_add(count, size, sem='relaxed')
        offset = tl.cumsum(keep.to(tl.int32)) - 1
        tl.store(selected + base + offset, i, keep)


@triton.jit
def begin_device(state, counts, bound, epoch):
    n = tl.load(state)
    stop = tl.load(state + 2)
    if (stop < 0) & (n > bound):
        tl.store(state + 2, epoch)
    tl.store(counts, 0)
    tl.store(counts + 1, 0)


@triton.jit(do_not_specialize=['bound', 'mask'])
def insert_device(keys, table, state, counts, bound, mask, W: tl.constexpr, B: tl.constexpr):
    before = tl.load(state)
    added = tl.load(counts)
    i = before + tl.program_id(0) * B + tl.arange(0, B)
    pending = (i < before + added) & (tl.load(state + 2) < 0)
    h = tl.full((B,), 0x9e3779b9, tl.uint32)
    for w in tl.static_range(W):
        h = _mix(h, tl.load(keys + i.to(tl.int64) * W + w, pending, 0).to(tl.uint32))
    slot = h & mask
    while tl.sum(pending.to(tl.int32), 0) > 0:
        old = _cas(table + slot, i, pending)
        pending &= old != -1
        slot = (slot + pending.to(tl.uint32)) & mask


@triton.jit
def advance_device(state, counts, cap, epoch):
    if tl.load(state + 2) < 0:
        n = tl.load(state) + tl.load(counts)
        live = tl.load(state + 1) + tl.load(counts + 1)
        tl.store(state, n)
        tl.store(state + 1, live)
        if live > cap:
            tl.store(state + 2, epoch + 1)
