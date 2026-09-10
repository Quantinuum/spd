"""On-demand algebra/analysis kernels; the rotation hot path is unchanged."""
import triton
import triton.language as tl
from .kernels import _mix, _popc


@triton.jit(do_not_specialize=['n', 'mask'])
def lookup(query, source, table, indices, matched, n, mask,
           W: tl.constexpr, MARK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    live = i < n
    h = tl.full((B,), 0x9e3779b9, tl.uint32)
    for w in tl.static_range(W):
        h = _mix(h, tl.load(query + i.to(tl.int64)*W+w, live, 0).to(tl.uint32))
    slot = h & mask
    pending = live
    result = tl.full((B,), -1, tl.int32)
    while tl.sum(pending.to(tl.int32), 0) > 0:
        idx = tl.load(table + slot, pending, -1)
        equal = pending & (idx >= 0)
        for w in tl.static_range(W):
            q = tl.load(query + i.to(tl.int64)*W+w, live, 0)
            s = tl.load(source + idx.to(tl.int64)*W+w, pending & (idx >= 0), 0)
            equal &= q == s
        result = tl.where(equal, idx, result)
        pending &= (idx >= 0) & ~equal
        slot = (slot + pending.to(tl.uint32)) & mask
    tl.store(indices+i, result, live)
    if MARK:
        # Query/source keys are unique, so matching stores cannot race.
        tl.store(matched+result, 1, live & (result >= 0))


@triton.jit(do_not_specialize=['n'])
def weights(keys, out, n, W: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B+tl.arange(0, B)
    value = tl.full((B,), 0, tl.int32)
    for w in tl.static_range(W//2):
        x = tl.load(keys+i.to(tl.int64)*W+w, i<n, 0)
        z = tl.load(keys+i.to(tl.int64)*W+W//2+w, i<n, 0)
        value += _popc(x | z)
    tl.store(out+i, value, i<n)


@triton.jit(do_not_specialize=['n'])
def translate(keys, out, n, shift, W: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B+tl.arange(0, B)
    for half in tl.static_range(2):
        base = i.to(tl.int64)*W + half*(W//2)
        for w in tl.static_range(W//2):
            original = tl.load(keys+base+w, i<n, 0).to(tl.uint32)
            if w*32 < S:
                if S < 32:
                    bits = original >> (32-S)
                    rotated = (bits >> shift) | (bits << (S-shift))
                    value = (rotated << (32-S)) | (original & ((1 << (32-S))-1))
                else:
                    start = (w*32-shift+S) % S
                    offset = start % 32
                    a = tl.load(keys+base+start//32, i<n, 0).to(tl.uint32)
                    b = tl.load(keys+base+start//32+1, (i<n) & (start//32+1 < W//2), 0).to(tl.uint32)
                    sequence = (a << offset) | tl.where(offset == 0, 0, b >> (32-offset))
                    first = tl.minimum(32, S-start)
                    first_mask = tl.full((), 0xffffffff, tl.uint32) << (32-first)
                    head = tl.load(keys+base, i<n, 0).to(tl.uint32)
                    sequence = (sequence & first_mask) | tl.where(first < 32, head >> first, 0)
                    valid = tl.minimum(32, S-w*32)
                    mask = tl.full((), 0xffffffff, tl.uint32) << (32-valid)
                    value = (sequence & mask) | (original & ~mask)
                tl.store(out+base+w, value, i<n)
            else:
                tl.store(out+base+w, original, i<n)


@triton.jit(do_not_specialize=['n'])
def product(left, right, out, phase, n, W: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B+tl.arange(0, B)
    count = tl.full((B,), 0, tl.int32)
    for w in tl.static_range(W//2):
        x = tl.load(left+w).to(tl.uint32)
        z = tl.load(left+W//2+w).to(tl.uint32)
        y = tl.load(right+i.to(tl.int64)*W+w, i<n, 0).to(tl.uint32)
        t = tl.load(right+i.to(tl.int64)*W+W//2+w, i<n, 0).to(tl.uint32)
        count += 2*_popc(x&t) + _popc(x&z) + _popc(y&t) - _popc((x^y)&(z^t))
        tl.store(out+i.to(tl.int64)*W+w, x^y, i<n)
        tl.store(out+i.to(tl.int64)*W+W//2+w, z^t, i<n)
    tl.store(phase+i, count & 3, i<n)


@triton.jit(do_not_specialize=['n', 'q', 'r'])
def susceptibility(keys, c, g, partials, n, q, r, W: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B+tl.arange(0, B)
    active = tl.full((B,), False, tl.int1)
    for j in tl.static_range(2):
        site = q if j == 0 else r
        x = tl.load(keys+i.to(tl.int64)*W+site//32, i<n, 0)
        z = tl.load(keys+i.to(tl.int64)*W+W//2+site//32, i<n, 0)
        active |= ((x | z).to(tl.uint32) & (1 << (31-site%32)).to(tl.uint32)) != 0
    c_i = tl.load(c+i, i<n, 0)
    g_i = tl.load(g+i, i<n, 0)
    tl.store(partials+tl.program_id(0), -tl.sum(tl.where(active, c_i*g_i, 0).to(tl.float64), 0))
