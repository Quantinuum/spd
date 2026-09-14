"""Reduce nonzero Pauli support on-device without exporting mutable storage."""
import torch
import triton
import triton.language as tl


@triton.jit
def _or(a, b):
    return a | b


@triton.jit(do_not_specialize=['n'])
def _reduce_support(keys, coeff, out, n, KS: tl.constexpr, WS: tl.constexpr, CS: tl.constexpr,
                    W: tl.constexpr, BW: tl.constexpr, B: tl.constexpr):
    rows = tl.program_id(0) * B + tl.arange(0, B)
    words = tl.arange(0, BW)
    c = tl.load(coeff + rows.to(tl.int64) * CS, rows < n, 0)
    active = (rows < n) & (c != 0)
    offsets = rows[:, None].to(tl.int64) * KS + words[None, :] * WS
    valid = active[:, None] & (words[None, :] < W)
    x = tl.load(keys + offsets, valid, 0).to(tl.uint32)
    z = tl.load(keys + offsets + W * WS, valid, 0).to(tl.uint32)
    mask = tl.reduce(x | z, 0, _or)
    tl.atomic_or(out + words, mask.to(tl.int32), words < W, sem='relaxed')
    invalid = (c != c) | (tl.abs(c) == float('inf'))
    tl.atomic_or(out + W, tl.max(invalid.to(tl.int32), 0), sem='relaxed')


def support_mask(keys, coefficients):
    """Return W support words plus a nonfinite flag; copy only W+1 int32s.

    Zero-coefficient rows (including dead storage slots) do not seed support.
    Reads raw storage without compaction, mutation, or state-sized temporaries.
    """
    words = keys.shape[1] // 2
    with torch.cuda.device(keys.device):
        result = torch.zeros(words + 1, dtype=torch.int32, device=keys.device)
        if len(coefficients):
            _reduce_support[(triton.cdiv(len(coefficients), 256),)](
                keys, coefficients, result, len(coefficients),
                keys.stride(0), keys.stride(1), coefficients.stride(0), words,
                triton.next_power_of_2(words), 256,
            )
        return result.cpu().numpy().view('uint32')
