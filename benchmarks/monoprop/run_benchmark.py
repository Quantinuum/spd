"""Reproduce the published MonoProp workload using Triton or cuPauliProp only.

Model provenance: Algorithmiq/monoprop, packages/bench-third-party/pauli_prop,
revision recorded in reference/provenance.json. No MonoProp code is executed.
"""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def model(n, observable):
    """Authoring order: row-major right/down ZZ bonds, all Z, then all X."""
    nq = n * n
    gates = []
    for row in range(n):
        for col in range(n):
            q = row * n + col
            if col + 1 < n:
                gates.append(('ZZ', [q, q + 1], .05 * 1.5))
            if row + 1 < n:
                gates.append(('ZZ', [q, q + n], .05 * 1.5))
    gates += [('Z', [q], .05) for q in range(nq)]
    gates += [('X', [q], .05) for q in range(nq)]
    pair = [20, 21] if observable == 'fixed' else [n * (n // 2) + max(n // 2 - 1, 0), n * (n // 2) + max(n // 2 - 1, 0) + 1]
    if max(pair) >= nq:
        raise ValueError('fixed observable [20,21] requires at least 22 qubits')
    return list(reversed(gates)), pair


def label(paulis, qubits, nq):
    chars = ['I'] * nq
    for p, q in zip(paulis, qubits):
        chars[q] = p
    return ''.join(chars)


def versions():
    out = {}
    for name in ('torch', 'triton', 'cupy-cuda12x', 'cuquantum-python-cu12'):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('triton', 'cupauliprop'), required=True)
    parser.add_argument('--n', type=int, default=12)
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--observable', choices=('fixed', 'central'), default='fixed')
    parser.add_argument('--cutoff', type=float, default=1e-6)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dump', action='store_true', help='Save canonical coefficients for SMALL comparisons only')
    args = parser.parse_args()
    if args.n < 2 or args.steps < 1 or args.cutoff < 0:
        parser.error('n >= 2, steps >= 1, cutoff >= 0 required')
    import numpy as np
    gates, pair = model(args.n, args.observable)
    nq = args.n ** 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(backend=args.backend, n=args.n, num_qubits=nq, steps=args.steps,
        observable=pair, hx=1., hz=1., j=1.5, dt=.05, cutoff=args.cutoff,
        precision='float64', weight_cutoff=None, term_cap=None,
        timing='synchronized propagation plus Z expectation and count; tiny disposable compile first; no full-workload warm-up',
        host=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0)), versions=versions(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
        spd_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        allocator_environment={k:v for k,v in os.environ.items() if k in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF')})
    warm_start = time.perf_counter()
    if args.backend == 'triton':
        import torch
        from spd import triton_backend as tb
        from spd.circuit_ir import PauliRotation
        operations = [PauliRotation('rotation',label(p,q,nq),theta) for p,q,theta in gates]
        def create():
            return tb.create_op({label('ZZ',pair,nq):1.}, precision='double')
        def advance(state):
            for op in operations:
                state.apply_in_place(op, args.cutoff, None, diagnostics=False)
            return state
        sync = torch.cuda.synchronize
        def observe(state):
            return float(state.get_expectation_value('Z')), state.get_size()
        def start_memory():
            torch.cuda.reset_peak_memory_stats()
        def memory():
            return dict(device_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                        device_reserved_peak_bytes=torch.cuda.max_memory_reserved())
        # Compile all packed-width specializations and gate metadata on tiny support.
        warm = advance(create())
        observe(warm)
        warm._storage.compact(final=True)
        sync()
        del warm
        def dump(state):
            keys, c = state.to_host()
            w = keys.shape[1] // 2
            return {''.join('IXZY'[((int(k[q//32]) >> (31-q%32)) & 1) + 2*((int(k[w+q//32]) >> (31-q%32)) & 1)] for q in range(nq)):float(v) for k,v in zip(keys,c)}
    else:
        import cupy as cp
        from cuquantum.pauliprop.experimental import (LibraryHandle, PauliExpansion,
            PauliExpansionOptions, PauliRotationGate, Truncation, get_num_packed_integers)
        handle = LibraryHandle()
        width = get_num_packed_integers(nq)
        def create():
            bits = np.zeros((1, 2*width), dtype=np.uint64)
            for q in pair:
                bits[0,width+q//64] |= np.uint64(1) << np.uint64(q%64)
            return PauliExpansion(library_handle=handle,num_qubits=nq,num_terms=1,
                xz_bits=cp.asarray(bits),coeffs=cp.ones(1,dtype=cp.float64),
                options=PauliExpansionOptions(memory_limit='80%',blocking=True))
        operations = [PauliRotationGate(theta,list(p),q) for p,q,theta in gates]
        truncation = Truncation(pauli_coeff_cutoff=args.cutoff,pauli_weight_cutoff=nq)
        def advance(state):
            for op in operations:
                state = state.apply_gate(op,truncation=truncation,adjoint=True,
                    sort_order=None,keep_duplicates=False)
            return state
        sync = cp.cuda.Device().synchronize
        def observe(state):
            significand, exponent = state.trace_with_zero_state()
            return float(significand * np.exp2(exponent)), state.num_terms
        class Peak(cp.cuda.MemoryHook):
            def __init__(self):
                self.used = self.reserved = 0
            def malloc_postprocess(self, **kwargs):
                pool = cp.get_default_memory_pool()
                self.used = max(self.used,pool.used_bytes())
                self.reserved = max(self.reserved,pool.total_bytes())
        peak = Peak()
        def start_memory():
            peak.used = cp.get_default_memory_pool().used_bytes()
            peak.reserved = cp.get_default_memory_pool().total_bytes()
        def memory():
            return dict(device_allocated_peak_bytes=peak.used,device_reserved_peak_bytes=peak.reserved)
        warm = advance(create())
        observe(warm)
        sync()
        del warm
        def dump(state):
            bits, c = cp.asnumpy(state.xz_bits[:state.num_terms]), cp.asnumpy(state.coeffs[:state.num_terms])
            return {''.join('IXZY'[((int(k[q//64]) >> (q%64)) & 1) + 2*((int(k[width+q//64]) >> (q%64)) & 1)] for q in range(nq)):float(v) for k,v in zip(bits,c)}
    gc.collect()
    metadata['warmup_seconds'] = time.perf_counter() - warm_start
    rows = []
    state = create()
    sync()
    def save():
        args.output.write_text(json.dumps(dict(metadata=metadata, rows=rows,
            total_runtime_s=sum(r['runtime_s'] for r in rows),
            total_runtime_excl_first_s=sum(r['runtime_s'] for r in rows[1:])),indent=2)+'\n')
    for i in range(args.steps):
        gc.collect()
        sync()
        start_memory()
        if args.backend == 'cupauliprop':
            peak.__enter__()
        start = time.perf_counter()
        try:
            state = advance(state)
            expectation, count = observe(state)
            sync()
            elapsed = time.perf_counter() - start
        finally:
            if args.backend == 'cupauliprop':
                peak.__exit__(None,None,None)
        rows.append(dict(step=i+1,runtime_s=elapsed,num_terms=count,expectation=expectation,
            host_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**memory()))
        save()
        print(json.dumps(rows[-1]),flush=True)
    if args.dump:
        args.output.with_suffix('.coefficients.json').write_text(json.dumps(dump(state),sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
