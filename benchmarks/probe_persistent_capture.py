"""Bounded CUDA-graph feasibility probe on closed, fixed Pauli support.

This is not the growing 11x11 trajectory. Counts stay device-resident because
all XOR partners already exist, so no insertion/growth is needed. Both timing
paths use the identical kernels; the captured path only changes dispatch.
"""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from spd import triton_backend as tb
from spd.circuit_ir import PauliRotation
from spd.triton_backend.persistent import _Storage
from spd.triton_backend.experiment_kernels import update_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    result = []
    for nq in [3, 6]:
        labels = [''.join(p)+'I'*(128-nq) for p in itertools.product('IXYZ',repeat=nq)]
        rng = np.random.default_rng(482)
        state = tb.create_op(dict(zip(labels,rng.normal(size=len(labels)))), 128, 'double')
        storage = _Storage(state, .1)
        gates, ops = [], []
        for j in range(32):
            p = ['I']*128
            p[j%nq] = 'Z' if j%2 else 'X'
            if not j%2:
                p[(j+1)%nq] = 'X'
            label = ''.join(p)
            gates.append(tb._gate_data(label,.137,0.,128,torch.float64,state.c_array.device))
            ops.append(PauliRotation('rotation',label,.137))
        expected = tb.evolve_step(state, tuple(reversed(ops)), 0.)
        ek, ec = expected.to_host()
        reference = {tuple(k):c for k,c in zip(ek,ec)}
        def launch():
            for gate, scalars in gates:
                storage.counts.zero_()
                update_variant[((storage.n+127)//128,)](storage.keys,storage.coeff,storage.table,
                    gate,scalars,storage.counts,storage.n,storage.table_size-1,storage.width,128,
                    enable_fp_fusion=False)
        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        start = time.perf_counter()
        with torch.cuda.graph(graph):
            launch()
        capture_seconds = time.perf_counter()-start
        times = {}
        for name, fn in [('uncaptured',launch),('captured',graph.replay)]:
            trials=[]
            for _ in range(10):
                storage.coeff[:storage.n].copy_(state.c_array)
                torch.cuda.synchronize()
                start=time.perf_counter()
                fn()
                torch.cuda.synchronize()
                trials.append(time.perf_counter()-start)
            times[name]=dict(median_seconds=statistics.median(trials), trials=trials)
            keys=storage.keys[:storage.n].cpu().numpy().view(np.uint32)
            c=storage.coeff[:storage.n].cpu().numpy()
            assert set(map(tuple,keys))==reference.keys()
            np.testing.assert_allclose(c,[reference[tuple(k)] for k in keys],atol=2e-13,rtol=2e-13)
            assert storage.counts[0].item()==0
        result.append(dict(rows=storage.n,gates=32,capture_seconds=capture_seconds,
                           coefficients_match=True,timing=times))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
