"""Compare all final Pauli keys and coefficients, independent of GPU row order."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('baseline',type=Path)
    p.add_argument('pruned',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    keys=[np.load(path/'keys.npy',mmap_mode='r') for path in (args.baseline,args.pruned)]
    coeff=[np.load(path/'coeffs.npy',mmap_mode='r') for path in (args.baseline,args.pruned)]
    assert keys[0].shape==keys[1].shape
    assert coeff[0].shape==coeff[1].shape==(len(keys[0]),)
    started=time.perf_counter()
    order=[]
    for i,k in enumerate(keys):
        print(f'Sorting all {len(k):,} packed Pauli rows: input {i+1}',flush=True)
        order.append(np.lexsort(k.T[::-1]))
    hashes=[hashlib.sha256(),hashlib.sha256()]
    chashes=[hashlib.sha256(),hashlib.sha256()]
    max_error=0.
    different=0
    squared_error=0.
    for start in range(0,len(keys[0]),1000000):
        stop=min(start+1000000,len(keys[0]))
        a,b=[k[ix[start:stop]] for k,ix in zip(keys,order)]
        assert np.array_equal(a,b), f'Pauli support differs at chunk {start}'
        for h,k in zip(hashes,(a,b)):
            h.update(k.tobytes())
        ca,cb=[c[ix[start:stop]] for c,ix in zip(coeff,order)]
        assert np.all(np.isfinite(ca)) and np.all(np.isfinite(cb))
        delta=ca-cb
        max_error=max(max_error,float(np.max(np.abs(delta),initial=0.)))
        different+=int(np.count_nonzero(delta))
        squared_error+=float(np.dot(delta,delta))
        for h,c in zip(chashes,(ca,cb)):
            h.update(c.tobytes())
    result=dict(num_terms=len(keys[0]),keys_identical=True,max_abs_coefficient_error=max_error,
                num_coefficients_differing=different,coefficient_l2_error=squared_error**.5,
                canonical_key_sha256=[h.hexdigest() for h in hashes],
                canonical_coefficient_sha256=[h.hexdigest() for h in chashes],
                validation_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
