"""Collect timing, scalar validation, and memory from isolated option trials."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    a=parser.parse_args()
    ref_dir=a.directory.parent/'persistent_storage_20260910/candidate2'
    reference=pickle.loads(next(ref_dir.glob('*.pkl')).read_bytes())
    result=dict(reference=str(ref_dir),runs={})
    for path in sorted(a.directory.iterdir()):
        if not path.is_dir():
            continue
        pickles=list(path.glob('*.pkl'))
        if not pickles:
            continue
        d=pickle.loads(pickles[0].read_bytes())
        n=len(d['times'])
        meta=json.loads((path/'option.json').read_text()) if (path/'option.json').exists() else {}
        alloc=json.loads((path/'allocator.json').read_text()) if (path/'allocator.json').exists() else {}
        counters=alloc['steps'][-1]['counters'] if alloc.get('steps') else {}
        result['runs'][path.name]=dict(steps=n,variant=meta.get('variant'),
            dead_fraction=meta.get('dead_fraction'),growth_factor=meta.get('growth_factor'),
            total_seconds=sum(d['times']),final_step_seconds=d['times'][-1],
            counts_match=d['num_paulis']==reference['num_paulis'][:n],
            max_expectation_difference=float(np.max(np.abs(np.array(d['all_results'])-reference['all_results'][:n+1]))),
            max_norm_difference=float(np.max(np.abs(np.array(d['norms'])-reference['norms'][:n]))),
            peak_allocated_bytes=counters.get('allocated_bytes.all.peak'),
            peak_reserved_bytes=counters.get('reserved_bytes.all.peak'),
            allocation_retries=counters.get('num_alloc_retries'),ooms=counters.get('num_ooms'))
    (a.directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,r in result['runs'].items():
        print(name,r['steps'],round(r['total_seconds'],6),round(r['final_step_seconds'],6),
              r['counts_match'],round(r['peak_allocated_bytes']/1e9,3) if r['peak_allocated_bytes'] else None)


if __name__=='__main__':
    main()
