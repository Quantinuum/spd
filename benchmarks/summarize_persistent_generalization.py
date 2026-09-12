"""Compare cross-workload timings, scalars and saved full AFH coefficients."""
import json
from pathlib import Path
import sys
import numpy as np

REFERENCE_STATES = {}

def state_difference(reference, candidate):
    def load(path):
        with np.load(path) as data:
            keys=np.ascontiguousarray(data['keys'])
            coeff=data['coeff'].copy()
        key=keys.view(np.dtype((np.void,keys.dtype.itemsize*keys.shape[1]))).ravel()
        order=np.argsort(key)
        return key[order],coeff[order]
    if reference not in REFERENCE_STATES:
        REFERENCE_STATES[reference]=load(reference)
    rk,rc=REFERENCE_STATES[reference]
    if candidate==reference:
        ck,cc=rk,rc
    else:
        ck,cc=load(candidate)
    if np.array_equal(rk,ck):
        return dict(reference_rows=len(rk),candidate_rows=len(ck),missing_keys=0,extra_keys=0,
            max_common_coefficient_difference=float(np.max(np.abs(rc-cc),initial=0)),
            max_missing_coefficient=0.,max_extra_coefficient=0.)
    # Already-aligned keys are common and cannot appear elsewhere in these
    # unique sorted supports. Remove them before sorting the remaining union.
    nr,nc=len(rk),len(ck)
    aligned_max=0.
    if nr==nc:
        aligned=rk==ck
        aligned_max=float(np.max(np.abs(rc[aligned]-cc[aligned]),initial=0))
        rk,rc=rk[~aligned],rc[~aligned]
        ck,cc=ck[~aligned],cc[~aligned]
    _,ri,ci=np.intersect1d(rk,ck,assume_unique=True,return_indices=True)
    missing=np.ones(len(rk),dtype=bool); missing[ri]=False
    extra=np.ones(len(ck),dtype=bool); extra[ci]=False
    return dict(reference_rows=nr,candidate_rows=nc,
        missing_keys=int(missing.sum()),extra_keys=int(extra.sum()),
        max_common_coefficient_difference=max(aligned_max,float(np.max(np.abs(rc[ri]-cc[ci]),initial=0))),
        max_missing_coefficient=float(np.max(np.abs(rc[missing]),initial=0)),
        max_extra_coefficient=float(np.max(np.abs(cc[extra]),initial=0)))



def main():
    root=Path(sys.argv[1])
    timings_only='--timings-only' in sys.argv
    records=[]
    previous_path=root/'comparison.json'
    previous={r['name']:r for r in json.loads(previous_path.read_text())} if previous_path.exists() else {}
    failures=[]
    paths=list(root.glob('xxz21_*_19.json'))+list(root.glob('afh*_scan.json'))+list(root.glob('*_repeat.json'))
    for path in sorted(paths):
        result=json.loads(path.read_text())
        if result.get('timing_valid') is False:
            continue
        if result.get('status')=='out_of_memory':
            failures.append(dict(name=path.stem,status='out_of_memory',
                last_work=result.get('last_work'),peak_allocated=result.get('peak_allocated'),
                peak_reserved=result.get('peak_reserved'),error=result.get('error')))
            continue
        if not result.get('steps'):
            continue
        args=result['args']; xxz=args['workload']=='xxz'
        refname='xxz21_baseline_19.json' if xxz else f"afh{args['layers']}_baseline_{args['cap']}_scan.json"
        refpath=root/refname
        if not refpath.exists():
            continue
        reference=json.loads(refpath.read_text())
        count=min(len(reference['steps']),len(result['steps']))
        rows=result['steps'][:count] if count else result['steps']
        refs=reference['steps'][:count]
        record=dict(name=path.stem,status=result.get('status','running'),steps=len(result['steps']),
            seconds=sum(float(np.median(r['seconds'])) for r in rows),
            energy_seconds=sum(float(np.median(r['energy_seconds'])) for r in rows),
            final_seconds=float(np.median(rows[-1]['seconds'])),final_rows=rows[-1]['rows'],
            energy=rows[-1]['energy'],norm2=rows[-1]['norm2'],
            compared_steps=count,
            max_count_difference=max((abs(r['rows']-b['rows']) for r,b in zip(rows,refs)),default=None),
            max_energy_difference=max((abs(r['energy']-b['energy']) for r,b in zip(rows,refs)),default=None),
            max_norm2_difference=max((abs(r['norm2']-b['norm2']) for r,b in zip(rows,refs)),default=None),
            peak_allocated_gb=max(r['peak_allocated'] for r in rows)/1e9,
            peak_reserved_gb=max(r['peak_reserved'] for r in rows)/1e9,
            input_live_sum=sum(r['work']['input_live_sum'] for r in rows),
            gates_at_cap=sum(r['work']['at_cap'] for r in rows),
            circuit_matches=reference['circuit_sha256']==result['circuit_sha256'])
        if not timings_only and path.with_suffix('.npz').exists() and refpath.with_suffix('.npz').exists():
            record['coefficients']=state_difference(refpath.with_suffix('.npz'),path.with_suffix('.npz'))
        elif not timings_only and 'coefficients' in previous.get(record['name'],{}):
            old=previous[record['name']]
            fields=('seconds','final_rows','energy','norm2','compared_steps','circuit_matches')
            if all(old.get(k)==record.get(k) for k in fields):
                record['coefficients']=old['coefficients']
                record['coefficient_comparison_source']='Retained prior comparison; raw snapshots unavailable, not revalidated.'
        records.append(record)
        print(path.stem,record.get('coefficients','scalar comparison'),flush=True)
        if not timings_only:
            (root/'comparison.partial.json').write_text(json.dumps(records,indent=2)+'\n')
    (root/('comparison_timings.json' if timings_only else 'comparison.json')).write_text(json.dumps(records,indent=2)+'\n')
    (root/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    for r in records:
        difference='n/a' if r['max_energy_difference'] is None else f"{r['max_energy_difference']:.3g}"
        print(f"{r['name']:42s} {r['seconds']:9.4f}s {r['final_seconds']:8.4f}s {r['peak_allocated_gb']:7.3f}GB n={r['final_rows']:10d} dE={difference} dn={r['max_count_difference']}")

if __name__=='__main__':
    main()
