"""Compare AFH at 20/50M caps, promoting measured contenders."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/persistent_generalization_20260910'
LARGE=OUT/'cutoff1e5'
records=[]

def run(name,args,directory=LARGE):
    directory.mkdir(parents=True,exist_ok=True)
    print('START',name,flush=True)
    target=directory/(name+'.json')
    with (directory/(name+'.log')).open('w') as log:
        p=subprocess.run([sys.executable,str(ROOT/'benchmarks/benchmark_persistent_generalization.py'),
            *args,'--output',str(target)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    r=json.loads(target.read_text()) if target.exists() else {}
    records.append(dict(name=name,returncode=p.returncode,status=r.get('status')))
    (LARGE/'generic_scan_status.json').write_text(json.dumps(records,indent=2)+'\n')
    print('DONE',name,p.returncode,r.get('status'),flush=True)
    if p.returncode and r.get('status')!='out_of_memory':
        print((directory/(name+'.log')).read_text()[-3000:],flush=True)
        raise RuntimeError(name)
    return r

def afh(mode,cap,repeats=1):
    args=['--workload','afh','--size','8','--layers','2','--random-scale','1',
          '--cutoff','.00001','--cap',str(cap),'--variant',mode,'--repeats',str(repeats)]
    if cap==20000000:
        args+=['--save-state']
    return run(f'afh2_{mode}_{cap}_scan',args)

def seconds(r):
    return float(np.median(r['steps'][0]['seconds'])) if r.get('status')=='complete' else float('inf')

if __name__=='__main__':
    LARGE.mkdir(parents=True,exist_ok=True)
    with (OUT/'all_axes_tests.log').open('w') as log:
        t=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_triton_persistent_all_axes.py',
            'tests/test_triton_persistent_wide_options.py','tests/test_triton_persistent_options.py',
            'tests/test_triton_persistent.py','tests/test_triton_backend.py','tests/test_triton_invariants.py'],
            stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    print('TESTS',t.returncode,flush=True)
    if t.returncode:
        raise RuntimeError('Tests failed')
    twenty={'baseline':json.loads((LARGE/'afh2_baseline_20000000_scan.json').read_text())}
    assert twenty['baseline']['status']=='complete'
    for mode in ['local_all','select_all','inverted_all','select_all+local_all','inverted_all+local_all']:
        twenty[mode]=afh(mode,20000000)
    selector=min(['select_all','select_all+local_all'],key=lambda m:seconds(twenty[m]))
    indexed=min(['inverted_all','inverted_all+local_all'],key=lambda m:seconds(twenty[m]))
    fifty={'baseline':json.loads((LARGE/'afh2_baseline_50000000_scan.json').read_text())}
    assert fifty['baseline']['status']=='complete'
    modes=[indexed,selector]
    if seconds(twenty['local_all'])<=1.03*seconds(twenty[selector]):
        modes.append('local_all')
    for mode in modes:
        fifty[mode]=afh(mode,50000000)
    contender=min(modes,key=lambda m:seconds(fifty[m]))
    (LARGE/'generic_promotion.json').write_text(json.dumps(dict(selector=selector,indexed=indexed,
        contender=contender,rule='Compare all-axis variants at 20M. At 50M retain best maintained-index and active-scan variants; also local-only if within 3% of scan at 20M. Stop at 50M as requested.'),indent=2)+'\n')
    print('COMPLETE: stopped at the requested 50M cap',flush=True)
