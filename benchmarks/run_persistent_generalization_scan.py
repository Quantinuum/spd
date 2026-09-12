"""Sequential isolated-process scan; never overlaps GPU measurements."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/persistent_generalization_20260910'
OUT.mkdir(parents=True,exist_ok=True)
jobs=[]
for variant in ['baseline','control','inverted','select','local','fingerprint','soa']:
    jobs.append((f'xxz21_{variant}_19',['--workload','xxz','--size','21','--steps','19','--variant',variant]))
for layers,caps in [(2,[3000000,500,1500]),(3,[3000000,1000,5000])]:
    for cap in caps:
        modes=['baseline','control','inverted','select','device']
        if cap==3000000:
            modes+=['local','fingerprint','soa']
        for mode in modes:
            jobs.append((f'afh{layers}_{mode}_{cap}_scan',['--workload','afh','--size','8',
                '--layers',str(layers),'--cap',str(cap),'--variant',mode,'--repeats','2','--save-state']))
records=[]
for name,args in jobs:
    print('START',name,flush=True)
    with (OUT/(name+'.log')).open('w') as log:
        run=subprocess.run([sys.executable,str(ROOT/'benchmarks/benchmark_persistent_generalization.py'),
            *args,'--output',str(OUT/(name+'.json'))],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    records.append(dict(name=name,returncode=run.returncode))
    (OUT/'scan_status.json').write_text(json.dumps(records,indent=2)+'\n')
    print('DONE',name,run.returncode,flush=True)
    if run.returncode:
        print((OUT/(name+'.log')).read_text()[-3000:],flush=True)
