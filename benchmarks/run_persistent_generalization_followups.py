"""Harder AFH angles, wide-key repeats, and independent reference checks."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/persistent_generalization_20260910'

def run(name,args,directory=OUT):
    directory.mkdir(parents=True,exist_ok=True)
    print('START',name,flush=True)
    with (directory/(name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'benchmarks/benchmark_persistent_generalization.py'),
            *args,'--output',str(directory/(name+'.json'))],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    print('DONE',name,result.returncode,flush=True)
    if result.returncode:
        print((directory/(name+'.log')).read_text()[-3000:],flush=True)
        raise RuntimeError(name)

if __name__=='__main__':
    large=OUT/'large'
    for cap in [5000000,2000000,10000000]:
        modes=['control','inverted','select','local','fingerprint','soa','device'] if cap==5000000 else ['baseline','control','inverted','select']
        for mode in modes:
            run(f'afh2_{mode}_{cap}_scan',['--workload','afh','--size','8','--layers','2',
                '--random-scale','1','--cutoff','.0001','--cap',str(cap),'--variant',mode,
                '--repeats','2','--save-state'],large)
    for mode in ['baseline','inverted','inverted+local']:
        run(f'xxz21_{mode}_repeat',['--workload','xxz','--size','21','--steps','19',
            '--variant',mode,'--repeats','2'])
    run('afh2_reference_3000000_scan',['--workload','afh','--size','8','--layers','2',
        '--variant','reference','--cap','3000000','--save-state'])
    with (OUT/'wide_tests.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_triton_persistent_wide_options.py'],
            stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    print('TESTS',result.returncode,flush=True)
