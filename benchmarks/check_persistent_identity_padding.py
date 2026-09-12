"""Full coefficient check for identity padding on the default physical AFH circuit."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/persistent_key_scaling_20260911'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[];states=[];files=[]
    for pad in [0,32]:
        target=OUT/f'afh_default_pad{pad}.json'
        with target.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'benchmarks/benchmark_persistent_generalization.py'),
                '--workload','afh','--size','8','--layers','2','--variant','baseline',
                '--cap','3000000','--extra-padding-qubits',str(pad),'--save-state','--output',str(target)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads(target.read_text());assert result['status']=='complete'
        path=target.with_suffix('.npz');files.append(path)
        with np.load(path) as data:
            keys=data['keys'];coeff=data['coeff']
            half=keys.shape[1]//2
            assert not np.any(keys[:,16:half]) and not np.any(keys[:,half+16:])
            canonical=np.ascontiguousarray(np.concatenate([keys[:,:16],keys[:,half:half+16]],axis=1))
            packed=canonical.view(np.dtype((np.void,128))).ravel();order=np.argsort(packed)
            states.append((packed[order],coeff[order]))
        records.append(dict(run=str(target.relative_to(ROOT)),snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    same_keys=np.array_equal(states[0][0],states[1][0])
    same_coeff= same_keys and np.array_equal(states[0][1],states[1][1])
    diff=float(np.max(np.abs(states[0][1]-states[1][1]))) if same_keys else None
    result=dict(rows=[len(s[0]) for s in states],same_canonical_keys=same_keys,
        bit_identical_coefficients=same_coeff,max_coefficient_difference=diff,runs=records,
        snapshots='Removed after recording the comparison; rerun this script to regenerate.')
    (OUT/'padding_coefficient_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    assert same_keys and same_coeff,result
    for path in files:path.unlink()
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
