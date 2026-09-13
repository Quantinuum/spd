"""Run independent Triton processes, sequentially, with a 300-second limit each."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes',type=int,nargs='+',default=list(range(6,19,2)))
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output-dir',type=Path,default=HERE/'results')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for repeat in range(1,args.repeats+1):
        for n in args.sizes:
            output=args.output_dir/f'scaling_triton_L{n}_r{repeat}.json'
            if output.exists():
                old=json.loads(output.read_text())
                if len(old.get('rows',[]))==28:
                    print('Already complete:',output.name,flush=True)
                    continue
                raise RuntimeError(f'Inspect incomplete output before retrying: {output}')
            print('Starting',output.name,flush=True)
            with output.with_suffix('.log').open('w') as log:
                try:
                    run=subprocess.run([sys.executable,str(HERE/'run_benchmark.py'),
                        '--backend','triton','--n',str(n),'--observable','central',
                        '--output',str(output)],stdout=log,stderr=subprocess.STDOUT,timeout=300)
                    if run.returncode:
                        raise RuntimeError(f'Benchmark failed: {output.with_suffix(".log")}')
                except subprocess.TimeoutExpired:
                    output.with_suffix('.failure.json').write_text(json.dumps(dict(status='timeout',limit_s=300)))
                    raise
            print('Finished',output.name,flush=True)


if __name__=='__main__':
    main()
