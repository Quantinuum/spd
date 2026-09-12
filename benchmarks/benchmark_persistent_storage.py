"""Opt-in persistent-storage experiment using the unchanged stepwise timer."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['persistent', 'anti'], default='persistent')
    parser.add_argument('--num-steps', type=int, default=23)
    parser.add_argument('--dead-fraction', type=float, default=.1)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    a = parser.parse_args()
    import spd.triton_backend as tb
    from spd.triton_backend.persistent import evolve_step_persistent
    root = Path(__file__).resolve().parents[1]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(mode=a.mode, dead_fraction=a.dead_fraction, calls=[],
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  source_sha256=hashlib.sha256((root/'spd/triton_backend/persistent.py').read_bytes()).hexdigest(),
                  gpu_status=subprocess.check_output(['nvidia-smi', '-q'], text=True))
    if a.mode == 'persistent':
        def evolve(state, operations, trunc_val=0., max_num_str=None):
            stats = {'input_rows': state.get_size()}
            output = evolve_step_persistent(state, operations, trunc_val, max_num_str,
                                             dead_fraction=a.dead_fraction, stats=stats)
            stats['output_rows'] = output.get_size()
            result['calls'].append(stats)
            return output
        tb.evolve_step = evolve
    if a.profile:
        path = root/'benchmarks/profile_triton_forward.py'
        sys.argv = [str(path), '--steps', '3', '14', str(a.num_steps), '--output-dir', str(a.output_dir)]
    else:
        path = root/'benchmarks/investigate_triton_allocator.py'
        sys.argv = [str(path), '--num-steps', str(a.num_steps), '--output-dir', str(a.output_dir)]
    try:
        runpy.run_path(str(path), run_name='__main__')
    finally:
        (a.output_dir/'persistent.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
