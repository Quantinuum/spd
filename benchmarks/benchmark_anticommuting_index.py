"""Run the standard synchronized benchmark with the full or sparse forward index.

The full-index control substitutes only the index launch and ignores its gate
argument. All other production forward code and benchmark timing are identical.
Run GPU trials sequentially in separate processes.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', choices=['full', 'anti'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-steps', type=int, default=23)
    args = parser.parse_args()
    import torch
    import spd.triton_backend as tb
    from spd.triton_backend.kernels import build_index
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(index=args.index,
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip(),
        sources_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                       ['spd/triton_backend/kernels.py', 'spd/triton_backend/__init__.py',
                        'examples/benchmark_2d_obc_xx_z_stepwise.py', 'benchmarks/benchmark_anticommuting_index.py']},
        versions={p: importlib.metadata.version(p) for p in ['torch', 'triton', 'numpy', 'pytket']},
        gpu=torch.cuda.get_device_name(), allocator_backend=torch.cuda.get_allocator_backend(),
        allocator_env={k: os.environ.get(k) for k in ['PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF']},
        gpu_status=subprocess.check_output(['nvidia-smi', '-q'], text=True))
    (args.output_dir / 'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.index == 'full':
        class FullIndexLaunch:
            def __getitem__(self, grid):
                def launch(keys, table, gate, n, mask, width, block):
                    return build_index[grid](keys, table, n, mask, width, block)
                return launch
        tb.build_anticommuting_index = FullIndexLaunch()
    path = root / 'benchmarks/investigate_triton_allocator.py'
    sys.argv = [str(path), '--num-steps', str(args.num_steps), '--output-dir', str(args.output_dir)]
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
