"""Run one isolated persistent-storage option with the standard stepwise timer."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--steps', type=int, default=23)
    parser.add_argument('--dead-fraction', type=float, default=.1)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--growth-factor', type=float)
    args = parser.parse_args()
    from spd.triton_backend import persistent
    from spd.triton_backend.persistent_options import evolve_options
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(variant=args.variant, dead_fraction=args.dead_fraction, growth_factor=args.growth_factor,
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
        ['spd/triton_backend/persistent.py','spd/triton_backend/persistent_options.py',
         'spd/triton_backend/experiment_kernels.py']})
    (args.output_dir/'option.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.growth_factor is not None:
        import torch
        if args.variant != 'baseline' or args.growth_factor <= 1:
            raise ValueError('Growth sweep requires baseline and factor > 1')
        class GrowthStorage(persistent._Storage):
            def _reserve(self):
                needed = 2*self.n
                if needed <= self.capacity:
                    return
                self.capacity = max(needed, int(self.capacity*args.growth_factor))
                keys = torch.empty((self.capacity,self.width),dtype=torch.int32,device=self.device)
                coeff = torch.empty(self.capacity,dtype=self.dtype,device=self.device)
                keys[:self.n].copy_(self.keys[:self.n])
                coeff[:self.n].copy_(self.coeff[:self.n])
                self.keys,self.coeff = keys,coeff
                self.growths += 1
        persistent._Storage = GrowthStorage
    if args.variant != 'baseline':
        def evolve(state, operations, trunc_val=0., max_num_str=None, *, dead_fraction=.1, stats=None):
            return evolve_options(state, operations, trunc_val, max_num_str,
                mode=args.variant, dead_fraction=dead_fraction, stats=stats)
        persistent.evolve_step_persistent = evolve
    path = root/'benchmarks/benchmark_persistent_storage.py'
    sys.argv = [str(path), '--num-steps', str(args.steps), '--dead-fraction', str(args.dead_fraction),
                '--output-dir', str(args.output_dir)] + (['--profile'] if args.profile else [])
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
