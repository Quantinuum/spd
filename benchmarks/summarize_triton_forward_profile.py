"""Summarize exclusive GPU trace events, avoiding CPU/device attribution double counts."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--baseline-dir', type=Path, help='Defaults to DIRECTORY/baseline')
    args = parser.parse_args()
    root = args.directory
    profile = json.loads((root / 'profile/profile.json').read_text())
    baseline_dir = args.baseline_dir if args.baseline_dir is not None else root / 'baseline'
    baseline = pickle.loads(next(baseline_dir.glob('*.pkl')).read_bytes())
    summary = {'steps': [], 'trajectory_validation': {}}
    for record in profile['profiles']:
        trace = json.loads((root / f"profile/step_{record['step']}_trace.json").read_text())
        stages = {}
        for event in trace['traceEvents']:
            if event.get('ph') != 'X' or event.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
                continue
            name = event['name']
            stage = name if name in ('build_index', 'rotate') else ('fill' if 'FillFunctor' in name else name)
            stage_record = stages.setdefault(stage, {'calls': 0, 'seconds': 0.})
            stage_record['calls'] += 1
            stage_record['seconds'] += event['dur'] / 1e6
        sample = record['output_snapshot_anticommuting_fraction_by_gate']
        summary['steps'].append(dict(
            step=record['step'], input_rows=record['input_rows'], output_rows=record['output_rows'],
            baseline_seconds=baseline['times'][record['step']-1],
            instrumented_wall_seconds=record['instrumented_wall_seconds'], gpu_stages=stages,
            cpu_api_seconds={e['name']: e['cpu_total_us']/1e6 for e in record['events'] if e['name'] in
                             ['aten::empty', 'aten::full', 'aten::zeros', 'aten::item', 'cudaStreamSynchronize', 'cudaMemcpyAsync']},
            snapshot_anticommuting_fraction={k: float(np.mean(v)) for k,v in sample.items()},
            snapshot_all_gate_anticommuting_fraction=float(np.mean([x for v in sample.values() for x in v])),
            peak_allocated_bytes=record['peak_allocated_bytes'], peak_reserved_bytes=record['peak_reserved_bytes']))
    records = profile['trajectory']
    summary['trajectory_validation'] = dict(
        steps=len(records), counts_equal=all(r['count']==baseline['num_paulis'][i] for i,r in enumerate(records)),
        max_expectation_difference=max(abs(r['expectation']-baseline['all_results'][i+1]) for i,r in enumerate(records)),
        max_norm_difference=max(abs(r['norm']-baseline['norms'][i]) for i,r in enumerate(records)))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
