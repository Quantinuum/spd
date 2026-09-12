"""Compare recorded full and sparse index trajectories and allocator peaks."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    original_dir = args.directory.parent / 'forward_profile_20260910/baseline'
    baseline = pickle.loads(next(original_dir.glob('*.pkl')).read_bytes())
    result = {'reference': str(original_dir), 'runs': {}}
    for name in ['candidate1', 'candidate2', 'full_control']:
        path = args.directory / name
        d = pickle.loads(next(path.glob('*.pkl')).read_bytes())
        memory = json.loads((path / 'allocator.json').read_text())['steps'][-1]['counters']
        result['runs'][name] = dict(
            total_seconds=sum(d['times']), final_step_seconds=d['times'][-1],
            speedup_vs_original=sum(baseline['times'])/sum(d['times']),
            counts_equal=d['num_paulis']==baseline['num_paulis'],
            max_expectation_difference=float(np.max(np.abs(np.array(d['all_results'])-baseline['all_results']))),
            max_norm_difference=float(np.max(np.abs(np.array(d['norms'])-baseline['norms']))),
            peak_allocated_bytes=memory['allocated_bytes.all.peak'],
            peak_reserved_bytes=memory['reserved_bytes.all.peak'],
            allocation_retries=memory['num_alloc_retries'], ooms=memory['num_ooms'])
    result['candidate_mean_total_seconds'] = float(np.mean([result['runs'][n]['total_seconds'] for n in ['candidate1', 'candidate2']]))
    result['speedup_vs_full_control'] = result['runs']['full_control']['total_seconds']/result['candidate_mean_total_seconds']
    (args.directory / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
