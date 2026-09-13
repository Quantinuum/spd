"""Build the comparison tables/figure from saved results; no propagation."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REF=HERE/'reference'
RESULTS=HERE/'results'


def load(name):
    return json.loads((RESULTS/name).read_text())


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    small_a=load('small_triton.coefficients.json')
    small_b=load('small_cupauliprop.coefficients.json')
    assert small_a.keys()==small_b.keys(), 'Small-case support mismatch'
    small_error=max(abs(small_a[k]-small_b[k]) for k in small_a)
    assert small_error < 1e-12, small_error
    (RESULTS/'small_validation.json').write_text(json.dumps(dict(
        num_terms=len(small_a), keys_equal=True, max_coefficient_error=small_error),indent=2)+'\n')
    digits=list(csv.DictReader((REF/'digitized_scaling.csv').open()))
    gpu={r['num_qubits']:r for r in map(json.loads,(REF/'scaling_gpu.jsonl').read_text().splitlines()) if r.get('status')=='ok'}
    cpu={r['num_qubits']:r for r in map(json.loads,(REF/'scaling_cpu.jsonl').read_text().splitlines()) if r.get('status')=='ok' and r['backend']=='monoprop'}
    fig,ax=plt.subplots(figsize=(9.4,5.8),layout='constrained')
    styles={'monoprop':('#0072B2','o','MonoProp · published CPU'),
        'cuPauliProp (GPU)':('#CC79A7','o','cuPauliProp · published A100'),
        'PauliPropagation.jl':('#E69F00','o','PauliPropagation.jl · published CPU')}
    for engine,(color,marker,label) in styles.items():
        rows=[r for r in digits if r['engine']==engine]
        ax.plot([int(r['lattice_n']) for r in rows],[float(r['digitized_runtime_s']) for r in rows],
            color=color,marker=marker,linestyle='--',linewidth=1.8,label=label)
    comparisons=[]
    for n in range(6,19,2):
        paths=sorted(RESULTS.glob(f'scaling_triton_L{n}_r*.json'))
        runs=[json.loads(p.read_text()) for p in paths]
        if len(runs)!=3 or any(len(r['rows'])!=28 for r in runs):
            raise ValueError(f'Need three complete runs at L={n}')
        counts=[r['rows'][-1]['num_terms'] for r in runs]
        if any(c!=gpu[n*n]['final_num_terms'] for c in counts):
            raise ValueError(f'Final support count mismatch at L={n}: {counts}')
        err=max(abs(r['rows'][-1]['expectation']-gpu[n*n]['final_expval']) for r in runs)
        if err>1e-12:
            raise ValueError(f'Expectation mismatch at L={n}: {err}')
        times=[r['total_runtime_s'] for r in runs]
        median=statistics.median(times)
        comparisons.append(dict(lattice_n=n,num_qubits=n*n,
            triton_median_s=median,triton_min_s=min(times),triton_max_s=max(times),trials=len(runs),
            monoprop_published_s=cpu[n*n]['total_runtime_s'],
            cupauliprop_published_s=gpu[n*n]['total_runtime_s'],
            monoprop_over_triton=cpu[n*n]['total_runtime_s']/median,
            cupauliprop_over_triton=gpu[n*n]['total_runtime_s']/median,
            triton_final_terms=counts[0],cupauliprop_final_terms=gpu[n*n]['final_num_terms'],
            monoprop_final_terms=cpu[n*n]['final_num_terms'],max_expectation_error_vs_cupauliprop=err,
            triton_peak_allocated_bytes=max(max(s['device_allocated_peak_bytes'] for s in r['rows']) for r in runs),
            triton_peak_reserved_bytes=max(max(s['device_reserved_peak_bytes'] for s in r['rows']) for r in runs),
            triton_median_excl_first_s=statistics.median(r['total_runtime_excl_first_s'] for r in runs),
            monoprop_published_excl_first_s=cpu[n*n]['total_runtime_excl_first_s'],
            cupauliprop_published_excl_first_s=gpu[n*n]['total_runtime_excl_first_s']))
    x=[r['lattice_n'] for r in comparisons];y=[r['triton_median_s'] for r in comparisons]
    ax.errorbar(x,y,yerr=[[r['triton_median_s']-r['triton_min_s'] for r in comparisons],
        [r['triton_max_s']-r['triton_median_s'] for r in comparisons]],
        color='#009E73',marker='s',linewidth=2.6,capsize=4,label='SPD Triton · local A100 (median of 3)')
    ax.set_yscale('log');ax.set_xticks(x,[f'{n}×{n}' for n in x])
    ax.set_xlabel('Lattice size');ax.set_ylabel('Total propagation + expectation time (s)')
    ax.set_title('2D tilted-field Ising: 28 Trotter layers\nCoefficient cutoff 10⁻⁶ · no weight or term cap')
    ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=9,loc='lower right')
    fig.supxlabel('Dashed: digitized published data (≤0.64% pixel uncertainty). Solid: new measurements.\nDifferent reference hardware; local Triton compilation excluded. See benchmark methodology.',fontsize=8)
    write_csv(HERE/'scaling_comparison.csv',comparisons)
    local=load('fixed_cupauliprop.json');triton=load('fixed_triton.json')
    source=json.loads((REF/'results_gpu.json').read_text());key='cuPauliProp (GPU)'
    fixed_validation=dict(
        local_cupauliprop_counts_match_published=[r['num_terms'] for r in local['rows']]==source['num_terms'][key],
        triton_counts_match_published=[r['num_terms'] for r in triton['rows']]==source['num_terms'][key],
        local_cupauliprop_max_expectation_error=max(abs(r['expectation']-v) for r,v in zip(local['rows'],source['expvals'][key])),
        triton_max_expectation_error=max(abs(r['expectation']-v) for r,v in zip(triton['rows'],source['expvals'][key])))
    assert fixed_validation['local_cupauliprop_counts_match_published']
    assert fixed_validation['triton_counts_match_published']
    assert fixed_validation['local_cupauliprop_max_expectation_error'] < 1e-12
    assert fixed_validation['triton_max_expectation_error'] < 1e-12
    for ext in ('png','svg'):
        fig.savefig(HERE/f'pauli_scaling_comparison.{ext}',dpi=180)
    source_all=json.loads((REF/'results.json').read_text())
    fixed=[]
    for name in ('monoprop','cuPauliProp (GPU)','QuEra ppvm','PauliPropagation.jl','Qiskit pauli-prop'):
        d=source_all
        device=name=='cuPauliProp (GPU)'
        fixed.append(dict(engine=name,provenance='published',final_step_s=d['runtime'][name][-1],
            final_terms=d['num_terms'][name][-1],final_expectation=d['expvals'][name][-1],
            peak_working_memory_mib=max(d['native_memory'][name]) if device else max(d['memory'][name]),
            memory_kind='device allocated' if device else 'host RSS',reserved_device_mib='',trials=1))
    for name,r in [('cuPauliProp',local),('SPD Triton',triton)]:
        fixed.append(dict(engine=name,provenance='local A100-SXM4-80GB',final_step_s=r['rows'][-1]['runtime_s'],
            final_terms=r['rows'][-1]['num_terms'],final_expectation=r['rows'][-1]['expectation'],
            peak_working_memory_mib=max(x['device_allocated_peak_bytes'] for x in r['rows'])/2**20,
            memory_kind='device allocated',reserved_device_mib=max(x['device_reserved_peak_bytes'] for x in r['rows'])/2**20,trials=1))
    write_csv(HERE/'fixed_comparison.csv',fixed)
    validation=dict(small=load('small_validation.json'),fixed=fixed_validation,
        scaling_all_counts_match=True,scaling_max_expectation_error=max(r['max_expectation_error_vs_cupauliprop'] for r in comparisons))
    (HERE/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    lines=['# Measured comparison tables','','## Fixed 12×12 lattice, observable ZZ[20,21]','',
        '| Engine | Source | Final step (s) | Final terms | Peak memory (MiB) | Memory type |',
        '|---|---|---:|---:|---:|---|']
    for r in fixed:
        lines.append(f"| {r['engine']} | {r['provenance']} | {r['final_step_s']:.3f} | {r['final_terms']:,} | {r['peak_working_memory_mib']:.1f} | {r['memory_kind']} |")
    lines+=['','## Scaling, central ZZ bond, 28 layers','',
        '| Lattice | Triton median [min, max] (s) | Published MonoProp (s) | Published cuPauliProp (s) | MonoProp / Triton | cuPauliProp / Triton |',
        '|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['lattice_n']}×{r['lattice_n']} | {r['triton_median_s']:.3f} [{r['triton_min_s']:.3f}, {r['triton_max_s']:.3f}] | {r['monoprop_published_s']:.3f} | {r['cupauliprop_published_s']:.3f} | {r['monoprop_over_triton']:.2f}× | {r['cupauliprop_over_triton']:.2f}× |")
    lines+=['','Published table values use the exact source JSON; the figure uses raster-digitized reference points.',
        'Hardware, timing, memory, and truncation qualifications are in [README.md](README.md).']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [HERE/'run_benchmark.py',HERE/'run_sweep.py',HERE/'digitize.py',HERE/'plot_comparison.py',
                  *sorted(RESULTS.glob('*.json'))]}
    (HERE/'measurement_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
