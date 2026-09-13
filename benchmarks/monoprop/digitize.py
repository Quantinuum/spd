"""Digitize marker centers from the published raster, then check source JSONL.

Calibration is from image axis ticks, not from the underlying timing values.
The x axis is categorical lattice size; y is logarithmic. Pixel uncertainty
is conservatively +/- 1 pixel (about 0.64% in runtime).
"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REF = HERE / 'reference'


def main():
    image = np.asarray(Image.open(REF / 'pauli_scaling_runtime.png').convert('RGB'))
    if image.shape != (878, 1094, 3):
        raise ValueError('Calibration requires the pinned 1094x878 source image')
    # Tick/grid centers read from the raster: 10 s and 100 s, respectively.
    y10, y100 = 634.5, 273.0
    x_centers = [141, 290, 439, 588, 737, 885, 1034]
    colors = {'monoprop':(0,114,178),'cuPauliProp (GPU)':(204,121,167),
              'PauliPropagation.jl':(230,159,0)}
    exact = {}
    for filename in ('scaling_cpu.jsonl','scaling_gpu.jsonl'):
        # The first file wins across files; later entries within a file win.
        local = {}
        for line in (REF / filename).read_text().splitlines():
            r = json.loads(line)
            if r.get('status') == 'ok':
                local[(r['label'],r['num_qubits'])] = r['total_runtime_s']
        for key,value in local.items():
            exact.setdefault(key,value)
    rows = []
    for engine,color in colors.items():
        mask = np.max(np.abs(image.astype(int)-np.array(color)),axis=2) <= 3
        for n,x in zip(range(6,19,2),x_centers):
            if engine == 'PauliPropagation.jl' and n>10:
                continue
            # Narrow strip through the circular marker excludes legend/title.
            ys,xs = np.where(mask[75:735,x-2:x+3])
            if not len(ys):
                raise ValueError(f'Missing marker for {engine}, L={n}')
            y = float(np.median(ys+75))
            value = 10**(1+(y10-y)/(y10-y100))
            source = exact[(engine,n*n)]
            error = (value/source-1)*100
            if abs(error)>1.:
                raise ValueError(f'Digitization differs from source by {error:.2f}%')
            rows.append(dict(engine=engine,lattice_n=n,num_qubits=n*n,
                marker_x_px=x,marker_y_px=y,digitized_runtime_s=value,
                published_source_runtime_s=source,relative_error_percent=error,
                pixel_uncertainty=1,runtime_uncertainty_percent=(10**(1/(y10-y100))-1)*100))
    with (REF/'digitized_scaling.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (REF/'digitization.json').write_text(json.dumps(dict(image='pauli_scaling_runtime.png',
        axis='categorical x; log10 y',y_at_10s=y10,y_at_100s=y100,
        colors=colors,max_absolute_error_percent=max(abs(r['relative_error_percent']) for r in rows)),indent=2)+'\n')
    print('Digitized',len(rows),'markers; max source discrepancy',max(abs(r['relative_error_percent']) for r in rows),'%')


if __name__ == '__main__':
    main()
