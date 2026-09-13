# Measured comparison tables

## Fixed 12×12 lattice, observable ZZ[20,21]

| Engine | Source | Final step (s) | Final terms | Peak memory (MiB) | Memory type |
|---|---|---:|---:|---:|---|
| monoprop | published | 1.561 | 33,309,328 | 4275.4 | host RSS |
| cuPauliProp (GPU) | published | 12.233 | 31,164,633 | 11651.5 | device allocated |
| QuEra ppvm | published | 417.290 | 24,878,659 | 6955.8 | host RSS |
| PauliPropagation.jl | published | 433.162 | 31,177,867 | 11102.0 | host RSS |
| Qiskit pauli-prop | published | 507.740 | 33,309,328 | 34270.1 | host RSS |
| cuPauliProp | local A100-SXM4-80GB | 11.287 | 31,164,633 | 14175.4 | device allocated |
| SPD Triton | local A100-SXM4-80GB | 0.834 | 31,164,633 | 7076.8 | device allocated |

## Scaling, central ZZ bond, 28 layers

| Lattice | Triton median [min, max] (s) | Published MonoProp (s) | Published cuPauliProp (s) | MonoProp / Triton | cuPauliProp / Triton |
|---|---:|---:|---:|---:|---:|
| 6×6 | 1.865 [1.855, 1.870] | 6.346 | 26.818 | 3.40× | 14.38× |
| 8×8 | 2.837 [2.815, 2.838] | 6.718 | 37.981 | 2.37× | 13.39× |
| 10×10 | 4.887 [4.876, 4.918] | 7.983 | 69.822 | 1.63× | 14.29× |
| 12×12 | 6.778 [6.775, 6.786] | 9.516 | 108.074 | 1.40× | 15.95× |
| 14×14 | 11.047 [11.014, 11.148] | 10.976 | 152.930 | 0.99× | 13.84× |
| 16×16 | 20.163 [20.152, 20.304] | 11.606 | 167.792 | 0.58× | 8.32× |
| 18×18 | 26.613 [26.528, 26.650] | 14.104 | 282.897 | 0.53× | 10.63× |

Published table values use the exact source JSON; the figure uses raster-digitized reference points.
Hardware, timing, memory, and truncation qualifications are in [README.md](README.md).
