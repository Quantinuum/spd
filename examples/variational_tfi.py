"""Use a stable TFI ansatz and reduce gate gradients to shared parameters."""

import numpy as np

import spd
from spd.ansatz import tfi_1d_hva


params = np.asarray([0.1, 0.2])
brickwork_ansatz = tfi_1d_hva(params, system_size=4, basis="0")

initial_spo = spd.create_spo({"ZZII": -1.0, "XIII": -1.0})
final_spo, _ = spd.evolve(
    initial_spo,
    brickwork_ansatz.circuit,
    trunc_val=1e-12,
    max_num_str=1_000,
)
initial_spgo = spd.init_gradient_spo(final_spo, basis="0")
_, gate_gradients, _ = spd.backpropagate(
    initial_spgo,
    brickwork_ansatz.circuit,
    trunc_val=1e-12,
    max_num_str=1_000,
)

print("gate gradients:", gate_gradients)
print(
    "parameter gradients:",
    brickwork_ansatz.parameter_gradients(gate_gradients),
)
