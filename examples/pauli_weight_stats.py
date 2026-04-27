from spd import jax_backend, numpy_backend


def show_weight_stats(name, module):
    module.utils.set_packbit(32)
    spo = module.create_op({
        "IIII": 1.0,
        "XIII": 2.0,
        "YZZI": 3.0,
        "XYZX": 4.0,
    })

    print(name)
    print("distribution:", spo.get_pauli_weight_distribution())
    print("counts:", spo.get_pauli_weight_counts())


show_weight_stats("numpy", numpy_backend)
show_weight_stats("jax", jax_backend)
