import numpy as np
import spd

import tfi_setup


np.random.seed(0)


def combine_grads(grads, number_of_parameters, system_size):
    combined = []
    for i in range(number_of_parameters):
        combined.append(np.array(grads[i * system_size:(i + 1) * system_size]).sum() * np.pi)
    return np.array(combined)


if __name__ == "__main__":
    test_dict = {
        "Z": {
            "exp_val": -2.47822274333196,
            "grads": np.array([0.0, 5.222412994248933, 0.3545047836908741]),
        },
        "+": {
            "exp_val": -0.05237482630857131,
            "grads": np.array(
                [
                    np.float64(-0.1838549159941437),
                    np.float64(0.2915450928908909),
                    np.float64(0.3598574035127437),
                ]
            ),
        },
    }

    for backend_name in ["numpy", "jax"]:
        for basis in ["Z", "+"]:
            backend = spd.BackendAdapter.from_name(backend_name, packbit=32, precision="single")
            if backend_name == "jax":
                backend.module.set_algorithm("stack_sort_merge")

            run_dim = 2
            if run_dim == 2:
                system_size_x = 6
                system_size_y = 6
                system_size = system_size_x * system_size_y
                number_of_parameters = 3
            else:
                system_size = 36
                number_of_parameters = 4

            data_dict = {}
            random_thetas = [0.14589411, -0.06241279, 0.391773]
            if run_dim == 1:
                circ = tfi_setup.gen_1d_TFI_ansatz_circuit(random_thetas, system_size)
                ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=1.3)
            else:
                circ = tfi_setup.gen_2d_TFI_ansatz_circuit(random_thetas, system_size_x, system_size_y)
                ham_dict = tfi_setup.gen_2d_Hamiltonian_dict(system_size_x, system_size_y, g=3.1)

            trunc_val = 1e-7
            print("\n Truncation Value:", trunc_val)

            initial_spo = spd.create_spo(ham_dict, backend=backend)
            final_spo, _ = spd.evolve(initial_spo, circ, trunc_val, max_num_str=int(1e6), backend=backend)
            exp_val = final_spo.get_expectation_value(basis=basis)

            initial_spgo = spd.init_gradient_spo(final_spo, basis=basis, backend=backend)
            _, grads, _ = spd.backpropagate(
                initial_spgo,
                circ,
                trunc_val,
                max_num_str=int(1e6),
                backend=backend,
            )
            combined = combine_grads(grads, number_of_parameters, system_size)
            print("all spd grads = ", grads)
            print("combine_grads = ", combined)

            exact_exp_val = test_dict[basis]["exp_val"]
            exact_grads = test_dict[basis]["grads"]

            assert np.allclose(exp_val, exact_exp_val)
            assert np.allclose(combined, exact_grads, rtol=1e-5)
            print("Passed all assertions for basis", basis, "and backend", backend_name)

    if test_finite_difference := True:
        backend = spd.BackendAdapter.from_name("jax", packbit=32, precision="single")
        backend.module.set_algorithm("stack_sort_merge")
        basis = "Z"
        exact_exp_val = test_dict[basis]["exp_val"]
        exact_grads = test_dict[basis]["grads"]

        for trunc_val in [1e-4]:
            print("========================================================")
            print("========= Testing Truncation Value:", trunc_val, " =========")
            print("========================================================")

            initial_spo = spd.create_spo(ham_dict, backend=backend)
            final_spo, _ = spd.evolve(initial_spo, circ, trunc_val, max_num_str=int(1e6), backend=backend)
            exp_val = final_spo.get_expectation_value(basis=basis)
            print(f"\n Expectation Value[trunc={trunc_val}]:", exp_val)

            initial_spgo = spd.init_gradient_spo(final_spo, basis=basis, backend=backend)
            _, grads, _ = spd.backpropagate(
                initial_spgo,
                circ,
                trunc_val,
                max_num_str=int(1e6),
                backend=backend,
            )
            combined = combine_grads(grads, number_of_parameters, system_size)

            print("\n Combined SPD Computed Gradients:", combined)
            print(
                "Error to Exact Gradients:",
                np.linalg.norm(np.array(combined) - exact_grads) / np.linalg.norm(exact_grads),
            )

            data_dict["trunc_" + str(trunc_val)] = {
                "exp_val": exp_val,
                "spd_grads": np.array(combined),
            }

            for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                print("\n Testing Finite Difference Gradient Computation:", " Epsilon =", eps)
                gradients = []

                for i in range(number_of_parameters):
                    new_random_thetas = random_thetas.copy()
                    new_random_thetas[i] += eps
                    if run_dim == 1:
                        new_circ = tfi_setup.gen_1d_TFI_ansatz_circuit(new_random_thetas, system_size)
                        ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=1.3)
                    else:
                        new_circ = tfi_setup.gen_2d_TFI_ansatz_circuit(
                            new_random_thetas,
                            system_size_x,
                            system_size_y,
                        )
                        ham_dict = tfi_setup.gen_2d_Hamiltonian_dict(system_size_x, system_size_y, g=3.1)

                    new_initial_spo = spd.create_spo(ham_dict, backend=backend)
                    new_final_spo, _ = spd.evolve(
                        new_initial_spo,
                        new_circ,
                        trunc_val,
                        max_num_str=int(1e6),
                        backend=backend,
                    )
                    new_exp_val = new_final_spo.get_expectation_value(basis=basis)
                    gradients.append((new_exp_val - exp_val) / eps)

                print("\n Finite Difference Gradients:", gradients)
                print(
                    "\n Error of Finite Difference to Exact Gradients:",
                    np.linalg.norm(np.array(gradients) - exact_grads) / np.linalg.norm(exact_grads),
                )
                data_dict["trunc_" + str(trunc_val)]["fd_grads_eps_" + str(eps)] = np.array(gradients)
