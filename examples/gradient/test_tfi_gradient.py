import pickle
import pytket
from pytket import Circuit
import numpy as np
np.random.seed(0)
import spd
import tfi_setup

if __name__ == "__main__":

    test_dict = {'Z': {'exp_val': -2.47822274333196,
                       'grads': np.array([0.0, 5.222412994248933, 0.3545047836908741])},
                 '+': {'exp_val': -0.05237482630857131,
                       'grads': np.array([np.float64(-0.1838549159941437), np.float64(0.2915450928908909), np.float64(0.3598574035127437)]),
                       }
                 }

    for backend_name in ['numpy', 'jax']:
        for basis in ['Z', '+']:
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

            # random_thetas = (np.random.rand(number_of_parameters) - 0.5)
            random_thetas = [ 0.14589411, -0.06241279,  0.391773]
            if run_dim == 1:
                circ = tfi_setup.gen_1d_TFI_ansatz_circuit(random_thetas,
                                                           system_size,
                                                           )
                ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=1.3)
            else:
                circ = tfi_setup.gen_2d_TFI_ansatz_circuit(random_thetas,
                                                           system_size_x,
                                                           system_size_y,
                                                           )
                ham_dict = tfi_setup.gen_2d_Hamiltonian_dict(system_size_x, system_size_y, g=3.1)

            trunc_val = 1e-7
            print("\n Truncation Value:", trunc_val)

            exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val,
                                                        basis=basis, backend_name=backend_name)
            print("\n Expectation Value:", exp_val)
            exact_exp_val = exp_val

            grads, backward_final_spo = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val,
                                                                        basis=basis, backend_name=backend_name)
            # print("\n SPD Computed Gradients:", grads)
            combine_grads = []
            for i in range(number_of_parameters):
                combine_grads.append(np.array(grads[i*system_size:(i+1)*system_size]).sum() * np.pi)

            exact_grads = np.array(combine_grads)
            print("all spd grads = ", grads)
            print("combine_grads = ", combine_grads)

            assert np.allclose(exact_exp_val, test_dict[basis]['exp_val']), f"Expectation value does not match for basis {basis} and backend {backend_name}"
            assert np.allclose(exact_grads, test_dict[basis]['grads'], rtol=1e-5), f"Gradients do not match for basis {basis} and backend {backend_name}"

            # data_dict['exact'] = {'exp_val': exact_exp_val,
            #                       'grads': exact_grads,
            #                       }
            print("Passed all assertions for basis", basis, "and backend", backend_name)

    exit()


    if test_finite_difference := True:
        # for trunc_val in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        for trunc_val in [1e-4]:
            print("========================================================")
            print("========= Testing Truncation Value:", trunc_val, " =========")
            print("========================================================")

            exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val, backend_name=backend_name)
            print(f"\n Expectation Value[trunc={trunc_val}]:", exp_val)
            print("Error to Exact Expectation Value:", abs(exp_val - exact_exp_val) / abs(exact_exp_val))

            grads = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val, backend_name=backend_name)
            # print("\n SPD Computed Gradients:", grads)
            combine_grads = []
            for i in range(number_of_parameters):
                combine_grads.append(np.array(grads[i*system_size:(i+1)*system_size]).sum() * np.pi)

            print("\n Combined SPD Computed Gradients:", grads)
            print("Error to Exact Gradients:", np.linalg.norm(np.array(combine_grads) - exact_grads) / np.linalg.norm(exact_grads))

            data_dict['trunc_' + str(trunc_val)] = {'exp_val': exp_val,
                                                    'spd_grads': np.array(combine_grads),
                                                    }


            # for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            for eps in [1e-2, 1e-5]:
                print("\n Testing Finite Difference Gradient Computation:", " Epsilon =", eps)
                # Compute the finite difference gradient
                gradients = []
                # eps = 1e-2

                for i in range(number_of_parameters):
                    new_random_thetas = random_thetas.copy()
                    new_random_thetas[i] += eps
                    new_circ = tfi_setup.gen_1d_TFI_ansatz_circuit(new_random_thetas,
                                                                   system_size,
                                                                   )
                    new_exp_val, _ = spd.run_pytket_circuit(new_circ, ham_dict, trunc_val, backend_name=backend_name)
                    # print("new_exp_val = ", new_exp_val)
                    gradient = (new_exp_val - exp_val) / eps
                    # print(f" Gradient wrt theta[{i}]:", gradient)
                    gradients.append(gradient)

                print("\n Finite Difference Gradients:", gradients)
                print("\n Error of Finite Difference to Exact Gradients:", np.linalg.norm(np.array(gradients) - exact_grads) / np.linalg.norm(exact_grads))
                data_dict['trunc_' + str(trunc_val)]['fd_grads_eps_' + str(eps)] = np.array(gradients)

    # # Save the data_dict to a pickle file
    # with open('tfi_gradient_data.pkl', 'wb') as f:
    #     pickle.dump(data_dict, f)


