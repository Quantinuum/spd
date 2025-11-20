import pickle
import pytket
from pytket import Circuit
import numpy as np
np.random.seed(0)
import spd
import tfi_setup

if __name__ == "__main__":
    system_size = 36
    number_of_parameters = 4
    data_dict = {}

    random_thetas = (np.random.rand(number_of_parameters) - 0.5)
    circ = tfi_setup.gen_1d_TFI_ansatz_circuit(random_thetas,
                                               system_size,
                                               )
    ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=1)

    trunc_val = 1e-7
    print("\n Truncation Value:", trunc_val)

    exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val, backend_name='numpy')
    print("\n Expectation Value:", exp_val)
    exact_exp_val = exp_val

    grads = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val, backend_name='numpy')
    # print("\n SPD Computed Gradients:", grads)
    combine_grads = []
    for i in range(number_of_parameters):
        combine_grads.append(np.array(grads[i*system_size:(i+1)*system_size]).sum() * np.pi)

    exact_grads = np.array(combine_grads)

    data_dict['exact'] = {'exp_val': exact_exp_val,
                          'grads': exact_grads,
                          }


    if test_finite_difference := True:
        # for trunc_val in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        for trunc_val in [1e-4]:
            print("========================================================")
            print("========= Testing Truncation Value:", trunc_val, " =========")
            print("========================================================")

            exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val, backend_name='numpy')
            print(f"\n Expectation Value[trunc={trunc_val}]:", exp_val)
            print("Error to Exact Expectation Value:", abs(exp_val - exact_exp_val) / abs(exact_exp_val))

            grads = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val, backend_name='numpy')
            # print("\n SPD Computed Gradients:", grads)
            combine_grads = []
            for i in range(number_of_parameters):
                combine_grads.append(np.array(grads[i*system_size:(i+1)*system_size]).sum() * np.pi)

            print("\n Combined SPD Computed Gradients:", combine_grads)
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
                    new_exp_val, _ = spd.run_pytket_circuit(new_circ, ham_dict, trunc_val, backend_name='numpy')
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


