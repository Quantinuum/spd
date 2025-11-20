import pickle
import pytket
from pytket import Circuit
import numpy as np
# np.random.seed(0)
import spd
import tfi_setup

if __name__ == "__main__":

    system_size_x = 4
    system_size_y = 4
    system_size = system_size_x * system_size_y
    number_of_parameters = 3

    # system_size = 36
    # number_of_parameters = 18
    # data_dict = {}

    random_thetas = (np.random.rand(number_of_parameters) - 0.5) * 0.1
    # random_thetas = np.array([0.1655667, 0.20199909, 0.2995168, 0.10819875])
    run_dim = 2
    if run_dim == 1:
        circ = tfi_setup.gen_1d_TFI_ansatz_circuit(random_thetas,
                                                   system_size,
                                                   )
        ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=1.0)
    else:
        circ = tfi_setup.gen_2d_TFI_ansatz_circuit(random_thetas,
                                                   system_size_x,
                                                   system_size_y,
                                                   )
        ham_dict = tfi_setup.gen_2d_Hamiltonian_dict(system_size_x,
                                                     system_size_y,
                                                     g=3.1)


    trunc_val = 1e-3
    print("\n Truncation Value:", trunc_val)

    exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val, backend_name='numpy')
    print("\n Expectation Value:", exp_val)

    grads = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val, backend_name='numpy')
    print("\n SPD Computed Gradients:", grads)

    def combine_grads(grads, run_dim, number_of_parameters, system_size):
        combine_grads = []
        if run_dim == 1:
            for i in range(number_of_parameters):
                combine_grads.append(np.array(grads[i*system_size:(i+1)*system_size]).sum() * np.pi)
        else:
            # for i in range(number_of_parameters//2):
            #     combine_grads.append(np.array(grads[3*i*system_size:(3*i+1)*system_size]).sum() * np.pi)
            #     combine_grads.append(np.array(grads[(3*i+1)*system_size:(3*i+3)*system_size]).sum() * np.pi)
            for i in range(number_of_parameters//3):
                combine_grads.append(np.array(grads[4*i*system_size:(4*i+1)*system_size]).sum() * np.pi)
                combine_grads.append(np.array(grads[(4*i+1)*system_size:(4*i+2)*system_size]).sum() * np.pi)
                combine_grads.append(np.array(grads[(4*i+2)*system_size:(4*i+4)*system_size]).sum() * np.pi)

        return np.array(combine_grads)


    def get_f_g(thetas):
        if run_dim == 1:
            circ = tfi_setup.gen_1d_TFI_ansatz_circuit(thetas,
                                             system_size,
                                             )
        else:
            circ = tfi_setup.gen_2d_TFI_ansatz_circuit(thetas,
                                             system_size_x,
                                             system_size_y,
                                             )

        exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val, backend_name='numpy')
        raw_grads = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val, backend_name='numpy')
        grads = combine_grads(raw_grads, run_dim, number_of_parameters, system_size)

        print("\n Current Thetas:", thetas, " Expectation Value:", exp_val, " Gradients:", grads)
        return exp_val, grads


    import scipy.optimize
    # minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
    # from scipy.optimize import basinhopping
    # ret = basinhopping(get_f_g,
    #                    random_thetas,
    #                    minimizer_kwargs=minimizer_kwargs,
    #                    niter=200,
    #                    )
    # print(ret)
    # exit()


    result = scipy.optimize.minimize(get_f_g,
                                     random_thetas,
                                     method='L-BFGS-B',
                                     jac=True,
                                     options={'disp': True, 'gtol': 1e-4, 'maxiter': 100}
                                     )
    print(result)

