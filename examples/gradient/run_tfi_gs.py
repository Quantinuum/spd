import pickle
import pytket
from pytket import Circuit
import numpy as np
# np.random.seed(0)
import spd
import tfi_setup

if __name__ == "__main__":

    backend_name = 'jax'
    system_size_x = 10
    system_size_y = 10
    system_size = system_size_x * system_size_y
    number_of_parameters = 15
    basis = '+'

    # system_size = 36
    # number_of_parameters = 18
    # data_dict = {}

    random_thetas = (np.random.rand(number_of_parameters) - 0.5)
    random_thetas[:6] = np.array([0.2653, 0.73664796, 0.1287, 0.385, 0.0623, -0.178,])
    # random_thetas[:6] = [-2.73849968e-02, 1.14894080e-03, 4.75105253e-02, 2.74527440e-02,
    #                      2.50015058e-01, 7.22616242e-06]
    # random_thetas[-6:] = [-0.00903, -0.003, 0.0476,
    #                      0.00902,  0.245, -0.00146]
    # random_thetas = np.array([0.1655667, 0.20199909, 0.2995168, 0.10819875])
    run_dim = 2
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
        ham_dict = tfi_setup.gen_2d_Hamiltonian_dict(system_size_x,
                                                     system_size_y,
                                                     g=3.04438)


    trunc_val = 1e-3
    print("\n Truncation Value:", trunc_val)

    exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val,
                                                basis=basis, backend_name=backend_name)
    print("\n Expectation Value:", exp_val)

    grads, backward_final_spo = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val,
                                                                basis=basis, backend_name=backend_name)
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

        exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val,
                                                    basis=basis, backend_name=backend_name)
        raw_grads, backward_final_spo = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val,
                                                                        basis=basis, backend_name=backend_name)
        grads = combine_grads(raw_grads, run_dim, number_of_parameters, system_size)

        # Add weight regularization
        lambda_reg = 0.00
        cost = exp_val + lambda_reg * np.sum(thetas**2)
        grad_reg = 2 * lambda_reg * thetas
        print(f"cost: {cost}, <E>: {exp_val}, Reg: {0.03 * np.sum(thetas**2)}")
        print(f"||theta||: {np.linalg.norm(thetas)}, ||grad||: {np.linalg.norm(grads)}")
        grads[:6] = 0
        return exp_val, grads


    import optax
    optimizer = optax.adam(learning_rate=3e-2)
    opt_state = optimizer.init(random_thetas)
    thetas = random_thetas.copy()
    for _ in range(500):
        print("===="*30)
        print(f"Thetas[{_}]: ", thetas)
        print("===="*30)
        f, g = get_f_g(thetas)
        updates, opt_state = optimizer.update(g, opt_state)
        thetas = optax.apply_updates(thetas, updates)


    """
    import scipy.optimize
    minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
    from scipy.optimize import basinhopping
    ret = basinhopping(get_f_g,
                       random_thetas,
                       minimizer_kwargs=minimizer_kwargs,
                       niter=200,
                       )
    print(ret)
    exit()


    result = scipy.optimize.minimize(get_f_g,
                                     random_thetas,
                                     method='L-BFGS-B',
                                     jac=True,
                                     options={'disp': True, 'gtol': 1e-4, 'maxiter': 100}
                                     )
    print(result)
    """

