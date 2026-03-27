import pickle
import pytket
from pytket import Circuit
import numpy as np
# np.random.seed(0)
import spd
import sys
import tfi_setup

if __name__ == "__main__":

    backend_name = 'jax'

    method = 'basinhopping'
    number_of_parameters =  int(sys.argv[1])
    system_size_x = number_of_parameters + 3
    system_size_y = system_size_x
    system_size = system_size_x * system_size_y
    basis = '+'
    g = 3.1

    niter = int(sys.argv[2])


    random_thetas = (np.random.rand(number_of_parameters) - 0.5) * 0.1
    # run_dim = 2
    circ = tfi_setup.gen_2d_TFI_ansatz_circuit(random_thetas,
                                               system_size_x,
                                               system_size_y,
                                               )
    ham_dict = tfi_setup.gen_2d_Hamiltonian_dict(system_size_x,
                                                 system_size_y,
                                                 g=g,)
                                                 # g=3.04438)

    trunc_val = 1e-6
    max_num_str = 1e6
    print("\n Truncation Value:", trunc_val)
    # for trunc_val in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]:
    #     print(f"\n Truncation Value: {trunc_val} | max num str: {max_num_str}")

    #     exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val,
    #                                                 max_num_str=max_num_str,
    #                                                 basis=basis, backend_name=backend_name)
    #     print("\n Expectation Value:", exp_val)

    # grads, backward_final_spo = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val,
    #                                                             max_num_str=max_num_str,
    #                                                             basis=basis, backend_name=backend_name)
    # print("\n SPD Computed Gradients:", grads)


    def combine_grads(grads, number_of_parameters, system_size):
        combine_grads = []
        for i in range(number_of_parameters//2):
            combine_grads.append(np.array(grads[3*i*system_size:(3*i+2)*system_size]).sum() * np.pi)
            combine_grads.append(np.array(grads[(3*i+2)*system_size:(3*i+3)*system_size]).sum() * np.pi)
        # for i in range(number_of_parameters//3):
        #     combine_grads.append(np.array(grads[4*i*system_size:(4*i+1)*system_size]).sum() * np.pi)
        #     combine_grads.append(np.array(grads[(4*i+1)*system_size:(4*i+2)*system_size]).sum() * np.pi)
        #     combine_grads.append(np.array(grads[(4*i+2)*system_size:(4*i+4)*system_size]).sum() * np.pi)

        return np.array(combine_grads)


    history = []
    def get_f_g(thetas):
        circ = tfi_setup.gen_2d_TFI_ansatz_circuit(thetas,
                                         system_size_x,
                                         system_size_y,
                                         )

        exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val,
                                                    max_num_str=max_num_str,
                                                    basis=basis, backend_name=backend_name)
        raw_grads, backward_final_spo = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val,
                                                                        max_num_str=max_num_str,
                                                                        basis=basis, backend_name=backend_name)
        grads = combine_grads(raw_grads, number_of_parameters, system_size)

        # Add weight regularization
        lambda_reg = 0.00
        cost = exp_val + lambda_reg * np.sum(thetas**2)
        grad_reg = 2 * lambda_reg * thetas
        print(f"cost: {cost}, <E>: {exp_val}, Reg: {0.03 * np.sum(thetas**2)}")
        print(f"||theta||: {np.linalg.norm(thetas)}, ||grad||: {np.linalg.norm(grads)}")
        history.append(cost)
        return exp_val, grads


    if method == 'adam':
        import optax
        optimizer = optax.adam(learning_rate=1e-2)
        opt_state = optimizer.init(random_thetas)
        thetas = random_thetas.copy()
        for _ in range(500):
            print("===="*30)
            print(f"Thetas[{_}]: ", thetas)
            print("===="*30)
            f, g = get_f_g(thetas)
            updates, opt_state = optimizer.update(g, opt_state)
            thetas = optax.apply_updates(thetas, updates)

        exit()

    if method == 'basinhopping':
        import scipy.optimize
        minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
        from scipy.optimize import basinhopping
        ret = basinhopping(get_f_g,
                           random_thetas,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=niter,
                           )
        print(ret)
        np.savetxt(f'2D_bh_{system_size}_np_{number_of_parameters}_g_{g}_params.txt', ret.x,)
        np.savetxt(f'2D_bh_{system_size}_np_{number_of_parameters}_g_{g}_history.txt', history)

        import pickle
        pickle.dump(ret, open(f'2D_bh_L_{system_size}_np_{number_of_parameters}_g_{g}_result.pkl', 'wb'))
        exit()
    else:
        import scipy.optimize
        result = scipy.optimize.minimize(get_f_g,
                                         random_thetas,
                                         method='L-BFGS-B',
                                         jac=True,
                                         options={'disp': True, 'gtol': 1e-5, 'maxiter': 100}
                                         )
        print(result)
