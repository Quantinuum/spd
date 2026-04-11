import pickle
import random
import pytket
from pytket import Circuit
import numpy as np
# np.random.seed(43)
np.random.seed(0)
import spd
import tfi_setup
import sys

if __name__ == "__main__":

    backend_name = 'jax'
    # method = 'basinhopping'
    # method = 'reuse'
    method = 'adam'

    # num_param_list = [12, 14, 16, 18, 20]
    # for number_of_parameters in num_param_list:
    number_of_parameters = int(sys.argv[1])
    num_layers = int(number_of_parameters // 3)
    system_size = 2 * num_layers + 3
    print(f"========      system size = {system_size}     =======")


    g = float(sys.argv[2])
    niter = int(sys.argv[3])

    basis = '+'
    assert basis in ['0', '+']
    precision = 'double'
    trunc_val = 1e-8
    max_num_str = 1e6
    print(f"\n Truncation Value: {trunc_val} | max num str: {max_num_str} | precision: {precision} | basis: {basis} | backend: {backend_name}")


    base_filename = f'L_{system_size}_np_{number_of_parameters}_g_{g}_symm_breaking'

    random_thetas = (np.random.rand(number_of_parameters) - 0.5) * 0.1
    # good_thetas = np.loadtxt('L_100_np_10_g_1.1_params.txt')
    # for depth in range(num_layers):
    #     random_thetas[3*depth:3*depth+2] += good_thetas[2*depth:2*depth+2]

    # good_thetas = np.loadtxt(f'L_13_np_15_g_{(g-0.1):.1f}_symm_breaking_params.txt') % 2
    # random_thetas += good_thetas
    # print("initialize from previous g")


    # str_g = str(g)
    # str_g = str_g.replace('.', '-')
    # niter_data = 1000
    # # dirname = f'/teamspace/jobs/1d-bh-niter{niter_data}-np{number_of_parameters}-g{str_g}-sb-ansatz/artifacts/'
    # dirname = './'
    # L = int(number_of_parameters//3) * 2 + 3
    # filename = f'L_{L}_np_{number_of_parameters}_g_{g}_symm_breaking_result.pkl'
    # data = pickle.load(open(dirname + filename, 'rb'))
    # print("load good old result from", dirname, filename)
    # print("previous good result = ", data.lowest_optimization_result.fun)
    # good_thetas = data.lowest_optimization_result.x % 2
    # random_thetas += good_thetas
    # print("thetas = ", random_thetas)
    # method = ''


    # run_dim = 1
    circ = tfi_setup.gen_1d_TFI_symm_breaking_ansatz_circuit(random_thetas,
                                            system_size,
                                            )
    ham_dict = tfi_setup.gen_1d_Hamiltonian_dict(system_size, g=g, full=False)


    exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val,
                                                max_num_str=max_num_str, precision=precision,
                                                basis=basis, backend_name=backend_name)
    print("\n Expectation Value:", exp_val)

    grads, backward_final_spo = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val,
                                                                max_num_str=max_num_str, precision=precision,
                                                                basis=basis, backend_name=backend_name)
    print("\n SPD Computed Gradients:", grads)


    def combine_grads(grads, number_of_parameters, system_size):
        combine_grads = []
        for i in range(number_of_parameters):
            combine_grads.append(np.array(grads[i*system_size:(i+1)*system_size]).sum() * np.pi)

        return np.array(combine_grads)

    history = []
    def get_f_g(thetas):
        # lambda_ose = 3e-6
        # Consider an exponential decay lambda_ose that starts at 1e-1 and decays to 1e-6 over 500 iterations
        # lambda_ose = lambda_setup * np.exp(-len(history) * -np.log(1e-1)/100)
        # lambda_ose = lambda_setup
        # lambda_ose = 0
        lambda_ose = 1e-1 if len(history) < 100 else 0
        print("lambda_ose: ", lambda_ose)

        circ = tfi_setup.gen_1d_TFI_symm_breaking_ansatz_circuit(thetas,
                                        system_size,
                                        )

        exp_val, final_spo = spd.run_pytket_circuit(circ, ham_dict, trunc_val,
                                                    max_num_str=max_num_str, precision=precision,
                                                    basis=basis, backend_name=backend_name)
        raw_grads, backward_final_spo = spd.run_pytket_circuit_backward(circ, final_spo, trunc_val,
                                                                        max_num_str=max_num_str, precision=precision,
                                                                        basis=basis, lambda_ose=lambda_ose,
                                                                        backend_name=backend_name)
        grads = combine_grads(raw_grads, number_of_parameters, system_size)

        # Add weight regularization
        lambda_reg = 0.00
        cost = exp_val + lambda_ose * final_spo.get_OSE() + lambda_reg * np.sum(thetas**2)
        # grad_reg = 2 * lambda_reg * thetas
        print("step = ", len(history), "num_param", number_of_parameters)
        print(f"cost: {cost}, <E>: {exp_val}, Reg: {0.03 * np.sum(thetas**2)}")
        print(f"||theta||: {np.linalg.norm(thetas)}, ||grad||: {np.linalg.norm(grads)}")
        history.append(cost)
        return exp_val, grads


    if method == 'adam':
        import optax
        optimizer = optax.adam(learning_rate=3e-2)
        opt_state = optimizer.init(random_thetas)
        thetas = random_thetas.copy()
        for _ in range(100):
            print("===="*30)
            print(f"Thetas[{_}]: ", thetas)
            print("===="*30)
            f, g = get_f_g(thetas)
            updates, opt_state = optimizer.update(g, opt_state)
            thetas = optax.apply_updates(thetas, updates)

        random_thetas = thetas



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
        np.savetxt(base_filename + '_params.txt', ret.x,)
        np.savetxt(base_filename + '_history.txt', history)

        import pickle
        pickle.dump(ret, open(base_filename + '_result.pkl', 'wb'))
        exit()



    import scipy.optimize
    result = scipy.optimize.minimize(get_f_g,
                                    random_thetas,
                                    method='L-BFGS-B',
                                    jac=True,
                                    options={'disp': True, 'gtol': 1e-4, 'maxiter': 1000}
                                    )
    print(result)
    print("params: ", result.x)
    # print(history)
    np.savetxt(base_filename + '_params_adam_lbfgsb.txt', result.x,)
    np.savetxt(base_filename + '_history_adam_lbfgsb.txt', history)

    import pickle
    pickle.dump(result, open(base_filename + '_result_lbfgsb.pkl', 'wb'))
    exit()



    # np.savetxt(f'L_{system_size}_np_{number_of_parameters}_g_{g}_params.txt', result.x,)
    # np.savetxt(f'L_{system_size}_np_{number_of_parameters}_g_{g}_history.txt', history)

