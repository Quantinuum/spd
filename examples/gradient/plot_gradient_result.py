import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns


if __name__ == '__main__':

    # Load
    with open('tfi_gradient_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)


    exact_exp_val = data_dict['exact']['exp_val']
    exact_grads = data_dict['exact']['grads']


    # plot results
    # Merge fig1, fig2 into twin axes plot

    # fig1 error in expectation value vs truncation value
    trunc_vals = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    exp_val_errors = []
    for trunc_val in trunc_vals:
        exp_val = data_dict['trunc_' + str(trunc_val)]['exp_val']
        exp_val_errors.append(abs(exp_val - exact_exp_val) / abs(exact_exp_val))

    print("expectation value errors:", exp_val_errors)
    plt.figure()
    ax = plt.gca()
    twin_ax = ax.twinx()
    plt.sca(twin_ax)
    plt.loglog(trunc_vals, exp_val_errors, marker='s', mfc='none', label='Expectation Value Error ->', color='k')
    plt.xlabel('Truncation Value')
    plt.ylabel('Relative Error in Expectation Value')
    plt.legend(loc='upper right')


    # fig2 error in gradient vs truncation value for SPD gradients
    plt.sca(ax)

    spd_grad_errors = []
    fd_grad_errors = {eps: [] for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
    for trunc_val in trunc_vals:
        spd_grads = data_dict['trunc_' + str(trunc_val)]['spd_grads']
        spd_grad_errors.append(np.linalg.norm(spd_grads - exact_grads) / np.linalg.norm(exact_grads))
        for eps in fd_grad_errors.keys():
            fd_grads = data_dict['trunc_' + str(trunc_val)]['fd_grads_eps_' + str(eps)]
            fd_grad_errors[eps].append(np.linalg.norm(fd_grads - exact_grads) / np.linalg.norm(exact_grads))


    # using gradient color for different finite difference epsilons
    colors = sns.color_palette("viridis", n_colors=5)
    # plt.figure()
    plt.loglog(trunc_vals, spd_grad_errors, marker='o', color='r', label='SPD Gradients')
    for i, (eps, errors) in enumerate(fd_grad_errors.items()):
        plt.loglog(trunc_vals, errors, marker='o', ls='--', color=colors[i], label=f'FD Gradients (eps={eps})')

    plt.xlabel('Truncation Value')
    plt.ylabel('Relative Error in Gradients')
    plt.legend()
    plt.grid(True)

    # align y-axis limits
    # ax_ylim = twin_ax.get_ylim()
    # ax.set_ylim(ax_ylim)

    plt.title('L=36, depth=18, 1dTFI, g=1.1')
    plt.show()
