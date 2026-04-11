import numpy as np
import pickle
import matplotlib.pyplot as plt

# benchmark_2d_obc_xx_z_stepwise.py					tomislav_spd_benchmark_data_dt_0.04_total_t_0.92_threshold_log_18.pkl
# benchmark_data_dt_0.04_total_t_0.12_threshold_log_18.pkl

# Keys in the data: dict_keys(['num_paulis', 'avg_num_paulis', 'avg_speeds', 'times', 'norms', 'all_results'])


def check_result():
    # Load the data from the pickle file
    data_tomislav = pickle.load(open('benchmark_data_dt_0.04_total_t_0.92_threshold_log_18_tomislav_core4.pkl', 'rb'))
    data_jax = pickle.load(open('jax_benchmark_data_dt_0.04_total_t_0.92_threshold_log_18.pkl', 'rb'))

    print("From Tomislav's file:")
    print("Results from Tomislav's file:")
    print(np.array(data_tomislav['all_results']))
    print("Norm of the results from Tomislav's file:")
    print(np.array(data_tomislav['norms']))
    print("Speeds from Tomislav's file:")
    print(np.array(data_tomislav['avg_speeds']))

    print("=================================")
    print("From JAX file:")
    print("Results from JAX file:")
    print(np.array(data_jax['all_results']))
    print("Norm of the results from JAX file:")
    print(np.array(data_jax['norms']))
    print("Speeds from JAX file:")
    print(np.array(data_jax['avg_speeds']))

def plot_result():
    # data_tomislav = pickle.load(open('tomislav_spd_benchmark_data_dt_0.04_total_t_0.92_threshold_log_18.pkl', 'rb'))
    data_tomislav = pickle.load(open('benchmark_data_dt_0.04_total_t_0.92_threshold_log_18_tomislav_core4.pkl', 'rb'))
    data_jax = pickle.load(open('jax_benchmark_data_dt_0.04_total_t_0.92_threshold_log_18.pkl', 'rb'))

    # plot result
    plt.plot(data_tomislav['all_results'], 'o-', label='Tomislav')
    plt.plot(data_jax['all_results'], 's--', mfc='none', label='JAX')
    plt.legend()
    plt.show()

def plot_speeds():
    data_tomislav_core1 = pickle.load(open('benchmark_data_dt_0.04_total_t_0.92_threshold_log_18_tomislav_core1.pkl', 'rb'))
    data_tomislav_core4 = pickle.load(open('benchmark_data_dt_0.04_total_t_0.92_threshold_log_18_tomislav_core4.pkl', 'rb'))
    data_jax = pickle.load(open('jax_benchmark_data_dt_0.04_total_t_0.92_threshold_log_18.pkl', 'rb'))

    # plot speeds against the number of Paulis
    plt.plot(data_tomislav_core1['avg_num_paulis'], data_tomislav_core1['avg_speeds'], 'o-', label='Tomislav Core 1')
    plt.plot(data_tomislav_core4['avg_num_paulis'], data_tomislav_core4['avg_speeds'], 'x-', label='Tomislav Core 4')
    plt.plot(data_jax['avg_num_paulis'], data_jax['avg_speeds'], 's--', mfc='none', label='JAX')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    check_result()
    plot_result()
    plot_speeds()
