

from utilities import  load_results
import warnings
import torch
import networkx as nx
import numpy as np
import pickle
from d_statistic import compute_d_statistics, compute_d_statistics_one_sample, compute_pval, acceptance_ratio
from dynamics import Dynamics

def load_neural_nets(number_networks, name_dynamics , folder_tail )  :
    nn_list = []
    for i in range(number_networks):
        folder = f"results/er_experiment_{name_dynamics}_{i}_{folder_tail}"
        A1, training_params1, dynamics_config1, dyn1, func1, loss_list1, x_train1, y_train1, x_test1, y_test1 = load_results(folder)
        nn_list.append(func1)
    return  nn_list, A1, training_params1.train_distr, dynamics_config1


if __name__ == "__main__":

    name_dynamics = "MAK"
    number_networks = 20
    N= 100

    # load models
    folder_tail = f"size_{N}_std_reg_1.0_self_int_True_nbr_int_True_self_hidden_1_nbr_hidden_1_single_gnnlayer_True"
    neural_networks , adjacency_matrices , train_dist, dyn_config = load_neural_nets(number_networks, name_dynamics, folder_tail)

    #load data on which they were trained
    with open(f"results/multimodel_training_validation_data_{name_dynamics}_er_{N}.pkl", "rb") as f:
        x_train, y_train, x_test, y_test = pickle.load(f)

    # generate d-statistics from training data
    d_stat_full_sample = compute_d_statistics(list_of_experiments = neural_networks,
                                               x_test = x_train , M = 5,
                                               number_of_draws = 1000, adj = adjacency_matrices[0])

    # test on a new datapoint
    m2 = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1]))

    A_test =  np.load("er_n_100_p_06.npy")
    A_test = torch.FloatTensor(A_test)
    N_test = A_test.shape[0]
    delta = 0.1
    p = 0.6
    dyn = Dynamics(A_test, model=dyn_config.model_name, B=dyn_config.B, R=dyn_config.R,
                        H=dyn_config.H, F=dyn_config.F, a=dyn_config.a, b=dyn_config.b)


    xi = m2.sample([N_test]) + delta
    model_id = np.random.randint(number_networks)
    y_pred = neural_networks[model_id](0,xi[:,None],A_test).squeeze()
    y_true = dyn(0,xi).squeeze()


    scaling =  100/ ((p*N_test)**2) # k_train ^2/k_test^2

    di = compute_d_statistics_one_sample(neural_networks, xi, M = 5, adj = A_test)
    p_val_scaled = compute_pval(di*scaling, d_stat_full_sample)
    p_val = compute_pval(di, d_stat_full_sample)

    print("Statistic: ", di, " p-val scaled: ", p_val_scaled, " p-val non-scaled", p_val)
