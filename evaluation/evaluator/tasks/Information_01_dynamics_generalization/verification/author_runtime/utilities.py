#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:47:39 2023

@author: vvasiliau
"""


import torch
import numpy as np
from dynamics import Dynamics
# import torchdiffeq
from torch.distributions import Beta
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})

# plt.rc('text', usetex=True)  # Enable LaTeX rendering
plt.rc('font', family='serif')  # Optional: set the default font to serif


import os
# from NeuralPsi import ODEFunc
from NeuralPsiBlock import ODEFunc
# from torchdiffeq import odeint
import pickle
import io
import matplotlib.pyplot as plt
from torch import nn
import networkx as nx
import os
import sys
from torchdiffeq import odeint
import random

def set_seeds():
    """Sets seeds for reproducibility."""
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.data.fill_(.1)

        m.bias.data.fill_(.1)
        print(m.weight, m.bias)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def analytic_solution(x0, A, B, t):

    L = np.zeros_like(A)
    np.fill_diagonal(L, A.sum(0))
    L = L -A
    eig = np.linalg.eig(L)
    eigvals = eig[0]
    eigvec = eig[1]
    a0 =np.array(x0.T * eigvec)
    a_t = a0.T * np.exp(-B * eigvals[:,None] * t[None,:])

    x_t =np.array( eigvec * a_t)
    return torch.tensor( x_t .T)


def get_range(setting, labels):
    max_label = max([float(labels[i].max().detach().numpy()) for i in range(len(labels))])
    min_label = min([float(labels[i].min().detach().numpy()) for i in range(len(labels))])
    return min_label, max_label




def load_results(folder):

    # graph adjacency
    adjacencies = [ torch.load(f'{folder}/{f}') for f in os.listdir(f'{folder}') if "adjacency_matrix" in f]
    A = adjacencies[0]

    # training config
    with open(f'{folder}/training_config.pkl', 'rb') as f:
        training_params = pickle.load(f)

    #dynamics config
    with open(f'{folder}/dynamics_config.pkl', 'rb') as f:
        dynamics_config = pickle.load(f)

    with open(f"{folder}/loss.pkl", "rb") as f:
        loss_list = pickle.load(f)

    dyn = Dynamics(A, model=dynamics_config.model_name, B=dynamics_config.B, R=dynamics_config.R,
                        H=dynamics_config.H, F=dynamics_config.F, a=dynamics_config.a, b=dynamics_config.b)

    func = ODEFunc(A = A, d= training_params.d,
                   h = training_params.h,
                   h2 = training_params.h2, bias = training_params.bias , self_interaction=training_params.self_interaction,
                   nbr_interaction=training_params.nbr_interaction, hidden_self=training_params.self_hidden_layers,
                   hidden_nbr=training_params.nbr_hidden_layers, single_layer=training_params.single_layer
                   )
    func.load_state_dict(torch.load(folder +"/neural_network.pth"))

    with open(f"{folder}/data.pkl", "rb") as f:
        x_train, y_train, x_test, y_test = pickle.load(f)

    return adjacencies, training_params, dynamics_config, dyn, func, loss_list, x_train, y_train, x_test, y_test




def compute_d_statistics(list_of_experiments, x_test , M ,  direct_fun = False , A = None, number_of_draws = 100, n_id = None): #number_iterations = 1,
    pred_list = []
    nsamples = len(x_test)
    nnodes = x_test[0].shape[0]
    for i in range(number_of_draws):
        sample_idx = torch.randint(0,nsamples,[1])[0]
        if n_id == None:
            node_idx = torch.randint(0,nnodes,[1])[0]
        else:
            node_idx = n_id
            # print(n_id)
        xi = x_test[sample_idx]
        pred = compute_d_statistics_one_sample(list_of_experiments, xi, M , direct_fun, A )
        pred = torch.tensor(np.array(pred)).mean(0).detach().numpy()[node_idx]
        pred_list.append(float(pred))
    return pred_list


def compute_d_statistics_one_sample(list_of_experiments, xi , M ,  direct_fun = False , A = None):
    pred_list = []
    # for i in range(number_iterations):
    index = torch.randint(0,len(list_of_experiments),(1,M))
    pred = []
    for m in index[0]:
        experiment = list_of_experiments[m]
        if direct_fun :
            pred.append(experiment(None, xi[:,None], A))
        else:
            pred.append(experiment.func(None, xi[:,None],A))
    # perm = torch.randperm(xi.size(0))[:10]
    pred = torch.stack(pred).squeeze()#[:,perm]
    pred = (pred.var(0).detach()).numpy()
    pred_list.append(pred)
    return pred_list

def get_acc_ratio_sample_vs_null(null_samples, testing_samples, alpha):
    p_vals = []
    for niter in range(len(testing_samples)):
        dx = testing_samples[niter]
        p_val = compute_pval(dx, np.array(null_samples))
        p_vals.append(p_val)
    accepted = acceptance_ratio(p_vals, alpha)
    return accepted

def compute_pval( x_dval , d_stat_values):
    return np.sum( d_stat_values >= x_dval ) / len(d_stat_values)


def acceptance_ratio(p_vals, alpha):
    return sum(np.array( p_vals ) >= alpha )/len(p_vals)

# def compute_critical_val( d_stat_values , alpha):
#     x = 0.0
#     d_stat_values =  np.array(d_stat_values)
#     delta = 1/(len(d_stat_values))
#     while True:
#         pval = compute_pval(x, d_stat_values)
#         if pval <= alpha:
#             return x
#         x += delta


def compute_critical_val( d_stat_values , alpha):
    x, y = ecdf(d_stat_values)
    y = 1-y
    y[y < alpha] = 10
    return x[np.argwhere(y==10)[0]][0]

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y
