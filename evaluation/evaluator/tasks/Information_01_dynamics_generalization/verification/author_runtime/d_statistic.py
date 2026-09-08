import torch
import numpy as np


def compute_d_statistics(list_of_experiments, x_test , M , number_of_draws = 100 , adj = None):
    pred_list = []
    nsamples = len(x_test)
    for i in range(number_of_draws):
        xi = x_test[int(torch.randint(0,nsamples,[1])[0])]
        pred = compute_d_statistics_one_sample(list_of_experiments, xi, M  ,adj =adj)
        pred_list.append(float(pred))
    return np.array(pred_list)

def compute_d_statistics_one_sample(list_of_experiments, xi , M, adj = None , node = None):
    index = torch.randperm(len(list_of_experiments))[:M]
    pred = []
    for m in index:
        experiment = list_of_experiments[int(m)]
        pred.append(experiment(0, xi[:,None], adj = adj))
    pred = torch.stack(pred).squeeze()
    pred = pred.var(0)
    if node == None:
        return float(pred[torch.randint(0,len(xi),[1])[0]].detach())
    else:
        return float(pred[node].detach())


def get_acc_ratio_sample_vs_null(null_samples, testing_samples, alpha):
    p_vals = []
    for niter in range(len(testing_samples)):
        dx = testing_samples[niter]
        p_val = compute_pval(dx, np.array(null_samples))
        p_vals.append(p_val)
    accepted = acceptance_ratio(p_vals, alpha)
    return accepted

def compute_pval( x_dval , d_stat_values):
    return np.sum( d_stat_values >= np.float(x_dval))  / len(d_stat_values)

def acceptance_ratio(p_vals, alpha):
    return sum(np.array( p_vals ) >= alpha )/len(p_vals)

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
