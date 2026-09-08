#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:06:32 2024

@author: vvasiliau
"""

from utilities import load_results
import torch
import numpy as np
import networkx as nx
from dynamics import Dynamics
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from torchdiffeq import odeint
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
size = 100
scale = 100
name_dynamics = "MAK"
from mycolorpy import colorlist as mcp
colors = mcp.gen_color(cmap="rainbow",n=5)

generate_results = False
warnings.filterwarnings('ignore')
relative = True
std_reg = 1
folder ="results/er_experiment_MAK_0_size_100_std_reg_1.0_self_int_True_nbr_int_True_self_hidden_1_nbr_hidden_1_single_gnnlayer_True"


fig, ax = plt.subplots(1,1, figsize= (8,8))

# for index_dynamics in range(len(list_of_dynamics)):
print(name_dynamics)
print(folder)
A1, training_params1, dynamics_config1, dyn1, func1, loss_list1, x_train1, y_train1, x_test1, y_test1 = load_results(folder)
m1 = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1]))

# g = nx.erdos_renyi_graph(100, 1)
# A = nx.adjacency_matrix(g).todense()
# A = torch.FloatTensor(A)
# A1 = [A]
# dyn1.A =  A1[0]
T = 1
dt = 0.01
N_t = int(T/dt)  + 1
t = torch.linspace(0,T,N_t)
x0 = m1.sample([A1[0].shape[0]])
y = odeint( dyn1, x0, t, method="dopri5").squeeze().t().T
y_test_pred = odeint(func1, x0[:, None], t[:]).detach().squeeze()
y_test_pred = odeint(lambda y, t: func1(y, t, A1[0]), x0[:,None], t, method="dopri5"  ).squeeze().detach()

print(torch.mean(abs(func1(0,x0[:,None]).squeeze() - dyn1(0,x0).squeeze())))
loss = (y - y_test_pred)

y = y[:,::10]
y_test_pred = y_test_pred[:,::10]

colors = pl.cm.rainbow(np.linspace(0,1,y.shape[1]))

for i in range(y.shape[1]):
    plt.plot(t, y[:,i], alpha = 0.5, linewidth = 3, color = colors[i])
    plt.plot(t, y_test_pred[:,i], "--", alpha = 1, linewidth = 3, color = colors[i], dashes=(5, 5))
# plt.title(name_dynamics)

ax.set_xlabel('$t$')
ax.set_ylabel('$\\mathbf{x}(t)$')

if name_dynamics == "Diffusion":
    fig.suptitle("Heat")
else:
    fig.suptitle(name_dynamics)
# Remove top and right borders
# ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Make ticks face outward
ax.tick_params(direction='out', length=10, width=1)

# Offset the spines to start at the first tick
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))


# Create an inset axis
if name_dynamics == "PD":
    axins = inset_axes(ax, width="30%", height="30%", loc='upper left',
                        bbox_to_anchor=(0.15, 0.1, 1, 1), bbox_transform=ax.transAxes)
    ax.set_ylim(0,2)
else:
    axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
    axins.set_ylim(0, 0.1)
    ax.set_ylim(0,1)

axins.plot(t, torch.sum(torch.abs(y - y_test_pred),1), linewidth = 2, color= "k")
axins.set_ylabel('$\\Vert \\mathbf{x}(t) - \\hat{\\mathbf{x}}(t) \\Vert_1$')
axins.set_xlabel("$t$")
axins.spines['left'].set_position(('outward', 10))
axins.spines['bottom'].set_position(('outward', 10))
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)

plt.tight_layout()