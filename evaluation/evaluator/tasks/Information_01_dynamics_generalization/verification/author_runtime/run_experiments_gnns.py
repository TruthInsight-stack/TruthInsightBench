
from dynamics import Dynamics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import warnings
from NeuralPsiBlock import ODEFunc
import random
from tqdm import tqdm
from gnns_utilities import save_results, ModelParameters, GCN, GCN_single
from torch_geometric.utils import from_networkx

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    number_train_samples = 200
    size = 2
    if size == 2:
        G = nx.DiGraph()
        G.add_edge(0,1)
        A = torch.FloatTensor(np.array(nx.adjacency_matrix(G).todense())).to(device)
    else:
        G = nx.erdos_renyi_graph(size, 0.5)
        A = torch.FloatTensor(np.array(nx.adjacency_matrix(G).todense())).to(device)

    # gcn_models = ["NeuralPsi", "SAGEConv","GraphConv","ResGatedGraphConv","GATConv","ChebConv"]#,
    gcn_models = ["NeuralPsi"]#,   "GraphConv_single", "ResGatedGraphConv_single","GATConv_single","ChebConv_single" ]
    dynamics_models = ["MAK"]#, "MM","PD","SIS","Diffusion"]
    # setup training object

    train_distr= torch.distributions.Beta(torch.FloatTensor([5]),torch.FloatTensor([2]))
    train_samples = [train_distr.sample([size]).to(device) for i in range(number_train_samples)]
    for model_name in gcn_models:
        for dynamics_name in dynamics_models:
            print(model_name,dynamics_name)

            params = ModelParameters(
                dynamics_name = dynamics_name,
                model_name =model_name,
                train_distr= train_distr,
                epochs = 1000,
                train_samples = number_train_samples,
                size = size,
                lr = 0.005,  # learning rate for training
                weight_decay = 0, # weight decay for training
                h = 30, #self interaction embedding dim
                h2 = 30 ,#nbr interaction embedding dim
                h3 = 0,
                d = 1,
                Q_factorized = True,
                bias = True)


            # dynamics
            if params.dynamics_name == "MAK":
                params.B = 0.1
                params.R = 1
                params.F = 0.5
                params.b =1
                dyn = Dynamics(A=A, B= params.B, R= params.R, F=params.F, b=params.b,  model = params.dynamics_name)
            if params.dynamics_name == "SIS":
                params.B = 4
                params.R = 5
                dyn = Dynamics(A=A, B= params.B, R =params.R,  model = params.dynamics_name)
            if params.dynamics_name == "Diffusion":
                params.B = 0.5
                dyn = Dynamics(A=A, B= params.B,   model = params.dynamics_name)
            if params.dynamics_name == "PD":
                params.B = 2
                params.R = 0.3
                params.a = 1.5
                params.b = 3
                dyn = Dynamics(A=A, B= params.B, R= params.R, a= params.a, b= params.b,   model = params.dynamics_name)
            if params.dynamics_name == "MM":
                params.H = 3
                params.B = 4
                params.R = 0.5
                dyn = Dynamics(A=A, B= params.B, R = params.R, H=params.H,   model = params.dynamics_name)

            y_train = []
            x_train = []
            # generate data for neural psi
            if params.model_name == "NeuralPsi":
                func = ODEFunc(A = A, d = 1, h=params.h, h2=params.h2,h3 = params.h3,
                                    bias = True,  Q_factorized= True).to(device)
                for i in range(params.train_samples):
                    params.train_distr.sample([params.size]).to(device)
                    # x0 = params.train_distr.sample([params.size]).to(device)
                    x0 = train_samples[i].to(device)
                    y_train.append(dyn(0, x0))
                    x_train.append(x0)

            if ("Conv" in params.model_name ) and ("_single" not in params.model_name ):  # params.model_name in ["SAGEConv", "ChebConv", "GCNConv"]:

                func = GCN(in_channels = 1, hidden_channels=params.h, out_channels= 1,
                           model_name = params.model_name ).to(device)
                training_data = []
                for i in range(params.train_samples):
                    params.train_distr.sample([params.size]).to(device)
                    # x0 = params.train_distr.sample([params.size]).to(device) # features
                    x0 = train_samples[i].to(device)
                    y0 = dyn(0, x0) # labels
                    sample_data = from_networkx(G).to(device)
                    sample_data.x = x0
                    sample_data.y = y0
                    training_data.append(sample_data)
                    y_train.append(dyn(0, x0))
                    x_train.append(x0)




            if ("Conv" in params.model_name ) and ("_single"  in params.model_name ):  #if params.model_name in ["SAGEConv_single", "ChebConv_single", "GCNConv_single"]:
                func = GCN_single(in_channels = 1, hidden_channels=params.h, out_channels= 1,
                           model_name = params.model_name ).to(device)
                training_data = []
                for i in range(params.train_samples):
                    params.train_distr.sample([params.size]).to(device)
                    # x0 = params.train_distr.sample([params.size]).to(device) # features
                    x0 = train_samples[i].to(device)
                    y0 = dyn(0, x0) # labels
                    sample_data = from_networkx(G).to(device)
                    sample_data.x = x0
                    sample_data.y = y0
                    training_data.append(sample_data)
                    y_train.append(dyn(0, x0))
                    x_train.append(x0)

            warnings.filterwarnings('ignore')


            ## training
            optimizer = torch.optim.Adam(func.parameters(), lr=params.lr, weight_decay=params.weight_decay)
            epochs = params.epochs
            reg_lambda = 0.1

            for itr in tqdm(range(epochs)):
                optimizer.zero_grad()
                if params.model_name == "NeuralPsi":
                    pred_y = [func(0, x_train[i][:,None]) for i in range(params.train_samples)]
                    true_y = y_train
                    v1 = torch.squeeze(torch.cat(pred_y,1),2)
                    v2 = torch.cat(y_train,1)

                    # weigh wrt to variance in the loss
                    variance_reg = torch.abs(v1-v2).var(0).mean()
                    loss = torch.abs(v1-v2).mean() + reg_lambda*variance_reg
                else:
                    pred_y = [func(d) for d in training_data]
                    true_y = [d.y for d in training_data]
                    v1 = torch.cat(pred_y,1)
                    v2 = torch.cat(true_y,1)

                    variance_reg = torch.abs(v1 - v2).var(0).mean()
                    loss = torch.abs(v1 - v2).mean() + reg_lambda * variance_reg


                loss.backward()
                optimizer.step()
                if itr % 100 == 0:
                    print(params.model_name,itr, loss)

            folder_name = f"models/neural_network_{params.model_name}_dynamics_{params.dynamics_name}_graph_size_{params.size}_directed"
            save_results(model = func, folder_name = folder_name, adj = A, training_parameters = params,x_train = x_train, y_train = y_train)
