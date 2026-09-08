import torch
import torch.nn as nn
import networkx as nx
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

def make_layers(in_channels, hidden_channels, num_hidden_layers, bias):
    layers = [
        nn.Linear(in_channels, hidden_channels, bias=bias),
        nn.Tanh()
    ]
    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(hidden_channels, hidden_channels, bias=bias),
            nn.Tanh()
        ])
    return nn.Sequential(*layers)

class SelfInteraction(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers, bias=True):
        super(SelfInteraction, self).__init__()
        self.g = make_layers(in_channels, hidden_channels, num_hidden_layers, bias)
        self.final = nn.Linear(hidden_channels, out_channels, bias=bias)

    def forward(self,t, x):
        x = self.g(x)
        return self.final(x)

class NeighborInteraction(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers, A, bias=True):
        super(NeighborInteraction, self).__init__()
        self.A = A
        self.nn_fun1 = nn.Sequential(make_layers(in_channels, hidden_channels, num_hidden_layers-1, bias), nn.Linear(hidden_channels, hidden_channels, bias=bias))
        self.nn_fun2 = nn.Sequential(make_layers(in_channels, hidden_channels, num_hidden_layers-1, bias), nn.Linear(hidden_channels, hidden_channels, bias=bias))
        self.agg = nn.Linear(hidden_channels, out_channels, bias=bias)

    def forward(self,t, x, adj=None):
        y1 = self.nn_fun1(x)
        y2 = self.nn_fun2(x)
        y1_tmp = torch.transpose(y1, 0, 2)
        y2_tmp = torch.transpose(y2, 0, 2)
        xx = torch.bmm(torch.transpose(y1_tmp, 1, 2), y2_tmp)
        if adj is None:
            nn_Q_out = self.A * xx
        else:
            nn_Q_out = adj * xx
        nn_Q_flatten_input = torch.transpose(torch.unsqueeze(nn_Q_out.sum(1), 1), 0, 2).float()
        agg_out = self.agg(nn_Q_flatten_input)
        return agg_out


class ODEFunc(nn.Module):
    def __init__(self, A, d, h, h2, bias=True, self_interaction = True, nbr_interaction = True, hidden_self = 1, hidden_nbr =1, single_layer = True):
        super(ODEFunc, self).__init__()
        self.A = A

        self.self_interaction = self_interaction
        self.nbr_interaction = nbr_interaction

        self.fun_self = SelfInteraction(in_channels = d, out_channels = d, hidden_channels = h, num_hidden_layers= hidden_self)


        # nbr layers
        self.single_layer = single_layer
        if single_layer == True:
            self.layer1 = NeighborInteraction(in_channels = d, out_channels = d, hidden_channels= h2, num_hidden_layers=hidden_nbr, bias = bias, A = A)
        else:
            self.layer1 = NeighborInteraction(in_channels = d, out_channels = h2, hidden_channels= h2, num_hidden_layers=hidden_nbr, bias = bias, A = A)
            self.layer2 = NeighborInteraction(in_channels = h2, out_channels = h2, hidden_channels= h2, num_hidden_layers=hidden_nbr, bias = bias, A = A)
            self.layer3 = NeighborInteraction(in_channels = h2, out_channels = d, hidden_channels= h2, num_hidden_layers=hidden_nbr, bias = bias, A = A)


    def forward(self, t, x, adj=None):
        res = 0
        if self.nbr_interaction == True:
            out_nbr = self.layer1(t,x,adj=adj)
            if self.single_layer == False:
                out_nbr = self.layer2(t, out_nbr, adj=adj)
                out_nbr = self.layer3(t, out_nbr, adj=adj)
            res += out_nbr
        if self.self_interaction == True:
            out_self = self.fun_self(t,x)
            res += out_self
        return res



if __name__ == "__main__":

    d, h, h2, h3, h4 = 1, 40, 60, 50, 30
    n = 2
    x = torch.randn([n , 1 , 1], requires_grad= True) # n x 1 x d
    # g = nx.erdos_renyi_graph(n , 0.5)
    g = nx.Graph()
    g.add_edge(0,1)
    # g.add_edge(0,)
    A = torch.tensor(nx.to_numpy_array(g))
    A = torch.diag(A.sum(0)) - A
    func = ODEFunc(A, d, h, h2, h3, h4)
    print(func( 0 , x, A, self_int = False, nbr_int=True) )
