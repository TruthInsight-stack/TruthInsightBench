from dataclasses import dataclass
import numpy as np
import torch
import networkx as nx
import pickle
import os
# from NeuralPsi import ODEFunc
from NeuralPsiBlock import ODEFunc
from dynamics import Dynamics


@dataclass
class TrainingParameters:
    setting: int
    train_distr: torch.distributions
    test_distr: torch.distributions
    method: str = 'dopri5'  # numerical int method
    testitr: int = 100  # frequency for printing while training
    epochs: int = 1000
    lr: float = 1e-2  # learning rate for training
    weight_decay: float = 1e-3  # weight decay for training
    h: int = 6 #self interaction embedding dim
    h2: int = 8 #nbr interaction embedding dim
    h3: int = 9
    h4: int = 10
    d: int = 1
    train_samples: int = 10
    test_samples: int = 10
    nsample: int = 10
    self_interaction: bool = True
    nbr_interaction: bool = True
    self_hidden_layers: int = 1
    nbr_hidden_layers: int = 1
    single_layer: bool = True
    # Q_factorized: bool = True
    bias: bool = True

@dataclass
class DynamicsParameters:
    model_name: str
    T: float = 2
    dt: float = 0.01
    device: str = 'cpu'
    R: float = np.nan
    F: float = np.nan
    a: float = np.nan
    b: float = np.nan
    H: float = np.nan
    B: float = np.nan


class Experiment():
    def __init__(self, device, training_parameters: TrainingParameters, dynamics_parameters: DynamicsParameters, graph: nx.Graph = None):
        self.device = device
        self.size = graph.number_of_nodes()
        self.training_parameters = training_parameters
        self.dynamics_parameters = dynamics_parameters

        self.t = torch.linspace(0., dynamics_parameters.T, int(dynamics_parameters.T / dynamics_parameters.dt)).to(device)
        self.A = torch.FloatTensor(nx.to_numpy_array(graph)).to(device)

        self.func = ODEFunc(A = self.A, d = training_parameters.d,
                       h=training_parameters.h, h2=training_parameters.h2,
                       bias = training_parameters.bias,
                       self_interaction = training_parameters.self_interaction,
                       nbr_interaction = training_parameters.nbr_interaction,
                       hidden_self = training_parameters.self_hidden_layers,
                       hidden_nbr =training_parameters.nbr_hidden_layers,
                       single_layer = training_parameters.single_layer
                       # Q_factorized= training_parameters.Q_factorized
                       ).to(device)


        self.optimizer = torch.optim.Adam(self.func.parameters(), lr=training_parameters.lr, weight_decay=training_parameters.weight_decay)
        # self.optimizer = torch.optim.SGD(self.func.parameters(), lr =training_parameters.lr , momentum = 0.9 )
        # self.optimizer = torch.optim.Adagrad(self.func.parameters(), lr=training_parameters.lr, weight_decay=training_parameters.weight_decay)

        self.dyn = Dynamics(self.A, model=dynamics_parameters.model_name, B=dynamics_parameters.B, R=dynamics_parameters.R,
                            H=dynamics_parameters.H, F=dynamics_parameters.F, a=dynamics_parameters.a, b=dynamics_parameters.b)
        self.train_distr = training_parameters.train_distr
        self.test_distr = training_parameters.test_distr


    def generate_train_test_data(self , xy = None):

        x_list = []
        y_list = []
        if self.training_parameters.setting == 1:
            # Supervised learning  regression task where time-dependence
            # is broken and samples as iid
            train_samples = self.training_parameters.train_samples
            test_samples = self.training_parameters.test_samples
            # xy = self.get_uniform_mesh()


            for i in range(train_samples):
                if xy != None:
                    x0 = xy[:,i][:,None]
                else:
                    x0 = self.train_distr.sample([self.size]).to(self.device)
                y_list.append(self.dyn(0, x0))
                x_list.append(x0)
            # print(xy.shape, train_samples)
            if xy != None:
                xy = xy[:,train_samples:]
            # print(xy.shape)
            for i in range(test_samples-1):
                if xy != None:
                    x0 = xy[:,i][:,None]
                    # print(x0.shape)
                else:
                    x0 = self.test_distr.sample([self.size]).to(self.device)
                y_list.append(self.dyn(0, x0))
                x_list.append(x0)

            nsamples = 1

        y_train = y_list[:train_samples * nsamples]
        y_test = y_list[train_samples * nsamples:]
        x_train = x_list[:train_samples * nsamples]
        x_test = x_list[train_samples * nsamples:]

        return x_train, y_train, x_test, y_test

    def generate_arbitrary_data(self, A_new, nsamples, dist):
        x_list,y_list = [],[]
        alt_dyn = Dynamics(A_new, model=self.dynamics_parameters.model_name, B=self.dynamics_parameters.B, R=self.dynamics_parameters.R,
                            H=self.dynamics_parameters.H, F=self.dynamics_parameters.F, a=self.dynamics_parameters.a, b=self.dynamics_parameters.b)
        size = A_new.shape[0]
        for i in range(nsamples):
            x0 = dist.sample([size])
            y_list.append(alt_dyn(0, x0))
            x_list.append(x0)
        return x_list, y_list

    def generate_arbitrary_point(self, A_new, x0):
        alt_dyn = Dynamics(A_new, model=self.dynamics_parameters.model_name, B=self.dynamics_parameters.B, R=self.dynamics_parameters.R,
                            H=self.dynamics_parameters.H, F=self.dynamics_parameters.F, a=self.dynamics_parameters.a, b=self.dynamics_parameters.b)
        return alt_dyn(0, x0)


    def get_uniform_mesh(self, NY =10, ymax = 1):

        # NY = 10;
        ymin = 0;
        dy = (ymax - ymin) / (NY - 1.)
        NX = NY
        xmin = ymin
        xmax = ymax
        dx = (xmax - xmin) / (NX - 1.)
        y = torch.Tensor(np.array([ymin + float(i) * dy for i in range(NY)]))
        x = torch.Tensor(np.array([xmin + float(i) * dx for i in range(NX)]))
        x, y = torch.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        xy = torch.stack((x, y))
        return xy , x , y

    def get_custom_mesh(self, data):
        # data is a list of tensors of shape 2X1

        xy = torch.stack(data)[:,:,0].T
        return xy, xy[0], xy[1]



    def save(self, folder_name, loss_list, x_train, y_train, x_test, y_test, adjacency_matrices = None):
        folder =f"{folder_name}"
        print("saving files to: " ,folder)
        os.makedirs(folder, exist_ok=True)

        # NN
        checkpoint = self.func.state_dict()
        torch.save(checkpoint, f'{folder}/neural_network.pth')

        # graph
        if adjacency_matrices == None:
            torch.save(self.A, f'{folder}/adjacency_matrix.pt')
        else:
            for i in range(len(adjacency_matrices)):
                a = adjacency_matrices[i]
                torch.save(a, f'{folder}/adjacency_matrix_{i}.pt')

        # Loss
        with open(f"{folder}/loss.pkl","wb+") as f:
            pickle.dump(loss_list, f)

        # training config
        with open(f"{folder}/training_config.pkl","wb+") as f:
            pickle.dump(self.training_parameters, f)

        #dynamics config
        with open(f"{folder}/dynamics_config.pkl","wb+") as f:
            pickle.dump(self.dynamics_parameters, f)

        #data
        with open(f"{folder}/data.pkl","wb+") as f:
            pickle.dump([x_train, y_train, x_test, y_test], f)
