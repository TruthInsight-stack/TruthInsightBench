# from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from experiment_class import DynamicsParameters, TrainingParameters, Experiment
from utilities import set_seeds
import warnings
import torch
import networkx as nx
import numpy as np
from torchdiffeq import odeint
# import operator
import matplotlib.pyplot as plt
import random
from dynamics import Dynamics


def get_batch(y_data, t_data, batch_size=20, batch_time=11):
    # Number of time series (n_iter)
    num_series = y_data.shape[0]

    # Length of each time series
    series_length = y_data.shape[2]

    # Randomly select starting points for the batch
    series_indices = np.random.randint(0,num_series, size=batch_size)
    # Adjust how time indices are generated based on series length
    if series_length == 2:
        time_indices = np.zeros(batch_size, dtype=np.int64)  # always start at 0 when series length is 2
    else:
        time_indices = np.random.choice(np.arange(series_length - batch_time, dtype=np.int64), batch_size, replace=True)

    # Initialize empty lists for batch data
    x_batch = []
    y_batch = []
    t_batch = []

    for i, s_idx in enumerate(series_indices):
        # Get the time index for this batch element
        t_idx = time_indices[i]

        # Retrieve the initial state for this batch element (x0)
        x0 = y_data[s_idx, :, t_idx:t_idx + 1]

        # Retrieve the time points and corresponding y values for the current batch
        t_current = t_data[s_idx, t_idx:t_idx + batch_time]
        y_current = y_data[s_idx, :, t_idx:t_idx + batch_time]

        # Append to batch lists
        x_batch.append(x0)
        y_batch.append(y_current)
        t_batch.append(t_current)

    # Convert to PyTorch tensors and stack
    x_batch = torch.cat(x_batch, dim=1).squeeze()
    y_batch = torch.stack(y_batch, dim=0)
    t_batch = torch.stack(t_batch, dim=0)

    return x_batch, y_batch, t_batch


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    multiple_nn = False
    if multiple_nn:
        M_tot = 20
        bootstrap_fraction = 0.9
    else:
        M_tot = 1
        bootstrap_fraction  = 1

    model_name = "MAK"
    results_root = "results"
    network_name  = "er"
    N=100
    A = np.load("er_n_100_p_01.npy")
    g = nx.from_numpy_array(A)
    A = nx.adj_matrix(g).todense()
    A = torch.FloatTensor(A)

    ############################
    # Definition of regularizers
    ############################
    regularizer_lambda = 0 # regularizer that minimizes variance in the loss across nodes


    if model_name == "Diffusion":
        dynamics_params = DynamicsParameters(model_name = "Diffusion", B=0.5)
    if model_name == "MAK":
        dynamics_params = DynamicsParameters(model_name = "MAK", B=0.1, F = 0.5, R  = 1, b= 3)
    if model_name == "PD":
        dynamics_params = DynamicsParameters(model_name = "PD", B=2, R  = 0.3 , a =1.5 , b = 3 )
    if model_name == "MM":
        dynamics_params = DynamicsParameters(model_name = "MM", B=4, R  = 0.5 , H =3 )
    if model_name == "SIS":
        dynamics_params = DynamicsParameters(model_name = "SIS", B =4, R  = 0.5 )

    self_hidden_layers = 1
    nbr_hidden_layers = 1
    training_params = TrainingParameters(setting=1, train_distr = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1])),
                                         test_distr = torch.distributions.Beta(torch.FloatTensor([1]),torch.FloatTensor([1])),
                                         train_samples=1000,test_samples =0.1, epochs = 2000, lr=0.01, nsample = 10, weight_decay= 0.01,
                                         h=30, h2=30,  bias = True ,self_interaction = True, nbr_interaction = True, self_hidden_layers = self_hidden_layers,
                                         nbr_hidden_layers = nbr_hidden_layers, single_layer = True, method = "dopri5")
    experiment = Experiment(device = device, dynamics_parameters = dynamics_params, training_parameters = training_params,
                                  graph = g)

    ###########
    # Correlated data setting
    ###########

    niter = 100 # number of test time series
    niter_val = 50 # number of validation time series

    T = 0.1 # max T of time series
    dt = 0.01 # integration timestep


    # training batches
    batch_time = 5
    batch_size = 10

    ###########
    # end of correlated data setting
    ###########

    ###########
    # 2 timestep time series setting
    ###########
    #
    # niter = 1000
    # niter_val = 100
    #
    # T = 0.01
    # dt = 0.01
    #
    # # training batches
    # batch_time =2
    # batch_size = 10
    #
    ###########
    # end of 2 timestep time series setting
    ###########



    N_t = int(T/dt)  + 1
    training_params.train_samples =  int(niter* (1 - training_params.test_samples))
    training_params.test_samples =  int(niter* (training_params.test_samples))
    x_train , y_train, t_train = [], [], []
    x_val , y_val, t_val = [], [], []
    t = torch.linspace(0,T,N_t)
    print(t)
    scale_obs = 0.0 # Observational noise std
    irregular_sampling = True # Whether data is sampled at irregular time intervals . If true, we add some truncated gaussian noise to dt

    for i in range(niter+niter_val):
        if scale_obs != 0:
            m = torch.distributions.normal.Normal(loc = 0, scale = scale_obs)
            noise = m.sample([N, t.shape[0]])#* scale
        else:
            noise = torch.zeros([N,t.shape[0]])

        if irregular_sampling:
            mu = 0.2  # Mean of the Gaussian distribution
            sigma = 0.01  # Standard deviation of the Gaussian distribution
            lower_bound = 0.1  # Lower bound for truncation
            upper_bound = 0.3  # Upper bound for truncation
            # Generate Gaussian differences
            differences = torch.normal(mu, sigma, size=(N_t-1,))
            # Truncate differences
            differences = torch.clamp(differences, min=lower_bound, max=upper_bound)
            # Normalize differences to sum up to T
            differences = differences / differences.sum() * T
            # Compute the cumulative sum of differences to get the time steps
            t = torch.cat((torch.tensor([0]), differences.cumsum(dim=0)))

        x0 = torch.rand([N,1])
        y = odeint( experiment.dyn, x0, t, method="dopri5").squeeze().t()
        signal_power = torch.mean(abs(y) ** 2)
        noise_power = torch.mean(abs(noise) ** 2)
        snr_db = 10 * torch.log10(signal_power / (noise_power+1e-9))
        print(snr_db)
        y = y + noise
        if i < niter:
            x_train.append(x0)
            y_train.append(y)
            t_train.append(t)
        else:
            x_val.append(x0)
            y_val.append(y)
            t_val.append(t)

    for i in range(N):
        plt.scatter(t, y.T[:,i],s = 0.1)
    plt.show()

    y_train = torch.stack(y_train) # number time series , nodes , timesteps
    t_train = torch.stack(t_train) # number timeseries, timesteps
    y_val = torch.stack(y_val)
    t_val = torch.stack(t_val)

    #########################
    # Training
    #########################
    scheduler = ReduceLROnPlateau(experiment.optimizer, 'min', patience= 50, cooldown=10)
    loss_list = []

    val_loss_list = []
    best_val_loss = float('inf')
    no_improve_epochs = 0

    patience = 100  # Number of epochs to wait after which training will be stopped if no improvement
    early_stopping = False
    for itr in range(training_params.epochs + 1):
        experiment.optimizer.zero_grad()
        pred_y = []
        x_batch, y_batch, t_batch = get_batch(y_train,t_train,batch_size = batch_size, batch_time = batch_time)
        for i in range(batch_size):
            pred_yi = odeint(experiment.func, x_batch[:,i,None,None], t_batch[i], method=training_params.method).squeeze().T
            pred_y.append(pred_yi)
        pred_y = torch.stack(pred_y)
        loss = torch.sum(torch.abs(pred_y-y_batch))

        loss.backward()
        experiment.optimizer.step()
        loss_list.append(float(loss.detach()))

        # Validate
        if itr > 500 :
            x_batch_val, y_batch_val, t_batch_val = get_batch(y_val, t_val, batch_size=batch_size, batch_time=batch_time)
            pred_y_val = [odeint(experiment.func, x_batch_val[:, i, None, None], t_batch_val[i], method=training_params.method).squeeze().T for i in range(batch_size)]
            pred_y_val = torch.stack(pred_y_val)
            loss_val = torch.sum(torch.abs(pred_y_val - y_batch_val))
            val_loss_list.append(float(loss_val.detach()))

            # Step the scheduler on validation loss
            prev_lr = experiment.optimizer.param_groups[0]['lr']
            scheduler.step(loss_val)
            if prev_lr != experiment.optimizer.param_groups[0]['lr']:
                print(f"Learning rate scheduler update: {experiment.optimizer.param_groups[0]['lr']}")

        # Validate every 100 iterations
        if itr % 10 == 0:
            if early_stopping:
                # Early stopping check
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    print("Stopping early due to no improvement in validation loss.")
                    break

            # Optionally, plot the predictions
            if itr % 100 == 0:
                # Print training and validation loss
                print(f"Iteration {itr}: Training Loss {loss.detach().float()}")#", Validation Loss {loss_val.detach().float()}")
                y_test_pred = odeint(experiment.func, x0[:, None], t[:], method=training_params.method).detach().squeeze()
                plt.plot(t, y.T)
                plt.plot(t, y_test_pred, "--")
                plt.show()
    #######
    # Save
    #######
    experiment.save(f"{results_root}/neural_ode_{network_name}_experiment_{model_name}_size_{N}_std_reg_{regularizer_lambda}_ntraj_{niter}_dt_{dt}_T_{T}_{training_params.method}_irrsampl_{irregular_sampling}_noisesigma_{scale_obs}_early_Stopping_{early_stopping}_batchtime_{batch_time}", loss_list, y_train, t_train, [], [] , [A])
