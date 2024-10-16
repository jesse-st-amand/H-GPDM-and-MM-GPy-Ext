from HGPLVM.backconstraints.backconstraints_base import BC_Base
import numpy as np
from GPy.core import Param
import GPy

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

class ThreeLayerMLP(nn.Module):
    def __init__(self, D, H, d):
        super(ThreeLayerMLP, self).__init__()
        self.input_layer = nn.Linear(D, H)
        self.hidden_layer = nn.Linear(H, H)
        self.output_layer = nn.Linear(H, d)

        # Initialize the optimizer as None; it will be set up outside the class
        self.optimizer = None

    def forward(self, Y):
        H = F.relu(self.input_layer(Y))
        H = F.relu(self.hidden_layer(H))
        X = self.output_layer(H)
        return X

    def train_step(self, Y, X_target):
        """
        Performs a single training step, including the forward pass, loss calculation,
        and the optimizer step.

        Parameters:
        - Y: Input tensor of shape (N, D)
        - X_target: Target tensor of shape (N, d)

        Returns:
        - loss.item(): The loss value as a Python float
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Please set the optimizer before training.")

        # Zero the gradients (PyTorch accumulates gradients by default)
        self.optimizer.zero_grad()

        # Forward pass
        X_pred = self(torch.tensor(Y, dtype=torch.float))

        # Compute loss
        loss = F.mse_loss(X_pred, torch.tensor(X_target, dtype=torch.float))

        # Backward pass (computes gradients)
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss.item()





class MLP_BC(BC_Base):

    def __init__(self, GPNode, output_dim, param_dict, name=''):
        self.hidden_dim = output_dim
        super().__init__(GPNode, output_dim, param_dict, name=name)
        self.first_call = True

    def initialize_Y(self):
        self.Y = self.GPNode.Y
        self.Y1 = self.Y[:, self.param_dict['Y1 indices']]
        self.Y2 = self.Y[:, self.param_dict['Y2 indices']]


    def f(self):
        return self.model(torch.tensor(self.Y, requires_grad=True).float()).detach().numpy()

    def BC_f_new(self, Y_new, pred_group=0, **kwargs):
        if pred_group == 0:
            return self.model(torch.tensor(Y_new).float()).detach().numpy()
        else:
            raise ValueError('MLP BC has no pred var groups 1 and 2. Use GP map.')

    def get_X(self, X):
        return torch.tensor(X, requires_grad=True).float()

    def construct_mapping(self):
        # Model parameters
        D = self.input_dim  # Dimensionality of the input
        H = self.hidden_dim  # Number of neurons in the hidden layer
        d = self.output_dim  # Dimensionality of the output

        # Initialize the model
        self.model = ThreeLayerMLP(D, H, d)

        # Set up the optimizer
        self.model.optimizer = Adam(self.model.parameters(), lr=0.001)  # Learning rate

        # Number of epochs - how many times to iterate through the entire dataset
        num_epochs = self.param_dict['num_epochs']

        Ys, Xs = self.batching()

        # Assuming `train_loader` is a DataLoader object that loads batches of (Y, X_target) pairs

        for epoch in range(num_epochs):
            total_loss = 0
            for Y_batch, X_target_batch in zip(Ys, Xs):
                loss = self.model.train_step(Y_batch, X_target_batch)
                total_loss += loss
            avg_loss = total_loss / len(Ys)
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    def batching(self):

        k_set_size = int(self.GPNode.num_seqs / self.param_dict['num_acts'])
        k_set_eps = np.arange(k_set_size, self.GPNode.num_seqs + k_set_size, k_set_size, dtype=int)
        seq_eps_k_list = []
        seq_x0s_k_list = []

        for k_ep in k_set_eps:
            seq_eps_k_list.append(np.array(self.GPNode.seq_eps)[(k_ep - k_set_size):k_ep])
            seq_x0s_k_list.append(np.array(self.GPNode.seq_x0s)[(k_ep - k_set_size):k_ep])

        Y_batches = []
        X_batches = []

        for k in range(k_set_size):
            tempY = []
            tempX = []
            for seq_eps, seq_x0s in zip(seq_eps_k_list, seq_x0s_k_list):
                tempY.append(self.Y[seq_x0s[k]:seq_eps[k]+1,:])
                tempX.append(self.X[seq_x0s[k]:seq_eps[k]+1,:])
            Y_batches.append(np.vstack(tempY))
            X_batches.append(np.vstack(tempX))

        return Y_batches, X_batches

    def update_gradients(self, dL_dX):
        self.X = self.get_X(self.GPNode.X.values)
        dL_dX_tensor = torch.tensor(dL_dX, requires_grad=True).float()
        self.X.backward(dL_dX_tensor, retain_graph=True)
        for i,(param_label,param) in enumerate(zip(self.MLP_param_dict.keys(),self.model.parameters())):
            self.MLP_param_dict[param_label].gradient = param.grad
        return dL_dX


    def link_params(self):
        self.MLP_param_dict = {}
        for i, layer in enumerate(self.model.children()):
            for j, param in enumerate(layer.parameters()):
                param_label = 'layer' + str(i) + 'param' + str(j)
                self.MLP_param_dict[param_label] = param.detach()
                self.link_parameter(Param(param_label, self.MLP_param_dict[param_label]))

    def constrain_params(self):
        pass


    def initialize_A(self):
        pass

