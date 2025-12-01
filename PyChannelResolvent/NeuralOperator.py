import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class NPYDataset(Dataset):

    def __init__(self, data_dirs, device='cpu'):
        self.data_dirs = data_dirs
        self.device = device

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        data_inputs_dir, data_target_dir = self.data_dirs[idx]
        data_inputs = torch.from_numpy(np.load(data_inputs_dir)).float().to(
            self.device)
        data_target = torch.from_numpy(np.load(data_target_dir)).float().to(
            self.device)
        return data_inputs, data_target


class BranchNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=8):
        super(BranchNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Create a list of layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]  # input layer
        for _ in range(num_hidden_layers - 1):  # hidden layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))  # output layer

        # Combine layers into a sequential module
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TrunkNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=8):
        super(TrunkNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Create a list of layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]  # input layer
        for _ in range(num_hidden_layers - 1):  # hidden layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))  # output layer

        # Combine layers into a sequential module
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralOperator(nn.Module):

    def __init__(self,
                 branch_input_dim,
                 branch_hidden_dim,
                 group_dim,
                 trunk_input_dim,
                 trunk_hidden_dim,
                 branch_num_hidden_layers=8,
                 trunk_num_hidden_layers=8,
                 group_size=1):
        super(NeuralOperator, self).__init__()
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.group_dim = group_dim
        self.group_size = group_size
        self.branch_net = BranchNet(branch_input_dim, branch_hidden_dim,
                                    group_dim * group_size,
                                    branch_num_hidden_layers)
        self.trunk_net = TrunkNet(trunk_input_dim, trunk_hidden_dim, group_dim,
                                  trunk_num_hidden_layers)

    def forward(self, x):
        # x: (batch_size, branch_input_dim+ trunk_input_dim)
        u, y = x[..., :self.branch_input_dim], x[..., self.branch_input_dim:]
        branch_output = self.branch_net(
            u)  # Shape (batch_size, group_dim * group_size)
        trunk_output = self.trunk_net(
            y)  # Shape (batch_size, group_dim*group_size)

        branch_output = branch_output.view(-1, self.group_dim, self.group_size)
        trunk_output = trunk_output.view(-1, self.group_dim, 1)
        outputs = torch.bmm(trunk_output.permute(0, 2, 1),
                            branch_output).squeeze(1)

        return outputs


def init_NeuralOperator(model_args, device):
    branch_input_dim_ = model_args['branch_input_dim']
    branch_hidden_dim_ = model_args['branch_hidden_dim']
    group_dim_ = model_args['group_dim']
    trunk_input_dim_ = model_args['trunk_input_dim']
    trunk_hidden_dim_ = model_args['trunk_hidden_dim']
    branch_num_hidden_layers_ = model_args['branch_num_hidden_layers']
    trunk_num_hidden_layers_ = model_args['trunk_num_hidden_layers']
    group_size_ = model_args['group_size']
    model = NeuralOperator(branch_input_dim=branch_input_dim_,
                           branch_hidden_dim=branch_hidden_dim_,
                           group_dim=group_dim_,
                           trunk_input_dim=trunk_input_dim_,
                           trunk_hidden_dim=trunk_hidden_dim_,
                           branch_num_hidden_layers=branch_num_hidden_layers_,
                           trunk_num_hidden_layers=trunk_num_hidden_layers_,
                           group_size=group_size_).to(device)
    return model


def load_NeuralOperator(path, model_args, device):
    branch_input_dim_ = model_args['branch_input_dim']
    branch_hidden_dim_ = model_args['branch_hidden_dim']
    group_dim_ = model_args['group_dim']
    trunk_input_dim_ = model_args['trunk_input_dim']
    trunk_hidden_dim_ = model_args['trunk_hidden_dim']
    branch_num_hidden_layers_ = model_args['branch_num_hidden_layers']
    trunk_num_hidden_layers_ = model_args['trunk_num_hidden_layers']
    group_size_ = model_args['group_size']
    model = NeuralOperator(branch_input_dim=branch_input_dim_,
                           branch_hidden_dim=branch_hidden_dim_,
                           group_dim=group_dim_,
                           trunk_input_dim=trunk_input_dim_,
                           trunk_hidden_dim=trunk_hidden_dim_,
                           branch_num_hidden_layers=branch_num_hidden_layers_,
                           trunk_num_hidden_layers=trunk_num_hidden_layers_,
                           group_size=group_size_).to(device)
    model.load_state_dict(
        torch.load(path, weights_only=True, map_location=device))
    model.eval()
    return model


def NeuralOperator_prediction(model, inputs, device):
    model.eval()
    inputs = torch.from_numpy(inputs).float().to(device)
    with torch.no_grad():
        predictions = model(inputs)
    return predictions.cpu().numpy()
