import torch
import torch.nn as nn
import torch.optim as optim


class QN(nn.Module):
    def __init__(self, input, output):
        super(QN, self).__init__()
        self.fc1 = nn.Linear(input, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU after first layer
        x = torch.relu(self.fc2(x))  # ReLU after first layer
        return self.fc3(x)  # output layer returns Q value
