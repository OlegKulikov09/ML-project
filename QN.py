import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input, output):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input, 64)  # Первый линейный слой
        self.fc2 = nn.Linear(64, 64) # Второй линейный слой
        self.fc3 = nn.Linear(64, output)  # Выходной слой

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Применяем ReLU после первого слоя
        x = torch.relu(self.fc2(x))  # Применяем ReLU после второго слоя
        return self.fc3(x)  # Выходной слой возвращает значения Q