import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import math

class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        """
        A custom PyTorch linear layer.
        """
        super(Linear, self).__init__()
        self.weights = Parameter(
            torch.zeros((output_size, input_size))
        )
        self.bias = Parameter(
            torch.zeros(output_size)
        )

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

        init.kaiming_uniform_(self.weights)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weights.T + self.bias

def relu(x):
    return torch.maximum(x, torch.tensor(0.0))

def softmax(x):
    # Subtract the maximum value in each row for numerical stability
    x_exp = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return x_exp / torch.sum(x_exp, dim=1, keepdim=True)

class MLP(nn.Module):
    def __init__(self):
        # Very brutal mlp with size of 3072, 128, 10
        super(MLP, self).__init__()
        self.layer1 = Linear(784, 3072)
        self.layer2 = Linear(3072, 128)
        self.layer3 = Linear(128, 10)

    def forward(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = softmax(self.layer3(x))
        return x