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

        # an idea similar to kaiming init: consider the size and activation
        # to modify the initilization scale
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

        init.kaiming_uniform_(self.weights)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weights.T + self.bias