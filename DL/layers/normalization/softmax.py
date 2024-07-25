import torch

def softmax(x):
    # Subtract the maximum value in each row for numerical stability
    x_exp = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return x_exp / torch.sum(x_exp, dim=1, keepdim=True)