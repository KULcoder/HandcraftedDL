"""
Several attention based models on MNIST tasks.
"""

import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from scaled_dot_product_attention import ScaledDotProductAttention

class MNISTSelfAttentionModel(nn.Module):
    def __init__(self, input_dim, d_k, num_classes):
        super(MNISTSelfAttentionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, d_k)
        self.attention = ScaledDotProductAttention(d_k)
        self.fc2 = nn.Linear(d_k, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        Q = K = V = x.unsqueeze(1) # Use the same input for Q, K, V
        attn_output, _ = self.attention(Q, K, V)
        attn_output = attn_output.squeeze(1)
        out = self.fc2(attn_output)
        return out

class MNISTMultiHeadAttentionModel(nn.Module):
    def __init__(self, num_heads, d_model, d_k, d_v, num_classes=10):
        # includes two convolutional layers to reduce dimension
        super(MNISTMultiHeadAttentionModel, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, d_model)
        self.attention = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.fc2 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)
        x = self.fc1(x).unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        x = self.fc2(x)
        return x