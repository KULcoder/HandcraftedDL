from scaled_dot_product_attention import ScaledDotProductAttention

import torch.nn as nn

"""
Classical and popular attention: multi-head attention. 
Can be used as one layer. See example usage in attention models.
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.q_linear = nn.Linear(d_model, d_k * num_heads)
        self.k_linear = nn.Linear(d_model, d_k * num_heads)
        self.v_linear = nn.Linear(d_model, d_v * num_heads)
        self.fc_out = nn.Linear(d_v * num_heads, d_model)

        self.attention = ScaledDotProductAttention(d_k)

        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.q_linear(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(V).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Apply attention on all the projected vectors in batch
        scores, attention_weights = self.attention(Q, K, V, mask)

        # concat attention output
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)

        # Final linear layer
        output = self.fc_out(concat)

        return output, attention_weights