import torch.nn as nn
import torch.nn.functional as F

"""
Classical and popular attention: scaled dot product attention. 
Can be used as one layer. See example usage in attention models.
"""

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        # d_k represents the dimension of encoding the input information
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # Q: Query, K: Key, V: Value
        # Q times K determines the relationship, times V produces the change of value after such relationship

        # Step 1: Compute the dot product between Q and K, scaled by the input dimension
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        # Step 2: Apply the Mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # assign negative inf to masked values
        # Step 3: Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        # Step 4: Compute the weighted sum of values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights