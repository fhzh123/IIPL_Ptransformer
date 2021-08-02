import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_size, ffn_hid_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(emb_size, ffn_hid_dim)
        self.w_2 = nn.Linear(ffn_hid_dim, emb_size)

    def forward(self, x):
        return self.w_2(F.gelu(self.w_1(x)))