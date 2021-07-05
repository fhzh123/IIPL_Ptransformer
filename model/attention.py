import torch
from util import *
import torch.nn as nn

def attention(query, key, value, mask=None, dropout=None):
    # query = [batch size, trg len, d_model] 
    # key, value = [batch size, src len, d_model]

    q_k = torch.bmm(query, key.transpose(-2, -1))

    # q_k = [batch size, trg len, src len]

    scores = q_k / math.sqrt(query.size(-1))

    if mask is not None:
        print(scores.shape, mask.shape)
        scores += mask

    output = torch.softmax(scores, dim=-1)

    if dropout is not None:
        output = dropout(output)

    output = torch.bmm(output, value)

    # output = [batch size, trg len, d_model]    
    
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, length = query.size(0), query.size(1)

        # 하버드nlp 에서 기용한 코드 (이유: nhead 만큼 나눠서 parallel하게 진행하기 위해서)
        # query, key, value = \
        #     [l(x).view(batch_size, -1, self.nhead * self.d_k)
        #      for l, x in zip(self.linears, (query, key, value))]

        # pytorch 에서 기용한 코드 (Shape이 다릅니다.)
        query = self.w_q(query).view(batch_size, -1, self.nhead * self.d_k)
        key = self.w_q(key).view(batch_size, -1, self.nhead * self.d_k)
        value = self.w_q(value).view(batch_size, -1, self.nhead * self.d_k)

        # 마스크를 boolean대신 -infinity로 채워서 더합니다.
        if mask is not None and mask.dtype == torch.bool:
            mask = mask.view(batch_size, 1, 1, length).expand(-1, self.nhead, -1, -1).reshape(batch_size * self.nhead, 1, length)
            new_mask = torch.zeros_like(mask, dtype=torch.float)
            new_mask.masked_fill_(mask, float("-inf"))
            mask = new_mask

        # Scaled Dot-Product Attention layer
        x = attention(query, key, value, mask, self.dropout)

        # Concatenate the nhead layers of attention outputs
        x = x.transpose(0, 1).contiguous().view(batch_size, -1, self.nhead * self.d_k)

        # Final linear layer to get the final output
        return self.w_o(x)
        # return self.linears[3](x) 

