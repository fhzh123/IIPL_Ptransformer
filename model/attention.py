import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, nhead, dropout, device=None):
        super(MultiHeadAttention, self).__init__()
        self.d_k = emb_size // nhead
        self.h = nhead
        self.w_q = nn.Linear(emb_size, emb_size, device=device)
        self.w_k = nn.Linear(emb_size, emb_size, device=device)
        self.w_v = nn.Linear(emb_size, emb_size, device=device)
        self.w_o = nn.Linear(emb_size, emb_size, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # query = [ tgt_len, batch, emb ]
        # key, value = [ src_len, batch, emb ]
        tgt_len, batch, embed_dim = query.shape

        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        query = query.contiguous().view(tgt_len, batch * self.h, self.d_k).transpose(0, 1)
        value = value.contiguous().view(-1, batch * self.h, self.d_k).transpose(0, 1)
        key = key.contiguous().view(-1, batch * self.h, self.d_k).transpose(0, 1)

        # query = [ batch * nhead, tgt_len, d_k ] , now batch first
        # key, value = [ batch * nhead, src_len, d_k ]

        src_len = key.size(1)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(
                batch, 1, 1, src_len).expand(-1, self.h, -1, -1).reshape(batch * self.h, 1, src_len)

            if attn_mask is not None and attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)

            if attn_mask is not None and attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
        
        attn = scaled_dot_product_attention(
            query, key, value, attn_mask, self.dropout)
        attn = attn.transpose(0, 1).contiguous().view(
            tgt_len, batch, embed_dim)
        attn = self.w_o(attn)

        return attn, None


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    # query = [ batch * head, tgt_len, d_k ]
    # key, value = [ batch * nhead, src_len, d_k ]

    scores = torch.bmm(query, key.transpose(-2, -1))

    # scores = [ batch * nhead, tgt_len, src_len ]

    scores /= math.sqrt(query.size(2))

    if mask is not None:
        scores += mask

    scores = dropout(F.softmax(scores, dim=-1))
    attn = torch.bmm(scores, value)

    # attn = [ batch * nhead, tgt_len, d_k ]

    return attn