# -*- coding: utf-8 -*-

import math

import torch
from torch import nn
import torch.nn.functional as F


class PositionEncoder(nn.Module):
    def __init__(self, d_model, max_seq_size):
        super().__init__()

        self.positional_encoding = torch.zeros(max_seq_size, d_model)
        for pos in range(max_seq_size):
            for i in range(0, d_model // 2):
                self.positional_encoding[pos, 2*i] = math.sin(pos / 10000 ** (2*i / d_model))
                self.positional_encoding[pos, 2*i + 1] = math.cos(pos / 10000 ** ((2*i + 1) / d_model))
        self.positional_encoding = self.positional_encoding.unsqueeze(0)

    def forward(self, x):
        seq_size = x.size(1)

        x += self.positional_encoding[:, :seq_size]
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"Model dimension {d_model} not divisible by number "
                "of attention projections {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_attention = d_model // n_heads

        self.W_queries = nn.Linear(d_model, d_model, bias=False)
        self.W_keys = nn.Linear(d_model, d_model, bias=False)
        self.W_values = nn.Linear(d_model, d_model, bias=False)

        self.W_output = nn.Linear(d_model, d_model, bias=False)

    def attention(self, query, key, value):
        x = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_attention), dim=-1)
        x = torch.matmul(x, value)

        return x

    def forward(self, query_input, kv_input):
        batch_size = query_input.size(0)

        queries = self.W_queries(query_input)
        keys = self.W_keys(kv_input)
        values = self.W_values(kv_input)

        queries = queries.view(batch_size, -1, self.n_heads, self.d_attention).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.n_heads, self.d_attention).transpose(1, 2)
        values = values.view(batch_size, -1, self.n_heads, self.d_attention).transpose(1, 2)

        x = self.attention(queries, keys, values)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        x = self.W_output(x)

        return x
