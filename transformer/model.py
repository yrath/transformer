# -*- coding: utf-8 -*-

import math

import torch
from torch import nn
import torch.nn.functional as F

from transformer.sublayers import PositionEncoder, MultiHeadAttention

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


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, n_multi_head, dropout):
        super().__init__()

        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_multi_head)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.dropout2 = nn.Dropout(p=dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention block
        residual = x
        x = self.attention(x, x)
        x = self.dropout1(x)
        x += residual
        x = self.layer_norm1(x)

        # Feedforward block
        residual = x

        x = self.l2(F.relu(self.l1(x)))
        x = self.dropout2(x)
        x += residual
        x = self.layer_norm2(x)

        return x


class Encoder(nn.Module):

    def __init__(self, d_model, d_ff, n_multi_head, dropout, n_layers):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_ff=d_ff, n_multi_head=n_multi_head, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, n_multi_head, dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_multi_head)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.encoder_attention = MultiHeadAttention(d_model=d_model, n_heads=n_multi_head)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.dropout3 = nn.Dropout(p=dropout)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        # Self-attention block
        residual = x
        x = self.self_attention(x, x)
        x = self.dropout1(x)
        x += residual
        x = self.layer_norm1(x)

        # Encoder-attention block
        residual = x
        x = self.encoder_attention(x, encoder_output)
        x = self.dropout2(x)
        x += residual
        x = self.layer_norm2(x)

        # Feedforward block
        residual = x

        x = self.l2(F.relu(self.l1(x)))
        x = self.dropout3(x)
        x += residual

        x = self.layer_norm2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, d_model, d_ff, n_multi_head, dropout, n_layers):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_ff=d_ff, n_multi_head=n_multi_head, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x


class Transformer(nn.Module):

    def __init__(self, d_input_dict, d_output_dict, d_model=512, d_ff=2048,
        n_encoder_layers=6, n_decoder_layers=6, n_multi_head=8,
        dropout=0.1, max_seq_size=200):
        super().__init__()

        self.d_model = d_model

        self.input_embedding = nn.Embedding(d_input_dict, d_model)
        self.encoder_positional_encoding = PositionEncoder(d_model=d_model, max_seq_size=max_seq_size)
        self.encoder_dropout = nn.Dropout(p=dropout)
        self.output_embedding = nn.Embedding(d_output_dict, d_model)
        self.decoder_positional_encoding = PositionEncoder(d_model=d_model, max_seq_size=max_seq_size)
        self.decoder_dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(d_model=d_model, d_ff=d_ff, n_multi_head=n_multi_head,
            dropout=dropout, n_layers=n_encoder_layers)
        self.decoder = Decoder(d_model=d_model, d_ff=d_ff, n_multi_head=n_multi_head,
            dropout=dropout, n_layers=n_decoder_layers)

        self.linear_output = nn.Linear(d_model, d_output_dict)

    def forward(self, inputs, outputs):
        inputs = self.input_embedding(inputs) * math.sqrt(self.d_model)
        inputs = self.encoder_positional_encoding(inputs)
        inputs = self.encoder_dropout(inputs)

        outputs = self.output_embedding(outputs) * math.sqrt(self.d_model)
        outputs = self.decoder_positional_encoding(outputs)
        outputs = self.decoder_dropout(outputs)

        encoder_output = self.encoder(inputs)
        decoder_output = self.decoder(outputs, encoder_output)

        outp = F.softmax(self.linear_output(decoder_output))

        return outp
