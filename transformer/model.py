# -*- coding: utf-8 -*-

import math

import torch
from torch import nn
import torch.nn.functional as F

from transformer.sublayers import PositionEncoder, MultiHeadAttention


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

    def forward(self, x, mask):
        # Attention block
        residual = x
        x = self.attention(x, x, mask)
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

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
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

    def forward(self, x, encoder_output, src_mask, target_mask):
        # Self-attention block
        residual = x
        x = self.self_attention(x, x, target_mask)
        x = self.dropout1(x)
        x += residual
        x = self.layer_norm1(x)

        # Encoder-attention block
        residual = x
        x = self.encoder_attention(x, encoder_output, src_mask)
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
            DecoderLayer(d_model=d_model, d_ff=d_ff, n_multi_head=n_multi_head,
                dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return x


class Transformer(nn.Module):

    def __init__(self, d_vocab, d_model=512, d_ff=2048,
        n_encoder_layers=6, n_decoder_layers=6, n_multi_head=8,
        dropout=0.1, max_seq_size=200):
        super().__init__()

        self.d_model = d_model

        self.input_embedding = nn.Embedding(d_vocab, d_model)
        self.encoder_positional_encoding = PositionEncoder(d_model=d_model, max_seq_size=max_seq_size)
        self.encoder_dropout = nn.Dropout(p=dropout)
        self.output_embedding = nn.Embedding(d_vocab, d_model)
        self.decoder_positional_encoding = PositionEncoder(d_model=d_model, max_seq_size=max_seq_size)
        self.decoder_dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(d_model=d_model, d_ff=d_ff, n_multi_head=n_multi_head,
            dropout=dropout, n_layers=n_encoder_layers)
        self.decoder = Decoder(d_model=d_model, d_ff=d_ff, n_multi_head=n_multi_head,
            dropout=dropout, n_layers=n_decoder_layers)

        self.linear_output = nn.Linear(d_model, d_vocab, bias=False)
        self.linear_output.weight = self.output_embedding.weight

    def forward(self, inputs, outputs, src_mask, target_mask):
        inputs = self.input_embedding(inputs) * math.sqrt(self.d_model)
        inputs = self.encoder_positional_encoding(inputs)
        inputs = self.encoder_dropout(inputs)

        outputs = self.output_embedding(outputs) * math.sqrt(self.d_model)
        outputs = self.decoder_positional_encoding(outputs)
        outputs = self.decoder_dropout(outputs)

        encoder_output = self.encoder(inputs, src_mask)
        decoder_output = self.decoder(outputs, encoder_output, src_mask, target_mask)

        outp = self.linear_output(decoder_output)

        return outp
