#!/usr/bin/env python
# coding=utf-8
import torch
from torch import nn
import numpy as np

class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, query, key, value, q_mask, k_mask):
        '''
        query: [batch, seq1, hidden_size]
        key: [batch, seq2, hidden_size * num_hidden_state]
        value: [batch, seq2, hidden_size * num_hidden_state]
        q_mask: [batch, seq1]
        k_mask: [batch, seq2]

        return:
            [batch, seq1, hidden_size]
        '''
        batch, seq1, hidden_size1 = query.size()
        _, seq2, hidden_size2 = key.size()

        mask = q_mask.unsqueeze(2).matmul(k_mask.unsqueeze(1))
        assert mask.size() == torch.Size((batch, seq1, seq2))
        
        query_e = query.unsqueeze(2).expand(-1, -1, seq2, -1)
        key_e = key.unsqueeze(1).expand(-1, seq1, -1, -1)
        stack = torch.cat([query_e, key_e], dim=-1)
        assert stack.size() == torch.Size((batch, seq1, seq2, hidden_size1 + hidden_size2))
        # [batch, seq1, seq2]
        A = self.mlp(stack) \
                .squeeze(-1) \
                .masked_fill(mask == 0, float('-inf')) \
                .exp()
        A_sum = A.sum(dim=-1, keepdim=True).clamp(min=2e-15)
        attn = A.div(A_sum)
        assert A.size() == torch.Size((batch, seq1, seq2))
        return attn.matmul(value)


class ScoreLayer(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.2):
        super(ScoreLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
            feature_in = m.weight.size(1)
            bound = 1 / np.sqrt(feature_in)
            nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        '''
        x: [batch, seq, left_dim]
        return: [batch, seq, num_labels]
        '''
        return self.mlp(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, q_mask, k_mask, scale=None):
        '''
        q: [B, L_q, D_q]
        k: [B, L_k, D_k]
        v: [B, L_v, D_v]
        q_mask: [B, L_q]
        k_mask: [B, L_k]
        '''
        batch, L_q, D_q = query.size()
        _, L_k, D_k = key.size()

        if scale is None:
            scale = D_q

        mask = q_mask.unsqueeze(2).matmul(k_mask.unsqueeze(1))
        assert mask.size() == torch.Size((batch, L_q, L_k))

        # [batch, L_q, L_k]
        A = query.matmul(key.transpose(1, 2)) \
                .div(np.sqrt(scale)) \
                .masked_fill(mask == 0, float('-inf')) \
                .exp()
        A_sum = A.sum(dim=-1, keepdim=True).clamp(min=2e-15)
        attn = A.div(A_sum)
        assert attn.size() == torch.Size((batch, L_q, L_k))
        return attn.matmul(value)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, nheads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim_head = dim // nheads
        self.nheads = nheads
        self.linear_k = nn.Linear(dim, self.dim_head * nheads)
        self.linear_v = nn.Linear(dim, self.dim_head * nheads)
        self.linear_q = nn.Linear(dim, self.dim_head * nheads)

        self.dot_product_attn = ScaledDotProductAttention()
        self.linear_final = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(dim)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
            feature_in = m.weight.size(1)
            bound = 1 / np.sqrt(feature_in)
            nn.init.uniform_(m.bias, -bound, bound)
            #nn.init.xavier_normal_(m.weight)
            #m.bias.data.fill_(0)

    def forward(self, query, key, value, q_mask, k_mask):
        '''
        key: [B, L_k, D_k]
        value: [B, L_v, D_v]
        query: [B, L_q, D_q]
        q_mask: [B, L_q]
        k_mask: [k_q]
        '''
        residual = query
        batch = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch * self.nheads, -1, self.dim_head)
        value = value.view(batch * self.nheads, -1, self.dim_head)
        query = query.view(batch * self.nheads, -1, self.dim_head)

        q_mask = q_mask.repeat(self.nheads, 1)
        k_mask = k_mask.repeat(self.nheads, 1)

        context = self.dot_product_attn(query=query,
                                        key=key,
                                        value=value,
                                        q_mask=q_mask,
                                        k_mask=k_mask,
                                        scale=self.dim_head)
        context = context.view(batch, -1, self.nheads * self.dim_head)
        
        output = self.linear_final(context)
        output = self.dropout(output)

        output = self.layer_norm(residual + output)

        return output

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, dim=512, ffn_dim=2048, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(dim, ffn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        '''
        x: [B, S, D]
        '''
        output = x.transpose(1, 2)
        output = self.w2(torch.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        return self.layer_norm(x + output)

class Transformer(nn.Module):
    def __init__(self, dim, nheads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.nheads = nheads
        self.attention = MultiHeadAttention(dim=dim, nheads=nheads, dropout=dropout)
        dim = (dim // nheads) * nheads
        self.pos_fc = PositionalWiseFeedForward(dim=dim, dropout=dropout, ffn_dim=dim)
    
    #def init_weights(self):
    #    initrange = 0.1
    #    nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    #    for module in self.decoder.modules():
    #        nn.init.zeros_(module.weight)
    #        nn.init.uniform_(module.weight, -initrange, initrange)
    #    if self.dueling:
    #        nn.init.zeros_(self.value_layer.weight)
    #        nn.init.uniform_(self.value_layer, -initrange, initrange)
    
    def forward(self, query, q_mask, key=None, value=None, k_mask=None):
        '''
        query: [B, L_q, D_q]
        key: [B, L_k, D_k]
        value: [B, L_v, D_v]
        q_mask: [B, L_q]
        k_mask: [B, L_k]
        '''
        if key is None:
            key = query
            value = query
            k_mask = q_mask

        B, L_q, dim = query.size()
        L_k = key.size(1)

        output = self.attention(
            query=query,
            key=key,
            value=value,
            q_mask=q_mask,
            k_mask=k_mask
        )
        dim = (dim // self.nheads) * self.nheads
        assert output.size() == torch.Size((B, L_q, dim))
        output = self.pos_fc(output)
        return output, q_mask
