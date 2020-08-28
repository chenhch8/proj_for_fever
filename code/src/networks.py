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
    def __init__(self, left_dim, right_dim, num_labels):
        super(ScoreLayer, self).__init__()
        self.num_labels = num_labels
        self.weight = Parameter(torch.Tensor(num_labels,
                                             left_dim,
                                             right_dim))
        self.bias = Parameter(torch.Tensor(num_labels))
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            feature_in = self.weight.size(2)
            bound = 1 / np.sqrt(feature_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, left_inputs, right_inputs):
        '''
        left_inputs: [batch, seq, left_dim]
        right_inputs: [batch, seq, right_dim]
        
        return: [batch, seq, num_labels]
        '''
        batch, seq, left_dim = left_inputs.size()
        _, _, right_dim = right_inputs.size()
        num_labels = self.num_labels

        ws = self.weight.unsqueeze(0).matmul(right_inputs.unsqueeze(1).transpose(3, 2))
        assert ws.size() == torch.Size((batch, num_labels, left_dim, seq))
        
        scores = left_inputs.transpose(2, 1).unsqueeze(1).mul(ws).sum(dim=2) + self.bias[None,:,None]
        scores = scores.permute(0, 2, 1)
        assert scores.size() == torch.Size((batch, seq, num_labels))

        return scores
