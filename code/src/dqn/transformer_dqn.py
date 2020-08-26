#!/usr/bin/env python3
# coding=utf-8
from tqdm import tqdm, trange
from functools import reduce
from copy import deepcopy
import pdb
import math
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
#from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam, AdamW, lr_scheduler
#from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
#from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AlbertConfig,
    BertConfig,
    XLNetConfig,
    RobertaConfig,
)

from .base_dqn import BaseDQN
from .lstm_dqn import lstm_load_and_process_data
from data.structure import *

CONFIG_CLASSES = {
    'bert': BertConfig,
    'albert': AlbertConfig,
    'xlnet': XLNetConfig,
    'roberta': RobertaConfig
}


transformer_load_and_process_data = lstm_load_and_process_data

def convert_tensor_to_transformer_inputs(batch_claims: List[List[float]],
                                         batch_evidences: List[torch.Tensor],
                                         device=None) -> dict:
    device = device if device != None else torch.device('cpu')
    
    evi_len = [evi.size(0) for evi in batch_evidences]
    evi_max = max(evi_len)

    claims_tensor = torch.tensor(batch_claims, device=device)

    evidences_pad = pad_sequence(batch_evidences, batch_first=True).to(device)

    evidences_mask = torch.tensor(
        [[1] * size + [0] * (evi_max - size) for size in evi_len],
        dtype=torch.float,
        device=device
    )

    return {
        'claims': claims_tensor,
        'evidences': evidences_pad,
        'evidences_mask': evidences_mask
    }


#class PositionalEncoding(nn.Module):
#    r"""Inject some information about the relative or absolute position of the tokens
#        in the sequence. The positional encodings have the same dimension as
#        the embeddings, so that the two can be summed. Here, we use sine and cosine
#        functions of different frequencies.
#    .. math::
#        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
#        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
#        \text{where pos is the word position and i is the embed idx)
#    Args:
#        d_model: the embed dim (required).
#        dropout: the dropout value (default=0.1).
#        max_len: the max. length of the incoming sequence (default=5000).
#    Examples:
#        >>> pos_encoder = PositionalEncoding(d_model)
#    """
#
#    def __init__(self, d_model, dropout=0.1, max_len=5000):
#        super(PositionalEncoding, self).__init__()
#        self.dropout = nn.Dropout(p=dropout)
#
#        pe = torch.zeros(max_len, d_model)
#        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#        pe[:, 0::2] = torch.sin(position * div_term)
#        pe[:, 1::2] = torch.cos(position * div_term)
#        pe = pe.unsqueeze(0).transpose(0, 1)
#        self.register_buffer('pe', pe)
#
#    def forward(self, x):
#        r"""Inputs of forward function
#        Args:
#            x: the sequence fed to the positional encoder model (required).
#        Shape:
#            x: [sequence length, batch size, embed dim]
#            output: [sequence length, batch size, embed dim]
#        Examples:
#            >>> output = pos_encoder(x)
#        """
#
#        x = x + self.pe[:x.size(0), :]
#        return self.dropout(x)

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

class QNetwork(nn.Module):
    def __init__(self,
                 num_labels,
                 hidden_size,
                 dropout=0.1,
                 nheads=8,
                 num_layers=3):
                 #dueling=False):
        super(QNetwork, self).__init__()
        # Transformer
        #self.pos_encoder = PositionalEncoding(hidden_size,
        #                                      dropout=dropout,
        #                                      max_len=6)
        #encoder_layers = TransformerEncoderLayer(d_model=hidden_size,
        #                                         nhead=nhead,
        #                                         dropout=dropout)
        #self.encoder = TransformerEncoder(encoder_layers,
        #                                  num_layers=num_layers)
        ## Value
        #if dueling:
        #    self.value_layer = nn.Linear(hidden_size, 1)
        self.nheads = nheads
        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, 'transf:%d' % i, Transformer(dim=hidden_size,
                                                       nheads=nheads,
                                                       dropout=dropout))
        self.transformer_2 = Transformer(dim=hidden_size,
                                         nheads=nheads,
                                         dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, claims, evidences, evidences_mask):
        '''
        claims: [batch, hidden_size]
        evidences: [batch, seq, hidden_size]
        evidences_mask: [batch, seq]

        return:
            [batch * available_seq2, num_labels]
        '''
        batch, seq, hidden_size = evidences.size()
        num_labels, nheads = 3, self.nheads
        for i in range(self.num_layers):
            layer = getattr(self, 'transf:%d' % i)
            evidences, _ = layer(evidences, evidences_mask)
        assert evidences.size() == torch.Size((batch, seq, hidden_size))
        output, _ = self.transformer_2(
            query = claims.unsqueeze(1),
            key=evidences,
            value=evidences,
            q_mask=torch.ones([batch, 1], dtype=torch.float).to(evidences_mask),
            k_mask=evidences_mask
        )
        assert output.size() == torch.Size((batch, 1, hidden_size))
        assert claims.size() == torch.Size((batch, hidden_size))
        q_value = self.mlp(torch.cat([claims, output.squeeze(1)], dim=-1))
        assert q_value.size() == torch.Size((batch, 3))
        #pdb.set_trace()

        return (q_value,)


class TransformerDQN(BaseDQN):
    def __init__(self, args):
        super(TransformerDQN, self).__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        config_class = CONFIG_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.model_name_or_path)
        # q network
        self.q_net = QNetwork(
            hidden_size=config.hidden_size,
            #dropout=config.dropout,
            num_labels=args.num_labels,
            nheads=args.nhead,
            num_layers=args.num_layers
        )
        # Target network
        self.t_net = deepcopy(self.q_net) if args.do_train else self.q_net
        self.q_net.zero_grad()

        self.set_network_untrainable(self.t_net)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        
        #self.optimizer = SGD(self.q_net.parameters(), lr=args.learning_rate, momentum=0.9)
        self.optimizer = AdamW(self.q_net.parameters(), lr=args.learning_rate)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer,
                                               lr_lambda=lambda epoch: max(np.power(0.5, epoch // 100), 5e-6 / args.learning_rate))
        #self.optimizer = Adam(self.q_net.parameters(), lr=args.learning_rate)


    def convert_to_inputs_for_select_action(self, batch_state: List[State], batch_actions: List[List[Action]]) -> List[dict]:
        assert len(batch_state) == len(batch_actions)
        batch_claims, batch_evidences = [], []
        for state, actions in zip(batch_state, batch_actions):
            # tokens here is actually the sentence embedding
            ## [1, dim]
            batch_claims.extend([state.claim.tokens] * len(actions))
            ## [seq, dim]
            evidence = [sent.tokens for sent in state.candidate]
            batch_evidences.extend(
                [torch.tensor(evidence + [action.sentence.tokens],
                              dtype=torch.float) for action in actions],
            )
        return convert_tensor_to_transformer_inputs(batch_claims, batch_evidences)
    

    def convert_to_inputs_for_update(self, states: List[State], actions: List[Action]) -> dict:
        assert len(states) == len(actions)
        batch_claims, batch_evidences = [], []
        for state, action in zip(states, actions):
            ## [1, dim]
            batch_claims.append(state.claim.tokens)
            ## [seq, dim]
            evidence = [sent.tokens for sent in state.candidate]
            batch_evidences.append(
                torch.tensor(evidence + [action.sentence.tokens],
                             dtype=torch.float)
            )
        return convert_tensor_to_transformer_inputs(batch_claims, batch_evidences, self.device)

