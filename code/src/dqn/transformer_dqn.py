#!/usr/bin/env python3
# coding=utf-8
from tqdm import tqdm, trange
from functools import reduce
from copy import deepcopy
import pdb
import math

import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam, AdamW
#from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
#from torch.utils.data.distributed import DistributedSampler

from .base_dqn import BaseDQN
#from .lstm_dqn import lstm_load_and_process_data, convert_tensor_to_lstm_inputs
from .lstm_dqn import lstm_load_and_process_data
from data.structure import *


transformer_load_and_process_data = lstm_load_and_process_data
convert_to_inputs_for_select_action = None # TODO


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class QNetwork(nn.Module):
    def __init__(self,
                 num_labels,
                 hidden_size,
                 dropout=0.1,
                 nhead=8,
                 num_layers=3,
                 dueling=True):
        super(QNetwork, self).__init__()
        self.dueling = dueling
        # Transformer
        self.pos_encoder = PositionalEncoding(hidden_size,
                                              dropout=dropout,
                                              max_len=6)
        encoder_layers = TransformerEncoderLayer(d_model=hidden_size,
                                                 nhead=nhead,
                                                 dropout=dropout)
        self.transformers = TransformerEncoder(encoder_layers,
                                               num_layers=num_layers)
        # attention paramters
        self.attn_layer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.ReLU(True)
        )
        # Value
        if dueling:
            self.value_layer = nn.Linear(hidden_size, 1)
        # Advantage
        self.weight = Parameter(torch.Tensor(num_labels,
                                             hidden_size,
                                             hidden_size))
        self.bias = Parameter(torch.Tensor(num_labels))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            feature_in = self.weight.size(2)
            bound = 1 / np.sqrt(feature_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def attention_aggregate(self, query, key, value, q_mask, k_mask):
        '''
        query: [batch, seq1, hidden_size]
        key: [batch, seq2, hidden_size]
        value: [batch, seq2, hidden_size]
        q_mask: [batch, seq1]
        k_mask: [batch, seq2]

        return:
            [batch, seq1, hidden_size]
        '''
        batch, seq1, hidden_size = query.size()
        seq2 = key.size(1)

        mask = q_mask.unsqueeze(2).matmul(k_mask.unsqueeze(1))
        assert mask.size() == torch.Size((batch, seq1, seq2))
        
        query_e = query.unsqueeze(2).expand(-1, -1, seq2, -1)
        key_e = key.unsqueeze(1).expand(-1, seq1, -1, -1)
        stack = torch.cat([query_e, key_e], dim=-1)
        assert stack.size() == torch.Size((batch, seq1, seq2, hidden_size * 2))
        # [batch, seq1, seq2]
        A = self.attn_layer(stack) \
                .squeeze(-1) \
                .masked_fill(torch.logical_not(mask), float('-inf')) \
                .exp()
        A_sum = A.sum(dim=-1, keepdim=True).clamp(min=2e-15)
        attn = A.div(A_sum)
        assert A.size() == torch.Size((batch, seq1, seq2))
        return attn.matmul(value)
        

    def forward(self, states, state_mask, actions, actions_mask):
        '''
        states: [batch, seq, hidden_size]
        state_mask: [batch, seq]
        actions: [batch, seq2, hidden_size]
        actions_mask: [batch, seq2]

        return:
            [batch * available_seq2, num_labels]
        '''
        batch, seq, hidden_size = states.size()
        seq2, num_labels = actions.size(1), 3
        # [batch, seq, hidden_size]
        states = self.pos_encoder(states.permute(1, 0, 2)).permute(1, 0, 2)
        out = self.transformers(states)
        assert out.size() == torch.Size((batch, seq, hidden_size))
        
        # [batch, seq2, hidden_size]
        states_feat = self.attention_aggregate(actions,
                                               out, out,
                                               actions_mask,
                                               state_mask)
        actions_num = actions_mask.sum(dim=1).view(-1, 1, 1).expand(-1, 1, hidden_size)
        states_feat_mean = states_feat.sum(dim=1, keepdim=True).div(actions_num)
        assert actions_num.size() == torch.Size((batch, 1, hidden_size))
        assert states_feat_mean.size() == torch.Size((batch, 1, hidden_size))
        assert states_feat.size() == torch.Size((batch, seq2, hidden_size))
        
        # [batch, num_labels, hidden_size, seq2]
        ws = self.weight.unsqueeze(0).matmul(states_feat.unsqueeze(1).transpose(3, 2))
        assert ws.size() == torch.Size((batch, num_labels, hidden_size, seq2))
        
        # Value - [batch, seq2, num_labels]
        if self.dueling:
            val_scores = self.value_layer(states_feat_mean)
            assert val_scores.size() == torch.Size((batch, 1, 1))

            val_scores = val_scores.expand(-1, seq2, num_labels)
            assert val_scores.size() == torch.Size((batch, seq2, num_labels))
        
        # Advantage - [batch, seq2, num_labels]
        adv_scores = actions.transpose(2, 1).unsqueeze(1).mul(ws).sum(dim=2) + self.bias[None,:,None]
        adv_scores = adv_scores.permute(0, 2, 1)
        assert adv_scores.size() == torch.Size((batch, seq2, num_labels))
        
        # Q value - [batch, seq2, num_labels]
        if self.dueling:
            q_value = val_scores + adv_scores - adv_scores.mean(dim=(2, 1), keepdim=True)
        else:
            q_value = adv_scores
        assert q_value.size() == torch.Size((batch, seq2, num_labels))
        
        # 去除padding的action对应的score
        no_pad = actions_mask.view(-1).nonzero().view(-1)
        q_value = q_value.reshape(-1, num_labels)[no_pad]
        return (q_value,)


class TransformerDQN(BaseDQN):
    def __init__(self, args):
        super(TransformerDQN, self).__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model_type = args.model_name_or_path.lower().split('/')[-1]
        HIDDEN_SIZE = {
            'bert-base-uncased': 768,
            'bert-base-cased': 768,
            'albert-large-v2': 1024
        }
        # q network
        self.q_net = QNetwork(
            hidden_size=HIDDEN_SIZE[model_type],
            num_labels=args.num_labels,
            dueling=args.dueling,
            nhead=args.nhead,
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
                                               lr_lambda=lambda epoch: max(np.power(0.5, epoch // 500), 2e-6 / args.learning_rate))
        #self.optimizer = Adam(self.q_net.parameters(), lr=args.learning_rate)


    def convert_to_inputs_for_select_action(self, batch_state: List[State], batch_actions: List[List[Action]]) -> List[dict]:
        assert len(batch_state) == len(batch_actions)
        batch_state_tensor, batch_actions_tensor = [], []
        for state, actions in zip(batch_state, batch_actions):
            # tokens here is actually the sentence embedding
            ## [seq, dim]
            state_tensor = torch.tensor([state.claim.tokens] + [sent.tokens for sent in state.candidate],
                                        dtype=torch.float)
            ## [seq2, dim]
            actions_tensor = torch.tensor([action.sentence.tokens for action in actions],
                                          dtype=torch.float)
            batch_state_tensor.append(state_tensor)
            batch_actions_tensor.append(actions_tensor)
        return convert_tensor_to_lstm_inputs(batch_state_tensor, batch_actions_tensor)
    

    def convert_to_inputs_for_update(self, states: List[State], actions: List[Action]) -> dict:
        assert len(states) == len(actions)
        batch_state_tensor, batch_actions_tensor = [], []
        for state, action in zip(states, actions):
            ## [seq, dim]
            state_tensor = torch.tensor([state.claim.tokens] + [sent.tokens for sent in state.candidate],
                                        dtype=torch.float)
            ## [seq2, dim]
            actions_tensor = torch.tensor([action.sentence.tokens], dtype=torch.float)
            batch_state_tensor.append(state_tensor)
            batch_actions_tensor.append(actions_tensor)
        return convert_tensor_to_lstm_inputs(batch_state_tensor, batch_actions_tensor, self.device)

