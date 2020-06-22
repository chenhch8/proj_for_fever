#!/usr/bin/env python
# coding=utf-8
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np
import os
import pdb

from transformers import (
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    XLNetModel,
    #XLNetForSequenceclassify
)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            AlbertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    #"xlnet": (XLNetConfig, XLNetForSequenceclassify, XLNetTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
}


class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(AttentionLayer, self).__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, query, key, value, q_mask, k_mask):
        '''
        query: [batch, seq1, hidden_size]
        key: [batch, seq2, hidden_size]
        value: [batch, seq2, hidden_size]
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


class QNetwork(nn.Module):
    def __init__(self, args, dropout=0.1, hidden_size=768):
        super(QNetwork, self).__init__()
        
        self.num_hidden_state = 2
        self.dueling = args.dueling
        self.aggregate = args.aggregate

        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            dropout=dropout,
                            num_layers=args.num_layers,
                            batch_first=True,
                            bidirectional=True,
                            bias=True)
        self.tanh = nn.Tanh()

        if self.aggregate.find('attn') != -1:
            # attention paramters
            self.attn_layer = AttentionLayer(
                hidden_size + hidden_size * self.num_hidden_state,
                hidden_size
            )
        # Value
        if self.dueling:
            self.val_score_layer = nn.Linear(
                hidden_size * self.num_hidden_state,
                1
            )
        # Advantage
        self.adv_score_layer = ScoreLayer(
            left_dim=hidden_size,
            right_dim=hidden_size * self.num_hidden_state,
            num_labels=args.num_labels
        )
    
    def calc_state_semantic(self, features, mask, mode):
        batch, _, hidden_size = features.size()
        if mode == 'mean':
            num = mask.sum(dim=1).clamp(min=1).view(-1, 1, 1).expand(-1, 1, hidden_size)
            state_semantic = features.sum(dim=1, keepdim=True).div(num)
            assert num.size() == torch.Size((batch, 1, hidden_size))
        elif mode == 'max':
            state_semantic = features \
                                .masked_fill(mask.unsqueeze(2) == 0,
                                             float('-inf')) \
                                .max(dim=1, keepdim=True)[0]
        else:
            raise ValueError('mode error')
        return state_semantic

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
        # [batch, seq, hidden_size * num_hidden_state]
        out, _ = self.lstm(states)
        out = self.tanh(out)
        assert out.size() == torch.Size((batch, seq, self.num_hidden_state * hidden_size))
        
        if self.aggregate.find('attn') != -1:
            # [batch, seq2, hidden_size * num_hidden_state]
            states_feat = self.attn_layer(
                query=actions,
                key=out,
                value=out,
                q_mask=actions_mask,
                k_mask=state_mask
            )
            state_semantic = self.calc_state_semantic(
                states_feat, actions_mask,
                'mean' if self.aggregate.find('mean') != -1 else 'max'
            )
        else:
            if self.aggregate in {'max', 'mean'}:
                state_semantic = self.calc_state_semantic(out, state_mask, self.aggregate)
            elif self.aggregate == 'last':
                last_step = state_mask.sum(dim=1) \
                                .sub(1) \
                                .type(torch.long) \
                                .view(-1, 1, 1) \
                                .expand(-1, -1, 2 * hidden_size)
                state_semantic = out.gather(dim=1, index=last_step)
            states_feat = state_semantic.expand(-1, seq2, -1)
        assert state_semantic.size() == torch.Size((batch, 1, self.num_hidden_state * hidden_size))
        assert states_feat.size() == torch.Size((batch, seq2, self.num_hidden_state * hidden_size))
        
        # Value - [batch, seq2, num_labels]
        if self.dueling:
            val_scores = self.val_score_layer(state_semantic)
            assert val_scores.size() == torch.Size((batch, 1, 1))

            val_scores = val_scores.expand(-1, seq2, num_labels)
            assert val_scores.size() == torch.Size((batch, seq2, num_labels))
        
        # Advantage - [batch, seq2, num_labels]
        adv_scores = self.adv_score_layer(left_inputs=actions, right_inputs=states_feat)
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


class AutoBertModel(nn.Module):
    def __init__(self, model_name_or_path, model_type, num_labels=3, config=None):
        super(AutoBertModel, self).__init__()

        self.model_type = model_type

        config_class, model_class, _ = MODEL_CLASSES[model_type]
        if config is None:
            config = config_class.from_pretrained(model_name_or_path)
        config.num_labels = num_labels
        
        self.encoder = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
        )
        self.fc_1 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.fc_2 = nn.Sequential(
            nn.Linear(4 * config.hidden_size, config.hidden_size, bias=True),
            nn.ReLU(True)
        )
        self.classify = nn.Sequential(
            nn.Linear(5 * config.hidden_size, num_labels, bias=True),
            nn.Dropout(0.1),
            torch.nn.Softmax(dim=1)
        )
        self.attn_1 = AttentionLayer(2 * config.hidden_size)
        self.attn_2 = AttentionLayer(2 * config.hidden_size)
    
    def init_parameters(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data, a=np.sqrt(5))
            feature_in = module.weight.size(2)
            bound = 1 / np.sqrt(feature_in)
            nn.init.uniform_(module.bias.data, -bound, bound)

    def from_pretrained(self, path):
        print(f'loading checkpoint from {path}')
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'))
        self.load_state_dict(state_dict)

    def save_pretrained(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(path, 'pytorch_model.bin'))

    def coarse_fine_grain_layer(self, premise, hypothesis, p_mask, h_mask):
        '''
        premise/hypothesis: [batch, N, d]
        p_mask/h_mask: [batch, N]
        '''
        #def _mask_mean_(features, mask):
        #    return features.sum(dim=1).div(mask.sum(dim=1).clamp(min=1).view(-1, 1))
        def _mask_max_(features, mask):
            return features.masked_fill(mask.unsqueeze(2) == 0, float('-inf')).max(dim=1)[0]
        
        # (batch, N, d)
        pre_attn = self.attn_1(query=premise,
                               key=hypothesis,
                               value=hypothesis,
                               q_mask=p_mask,
                               k_mask=h_mask)
        hyp_attn = self.attn_1(query=hypothesis,
                               key=premise,
                               value=premise,
                               q_mask=h_mask,
                               k_mask=p_mask)
        # (batch, N, d)
        pre_aware = self.fc_2(torch.cat([premise, pre_attn, premise - pre_attn, premise * pre_attn], dim=-1))
        hyp_aware = self.fc_2(torch.cat([hypothesis, hyp_attn, hypothesis - hyp_attn, hypothesis * hyp_attn], dim=-1))
        # (batch, N, d)
        pre_aware = self.attn_2(query=pre_aware,
                                key=pre_aware,
                                value=pre_aware,
                                q_mask=p_mask,
                                k_mask=p_mask)
        hyp_aware = self.attn_2(query=hyp_aware,
                                key=hyp_aware,
                                value=hyp_aware,
                                q_mask=h_mask,
                                k_mask=h_mask)
        # (batch, d)
        pre_out = _mask_max_(pre_aware, p_mask) # (batch, d)
        hyp_out = _mask_max_(hyp_aware, h_mask)
        
        return torch.cat([pre_out, hyp_out, torch.abs(pre_out - hyp_out), pre_out * hyp_out], dim=-1)

            
    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        '''
        input_ids: [batch, N]
        token_type_ids: [batch, N]
        attention_mask: [batch, N]
        '''
        outputs = self.encoder(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)
        hidden_states = outputs[0]
        batch, N, d = hidden_states.size()
        
        p_mask = token_type_ids.float()  # evidence, (batch, N)
        h_mask = (1 - p_mask) * attention_mask.float()  # claim, (batch, N)
        if self.model_type.find('xlnet') != -1:
            p_mask[:, -1] = 0  # 去除 [coarse_features]
            coarse_features = self.fc_1(hidden_states[:, -1])
        else:
            h_mask[:, 0] = 0  # 去除 [coarse_features]
            coarse_features = self.fc_1(hidden_states[:, 0])
        assert coarse_features.size() == torch.Size((batch, d))

        # Premise->evidence sentence
        # Hypothesis->claim
        premise = hidden_states * p_mask.unsqueeze(2)  # evidence, (batch, N, d)
        hypothesis = hidden_states * h_mask.unsqueeze(2) # claim, (batch, N, d)

        fine_grain_feaures = self.coarse_fine_grain_layer(premise=premise,
                                                          hypothesis=hypothesis,
                                                          p_mask=p_mask,
                                                          h_mask=h_mask)
        assert fine_grain_feaures.size() == torch.Size((batch, 4 * d))

        # (batch, 5d)
        coarse_fine_features = torch.cat([coarse_features, fine_grain_feaures], dim=1)

        scores_clc = self.classify(coarse_fine_features)
        outputs = (scores_clc, coarse_fine_features)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(scores_clc, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
