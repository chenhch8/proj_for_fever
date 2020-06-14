#!/usr/bin/env python3
# coding=utf-8
from tqdm import tqdm, trange
from functools import reduce
from typing import List, Tuple
from copy import deepcopy
import os
import json
import pickle
import pdb

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam, AdamW
#from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
#from torch.utils.data.distributed import DistributedSampler

from .base_dqn import BaseDQN
from data.structure import *
from data.dataset import FeverDataset

from transformers import (
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    #XLNetModel,
    XLNetForSequenceClassification
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
    #"xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
}

def initilize_bert(args):
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_labels
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.to(args.device)

    def model_output(**params):
        if args.model_type in {'bert', 'albert'}:
            return model(**params)[1]
        elif args.model_type in {'xlnet'}:
            output = model.transformer(**params)[0]
            return model.sequence_summary(output)
    
    def feature_extractor(texts: List[str]) -> List[List[float]]:
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if args.model_type in ['xlnet'] else 0
        pad_on_left = bool(args.model_type in ['xlnet'])
        
        texts = [texts[0]] + [[texts[0], text] for text in texts[1:]]
        inputs = tokenizer.batch_encode_plus(texts, max_length=256)
        # padding
        max_length = max([len(input_ids) for input_ids in inputs['input_ids']])
        if pad_on_left:
            inputs['input_ids'] = torch.tensor(
                [[pad_token] * (max_length - len(input_ids)) + input_ids for input_ids in inputs['input_ids']],
                dtype=torch.long
            )
            inputs['attention_mask'] = torch.tensor(
                [[0] * (max_length - len(mask)) + mask for mask in inputs['attention_mask']],
                dtype=torch.long
            )
            inputs['token_type_ids'] = torch.tensor(
                [[pad_token_segment_id] * (max_length - len(token_type)) + token_type \
                 for token_type in inputs['token_type_ids']],
                dtype=torch.long
            )
        else:
            inputs['input_ids'] = torch.tensor(
                [input_ids + [pad_token] * (max_length - len(input_ids)) for input_ids in inputs['input_ids']],
                dtype=torch.long
            )
            inputs['attention_mask'] = torch.tensor(
                [mask + [0] * (max_length - len(mask)) for mask in inputs['attention_mask']],
                dtype=torch.long
            )
            inputs['token_type_ids'] = torch.tensor(
                [token_type + [pad_token_segment_id] * (max_length - len(token_type)) \
                 for token_type in inputs['token_type_ids']],
                dtype=torch.long
            )
        
        with torch.no_grad():
            INTERVEL = 64
            outputs = [model_output(
                            **dict(map(lambda x: (x[0], x[1][i:i + INTERVEL].to(args.device)),
                                       inputs.items()))
                        ) for i in range(0, inputs['input_ids'].size(0), INTERVEL)]
            outputs = torch.cat(outputs, dim=0)
            assert outputs.size(0) == inputs['input_ids'].size(0)
        return outputs.detach().cpu().numpy().tolist()
    
    return feature_extractor

def lstm_load_and_process_data(args: dict, filename: str, token_fn: 'function', fake_evi: bool=False) \
        -> DataSet:
    if filename.find('train') != -1:
        mode = 'train'
    elif filename.find('dev') != -1:
        mode = 'dev'
    else:
        mode = 'test'
    cached_file = os.path.join(
        '/'.join(filename.split('/')[:-1]),
        'cached_{}_{}_v5+6'.format(
            mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop()
        )
    )
    
    data = None
    if not os.path.exists(cached_file):
        feature_extractor = initilize_bert(args)

        os.makedirs(cached_file, exist_ok=True)
        
        args.logger.info(f'Loading and processing data from {filename}')
        data = []
        skip, count, num = 0, 0, 0
        with open(filename, 'rb') as fr:
            for line in tqdm(fr.readlines()):
                instance = json.loads(line.decode('utf-8').strip())
                
                total_texts = [sentence for _, text in instance['documents'].items() \
                                            for _, sentence in text.items()]
                if mode == 'train' and len(total_texts) < 5:
                    skip += 1
                    continue
                count += 1
                
                total_texts = [instance['claim']] + total_texts
                semantic_embedding = feature_extractor(total_texts)
                
                claim = Claim(id=instance['id'],
                              str=instance['claim'],
                              tokens=semantic_embedding[0])
                sent2id = {}
                sentences = []
                text_id = 1
                for title, text in instance['documents'].items():
                    for line_num, sentence in text.items():
                        sentences.append(Sentence(id=(title, int(line_num)),
                                                  str=sentence,
                                                  tokens=semantic_embedding[text_id]))
                        sent2id[(title, int(line_num))] = len(sentences) - 1
                        text_id += 1
                assert text_id == len(semantic_embedding)
                
                if mode == 'train':
                    label = args.label2id[instance['label']]
                    evidence_set = [[sentences[sent2id[(title, int(line_num))]] \
                                        for title, line_num in evi] \
                                            for evi in instance['evidence_set']]
                    if not fake_evi and instance['label'] == 'NOT ENOUGH INFO':
                        evidence_set = []
                elif mode == 'dev':
                    label = args.label2id[instance['label']]
                    evidence_set = instance['evidence_set']
                else:
                    label = evidence_set = None
                data.append((claim, label, evidence_set, sentences))
                
                if count % 1000 == 0:
                    for item in data:
                        with open(os.path.join(cached_file, f'{num}.pk'), 'wb') as fw:
                            pickle.dump(item, fw)
                        num += 1
                    del data
                    data = []

            for item in data:
                with open(os.path.join(cached_file, f'{num}.pk'), 'wb') as fw:
                    pickle.dump(item, fw)
                num += 1
            del data
        args.logger.info(f'Process Done. Skip: {skip}({skip / count})')

    dataset = FeverDataset(cached_file, label2id=args.label2id)
    return dataset


def convert_tensor_to_lstm_inputs(batch_state_tensor: List[torch.Tensor],
                                  batch_actions_tensor: List[torch.Tensor],
                                  device=None) -> dict:
    device = device if device != None else torch.device('cpu')

    state_len = [state.size(0) for state in batch_state_tensor]
    actions_len = [action.size(0) for action in batch_actions_tensor]
    s_max, a_max = max(state_len), max(actions_len)

    state_pad = pad_sequence(batch_state_tensor, batch_first=True)
    actions_pad = pad_sequence(batch_actions_tensor, batch_first=True)

    state_mask = torch.tensor([[1] * size + [0] * (s_max - size) for size in state_len],
                             dtype=torch.float)
    actions_mask = torch.tensor([[1] * size + [0] * (a_max - size) for size in actions_len],
                                dtype=torch.float)
    return {
        'states': state_pad.to(device),
        'actions': actions_pad.to(device),
        'state_mask': state_mask.to(device),
        'actions_mask': actions_mask.to(device)
    }


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
            num = mask.sum(dim=1).view(-1, 1, 1).expand(-1, 1, hidden_size)
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


class LstmDQN(BaseDQN):
    def __init__(self, args):
        super(LstmDQN, self).__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model_type = args.model_type.lower()
        config_class, _, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.model_name_or_path)
        
        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case,
        )
        # q network
        self.q_net = QNetwork(
            args,
            hidden_size=config.hidden_size,
        )
        # Target network
        self.t_net = deepcopy(self.q_net) if args.do_train else self.q_net
        self.q_net.zero_grad()

        self.set_network_untrainable(self.t_net)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        
        self.optimizer = AdamW(self.q_net.parameters(), lr=args.learning_rate)
        #self.optimizer = Adam(self.q_net.parameters(), lr=args.learning_rate)

    def token(self, text_sequence: str, max_length: int=None) -> Tuple[int]:
        return tuple(self.tokenizer.encode(text_sequence,
                                           add_special_tokens=False,
                                           max_length=max_length))
    
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

