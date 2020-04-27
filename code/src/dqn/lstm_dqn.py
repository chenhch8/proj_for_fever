#!/usr/bin/env python3
# coding=utf-8
from tqdm import tqdm, trange
from functools import reduce
from typing import List, Tuple
import pdb

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD
#from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
#from torch.utils.data.distributed import DistributedSampler

from .base_dqn import BaseDQN
from data.structure import *

def lstm_load_and_process_data(args: dict, filename: str, token_fn: 'function', is_eval=False) \
        -> DataSet:
    from transformers import (
        AlbertConfig,
        AlbertModel,
        AlbertTokenizer,
        BertConfig,
        BertModel,
        BertTokenizer,
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
        "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    }
    
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    def feature_extractor(texts: List[str]) -> List[List[float]]:
        pass
    
    cached_file = os.path.join(
        '/'.join(filename.split('/')[:-1]),
        'cached_{}_{}_{}_lstm.pk'.format('train' if filename.find('train') != -1 else 'dev',
                                    list(filter(None, args.model_name_or_path.split('/'))).pop())
    )
    data = None
    if not os.path.exists(cached_file):
        args.logger.info(f'Loading and processing data from {filename}')
        data = []
        skip, count = 0, 0
        with open(filename, 'rb') as fr:
            for line in tqdm(fr.readlines()):
                instance = json.loads(line.decode('utf-8').strip())
                
                total_texts = [sentence for _, text in instance['documents'].items() \
                                            for _, sentence in text.items()]
                if not is_eval and totel_texts < 5:
                    skip += 1
                    continue
                count += 1

                total_texts = [instance['claim']] + total_texts
                texts_embedding = feature_extractor(total_texts)
                
                claim = Claim(id=instance['id'],
                              str=instance['claim'],
                              tokens=token_fn(instance['claim'], max_length=args.max_sent_length))
                sent2id = {}
                sentences = []
                for title, text in instance['documents'].items():
                    for line_num, sentence in text.items():
                        sentences.append(Sentence(id=(title, int(line_num)),
                                                  str=sentence,
                                                  tokens=token_fn(sentence, max_length=args.max_sent_length)))
                        sent2id[(title, int(line_num))] = len(sentences) - 1
                
                evidence_set = [[sentences[sent2id[(title, int(line_num))]] \
                                    for title, line_num in evi] \
                                        for evi in instance['evidence_set']] \
                                if not is_eval else instance['evidence_set']
                data.append((claim, args.label2id[instance['label']], evidence_set, sentences))
            with open(cached_file, 'wb') as fw:
                pickle.dump(data, fw)
        args.logger.info(f'skip: {skip}({skip / count})')
    else:
        args.logger.info(f'Loading data from {cached_file}')
        with open(cached_file, 'rb') as fr:
            data = pickle.load(fr)
    return data


def convert_tensor_to_lstm_inputs(batch_state_tensor: List[torch.Tensor],
                                  batch_actions_tensor: List[torch.Tensor],
                                  device=None) -> dict:
    device = device if device != None else torch.device('cpu')

    state_len = [state.size(0) for state in batch_state_tensor]
    actions_len = [action.size(0) for action in batch_actions_tensor]
    s_max, a_max = max(state_len), max(actions_len)

    state_pad = pad_sequence(batch_state_tensor, batch_first=True)
    actions_pad = pad_sequence(batch_actions_tensor, batch_first=True)

    state_mask = torch.torch([[1] * size + [0] * (s_max - size) for size in state_len],
                             dtype=torch.bool)
    actions_mask = torch.tensor([[1] * size + [0] * (a_max - size) for size in actions_len],
                                dtype=torch.bool)
    return {
        'states': state_pad.to(device),
        'actions': actions_pad.to(device),
        'state_mask': state_mask.to(device),
        'actions_mask': actions_mask.to(device)
    }


class QNetwork(nn.Module):
    def __init__(self,
                 input_size,
                 num_labels,
                 hidden_size=None,
                 dropout=0.1,
                 bidirectional=True,
                 num_layers=3):
        super(QNetwork, self).__init__()
        if hidden_size is None:
            hidden_size = input_size
        num_hidden_state = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            dropout=dropout,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            bias=True)
        # attention paramters
        self.a_weight = Parameter(torch.Tensor(input_size, hidden_size * num_hidden_state))
        # q value parameters
        self.q_weight = Parameter(torch.Tensor(num_labels,
                                               hidden_size,
                                               hidden_size * num_hidden_state))
        self.q_bias = Parameter(torch.Tensor(num_layers))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.q_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.a_weight, a=math.sqrt(5))
        if self.q_bias is not None:
            feature_in = self.q_weight.size(2)
            bound = 1 / math.sqrt(feature_in)
            init.uniform_(self.q_bias, -bound, bound)

    def attention_aggregate(self, query, key, value, mask):
        '''
        query: [batch, hidden_size]
        key: [batch, seq, hidden_size * num_hidden_state]
        value: [batch, seq, hidden_size * num_hidden_state]
        mask: [batch, seq]

        return:
            [batch, 1, hidden_size]
        '''
        batch, seq = query.size(0), key.size(1)
        # [batch, seq]
        A = query.unsqueeze(2)
            .matmul(self.a_weight.matmul(key.transpose(2, 1)))
            .squeeze()
            .masked_fill(torch.logical_not(mask), float('-inf'))
            .softmax(dim=1)
        assert A.size() == torch.Size((batch, seq))
        return A.unsqueeze(2).mm(value).sum(dim=1, keepdim=True)
        

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
        assert out.size() == torch.Size((batch, seq, 2 * hidden_size))
        # [batch, 1, hidden_size]
        states_embedding = self.attention_aggregate(states[:, 0], out, out, state_mask)
        assert out.size() == torch.Size((batch, 1, hidden_size))
        # [batch, num_labels, hidden_size, 1]
        ws = self.q_weight.unsqueeze(0).matmul(states_embedding.unsqueeze(1).transpose(3, 2))
        assert ws.size() == torch.Size(batch, num_labels, hidden_size, 1)
        # [batch, seq2, nums_labels]
        logits = actions.unsqueeze(1).matmul(ws).squeeze().transpose(2, 1) + self.bias[None, None, :]
        assert logits.size() == torch.Size(batch, seq2, num_labels)
        # 去除padding的action对应的score
        logits = logits.reshape(-1, num_labels)[actions_mask.view(-1)]
        return (logits,)


class LstmDQN(BaseDQN):
    def __init__(self, args):
        super(LstmDQN, self).__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.model_type = args.model_type.lower()
        HIDDEN_SIZE = {
            'bert-base-uncased': 768,
            'bert-base-cased': 768,
            'albert-large-v2': 1024
        }
        # q network
        self.q_net = QNetwork(
             input_size=HIDDEN_SIZE[args.model_type.lower()],
             hidden_size=HIDDEN_SIZE[args.model_type.lower()]
             num_labels=args.num_labels,
        )
        # Target network
        self.t_net = deepcopy(self.q_net) if args.do_train else self.q_net
        self.q_net.zero_grad()

        self.set_network_untrainable(self.sent_encoder)
        self.set_network_untrainable(self.t_net)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        
        self.optimizer = SGD(self.q_net.parameters(), lr=args.learning_rate, momentum=0.9)



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

