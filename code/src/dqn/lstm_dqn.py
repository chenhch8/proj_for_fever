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
from torch.optim import Adam, AdamW

from transformers import (
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    #DistilBertConfig,
    #DistilBertForSequenceClassification,
    #DistilBertTokenizer,
    #FlaubertConfig,
    #FlaubertForSequenceClassification,
    #FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    #XLMConfig,
    #XLMForSequenceClassification,
    #XLMRobertaConfig,
    #XLMRobertaForSequenceClassification,
    #XLMRobertaTokenizer,
    #XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    #get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    #"xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    #"distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    #"xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    #"flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}

from .base_dqn import BaseDQN
from data.structure import *
from data.dataset import FeverDataset
from models import QNetwork, AutoBertModel 


def initilize_bert(args):
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = AutoBertModel(
        args.model_name_or_path,
        args.model_type,
        args.num_labels,
        config=config
    )
    model.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    def feature_extractor(texts: List[str]) -> List[List[float]]:
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if args.model_type in ['xlnet'] else 0
        pad_on_left = bool(args.model_type in ['xlnet'])
        
        texts = [texts[0]] + [[texts[0], text] for text in texts[1:]]
        inputs = tokenizer.batch_encode_plus(texts, max_length=128)
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
            INTERVEL = 32
            outputs = [model(
                            **dict(map(lambda x: (x[0], x[1][i:i + INTERVEL].to(args.device)),
                                       inputs.items()))
                        )[1] for i in range(0, inputs['input_ids'].size(0), INTERVEL)]
            outputs = torch.cat(outputs, dim=0)
            outputs[0, config.hidden_size:] = 0  # 去除 claim 的 fine-grain
            #if torch.isinf(outputs).sum() or torch.isnan(outputs).sum():
            #    pdb.set_trace()
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
        'cached_{}_{}_v5+6.2'.format(
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


class LstmDQN(BaseDQN):
    def __init__(self, args):
        super(LstmDQN, self).__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model_type = args.model_type.lower()
        #config_class, _, tokenizer_class = MODEL_CLASSES[args.model_type]
        config_class, _, _ = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.model_name_or_path)
        
        #self.tokenizer = tokenizer_class.from_pretrained(
        #    args.model_name_or_path,
        #    do_lower_case=args.do_lower_case,
        #)
        # q network
        self.q_net = QNetwork(
            args,
            hidden_size=5 * config.hidden_size,
        )
        # Target network
        self.t_net = deepcopy(self.q_net) if args.do_train else self.q_net
        self.q_net.zero_grad()

        self.set_network_untrainable(self.t_net)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        
        self.optimizer = AdamW(self.q_net.parameters(), lr=args.learning_rate)
        #self.optimizer = Adam(self.q_net.parameters(), lr=args.learning_rate)

    #def token(self, text_sequence: str, max_length: int=None) -> Tuple[int]:
    #    return tuple(self.tokenizer.encode(text_sequence,
    #                                       add_special_tokens=False,
    #                                       max_length=max_length))
    
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

