#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import torch
#from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
#from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from functools import reduce
from typing import List
import pdb

from .base_dqn import BaseDQN
from data.structure import Action, State, Transition

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


def convert_tokens_to_bert_inputs(all_tokens_a: List[int],
                                  all_tokens_b: List[int],
                                  max_seq_len: int, 
                                  CLS: int, SEP: int, device=None) -> dict:
    device = torch.device('cpu') if device is None else device
    
    all_inputs_ids = []
    all_inputs_mask = []
    all_segment_ids = []
    for tokens_a, tokens_b in zip(all_tokens_a, all_tokens_b):
        b_len = max_seq_len - 3 - len(tokens_a)
        assert b_len > 0
        inputs_ids = [CLS] + tokens_a + [SEP] + tokens_b[:b_len] + [SEP]
        inputs_mask = [1] * len(inputs_ids)
        segment_ids = [0] * (2 + len(tokens_a)) + [1] * (len(inputs_ids) - len(tokens_a) - 2)
        assert len(inputs_ids) == len(inputs_mask) == len(segment_ids)
        
        padding = [0] * (max_seq_len - len(inputs_ids))
        all_inputs_ids.append(inputs_ids + padding)
        all_inputs_mask.append(inputs_mask + padding)
        all_segment_ids.append(segment_ids + padding)

    return {
        'input_ids': torch.tensor(all_inputs_ids, dtype=torch.long, device=device),
        'attention_mask': torch.tensor(all_inputs_mask, dtype=torch.long, device=device),
        'token_type_ids': torch.tensor(all_segment_ids, dtype=torch.long, device=device)
    }


class BertDQN(BaseDQN):
    def __init__(self, args):
        super(BertDQN, self).__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=args.num_labels,
            finetuning_task=args.task_name,
            #cache_dir=args.cache_dir if args.cache_dir else None,
        )
        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            #cache_dir=args.cache_dir if args.cache_dir else None,
        )
        # Q network
        self.q_net = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            #cache_dir=args.cache_dir if args.cache_dir else None,
        )
        # Target network
        self.t_net = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            #cache_dir=args.cache_dir if args.cache_dir else None,
        ) if args.do_train else self.q_net
        self.q_net.zero_grad()
        self.set_network_untrainable(self.t_net)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.q_net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        #self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    def token(self, text_sequence: str) -> List[int]:
        return self.tokenizer.encode(text_sequence)
    

    def convert_to_inputs_for_select_action(self, state: State, actions: List[Action]) -> List[dict]:
        condidate = reduce(lambda seq1, seq2: seq1 + seq2,
                           map(lambda sent: sent.tokens, state.candidate)) if len(state.candidate) else []
        length = self.max_seq_length - 3 - len(state.claim.tokens) - len(condidate)
        if length <= 0:
            self.logger.info(f'claim: {len(state.claim.tokens)}; condidate: {len(condidate)}; length: {length}')
        assert length > 0
        all_tokens_a = [state.claim.tokens] * len(actions)
        all_tokens_b = [condidate + action.sentence.tokens[:length] for action in actions]
        #max_seq_len = max([len(tokens) for tokens in all_tokens_b]) + len(state.claim.tokens) + 3

        CLS, SEP = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        
        interval = 10 * self.args.train_batch_size
        inputs = [convert_tokens_to_bert_inputs(all_tokens_a[i:i + interval],
                                                all_tokens_b[i:i + interval],
                                                self.max_seq_length, CLS, SEP) \
                  for i in range(0, len(all_tokens_b), interval)]

        return inputs
    

    def convert_to_inputs_for_update(self, states: List[State], actions: List[Action]) -> dict:
        assert len(states) == len(actions)
        all_tokens_a, all_tokens_b = [], []
        #pdb.set_trace()
        for state, action in zip(states, actions):
            tokens_a = state.claim.tokens
            condidate = reduce(lambda seq1, seq2: seq1 + seq2,
                               map(lambda sent: sent.tokens, state.candidate)) \
                            if len(state.candidate) else []
            tokens_b = condidate + action.sentence.tokens
            all_tokens_a.append(tokens_a)
            all_tokens_b.append(tokens_b)
        #max_seq_len = min(max([len(tokens_a) + len(tokens_b) + 3 \
        #                        for tokens_a, tokens_b in zip(all_tokens_a, all_tokens_b)]),
        #                  self.max_seq_length)
        
        CLS, SEP = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        
        return convert_tokens_to_bert_inputs(all_tokens_a, all_tokens_b,
                                             self.max_seq_length, CLS, SEP, self.device)

