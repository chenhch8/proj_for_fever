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
from torch.optim import SGD, Adam, AdamW
#from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
#from torch.utils.data.distributed import DistributedSampler

from .base_dqn import BaseDQN
from .lstm_dqn import lstm_load_and_process_data
from data.structure import *

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
    XLNetForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)


ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            AlbertConfig,
            RobertaConfig,
        )
    ),
    (),
)


MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    #"xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
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
        elif args.model_type in {'roberta'}:
            output = model.roberta(**params)[0]
            output = output[:, 0, :]  # take <s> token (equiv. to [CLS])
            return output
    
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
            ) if args.model_type in ['bert', 'xlnet', 'albert'] else None
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
            ) if args.model_type in ['bert', 'xlnet', 'albert'] else None
        
        with torch.no_grad():
            INTERVEL = 64
            outputs = [model_output(
                            **dict(map(lambda x: (x[0], x[1][i:i + INTERVEL].to(args.device) if x[1] is not None else x[1]),
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
        if args.do_fever2:
            mode = f'fever2_{mode}'
    else:
        mode = 'test'
    cached_file = os.path.join(
        '/'.join(filename.split('/')[:-1]),
        'cached_{}_{}_v5+6'.format(
            mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop()
        )
    )
    if args.do_fever2:
        cached_file += '.pk'
    
    data = None
    if not os.path.exists(cached_file):
        feature_extractor = initilize_bert(args)

        if not args.do_fever2:
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
                elif mode.find('dev') != -1:
                    label = args.label2id[instance['label']]
                    evidence_set = instance['evidence_set']
                else:
                    label = evidence_set = None
                data.append((claim, label, evidence_set, sentences))
                
                if count % 1000 == 0 and not args.do_fever2:
                    for item in data:
                        with open(os.path.join(cached_file, f'{num}.pk'), 'wb') as fw:
                            pickle.dump(item, fw)
                        num += 1
                    del data
                    data = []

            if not args.do_fever2:
                for item in data:
                    with open(os.path.join(cached_file, f'{num}.pk'), 'wb') as fw:
                        pickle.dump(item, fw)
                    num += 1
                del data
            else:
                with open(cached_file, 'wb') as fw:
                    pickle.dump(data, fw)
                del data
        args.logger.info(f'Process Done. Skip: {skip}({skip / count})')

    dataset = FeverDataset(cached_file, label2id=args.label2id)
    return dataset


def convert_tensor_to_lstm_inputs(batch_claims: List[List[float]],
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
    
    def forward(self, query, key, value, q_mask, k_mask):
        '''
        query: [B, L_q, D_q]
        key: [B, L_k, D_k]
        value: [B, L_v, D_v]
        q_mask: [B, L_q]
        k_mask: [B, L_k]
        '''
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
        return output


class QNetwork(nn.Module):
    def __init__(self, num_labels, num_layers=3, nheads=8, dropout=0.1, hidden_size=768):
        super(QNetwork, self).__init__()
        
        self.num_hidden_state = 2
        self.num_labels = num_labels

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            bias=True
        )
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(2 * hidden_size, hidden_size)

        self.transformer = Transformer(
            dim=hidden_size,
            nheads=nheads,
            dropout=dropout
        )

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
        num_labels = self.num_labels
        
        evidences, _ = self.lstm(evidences)
        evidences = self.fc(self.tanh(evidences))
        assert evidences.size() == torch.Size((batch, seq, hidden_size))
        assert claims.size() == torch.Size((batch, hidden_size))
        
        output = self.transformer(
            query = claims.unsqueeze(1),
            key=evidences,
            value=evidences,
            q_mask=torch.ones([batch, 1], dtype=torch.float).to(evidences_mask),
            k_mask=evidences_mask
        )
        assert output.size() == torch.Size((batch, 1, hidden_size))
        
        q_value = self.mlp(torch.cat([claims, output.squeeze(1)], dim=-1))
        assert q_value.size() == torch.Size((batch, 3))
        #pdb.set_trace()

        return (q_value,)


class LstmDQN(BaseDQN):
    def __init__(self, args):
        super(TransformerDQN, self).__init__(args)
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        config_class = CONFIG_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.model_name_or_path)
        # q network
        self.q_net = QNetwork(
            num_layers=args.num_layers
            hidden_size=config.hidden_size,
            num_labels=args.num_labels,
            nheads=args.nhead,
        )
        # Target network
        self.t_net = deepcopy(self.q_net) if args.do_train else self.q_net
        self.q_net.zero_grad()

        self.set_network_untrainable(self.t_net)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        
        #self.optimizer = SGD(self.q_net.parameters(), lr=args.learning_rate, momentum=0.9)
        self.optimizer = AdamW(self.q_net.parameters(), lr=args.learning_rate)
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

