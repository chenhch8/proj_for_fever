#!/usr/bin/env python
# coding=utf-8
from data_structure import *
from collections import Counter, defaultdict
from typing import Tuple, List
import numpy as np
import pickle
from pprint import pprint

DataList = List[Tuple[Claim, int, EvidenceSet, List[Sentence]]]

def load_data(filename: str) -> DataList:
    print(f'Loading data from {cache_file}')
    with open(filename, 'rb') as fr:
        data = pickle.load(fr)
    return data

def statics(data: DataList) -> None:
    sentences_size = []
    tokens_size = []
    for _, label, evidence_set, sentences in data:
        if len(sentences) == 0:
            print(f'label: {label}; evidence_set: {evidence_set}')
        sentences_size.append(len(sentences))
        tokens_size.extend([len(sent.tokens) for sent in sentences])
    sents_count = Counter(sentences_size)
    tokens_count = Counter(tokens_size)
    sents_np = np.asarray(sentences_size)
    tokens_np = np.asarray(tokens_np)s
    print(f'sentences: {sents_np.min()}-{sents_np.max()}, {sents_np.mean()}')
    print(f'tokens: {tokens_np.min()}-{tokens_np.max()}, {tokens_np.mean()}')
    print(f'sents_count: {sents_count[sents_np.min()]}-{sents_count[sents_np.max()]}')
    print(f'tokens_count: {tokens_count[tokens_np.min()]}-{tokens_count[tokens_np.max()]}')
    pprint(sents_count)
    pprint(tokens_count)

if __name__ == '__main__':
    statics(load_data('./data/dqn/cache_train.pk'))
