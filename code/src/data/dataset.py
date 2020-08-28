#!/usr/bin/env python
# coding=utf-8
import os
import pickle
import random
from typing import List

from torch.utils.data import Dataset

from .structure import *

class FeverDataset(Dataset):
    def __init__(self, file_name_or_path: str, label2id: dict=None, is_raw: bool=False):
        super(FeverDataset, self).__init__()
        self.is_dir = os.path.isdir(file_name_or_path)
        self.label2id = label2id
        self.is_raw = is_raw
        if self.is_dir:
            names = sorted(os.listdir(file_name_or_path), key=lambda x: int(x[:-3]))
            self.data = list(map(lambda f: os.path.join(file_name_or_path, f), names))
        else:
            self.data = self.load_data(file_name_or_path)
        if is_raw:
            random.shuffle(self.data)

    def load_data(self, filename: str):
        with open(filename, 'rb') as fr:
            data = pickle.load(fr)
        return data

    def __getitem__(self, index: int):
        if self.is_dir:
            claim, label, evidence_set, sentences = self.load_data(self.data[index])
        else:
            claim, label, evidence_set, sentences = self.data[index]
        if self.is_raw:
            return claim, label, evidence_set, sentences
        state = [State(claim=claim,
                      label=label,
                      evidence_set=evidence_set,
                      pred_label=idx,
                      candidate=[],
                      count=0
        ) for idx in self.label2id.values()]
        actions = [[Action(sentence=sent, label='F/T/N') for sent in sentences] for _ in range(len(self.label2id))]
        return state, actions

    def __len__(self):
        return len(self.data)

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)
    
    def __len__(self):
        return min(len(d) for d in self.datasets)

def collate_fn(batch):
    data = []
    for item in batch:
        data.append(item)
    return list(zip(*data))

