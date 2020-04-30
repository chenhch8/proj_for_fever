#!/usr/bin/env python
# coding=utf-8
import pickle
import os
from tqdm import tqdm

from data.structure import *

target = '/mnt/data/chenhch8/cache_train_albert-large-v2_lstm'
source = '/home/chenhch8/proj_for_fever/code/data/dqn/cached_train_albert-large-v2_lstm'

names = sorted(os.listdir(source), key=lambda x: int(x[:-3]))
filenames = list(map(lambda f: os.path.join(source, f), names))

count = 0
for filename in filenames:
    print(f'Processing {filename}')
    with open(filename, 'rb') as fr:
        data = pickle.load(fr)
    for item in tqdm(data):
        with open(os.path.join(target, f'{count}.pk'), 'wb') as fw:
            pickle.dump(item, fw)
        count += 1
