#!/usr/bin/env python
# coding=utf-8
import re
import pdb
from typing import List, Set
from tqdm import tqdm
from collections import Counter

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from torch.utils.data.dataloader import DataLoader

from .structure import *
from .dataset import FeverDataset, collate_fn

stop_words  = set(stopwords.words('english'))
english_punctuations = {',', '\.', ':', ';', '\?', '\(', '\)', 
                        '\[', '\]', '&', '!', '\*', '@', '#', 
                        '$', '%', '\+', '\'', '\"', '_', '-lrb-', 
                        '-lsb-', '-rsb-', '-rrb-', '-lcb-', 
                        '-rcb-', '-colon-'}
pattern = re.compile(r'({})'.format('|'.join(english_punctuations)))

wordnet_lemmatizer = WordNetLemmatizer()

def remove_punct(text: str) -> str:
    return pattern.sub('', text)

def remove_stop_words(words: List[str]) -> List[str]:
    return [word for word in words if word not in stop_words]

def lemmatizer(words: List[str]) -> Set[str]:
    lemma_word = []
    for word in words:
        word = wordnet_lemmatizer.lemmatize(word, pos='n')
        word = wordnet_lemmatizer.lemmatize(word, pos='v')
        word = wordnet_lemmatizer.lemmatize(word, pos=('a'))
        lemma_word.append(word)
    return set(lemma_word)

def iou(words1: Set[str], words2: Set[str]) -> float:
    return len(words1 & words2) / max(1, len(words1 | words2))

def calc_two_sentences_iou(sent1: str, sent2: str) -> float:
    sent1_tokens = lemmatizer(remove_stop_words(word_tokenize(remove_punct(sent1))))
    sent2_tokens = lemmatizer(remove_stop_words(word_tokenize(remove_punct(sent2))))
    return iou(sent1_tokens, sent2_tokens)

def main(filename: str):
    dataset = FeverDataset(filename, label2id={'NOT ENOUGH INFO': 2})
    data_loader = DataLoader(dataset, num_workers=0, collate_fn=collate_fn, batch_size=1)

    print(f'parsing {filename}')
    has_relation = []
    no_relation = []
    for batch_state, batch_actions in tqdm(data_loader):
        claim = batch_state[0].claim.str
        if filename.find('train') != -1:
            evidence_set = [tuple(sent.id) for evi in batch_state[0].evidence_set for sent in evi]
        else:
            evidence_set = [(sent[2], sent[3]) for evi in batch_state[0].evidence_set for sent in evi if sent[2]]
        evidence_set = set(evidence_set)

        sentences = [action.sentence for action in batch_actions[0]]
        document = {}
        for sent in sentences:
            title, num = sent.id
            num = int(num)
            if title not in document:
                document[title] = {}
            if num not in document[title]:
                document[title][num] = {}
            document[title][num] = lemmatizer(remove_stop_words(word_tokenize(remove_punct(sent.str))))
            #document[title][num] = set(remove_stop_words(word_tokenize(remove_punct(sent.str))))
        
        claim_tokens = lemmatizer(remove_stop_words(word_tokenize(remove_punct(claim))))
        #claim_tokens = set(remove_stop_words(word_tokenize(remove_punct(claim))))

        for title in document:
            for num in document[title]:
                if (title, num) in evidence_set:
                    has_relation.append(iou(claim_tokens, document[title][num]))
                else:
                    no_relation.append(iou(claim_tokens, document[title][num]))

    has_counter = Counter(has_relation)
    no_counter = Counter(no_relation)
    has_sum = sum(has_counter.values())
    no_sum = sum(no_counter.values())
    
    has_relation = np.asarray(has_relation)
    no_relation = np.asarray(no_relation)

    print('has-top_10', list(map(lambda x: [x[0], x[1], x[1] / has_sum], has_counter.most_common(n=10))))
    print('no-top_10', list(map(lambda x: [x[0], x[1], x[1] / no_sum], no_counter.most_common(n=10))))

    print(f'has: min={has_relation.min()} max={has_relation.max()} mean={has_relation.mean()} std={has_relation.std()}')
    print(f'no: min={no_relation.min()} max={no_relation.max()} mean={no_relation.mean()} std={no_relation.std()}')

