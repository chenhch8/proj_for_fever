#!/usr/bin/env python
# coding=utf-8
from typing import List, Tuple
from tqdm import tqdm
from collections import defaultdict
import json

from .scorer import fever_score


def calc_test_result(predicted_list: List[dict], true_file: str, logger=None) -> List[dict]:
    predicted_dict = {int(item['id']): item for item in predicted_list}
    if logger:
        logger.info('Calculating test result')
        logger.info(f'Loading true data from {true_file}')
    result = []
    with open(true_file, 'r') as fr:
        for line in tqdm(fr.readlines()):
            instance = json.loads(line.strip())
            idx = int(instance['id'])
            if idx in predicted_dict:
                label = predicted_dict[idx]['predicted_label']
                evidence = predicted_dict[idx]['predicted_evidence']
            else:
                label = 'NOT ENOUGH INFO'
                evidence = []
            result.append({
                'id': idx,
                'predicted_label': label,
                'predicted_evidence': evidence
            })
    assert len(result) == 19998
    return result

def calc_fever_score(predicted_list: List[dict], true_file: str, logger=None) \
        -> Tuple[List[dict], float, float, float, float, float]:
    ids = set(map(lambda item: int(item['id']), predicted_list))
    if logger:
        logger.info('Calculating FEVER score')
        logger.info(f'Loading true data from {true_file}')
    with open(true_file, 'r') as fr:
        for line in tqdm(fr.readlines()):
            instance = json.loads(line.strip())
            if int(instance['id']) not in ids:
                predicted_list.append({
                    'id': instance['id'],
                    'label': instance['label'],
                    'evidence': instance['evidence'],
                    'predicted_label': 'NOT ENOUGH INFO',
                    'predicted_evidence': []
                })
    assert len(predicted_list) == 19998
    
    predicted_list_per_label = defaultdict(list)
    for item in predicted_list:
        predicted_list_per_label[item['label']].append(item)
    predicted_list_per_label = dict(predicted_list_per_label)

    scores = {}
    strict_score, label_accuracy, precision, recall, f1 = fever_score(predicted_list)
    scores['dev'] = (strict_score, label_accuracy, precision, recall, f1)
    if logger:
        logger.info(f'[Dev] FEVER: {strict_score}\tLA: {label_accuracy}\tACC: {precision}\tRC: {recall}\tF1: {f1}')
    for label, item in predicted_list_per_label.items():
        strict_score, label_accuracy, precision, recall, f1 = fever_score(item)
        scores[label] = (strict_score, label_accuracy, precision, recall, f1)
        if logger:
            logger.info(f'[{label}] FEVER: {strict_score}\tLA: {label_accuracy}\tACC: {precision}\tRC: {recall}\tF1: {f1}')
    return predicted_list, scores


def truncate_q_values(predicted_state_seq: List, thred: float=0.1, is_test: bool=False):
    predicted_list = []
    for idx, state_seq in predicted_state_seq:
        score_seq = [score for score, _, _, _ in state_seq]
        score_gap = [score_seq[t] - score_seq[t - 1] for t in range(1, len(score_seq))] # cur - pre
        ptr = len(score_seq) - 1
        for t in range(len(score_gap) - 1, -1, -1):
            if score_gap[t] >= -thred and score_gap[t] <= thred:
                ptr = t
            else:
                break
        predicted_list.append({
            'id': idx,
            'label': state_seq[ptr][1][0],
            'evidence': state_seq[ptr][2],
            'predicted_label': state_seq[ptr][1][1],
            'predicted_evidence': state_seq[ptr][3]
        } if not is_test else {
            'id': idx,
            'predicted_label': state_seq[ptr][1][1],
            'predicted_evidence': state_seq[ptr][3]
        })
    return predicted_list
