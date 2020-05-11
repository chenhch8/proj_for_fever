#!/usr/bin/env python
# coding=utf-8
from typing import List, Tuple

from .scorer import fever_score


def calc_fever_score(predicted_list: List[dict], true_file: str, logger) \
        -> Tuple[List[dict], float, float, float, float, float]:
    ids = set(map(lambda item: int(item['id']), predicted_list))
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
    logger.info(f'[Dev] FEVER: {strict_score}\tLA: {label_accuracy}\tACC: {precision}\tRC: {recall}\tF1: {f1}')
    for label, item in predicted_list_per_label.items():
        strict_score, label_accuracy, precision, recall, f1 = fever_score(item)
        scores[label] = (strict_score, label_accuracy, precision, recall, f1)
        logger.info(f'[{label}] FEVER: {strict_score}\tLA: {label_accuracy}\tACC: {precision}\tRC: {recall}\tF1: {f1}')
    return predicted_list, scores


def truncate_q_values(predicted_state_seq: List, thred: float=0.1):
    predicted_list = []
    for idx, state_seq in predicted_state_seq:
        score_seq = [score for score, _, _, _ in state_seq]
        score_gap = [score_seq[t] - score[t - 1] for t in range(len(score_seq) - 1, 1, -1)] # cur - pre
        ptr = len(score_seq) - 1
        for t in range(ptr, -1, -1):
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
        })
    return predicted_list
