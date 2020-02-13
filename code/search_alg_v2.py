#!/usr/bin/env python
# coding=utf-8
from feverdataset import Example, examples_to_tensors
# from processor import convert_example_to_feature

import torch
import numpy as np
import pdb

def evidence_generate(cur_sents, pool):
    evi_set = [[*cur_sents, sent] for sent in pool if sent not in cur_sents]
    return evi_set


def convert_example(claim, sents_id, id2sent):
    sents = [id2sent[doc_id][sent_id] for doc_id, sent_id in sents_id]
    return Example(sents_id, claim, ' '.join(sents))


def calculate_scores(data, evi_set, model, device, max_seq_len, tokenizer, THRED=32):
    def helper_fn(sub_evi_set):
        # 构造model输入
        examples = [convert_example(data['claim'],
                                    sents_id,
                                    data['sents']) \
                    for sents_id in sub_evi_set]

        _, input_ids, input_masks, segment_ids = examples_to_tensors(examples, tokenizer, max_seq_len)
        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        segment_ids = segment_ids.to(device)
        # 计算得分
        scores, _ = model(input_ids=input_ids,
                          token_type_ids=segment_ids,
                          attention_mask=input_masks)
        del input_ids, input_masks, segment_ids
        return scores

    if len(evi_set) < THRED:
        scores = helper_fn(evi_set)
    else:
        scores = []
        for start in range(0, len(evi_set), THRED):
            end = start + THRED
            sub_scores = helper_fn(evi_set[start:end])
            scores.append(sub_scores)
            del sub_scores
        scores = torch.cat(scores, dim=0)
    assert scores.size(1) == 3
    assert scores.size(0) == len(evi_set)

    return scores.data


def predict_N_or_TF(evi_set, label2id, label_map, scores, top_k=1):
    pos = [label2id[label_map[1]], label2id[label_map[2]]]  # 'SUPPORTS', 'REFUTES'
    neg = [label2id[label_map[0]]]                          # 'NOT ENOUGH INFO'
    TF = scores[:, pos]
    N = scores[:, neg]

    evidences, labels, max_scores = [], [], []
    if torch.prod(TF <= N):
        evidences.append([])
        labels.append(label2id[label_map[0]])
    else:
        indics = (TF > N).sum(dim=1).nonzero().view(-1)
        _, max_inds = scores[:, pos][indics].view(-1).topk(k=top_k, largest=True)
        for ind in max_inds:
            x, y = ind // 2, ind % 2
            labels.append(y + 1)
            evidences.append(evi_set[indics[x]])
            max_scores.append(scores[:, pos][indics[x], y].item())
    return evidences, labels, max_scores


def predict_T_or_F(evi_set, label2id, label_map, scores, top_k=1):
    position = [label2id[label_map[1]], label2id[label_map[2]]]  # 'SUPPORTS', 'REFUTES'
    scores = scores[:, position]

    _, indics = scores.view(-1).topk(k=top_k, largest=True)
    evidences, labels, max_scores = [], [], []
    for ind in indics:
        x, y = ind // len(position), ind % len(position)
        evidences.append(evi_set[x])
        labels.append(y + 1)
        max_scores.append(scores[x, y].item())
    
    return evidences, labels, max_scores


def greedy_search(data, model, tokenizer, label2id, device, label_map, max_seq_len=162, max_sent=5, THRED=32):
    '''
    label_map: ['NOT ENOGH INFO', 'SUPPORTS', 'REFUTES']
    '''
    pool = [[doc_id, sent_id] \
             for doc_id in data['sents'] \
               for sent_id in data['sents'][doc_id] \
                if len(data['sents'][doc_id][sent_id]) > 0]
    evidence, label = [], None
    max_score, flag = None, True
    #pdb.set_trace()
    with torch.no_grad():
        for _ in range(max_sent):
            evi_set = evidence_generate(evidence, pool)
            if len(evi_set) == 0: break

            scores = calculate_scores(data, evi_set, model, device, max_seq_len, tokenizer, THRED=THRED)
            
            if flag:
                flag = False
                cur_evidences, cur_labels, cur_max_scores = predict_N_or_TF(evi_set,
                                                                            label2id,
                                                                            label_map,
                                                                            scores,
                                                                            top_k=1)
                if cur_labels[0] == label2id[label_map[0]]:  # NOT ENOUGH INFO
                    evidence = []
                    label = cur_labels[0]
                    break
                else:
                    evidence = cur_evidences[0]
                    label = cur_labels[0]
                    max_score = cur_max_scores[0]
            else:
                cur_evidences, cur_labels, cur_max_scores = predict_T_or_F(evi_set,
                                                                           label2id,
                                                                           label_map,
                                                                           scores,
                                                                           top_k=1)
                #if cur_max_scores[0] <= max_score: break
                #else:
                #    max_score = cur_max_scores[0]
                #    label = cur_labels[0]
                #    evidence = cur_evidences[0]
                if cur_max_scores[0] >= max_score:
                    max_score = cur_max_scores[0]
                    label = cur_labels[0]
                evidence = cur_evidences[0]
            del scores
    return evidence, label


def beam_search(data, model, tokenizer, label2id, device, label_map, max_seq_len=162, max_sent=5, top_k=2, THRED=32):
    '''
    label_map: ['NOT ENOGH INFO', 'SUPPORTS', 'REFUTES']
    '''
    pool = [[doc_id, sent_id] \
             for doc_id in data['sents'] \
               for sent_id in data['sents'][doc_id] \
                 if len(data['sents'][doc_id][sent_id]) > 0]
    evidences, labels = [], []
    max_scores, flag = [], True
    with torch.no_grad():
        for _ in range(max_sent):
            if flag:
                flag = False
                evi_set = evidence_generate([], pool)
                scores = calculate_scores(data, evi_set, model, device, max_seq_len, tokenizer, THRED=THRED)

                cur_evidences, cur_labels, cur_max_scores = predict_N_or_TF(evi_set,
                                                                            label2id,
                                                                            label_map,
                                                                            scores,
                                                                            top_k=top_k)
                evidences = cur_evidences
                labels = cur_labels
                max_scores = cur_max_scores
                
                if label == label2id[label_map[0]]: break  # NOT ENOUGH INFO
            else:
                cur_evidences, cur_labels = [], []
                cur_max_scores = []
                for evidence in evidences:
                    evi_set = evidence_generate(evidence, pool)
                    if len(evi_set) == 0: break

                    scores = calculate_scores(data, evi_set, model, device, max_seq_len, tokenizer, THRED=THRED)

                    _cur_evidences, _cur_labels, _cur_max_scores = predict_T_or_F(evi_set,
                                                                                  label2id,
                                                                                  label_map,
                                                                                  scores,
                                                                                  top_k=top_k)
                    cur_evidences.extend(_cur_evidences)
                    cur_labels.extend(_cur_labels)
                    cur_max_scores.extend(_cur_max_scores)
                
                ## 搜索终止条件
                #local_max = np.array(cur_max_scores).max()
                #global_min = np.array(max_scores).min()
                #if local_max <= global_min: break

                # 保留得分最大的top_k个候选解
                #_evidences = evidences + _cur_evidences
                #_labels = labels + _cur_labels
                #_max_scores = max_scores + _cur_max_scores
                #top_k_indics = np.argsort(_max_scores)[-top_k:]
                top_k_indics = np.argsort(cur_max_scores)[-top_k:]
                evidences, labels, max_scores = [], [], []
                for ind in top_k_indics:
                    #evidences.append(_evidences[ind])
                    #labels.append(_labels[ind])
                    #max_scores.append(_max_scores[ind])
                    evidences.append(cur_evidences[ind])
                    labels.append(cur_labels[ind])
                    max_scores.append(cur_max_scores[ind])
    ind = np.array(max_scores).argmax()
    return evidences[ind], labels[ind]
