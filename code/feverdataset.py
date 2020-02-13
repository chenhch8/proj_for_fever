#!/usr/bin/env python
# coding=utf-8
import torch
from torch.utils.data import Dataset
import json
import pickle
from tqdm import tqdm
import numpy as np
import random
from pprint import pprint
from processor import convert_example_to_feature
from itertools import combinations

class Example:
    def __init__(self, guid, text_a, text_b=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        
LABEL_MAP = {
    'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2,
    'N': 0, 'T': 1, 'F': 2
}


def examples_to_tensors(examples, tokenizer, max_seq_len):
    _max_seq_len = max([len(tokenizer.tokenize(example.text_a)) + len(tokenizer.tokenize(example.text_b)) for example in examples])
    _max_seq_len = min(_max_seq_len + 3, max_seq_len)

    features = [convert_example_to_feature((example, _max_seq_len, tokenizer)) for example in examples]
    all_evi_ids = [f.guid for f in features]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    return all_evi_ids, all_input_ids, all_input_mask, all_segment_ids


class FeverDataset(Dataset):
    def __init__(self, raw_filename=None, case_filename=None, label_map=LABEL_MAP, nclass=3):
        super(FeverDataset, self).__init__()
        self.label_map = label_map
        self.counts = np.zeros(nclass, dtype=np.float)

        if case_filename is None:
            self.examples = []  # 保存example
            self.case_data = []    # 保存example数据那个约束
            self.load_and_process_data(raw_filename)
        elif raw_filename is None:
            self.examples = None
            self.case_data = None
            self.load_data(case_filename)
        print(f'case_data:{len(self.case_data)}; examples:{len(self.examples)}')

    def __len__(self):
        return len(self.examples)


    def load_and_process_data(self, filename):
        size = { 'train': 145449, 'dev': 19998 }
        with open(filename, 'r') as f:
            print(f'loading {filename} and constructing data ...')
            for line in tqdm(f, total=size[filename.split('/')[-1].split('_')[0]]):
                instance = json.loads(line.strip())
                label, claim, sents_dict = instance['label'], instance['claim'], instance['sents']
                evidence_list = instance['data'] # claim对应的证据集
                cache = [[], [], []]  # 缓存已经出现过的候选证据
                for raw_data in evidence_list:  # for each evidence
                    # 将数据构造结果存放到cache中
                    self.construct_case_data_for_each_evidence(label, claim, raw_data, sents_dict, cache)
                self.examples.extend(cache[0])
                self.case_data.extend(cache[2])
        assert len(self.case_data) == len(self.examples)
        profix_name = '.'.join(filename.split('.')[:-1])
        case_filename = f'{profix_name}-dataset-v13.pk'
        print(f'saving to {case_filename} ...')
        with open(case_filename, 'wb') as fw:
            pickle.dump({
                'examples': self.examples,
                'case_data': self.case_data,
                'counts': self.counts
            }, fw)


    def load_data(self, filename):
        print(f'loading {filename} ...')
        with open(filename, 'rb') as fr:
            data = pickle.load(fr)
        self.examples = data['examples']
        self.case_data = data['case_data']
        self.counts = data['counts']


    @property
    def weights(self):
        prob_per_class = self.counts.sum() / self.counts
        labels = list(map(lambda example: int(example[0]), self.examples))
        return prob_per_class[labels]


    #def shuffle(self, data):
    #    indics = list(range(len(data)))
    #    random.shuffle(indics)
    #    return map(lambda ind: data[ind], indics)

    
    def construct_case_data_for_each_evidence(self, label, claim, raw_data, sents_dict, cache):
        '''
        Input:
            label
            claim
            raw_data: [evidence, subset_list, supset_list], 其中
                evidence  # claim对应的一个证据
                subset_list: {
                                '1': [...],  # evidence 对应大小为 1 的子集列表, 列表中的每个元素为 [subset, fake_subset_list],
                                             # 其中 subset 为 evidence 的一个大小为 1 的子集,
                                             # fake_subset_list=[...] 为 subset 对应的负例列表, 
                                             # 该列表中的每个元素 fake_subset 具有如下性质: |subset| = |fake_subset| 且 |subset - fake_subset| = 1
                                '2': [...],  # evidence 对应大小为2的子集列表
                                '|evidence|': [...]
                             }
                supset_list: [...]  # evidence对应的超集列表, 列表中的每个元素 supset 具有如下性质: |supset - evidence| = 1 且 |supset| = |evidence| + 1
            sents_dict: {doc_id: {sent_id: sentence}}
            cache: [[],[],[]]  # 缓存已出现的候选证据
        '''

        assert label in self.label_map.keys()

        examples, evi_list, cases = cache

        def update(evi, case_id, label_id, count_inc=True):
            if evi not in evi_list:
                index = len(examples)
                evi_list.append(evi)
                examples.append((label_id,
                                 Example(evi,
                                         claim,
                                         ' '.join(list(map(lambda x: sents_dict[x[0]][str(x[1])], evi))))))
                cases.append([case_id])
                if count_inc:
                    self.counts[label_id] += 1
            else:
                index = evi_list.index(evi)
                if case_id not in cases[index]:
                    cases[index].append(case_id)
            return index + len(self.examples)  # 加上偏移量
        
        evidence, subset_list, supset_list = raw_data
        
        if label == 'NOT ENOUGH INFO':
            # 约束1
            _, fake_subset_list = subset_list['1'][0]
            assert len(fake_subset_list) > 0
            for [[doc_id, sent_id]] in fake_subset_list:
                update([[doc_id, sent_id]], (1, None), self.label_map[label])
            sample_num = min(5, len(fake_subset_list))
            samples = list(map(lambda x: [x[0][0], x[0][1]], random.sample(fake_subset_list, sample_num)))
            for i in range(1, sample_num + 1):
                for item in combinations(samples, i):
                    update(item, (1, None), self.label_map[label])
        else:
            assert len(evidence) > 0
            assert len(subset_list) > 0
            ## 约束2
            #for [doc_id, sent_id] in evidence:
            #    update([[doc_id, sent_id]], case2, self.label_map[label])
            # 约束3
            update(evidence, (3, None), self.label_map[label])
            # 约束6
            ### pos
            #update(evidence, case6)
            ## neg
            for supset in supset_list:
                update(supset, (6, 'neg'), self.label_map[label])
            # 约束4
            flag = False
            for key in subset_list:
                if key == str(len(evidence)):
                    #print(subset_list[key][0][0], evidence)
                    flag = subset_list[key][0][0] == evidence
                for subset, fake_subset_list in subset_list[key]:
                    assert len(subset) > 0
                    assert len(fake_subset_list) > 0
                    ## pos
                    pos_ind = update(subset, (4, 'pos'), self.label_map[label])
                    ## neg
                    for fake_subset in fake_subset_list:
                        _label = label if len(fake_subset) > 1 else 'NOT ENOUGH INFO'
                        update(fake_subset,
                               (4, 'neg', pos_ind) if len(fake_subset) > 1 else (4, 'neg'),
                               self.label_map[_label])
            assert flag
            ## 约束5
            #if len(subset_list) > 1:
            #    ## pos
            #    update(evidence, case5)
            #    ## neg
            #    for key in subset_list:
            #        if key == str(len(evidence)): continue
            #        for subset, _ in subset_list[key]:
            #            update(subset, case5)
        #return case1, case2, case3, case4, case5, case6
        #return case1, case2, case3, case4

    
    def __getitem__(self, index):
        '''
        获取第index个数据, “采样操作”在此处进行
        '''
        example, cases = self.examples[index], self.case_data[index]
        _cases, _cases_for_4 = [], []
        for case in cases:
            if len(case) == 3:
                _cases_for_4.append(case)
            else:
                _cases.append(case)
        if len(_cases_for_4):
            _case = random.choice(_cases_for_4)
            cases = [_case, *_cases]
            example = [self.examples[_case[2]], example]
        return example, cases, index

def collate_fn(tokenizer, max_seq_len=128, batch_size=32, label_map=LABEL_MAP):
    keys = {
        (1, None): 0,
        (3, None): 0,
        (4, 'neg'): 0,
        (4, 'pos'): 1,
        (6, 'neg'): 0
    }
    def callback(batch):
        examples = []
        labels = []
        ids = [[], [], [[], []]] # 三类索引: 分类, max, margin
        id2id = {}

        def update(idx, example):
            if idx not in id2id:
                id2id[idx] = len(examples)
                examples.append(example)
        
        for data, cases, idx in batch:
            if len(examples) > batch_size: break
            for case in cases:
                if len(case) == 3:
                    assert case[0] == 4 and case[1] == 'neg'
                    assert isinstance(data, list)
                    pos_example, neg_example = data
                    update(idx, neg_example[1])
                    update(case[2], pos_example[1])
                    if id2id[idx] not in ids[1]:
                        ids[1].append(id2id[idx])
                    if id2id[case[2]] not in ids[1]:
                        ids[1].append(id2id[case[2]])
                    ids[2][0].append(id2id[case[2]])
                    ids[2][1].append(id2id[idx])
                else:
                    key = keys[case]
                    label, example = data[1] if isinstance(data, list) else data
                    if case == (4, 'neg'):
                        assert label == label_map['N']
                    update(idx, example)
                    if id2id[idx] not in ids[key]:
                        ids[key].append(id2id[idx])
                        if key == 0:
                            labels.append(label)
        #for example, cases in batch:
            #label, case = example
            #idx = len(examples)
            #examples.append(case)
            #for case_id in cases:
            #    key = keys[case_id]
            #    if case_id == (4, 'neg') and label == label_map['N']:
            #        key = 0
            #    ids[key].append(idx)
            #    if key == 0:
            #        labels.append(label)
        assert len(labels) == len(ids[0])
        # tokenize
        evi_ids, input_ids, input_masks, segment_ids = examples_to_tensors(examples, tokenizer, max_seq_len)
        labels = torch.tensor(labels, dtype=torch.long) if len(labels) else []
        for i in range(len(ids) - 1):
            ids[i] = torch.tensor(ids[i], dtype=torch.long) if len(ids[i]) else []
        ids[-1] = torch.tensor(ids[-1], dtype=torch.long) if len(ids[-1][0]) else []

        return [evi_ids, input_ids, input_masks, segment_ids], labels, ids, examples
    
    return callback


if __name__ == '__main__':
    #from multiprocessing import cpu_count
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import WeightedRandomSampler
    batch_size = 82
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #train_dataset = FeverDataset(case_filename='./data/train_process(5)-v3-dataset-v12.pk')
    ##train_dataset = FeverDataset(raw_filename='./data/train_process(5)-v3.jsonl')
    ##dev_dataset = FeverDataset(raw_filename='./data/dev_process(5)-v3.jsonl')
    #sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset))
    #dataloader = DataLoader(train_dataset, batch_size=batch_size,
    #                        collate_fn=collate_fn(tokenizer, 128),
    #                        #sampler=sampler, num_workers=cpu_count() - 1)
    #                        sampler=sampler, num_workers=1)
    ##dataloader = DataLoader(train_dataset, batch_size=62,
    ##                        collate_fn=collate_fn(tokenizer, 128),
    ##                        shuffle=True)
    #
    #counts = np.zeros(3)
    #counts2 = np.zeros(2)
    #step = 0
    #for input_tensors, labels, ids, examples in tqdm(dataloader):
    #    for label in labels:
    #        counts[label] += 1
    #    counts2[0] += len(ids[0])
    #    counts2[1] += len(ids[1])
    #    step += 1
    #print(counts / step)
    #print(counts2 / step)
    
    dev_dataset = FeverDataset(case_filename='./data/dev_process(5)-v3-dataset-v13.pk')
    #dev_dataset = FeverDataset(raw_filename='./data/dev_process(5)-v3.jsonl')
    sampler = WeightedRandomSampler(dev_dataset.weights, len(dev_dataset))
    dataloader = DataLoader(dev_dataset, batch_size=batch_size,
                            collate_fn=collate_fn(tokenizer, 128),
                            #sampler=sampler, num_workers=cpu_count() - 1)
                            sampler=sampler, num_workers=1)
    #dataloader = DataLoader(dev_dataset, batch_size=62,
    #                        collate_fn=collate_fn(tokenizer, 128),
    #                        shuffle=True)
    
    counts = np.zeros(3)
    counts2 = np.zeros(2)
    step = 0
    for input_tensors, labels, ids, examples in tqdm(dataloader):
        for label in labels:
            counts[label] += 1
        counts2[0] += len(ids[0])
        counts2[1] += len(ids[1])
        step += 1
    print(counts / step)
    print(counts2 / step)
