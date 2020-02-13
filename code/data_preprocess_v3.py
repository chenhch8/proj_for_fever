import sqlite3
import pprint
import json
from collections import defaultdict
from tqdm import tqdm
import random
from pprint import pprint
import numpy as np
from itertools import combinations
from collections import deque
from copy import deepcopy
import sys
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', default='data/nn_doc_retr_1_train.jsonl', type=str)
parser.add_argument('--dev', default='data/nn_doc_retr_1_shared_task_dev.jsonl', type=str)
parser.add_argument('--db', default='data/fever.db', type=str)
parser.add_argument('--max_evidence_length', default=5, type=int)
parser.add_argument('--version', default='v5', type=str)
args = parser.parse_args()

args.to_train = 'data/train_process(%d)-%s.jsonl' % (args.max_evidence_length, args.version)
args.to_dev = 'data/dev_process(%d)-%s.jsonl' % (args.max_evidence_length, args.version)

pprint(vars(args))

THERD = 5  # 指定候选文档中非期望文档数量
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']

def ne_tagging(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append("_".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = "_".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    return continuous_chunk

class FeverDataBase:
    def __init__(self, db_name):
        print('connecting to %s ' % db_name)
        self.db = sqlite3.connect(db_name)
        self.cursor = self.db.cursor()
    def close(self):
        self.db.close()
    def query_by_doc_ids(self, doc_ids):
        if not (isinstance(doc_ids, list) or isinstance(doc_ids, tuple)):
            doc_ids = [doc_ids]
        placeholder = (', '.join(['?' for _ in range(len(doc_ids))])).format(*doc_ids)
        try:
            self.cursor.execute('SELECT id, lines_json FROM documents WHERE id in (%s)' % placeholder,  doc_ids)
        except:
            print('SELECT id, lines_json FROM documents WHERE id in (%s)' % doc_ids)
        return list(map(lambda items: (items[0], json.loads(items[1])), self.cursor.fetchall()))
    def query_all_doc_ids(self):
        self.cursor.execute('SELECT id FROM documents')
        return self.cursor.fetchall()


class Data:
    def __init__(self, name):
        print('loading %s' % name)
        self.data = []
        with open(name, 'r') as f:
            for line in tqdm(f.readlines()):
                instance = json.loads(line.strip())
                instance['prioritized_docids'] = sorted(instance['prioritized_docids'],
                                                        key=lambda x: x[1], reverse=True)
                # doc_ids
                doc_ids = list(set([sent[2] for evi in instance['evidence'] for sent in evi if sent[2] is not None]))
                pred_doc = [doc[0] for doc in instance['prioritized_docids']]
                for i in range(min(len(pred_doc), THERD)):
                    if pred_doc[i] in doc_ids: continue
                    doc_ids.append(pred_doc[i])
                if len(doc_ids) == 0:
                    doc_ids = ne_tagging(instance['claim'])  # 实体识别
                instance['doc_ids'] = doc_ids
                # process_evidence, 将每个claim对应的所有证据合并为一个
                evidence_list = sorted(instance['evidence'], key=lambda evi: len(evi), reverse=False)
                process_evidence = []
                for evidence in evidence_list:
                    for sent in evidence:
                        if sent[2] is None: break
                        if [sent[2], sent[3]] in process_evidence: continue
                        process_evidence.append([sent[2], sent[3]])
                instance['process_evidence'] = process_evidence
                for key in ['structured_docids', 'claim_lemmas', \
                            'claim_tokens', 'processed_claim', \
                            'predicted_docids']:
                    del instance[key]
                self.data.append(instance)
    def __len__(self):
        return len(self.data)
    def loader(self):
        for item in self.data:
             yield item


def evidence_construct(doc2sents, evidence=None, max_evidence_length=5):
    def pool_condition(sentence):
        words = word_tokenize(sentence) if len(sentence) else []
        return len([word for word in words if word not in english_punctuations]) > 0
    result = None
    pool = [[sent['sentences'], [doc_id, sent['line_num']]] \
                for doc_id, sents in doc2sents.items() \
                    for sent in sents if pool_condition(sent['sentences'])]
    pool = list(dict(pool).values())  # 去重
    assert len(pool) > 0

    if len(evidence) == 0:
        result = [
            None, {1: [[None, [[sent] for sent in pool]]]}, None]
    else:
        clip_evidence = evidence[:max_evidence_length]
        all_subset_list = defaultdict(list)  # 与证据子集相差一个元素但大小相同的子集
        all_supset_list = []                 # 证据超集
        # 构造证据子集及“差一”负子集
        # 遍历所有长度的证据子集
        for i in range(1, len(clip_evidence) + 1):
            # 遍历每个长度为 i 的证据子集
            for subset in combinations(clip_evidence, i):
                subset = list(subset)
                fake_subset_list = []
                # 构造“差一”负子集
                for j in range(len(subset)):
                    for sent in pool:
                        if sent in evidence: continue
                        fake_subset = deepcopy(subset)
                        fake_subset[j] = sent
                        fake_subset_list.append(fake_subset)

                if len(fake_subset_list) == 0:
                    pprint(pool)
                    pprint(evidence)
                    pprint(subset)
                    assert len(fake_subset_list) > 0

                all_subset_list[i].append([subset, fake_subset_list])
        # 构造证据的“加一”超集
        for sent in pool:
            if sent in evidence: continue
            supset = clip_evidence + [sent]
            all_supset_list.append(supset)

        if len(all_supset_list) == 0:
            pprint(pool)
            pprint(evidence)
            assert len(all_supsert_list) > 0

        result = [clip_evidence, dict(all_subset_list), all_supset_list]
    
    if len(result) == 0:
        pprint(doc2sents)
        pprint(pool)
        pprint(evidence_set)
        assert len(result) > 0

    return result


def sents_id2sents_str(doc2sents, instance):
    id2sent = defaultdict(dict)

    def fn(sent_list):
        for sent in sent_list:
            doc_id, sent_id = sent
            try:
                id2sent[doc_id][sent_id] = doc2sents[doc_id][sent_id]['sentences']
            except:
                from pprint import pprint
                print(sent)
                pprint(doc2sents)
                assert False

    evidence, all_subset_list, all_supset_list = instance
    if evidence is not None:
        fn(evidence)
        for supset in all_supset_list:
            fn(supset)
    for subset_list in all_subset_list.values():
        for subset, fake_subset_list in subset_list:
            if subset is not None: fn(subset)
            for fake_subset in fake_subset_list:
                fn(fake_subset)
    return dict(id2sent)

#def build_doc2sents(doc2sents):
#    id2sent = defaultdict(dict)
#    for doc_id in doc2sents:
#        document = doc2sents[doc_id]
#        for i, sentence in enumerate(document):
#            assert i == int(sentence['line_num'])
#            id2sent[doc_id][str(i)] = sentence['sentences']
#    return dict(id2sent)

fever_db = FeverDataBase(args.db)
doc_ids = fever_db.query_all_doc_ids()
doc_ids = list(map(lambda x: x[0], doc_ids))

train = Data(args.train)
dev = Data(args.dev)

train_process = []
for data in tqdm(train.loader(), total=len(train)):
    query_ids = data['doc_ids'] if len(data['doc_ids']) > 0 else random.sample(doc_ids, THERD)
    if len(query_ids) < THERD:
        query_ids += random.sample(doc_ids, THERD - len(query_ids))
#     print(query_ids)
    doc2sents = fever_db.query_by_doc_ids(query_ids)
    if len(doc2sents) == 0:
        query_ids = random.sample(doc_ids, THERD)
        doc2sents = fever_db.query_by_doc_ids(query_ids)
    doc2sents = dict(doc2sents)
    instance = evidence_construct(doc2sents, data['process_evidence'],
                                  max_evidence_length=args.max_evidence_length)
    sents = sents_id2sents_str(doc2sents, instance)
    #doc2sents = build_doc2sents(doc2sents)
    train_process.append(json.dumps({
        'claim': data['claim'],
        'evidence': data['evidence'],
        'id': data['id'],
        'label': data['label'],
        'doc_ids': data['doc_ids'],
        'process_evidence': data['process_evidence'],
        'verifiable': data['verifiable'],
        'data': instance,
        'doc2sents': sents
    }))
with open(args.to_train, 'w') as f:
    print('Saving to %s ...' % args.to_train)
    for item in train_process:
        f.write(item + '\n')
del train_process

dev_process = []
for data in tqdm(dev.loader(), total=len(dev)):
    query_ids = data['doc_ids'] if len(data['doc_ids']) > 0 else random.sample(doc_ids, THERD)
    if len(query_ids) < THERD:
        query_ids += random.sample(doc_ids, THERD - len(query_ids))
#     print(query_ids)
    doc2sents = fever_db.query_by_doc_ids(query_ids)
    if len(doc2sents) == 0:
        query_ids = random.sample(doc_ids, THERD)
        doc2sents = fever_db.query_by_doc_ids(query_ids)
    doc2sents = dict(doc2sents)
    instance = evidence_construct(doc2sents, data['process_evidence'],
                                  max_evidence_length=args.max_evidence_length)
    sents = sents_id2sents_str(doc2sents, instance)
    #doc2sents = build_doc2sents(doc2sents)
    dev_process.append(json.dumps({
        'claim': data['claim'],
        'evidence': data['evidence'],
        'id': data['id'],
        'label': data['label'],
        'doc_ids': data['doc_ids'],
        'process_evidence': data['process_evidence'],
        'verifiable': data['verifiable'],
        'data': instance,
        'doc2sents': sents
    }))
with open(args.to_dev, 'w') as f:
    print('Saving to %s ...' % args.to_dev)
    for item in dev_process:
        f.write(item + '\n')
del dev_process

