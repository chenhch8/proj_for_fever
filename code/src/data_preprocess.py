#!/usr/bin/env python
# coding=utf-8
import json
import sqlite3
from tqdm import tqdm
import unicodedata
from collections import defaultdict
#import pdb

ENCODING = 'utf-8'
DATABASE = './data/fever/fever.db'

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

def normalize(text: str) -> str:
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def data_process(in_file: str, out_file: str, is_train: bool=True) -> None:
    print(f'Loading {in_file}')
    instances = []
    with open(in_file, 'rb') as fr:
        for line in tqdm(fr.readlines()):
            instance = json.loads(line.decode(ENCODING).strip('\r\n'))
            
            evidence_set = []
            for evidence in instance['evidence']:
                process_evidence = []
                for sent in evidence:
                    if sent[2] is None: break
                    process_evidence.append([sent[2], sent[3]])
                if len(process_evidence) and process_evidence not in evidence_set:
                    evidence_set.append(process_evidence)

            instances.append({
                'id': instance['id'],
                'label': instance['label'],
                'claim': instance['claim'],
                'evidence_set': evidence_set,
                'predicted_pages': instance['predicted_pages']
             })

    print(f'Processing and writing to {out_file}')
    with open(out_file, 'wb') as fw:
        for instance in tqdm(instances):
            titles = instance['predicted_pages']
            if is_train:
                titles += [title for evidence in instance['evidence_set'] for title, _ in evidence]
            titles = list(set(titles))
            documents = defaultdict(dict)
            for title in titles:
                cursor.execute(
                    'SELECT * FROM documents WHERE id = ?',
                    (normalize(title),)
                )
                for row in cursor:
                    sentences = row[2].split('\n')
                    for sentence in sentences:
                        if sentence == '': continue
                        arr = sentence.split('\t')
                        if not arr[0].isdigit():
                            print(('Warning: this line from article %s for claim %d is not digit %s\r\n' % (title, instance['id'], sentence)).encode(ENCODING))
                            continue
                        line_num = int(arr[0])
                        if len(arr) <= 1: continue
                        sentence = ' '.join(arr[1:])
                        if sentence == '': continue
                        documents[title][line_num] = sentence
            fw.write((json.dumps({
                'id': instance['id'],
                'claim': instance['claim'],
                'label': instance['label'],
                'evidence_set': instance['evidence_set'],
                'documents': documents
            }) + '\n').encode(ENCODING))

if __name__ == '__main__':
    data_process('./data/retrieved/train.wiki7.jsonl', './data/dqn/train.jsonl', is_train=True)
    data_process('./data/retrieved/dev.wiki7.jsonl', './data/dqn/dev.jsonl', is_train=False)
    #data_process('./data/retrieved/test.wiki7.jsonl', './data/dqn/test.jsonl', is_train=False)
