# coding: utf-8
#import csv
import argparse
from pprint import pprint
from tqdm import tqdm
import sqlite3
import unicodedata
from collections import defaultdict
#import pdb

ENCODING = 'utf-8'
DATABASE = './data/fever/fever.db'
english_punctuations = {',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%'}

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()
#import sys

import pdb

#csv.field_size_limit(sys.maxsize)

def process_data(instances):
    new_instances = []
    for instance in tqdm(instances):
        evidence_set = set(map(lambda evidence: tuple(map(lambda sent: (sent[2], sent[3]),
                                                          evidence)),
                               instance['evidence'])) if label != 'NOT ENOUGH INFO' else []
        pred_evidence = tuple(map(lambda sent: (sent[0], sent[1]), instance['predicted_evidence']))
        
        titles = [sent[0] for evidence in evidence_set for sent in evidence] + \
                 [sent[0] for sent in pred_evidence]
        titles = set(titles)
        
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
                    if sentence == '' or sentence in english_punctuations: continue
                    documents[title][line_num] = sentence
        documents = dict(documents)

        for evidence in evidence_set:
            evi_str = ' '.join([documents[title][line_num] for title, line_num in evidence[:5]])
            label = instance['label'] if len(evidence) <= 5 else 'NOT ENOUGH INFO'
            new_instances.append({
                'id': instance['id'],
                'claim': instance['claim'],
                'evidence': evi_str,
                'label': label
            })
        for title, line_num in pred_evidence:
            if ((title, line_num),) in evidence_set or line_num not in documents[title]:
                continue
            new_instances.append({
                'id': instance['id'],
                'claim': instance['claim'],
                'evidence': documents[title][line_num],
                'label': 'NOT ENOUGH INFO'
            })
    return new_instances


def read_jsonl(filename):
    print(f'reading {filename}')
    instances = []
    with open(filename, 'r') as fr:
        for line in tqdm(fr.readlines()):
            instances.append(json.loads(line.strip()))
    return instances

def write_csv(filename, data):
    print(f'writting to {filename}')
    outputs = []
    for items in data:
        outputs.append('\t'.join(items))
    with open(filename, 'w') as fw:
        fw.write('\n'.join(outputs))

def convert_to_glue_input(instances):
    print('converting to glue input')
    head = [''] * 12
    head[0] = 'id'
    head[8] = 'claim'
    head[9] = 'evidence'
    head[-1] = 'label'
    outputs = [head]
    for instance in tqdm(instances):
        output = [''] * 12
        output[0] = instance['id']
        output[8] = instance['claim']
        output[9] = instance['evidence']
        output[-1] = instance['label']
        outputs.append(output)
    return outputs

def main(args):
    instances = read_jsonl(args.s)
    instances = process_data(instances)
    #pdb.set_trace()
    instances = convert_to_glue_input(instances)
    write_csv(args.o, instances)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', required=True, type=str)
    parser.add_argument('-o', required=True, type=str)
    args = parser.parse_args()
    print(vars(args))
    main(args)
