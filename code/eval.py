# coding: utf-8
import torch
from transformers import BertTokenizer
import json
import os
from scorer import fever_score
from tqdm import  tqdm
from collections import OrderedDict, defaultdict

import argparse
from pprint import pprint

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--version', default='v2', choices=['v1', 'v2', 'v2_2', 'v2_3', 'v2_4', 'v2_5' ,'v2_6'])
parser.add_argument('--data_file', default='./data/dev_process(5)-v3.jsonl', type=str)
parser.add_argument('--bert_model', default='bert-base-uncased', type=str)
parser.add_argument('--do_lower_case', action='store_true')
parser.add_argument('--bert_pretrained_path', default='./data/bert-base-uncased/', type=str)
parser.add_argument('--max_seq_len', default=128, type=int)
parser.add_argument('--max_sent', default=5, type=int)
parser.add_argument('--thred', default=128, type=int)
parser.add_argument('--cuda', default=-1, type=int)
parser.add_argument('--multi_cuda', action='store_true')
parser.add_argument('--pos_ids', action='store_true')
parser.add_argument('--search', default='greedy', choices=['greed', 'beam'])
parser.add_argument('--model_list', type=str)
parser.add_argument('--architecture', default='simple', choices=['simple', 'complex'])

args = parser.parse_args()
args.model_list = args.model_list.split('+')
args.root_dir = '/'.join(args.model_list[0].split('/')[:-1])
args.output_dir = '%s/eval' % args.root_dir
args.label_map = ['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES']

if args.architecture == 'simple':
    from model import BertForFEVER
else:
    from model2 import BertForFEVER

if args.version == 'v1':
    from search_alg_v1 import greedy_search, beam_search
elif args.version == 'v2':
    from search_alg_v2 import greedy_search, beam_search
elif args.version == 'v2_2':
    from search_alg_v2_2 import greedy_search, beam_search
elif args.version == 'v2_3':
    from search_alg_v2_3 import greedy_search, beam_search
elif args.version == 'v2_4':
    from search_alg_v2_4 import greedy_search, beam_search
elif args.version == 'v2_5':
    from search_alg_v2_5 import greedy_search, beam_search
elif args.version == 'v2_6':
    from search_alg_v2_6 import greedy_search, beam_search


pprint(vars(args))

SEARCH_FUN = greedy_search if args.search is 'greedy' else beam_search

dataset = []

with open(args.data_file, 'r') as f:
    print('loading %s' % args.data_file)
    for line in tqdm(f.readlines()):
        dataset.append(json.loads(line.strip()))

device = torch.device(f'cuda:{args.cuda}') if args.cuda != -1 and torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
label2id = {v: k for k, v in enumerate(args.label_map)}

def save(name, strict_score, label_accuracy, precision, recall, f1, instances=None):
    if instances is not None:
        with open(f'{args.output_dir}/eval_instances_{name}_{args.version}.json', 'w') as fw:
            fw.write(json.dumps(instances))
    if not os.path.exists(f'{args.output_dir}/fever_score_{args.version}.txt'):
        with open(f'{args.output_dir}/fever_score_{args.version}.txt', 'w') as fw:
            fw.write('mode\tstrict_score\tlabel_accuracy\tprecision\trecall\tf1\n')
    with open(f'{args.output_dir}/fever_score_{args.version}.txt', 'a+') as fw:
        fw.write('%s\t%f\t%f\t%f\t%f\t%f\n' % (name, strict_score, label_accuracy, precision, recall, f1))

def evaluate_fever_score(data, name):
    instances = []
    #wrong = [2, 5, 6, 9, 10, 11, 12, 16, 17, 20, 26, 29, 30, 31, 32, 36, 38, 40, 44, 45]
    for instance in tqdm(data):
    #for i, instance in enumerate(tqdm(data)):
        #if len(instance['process_evidence']) > 1:
        #    pdb.set_trace()
        evidence, label = SEARCH_FUN(instance, model, tokenizer,
                                     label2id, device, args.label_map,
                                     args.max_seq_len, args.max_sent, THRED=args.thred, pos_ids=args.pos_ids)
        if len(evidence) > 0:
            evidence = list(map(lambda x: [x[0], int(x[1])], evidence))
        instances.append({'label': instance['label'],
                          'evidence': instance['evidence'],
                          'predicted_label': args.label_map[label],
                          'predicted_evidence': evidence})
    strict_score, label_accuracy, precision, recall, f1 = fever_score(instances)
    print(strict_score, label_accuracy, precision, recall, f1)
    save(name, strict_score, label_accuracy, precision, recall, f1, instances)
    return instances

model = BertForFEVER(args.bert_pretrained_path)
model.to(device)
if args.cuda != -1 and args.multi_cuda and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for pretrain_model in args.model_list:
    print('loading pretrained model %s ...' % pretrain_model)
    state_dict = torch.load(pretrain_model, map_location=lambda storage, loc: storage)
    model_dict = OrderedDict()
    for k, v in state_dict['model_dict'].items():
        if k.startswith('module.'):
            key = k if args.multi_cuda and torch.cuda.device_count() > 1 else k[7:]
        else:
            key = k if not args.multi_cuda else f'module.{k}'
        model_dict[key] = v
    model.load_state_dict(model_dict)
    model.eval()

    name = pretrain_model.split('/')[1][:-4]
    mode = args.data_file.split('/')[-1][:-6]
    print(f'evaluating {name} ...')
    instances = evaluate_fever_score(dataset, '[%s]-[%s]-%s' % (args.search, mode, name))

    data_map = defaultdict(list)
    for instance in instances:
        data_map[instance['label']].append(instance)
    data_map = dict(data_map)

    for key in data_map:
        _name = '[%s]-[%s-%s]-%s' % (args.search, mode, key, name)
        print(f'evaluating {_name} ...')
        strict_score, label_accuracy, precision, recall, f1 = fever_score(data_map[key])
        print(strict_score, label_accuracy, precision, recall, f1)
        save(_name, strict_score, label_accuracy, precision, recall, f1)

