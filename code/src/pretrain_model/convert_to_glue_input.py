# coding: utf-8
#import csv
import argparse
from pprint import pprint
from tqdm import tqdm
#import sys

import pdb

#csv.field_size_limit(sys.maxsize)

def read_csv(filename):
    print(f'reading {filename}')
    data = []
    with open(filename, 'r') as fr:
        for line in tqdm(fr.readlines()):
            items = line.strip().split('\t')
            data.append(items)
    return data

def write_csv(filename, data):
    print(f'writting to {filename}')
    outputs = []
    for items in data:
        outputs.append('\t'.join(items))
    with open(filename, 'w') as fw:
        fw.write('\n'.join(outputs))

def convert_to_glue_input(inputs):
    print('converting to glue input')
    head = [''] * 12
    head[0] = 'id'
    head[8] = 'claim'
    head[9] = 'evidence'
    head[-1] = 'label'
    outputs = [head]
    for row in tqdm(inputs):
        output = [''] * 12
        output[0] = row[0]
        output[8] = row[2]
        output[9] = ' '.join(row[3:])
        output[-1] = row[1]
        outputs.append(output)
    return outputs

def main(args):
    data = read_csv(args.s)
    #pdb.set_trace()
    data = convert_to_glue_input(data)
    write_csv(args.o, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', required=True, type=str)
    parser.add_argument('-o', required=True, type=str)
    args = parser.parse_args()
    print(vars(args))
    main(args)
