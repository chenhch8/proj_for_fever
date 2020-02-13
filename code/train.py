# coding: utf-8

import argparse
import collections
import logging
import os
from pprint import pprint
import random
from collections import OrderedDict
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
#from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm
import json
#from tensorboardX import SummaryWriter

from transformers import BertTokenizer
from transformers.optimization import AdamW, WarmupLinearSchedule, WarmupConstantSchedule, WarmupCosineSchedule

import datetime
import argparse

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# 运行参数
parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument('--task_name', default='FEVER', type=str)
parser.add_argument('--train_data', default='data/train_process(5)-v3.jsonl', type=str)
parser.add_argument('--dev_data', default='data/dev_process(5)-v3.jsonl', type=str)
parser.add_argument('--is_pickle_file', action='store_true')
parser.add_argument('--output_dir', default='results', type=str)
parser.add_argument('--bert_model', default='bert-base-uncased', type=str)
parser.add_argument('--architecture', default='simple', choices=['simple', 'complex'])
# args.init_checkpoint = 'data/bert-base-uncased/bert-base-uncased-pytorch_model.bin'
parser.add_argument('--init_checkpoint', default='data/bert-base-uncased/', type=str)
parser.add_argument('--pretrained_model', default=None)
## Other parameters
parser.add_argument('--eval_test', action='store_true',  help='Whether to run eval on the test set.')
parser.add_argument('--do_lower_case', action='store_true')
parser.add_argument('--max_seq_length', default=128, type=int)
parser.add_argument('--train_batch_size', default=50, type=int)
parser.add_argument('--eval_batch_size', default=50, type=int)

parser.add_argument('--lr', default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument('--num_train_epochs', default=4.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--adam_epsilon', default=1e-8, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--warmup_steps', default=0, type=int)

parser.add_argument('--cuda', default=-1, type=int)
parser.add_argument('--multi_cuda', action='store_true')
parser.add_argument('--accumulate_gradients', default=1, type=int, help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
parser.add_argument('--local_rank', default=-1, type=int, help="local_rank for distributed training on gpus")
parser.add_argument('--seed', default=42)
parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help="Number of updates steps to accumualte before performing a backward/update pass.")

args = parser.parse_args()

pprint(vars(args))

if args.architecture == 'simple':
    from model import BertForFEVER
else:
    from model2 import BertForFEVER

from feverdataset import FeverDataset, collate_fn, Example

# 6 个约束对应的损失函数
def logistic_loss(x):
    return -torch.log(x)

def margin_loss(pos, neg, alpha=0.1):
    return torch.relu(neg - pos + alpha)

def loss1(scores, ids, labels):
    target = scores[ids].gather(1, labels.view(-1, 1))
    assert len(ids) == len(labels)
    assert target.size(1) == 1 and target.size(0) > 0
    return logistic_loss(target).mean()

def loss2(scores, ids, position=[1, 2]):
    target, _ = torch.max(scores[ids][:, position], dim=1, keepdim=True)
    assert target.size(1) == 1 and target.size(0) > 0
    return logistic_loss(target).mean()

def loss3(scores, ids, position=[1, 2]):
    assert len(ids[0]) == len(ids[1])
    pos, _ = torch.max(scores[ids[0]][:, position], dim=1, keepdim=True)
    neg, _ = torch.max(scores[ids[1]][:, position], dim=1, keepdim=True)
    return margin_loss(pos, neg).mean()

def correct_prediction(scores, labels, thred=None):
    pred_labels = scores.max(dim=1)[1]
    if thred is not None:
        pred_labels = torch.where(pred_labels <= thred, pred_labels, thred)
    return pred_labels.eq(labels).double().sum() / pred_labels.size(0)

def correct_prediction2(scores, ids, position=[1, 2]):
    assert len(ids[0]) == len(ids[1])
    pos, _ = torch.max(scores[ids[0]][:, position], dim=1, keepdim=True)
    neg, _ = torch.max(scores[ids[1]][:, position], dim=1, keepdim=True)
    return pos.gt(neg).double().sum() / pos.size(0)

# 主函数
def main():
    n_gpu = 0
    if args.local_rank == -1 or args.cuda != -1:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #n_gpu = torch.cuda.device_count()
        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda != -1 else "cpu")
        n_gpu = torch.cuda.device_count() if args.multi_cuda else 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # training set
    label_map = {
        'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2,
        'N': 0, 'T': 1, 'F': 2
    }
    
    def init_dataloader(dataset, batch_size):
        sampler = WeightedRandomSampler(dataset.weights, len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                collate_fn=collate_fn(tokenizer,
                                                      max_seq_len=128,
                                                      batch_size=batch_size,
                                                      label_map=label_map),
                                #sampler=sampler, num_workers=cpu_count() - 1)
                                sampler=sampler, num_workers=min(1, cpu_count() - 1))
        return dataloader
    
    if args.is_pickle_file:
        train_data = FeverDataset(case_filename=args.train_data, label_map=label_map)
    else:
        train_data = FeverDataset(raw_filename=args.train_data, label_map=label_map)
    train_dataloader = init_dataloader(train_data, args.train_batch_size)

    # test set
    if args.eval_test:
        if args.is_pickle_file:
            dev_data = FeverDataset(case_filename=args.dev_data, label_map=label_map)
        else:
            dev_data = FeverDataset(raw_filename=args.dev_data, label_map=label_map)
        dev_dataloader = init_dataloader(dev_data, args.eval_batch_size)

    num_train_steps = int(
        len(train_data) / args.train_batch_size * args.num_train_epochs)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    
    
    # model and optimizer

    model = BertForFEVER(args.init_checkpoint, num_labels=3)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        #model = DataParallelModel(model)
        #global loss1, loss2
        #loss1 = DataParallelCriterion(loss1)
        #loss2 = DataParallelCriterion(loss2)

  # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr,
                      eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_steps)
#     scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
#     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_steps)
    
    def run_model(batch, step=None, training=True):
        input_tensors, labels, ids, _ = batch
        acc1, acc2, acc3 = None, None, None

        input_ids, input_mask, segment_ids = [tensor.to(device) for tensor in input_tensors[1:]]
        try:
            scores, _ = model(input_ids=input_ids,
                              token_type_ids=segment_ids,
                              attention_mask=input_mask)
        except:
            print(input_ids.size())
            assert False
        del input_ids, input_mask, segment_ids
        
        loss = 0
        if len(ids[0]):
            loss += loss1(scores, ids[0], labels.to(device))
            acc1 = correct_prediction(scores.detach()[ids[0]], labels.to(device))
        if len(ids[1]):
            loss += loss2(scores, ids[1], [label_map['T'], label_map['F']])
            acc2 = correct_prediction(scores.detach()[ids[1]], 1, torch.tensor(1, dtype=torch.long).to(device))
        if len(ids[2]):
            loss += loss3(scores, ids[2], [label_map['T'], label_map['T']])
            acc3 = correct_prediction2(scores.detach(), ids[2], [label_map['T'], label_map['T']])
        #if n_gpu > 1:
        #    loss = loss.mean() # mean() to average on multi-gpu.
#        acc_loss /= acc_num  # 每个样本的平均loss
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if training:
            loss.backward(retain_graph=True)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

        return loss.cpu().data.item(),\
                acc1.cpu().data.item() if acc1 is not None else 0,\
                acc2.cpu().data.item() if acc2 is not None else 0,\
                acc3.cpu().data.item() if acc3 is not None else 0

    # load pretrained model
    if args.pretrained_model:
        print('loading pretrained model %s ...' % args.pretrained_model)
        if args.cuda == -1 or not torch.cuda.is_available():
            state_dict = torch.load(args.pretrained_model, map_location=lambda storage, loc: storage)
        else:
            state_dict = torch.load(args.pretrained_model)
        model_dict = OrderedDict()
        for k, v in state_dict['model_dict'].items():
            if k.startswith('module.'):
                key = k if args.multi_cuda else k[7:]
            else:
                key = k if not args.multi_cuda else f'module.{k}'
            model_dict[key] = v
        model.load_state_dict(model_dict)
        optimizer.load_state_dict(state_dict['optim_dict'])
        scheduler.load_state_dict(state_dict['scheduler_dict'])
        global_step = state_dict['global_step']
        init_epoch = state_dict['epoch']
    else:
        global_step = 0
        init_epoch = 0
    
    #writer = SummaryWriter(args.output_dir)
    for epoch in range(init_epoch, int(args.num_train_epochs)):
        model.train()
        acc_loss = 0
        acc_step = 0

        progressbar = tqdm(train_dataloader, desc="Train", leave=True)
        for step, batch in enumerate(progressbar):
            #if step < 86001 and step > 100:
            #    progressbar.refresh()
            #    continue
            cur_loss, acc1, acc2, acc3 = run_model(batch, step=step, training=True)
            
            global_step += 1
            acc_loss += cur_loss
            acc_step += 1
            
            _acc_loss = acc_loss / acc_step

            progressbar.set_description("Train (%.4f/%.4f/%.4f,%.4f,%.4f)" % (_acc_loss, cur_loss, acc1, acc2, acc3))
            progressbar.refresh()
            #writer.add_scalar('data/train_loss', _acc_loss, global_step)

            if step % 1000 == 0 and step:
                save_dict = {
                    'model_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'scheduler_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch
                }
                torch.save(save_dict, '%s/model_%d.pkl' % (args.output_dir, epoch))
        print(f'train loss: {acc_loss / acc_step}')
        
        if args.eval_test:
            model.eval()
            dev_loss = 0
            dev_step = 0
            progressbar = tqdm(dev_dataloader, desc="Dev", leave=True)
            with torch.no_grad():
                for step, batch in enumerate(tqdm(dev_dataloader, desc="Dev")):
                    cur_loss, acc1, acc2, acc3 = run_model(batch, training=False, step=step)
                    
                    dev_loss += cur_loss
                    dev_step += 1
                    
                    progressbar.set_description("Dev (%.4f/%.4f/%.4f,%.4f,%.4f)" % (dev_loss / dev_step, cur_loss, acc1, acc2, acc3))
                    progressbar.refresh()
                    
            dev_loss = dev_loss / dev_step
            #writer.add_scalar('data/dev_loss', dev_loss, epoch)
            print(f'dev loss: {dev_loss}')
        
        # save model
        save_dict = {
            'model_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
            'global_step': global_step,
            'train_loss': (acc_loss / acc_step),
            'epoch': epoch + 1
        }
        name = f'{acc_loss / acc_step}'
        if args.eval_test:
            name =  f'{name}-{dev_loss}'
            save_dict['dev_loss'] = dev_loss
        torch.save(save_dict, '%s/model_%d_%s.pkl' % (args.output_dir, epoch + 1, name))
        
if __name__ == '__main__':
    main()
