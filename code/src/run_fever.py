#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np
import logging
import os
import argparse
import json
import pickle
import random
from typing import List, Tuple
from tqdm import tqdm, trange
from multiprocessing import cpu_count, Pool

from dqn.bert_dqn import BertDQN
from environment import DuEnv
from replay_memory import ReplayMemory
from data_structure import Transition, Action, State, Claim, Sentence, Evidence, EvidenceSet
from config import set_com_args, set_dqn_args, set_bert_args


logger = logging.getLogger(__name__)


Agent = BertDQN
Env = DuEnv


def set_random_seeds(random_seed):
    """Sets all possible random seeds so results can be reproduced"""
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    # tf.set_random_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)


def load_and_process_data(filename: str, label2id: dict, max_sent_length: int, token_fn: 'function') \
        -> List[Tuple[Claim, int, EvidenceSet, List[Sentence]]]:
    cached_file = os.path.join(
        '/'.join(filename.split('/')[:-1]),
        'cached_{}_{}.pk'.format(max_sent_length, 'train' if filename.find('train') != -1 else 'dev')
    )
    data = None
    if not os.path.exists(cached_file):
        logger.info(f'Loading and processing data from {filename}')
        data = []
        with open(filename, 'rb') as fr:
            for line in tqdm(fr.readlines()):
                instance = json.loads(line.decode('utf-8').strip())
                claim = Claim(str=instance['claim'],
                              tokens=token_fn(instance['claim'])[:max_sent_length])
                sent2id = {}
                sentences = []
                for title, text in instance['documents'].items():
                    for line_num, sentence in text.items():
                        sentences.append(Sentence(id=(title, int(line_num)),
                                                  str=sentence,
                                                  tokens=token_fn(sentence)[:max_sent_length]))
                        sent2id[(title, int(line_num))] = len(sentences) - 1
                evidence_set = [[sentences[sent2id[(title, int(line_num))]] \
                                    for title, line_num in evi] \
                                        for evi in instance['evidence_set']]
                data.append((claim, label2id[instance['label']], evidence_set, sentences))
            with open(cached_file, 'wb') as fw:
                pickle.dump(data, fw)
    else:
        logger.info(f'Loading data from {cached_file}')
        with open(cached_file, 'rb') as fr:
            data = pickle.load(fr)
    return data


def run_dqn(args) -> None:
    env = Env(args.max_evi_size)
    agent = Agent(args)
    agent.to(args.device)
    if args.do_train:
        memory = ReplayMemory(args.capacity)
        train_data = load_and_process_data(os.path.join(args.data_dir, 'train.jsonl'),
                                           args.label2id, args.max_sent_length, agent.token)
        train_ids = list(range(len(train_data)))
        if args.checkpoint:
            agent.load(os.path.josin(args.output_dir, args.checkpoint))
        train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])
        for epoch in train_iterator:
            random.shuffle(train_ids)
            epoch_iterator = tqdm(train_ids, desc='[Train]Epoch:0', disable=args.local_rank not in [-1, 0])
            t_loss, t_steps = 0, 0
            for i, idx in enumerate(epoch_iterator):
                claim, label_id, evidence_set, sentences = train_data[idx]
                state = State(claim=claim,
                              label=label_id,
                              evidence_set=evidence_set,
                              pred_label=args.label2id['NOT ENOUGH INFO'],
                              candidate=[])
                actions = [Action(sentence=sent, label='F/T/N') for sent in sentences]
                while True:
                    action, _ = agent.select_action(state, actions, net=agent.q_net)
                    state_next, reward, done = env.step(state, action)
                    next_actions = list(filter(lambda x: action.sentence.id != x.sentence.id, actions)) \
                            if not done else []
                    if len(next_actions) == 0 and not done:
                        state_next = None
                        done = True
                    memory.push(Transition(state=state,
                                           action=action,
                                           next_state=state_next,
                                           reward=reward,
                                           next_actions=next_actions))
                    state = state_next
                    actions = next_actions
                    # sample batch data and optimize model
                    if len(memory) >= args.train_batch_size:
                        batch = memory.sample(args.train_batch_size)
                        loss = agent.update(batch)
                        t_loss += loss
                        t_steps += 1
                        epoch_iterator.set_description('[Train]Epoch:%d Loss:%.5f)' % (epoch, loss))
                        epoch_iterator.refresh()
                    if done:
                        break
                if i and i % args.target_update == 0:
                    agent.soft_update_of_target_network(args.tau)
            epoch_iterator.close()
            agent.save(os.join(args.output_dir, 'dqn', f'{epoch + 1}_{t_loss / t_steps}'))
        train_iterator.close()
    if args.do_eval:
        dev_data = load_and_process_data(os.path.join(args.data_dir, 'dev.jsonl'),
                                         args.label2id, args.max_sent_length, agent.token)
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    set_com_args(parser)
    set_dqn_args(parser)
    set_bert_args(parser)
    args = parser.parse_args()
    args.logger = logger
    args.label2id = {
        'NOT ENOUGH INFO': 2,
        'SUPPORTS': 1,
        'REFUTES': 0
    }
    logger.info(vars(args))

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        #"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        #args.fp16,
    )

    # Set seed
    set_random_seeds(args.seed)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    logger.info("Training/evaluation parameters %s", args)

    # run dqn
    run_dqn(args)

if __name__ == '__main__':
    main()

