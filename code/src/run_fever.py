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
from functools import reduce
from collections import defaultdict
#from multiprocessing import cpu_count, Pool

from dqn.bert_dqn import BertDQN
from environment import DuEnv, ChenEnv
from replay_memory import ReplayMemory, PrioritizedReplayMemory
from data.structure import Transition, Action, State, Claim, Sentence, Evidence, EvidenceSet
from config import set_com_args, set_dqn_args, set_bert_args

from scorer import fever_score
import pdb

logger = logging.getLogger(__name__)

Agent = BertDQN
Env = {'DuEnv': DuEnv, 'ChenEnv': ChenEnv}
Memory = {'random': ReplayMemory, 'priority': PrioritizedReplayMemory}

DataSet = List[Tuple[Claim, int, Evidence, List[Sentence]]]

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


def load_and_process_data(args: dict, filename: str, token_fn: 'function', is_eval=False) \
        -> DataSet:
    cached_file = os.path.join(
        '/'.join(filename.split('/')[:-1]),
        'cached_{}_{}_{}.pk'.format('train' if filename.find('train') != -1 else 'dev',
                                    list(filter(None, args.model_name_or_path.split('/'))).pop(),
                                    args.max_sent_length)
    )
    data = None
    if not os.path.exists(cached_file):
        logger.info(f'Loading and processing data from {filename}')
        data = []
        with open(filename, 'rb') as fr:
            for line in tqdm(fr.readlines()):
                instance = json.loads(line.decode('utf-8').strip())
                claim = Claim(id=instance['id'],
                              str=instance['claim'],
                              tokens=token_fn(instance['claim'])[:args.max_sent_length])
                sent2id = {}
                sentences = []
                for title, text in instance['documents'].items():
                    for line_num, sentence in text.items():
                        sentences.append(Sentence(id=(title, int(line_num)),
                                                  str=sentence,
                                                  tokens=token_fn(sentence)[:args.max_sent_length]))
                        sent2id[(title, int(line_num))] = len(sentences) - 1
                evidence_set = [[sentences[sent2id[(title, int(line_num))]] \
                                    for title, line_num in evi] \
                                        for evi in instance['evidence_set']] \
                                if not is_eval else instance['evidence_set']
                data.append((claim, args.label2id[instance['label']], evidence_set, sentences))
            with open(cached_file, 'wb') as fw:
                pickle.dump(data, fw)
    else:
        logger.info(f'Loading data from {cached_file}')
        with open(cached_file, 'rb') as fr:
            data = pickle.load(fr)
    return data


def calc_fever_score(predicted_list: List[dict], true_file: str) \
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


def train(args,
          agent: Agent,
          train_data: DataSet,
          epochs_trained: int=0,
          loss_trained_in_current_epoch: float=0,
          steps_trained_in_current_epoch: int=0) -> None:
    env = Env[args.env](args.max_evi_size)
    memory = Memory[args.mem](args.capacity) 
    
    train_ids = list(range(len(train_data)))
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])
    for epoch in train_iterator:
        random.shuffle(train_ids)
        if epochs_trained > 0:
            epochs_trained -= 1
            continue
        epoch_iterator = tqdm(train_ids,
                              desc='[Train]Loss:---------',
                              disable=args.local_rank not in [-1, 0])
        t_loss, t_steps = loss_trained_in_current_epoch, steps_trained_in_current_epoch
        for i, idx in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                if steps_trained_in_current_epoch == t_steps:
                    print('\nRestoring:', end='')
                steps_trained_in_current_epoch -= 1
                present = (t_steps - steps_trained_in_current_epoch) / float(t_steps)
                if present in [float(i) / 10. for i in range(1, 11)]:
                    print(f'{int(present * 100)}%', end='->' if steps_trained_in_current_epoch else '\n')
                continue
            claim, label_id, evidence_set, sentences = train_data[idx]
            state = State(claim=claim,
                          label=label_id,
                          evidence_set=evidence_set,
                          pred_label=args.label2id['NOT ENOUGH INFO'],
                          candidate=[])
            actions = [Action(sentence=sent, label='F/T/N') for sent in sentences]
            while True:
                action, _ = agent.select_action(state, actions, net=agent.q_net, is_eval=False)
                state_next, reward, done = env.step(state, action)
                next_actions = list(filter(lambda x: action.sentence.id != x.sentence.id, actions)) \
                        if not done else []
                if done: action = None
                memory.push(Transition(state=state,
                                       action=action,
                                       next_state=state_next,
                                       reward=reward,
                                       next_actions=next_actions))
                state = state_next
                actions = next_actions
                # sample batch data and optimize model
                if len(memory) >= args.train_batch_size:
                    if args.prioritized_replay:
                        tree_idx, isweights, batch = memory.sample(args.train_batch_size)
                    else:
                        batch = memory.sample(args.train_batch_size)
                        isweights = None
                    loss = agent.update(batch, isweights)
                    if args.prioritized_replay:
                        memory.batch_update_sumtree(tree_idx, loss.tolist())
                    loss = loss.mean().item()
                    t_loss += loss
                    t_steps += 1
                    epoch_iterator.set_description('[Train]Loss:%.8f' % loss)
                    epoch_iterator.refresh()
                if done or len(actions) == 0: break
            if i and i % args.target_update == 0:
                agent.soft_update_of_target_network(args.tau)
            
            if i and i % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f'{epoch}-{i + 1}-{t_loss / t_steps}')
                agent.save(save_dir)
                with open(os.path.join(save_dir, 'memory.pk'), 'wb') as fw:
                    pickle.dump(memory, fw)

        epoch_iterator.close()
        
        if steps_trained_in_current_epoch == 0:
            save_dir = os.path.join(args.output_dir, f'{epoch + 1}-0-{t_loss / t_steps}')
            agent.save(save_dir)
            with open(os.path.join(save_dir, 'memory.pk'), 'wb') as fw:
                pickle.dump(memory, fw)
    train_iterator.close()


def evaluate(args: dict, agent: Agent, save_dir: str, dev_data: DataSet=None):
    agent.eval()
    if dev_data is None:
        dev_data = load_and_process_data(args,
                                         os.path.join(args.data_dir, 'dev.jsonl'),
                                         agent.token)
    results = []
    logger.info('Evaluating')
    with torch.no_grad():
        for claim, label_id, evidence_set, sentences in tqdm(dev_data):
            state = State(claim=claim,
                          label=label_id,
                          evidence_set=evidence_set,
                          pred_label=args.label2id['NOT ENOUGH INFO'],
                          candidate=[])
            actions = [Action(sentence=sent, label='F/T/N') for sent in sentences]
            q_values, states = [], []
            for _ in range(args.max_evi_size):
                action, q_value = agent.select_action(state, actions, net=agent.q_net, is_eval=True)
                state_next = DuEnv.new_state(state, action)
                next_actions = list(filter(lambda x: action.sentence.id != x.sentence.id, actions))
                q_values.append(q_value)
                states.append(state)
                state = state_next
                actions = next_actions
                if len(next_actions) == 0: break
            
            score_t = q_values[-1]
            max_score = score_t
            max_t = len(q_values) - 1
            for t in range(len(states) - 2, -1, -1):
                score_t = q_values[t] - args.eps_gamma * q_values[t + 1] + score_t
                if max_score < score_t:
                    max_score = score_t
                    max_t = t
            #if states[max_t].pred_label != 2:
            #    pdb.set_trace()
            results.append({
                'id': claim.id,
                'label': args.id2label[states[max_t].label],
                'evidence': states[max_t].evidence_set,
                'predicted_label': args.id2label[states[max_t].pred_label],
                'predicted_evidence': \
                    reduce(lambda seq1, seq2: seq1 + seq2,
                           map(lambda sent: [list(sent.id)], states[max_t].candidate)) \
                        if len(states[max_t].candidate) else []
            })

    with open(os.path.join(save_dir, 'predicted_result.json'), 'w') as fw:
        json.dump(results, fw)
    
    predicted_list, scores = calc_fever_score(results, args.dev_true_file)
    with open(os.path.join(save_dir, 'predicted_result.json'), 'w') as fw:
        json.dump({
            'scores': scores,
            'predicted_list': predicted_list
        }, fw)
    logger.info(f'Results are saved in {os.path.join(save_dir, "predicted_result.json")}')


def run_dqn(args) -> None:
    agent = Agent(args)
    agent.to(args.device)
    if args.do_train:
        train_data = load_and_process_data(args,
                                           os.path.join(args.data_dir, 'train.jsonl'),
                                           agent.token)
        epochs_trained = 0
        loss_trained_in_current_epoch = 0
        steps_trained_in_current_epoch = 0
        if args.checkpoint:
            names = list(filter(lambda x: x != '', args.checkpoint.split('/')))[-1].split('-')
            epochs_trained = int(names[0])
            steps_trained_in_current_epoch = int(names[1])
            loss_trained_in_current_epoch = float('.'.join(names[2].split('.')[:-1])) * steps_trained_in_current_epoch
            agent.load(args.checkpoint)
            with open(os.path.join(args.checkpoint, 'memory.pk'), 'rb') as fr:
                memory = pickle.load(fr)
        train(args, agent, train_data,
              epochs_trained,
              loss_trained_in_current_epoch,
              steps_trained_in_current_epoch)
        
    if args.do_eval:
        assert args.checkpoint is not None
        agent.load(args.checkpoint)
        dev_data = load_and_process_data(args,
                                         os.path.join(args.data_dir, 'dev.jsonl'),
                                         agent.token,
                                         is_eval=True)
        evaluate(args, agent, args.checkpoint, dev_data)


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
    args.id2label = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO']
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

