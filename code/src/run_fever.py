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
from environment import BaseEnv, DuEnv, ChenEnv
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


def load_and_process_data(args: dict, filename: str, token_fn: 'function', is_eval=False, env: str=None) \
        -> DataSet:
    env = None if is_eval else Env[env](5)
    cached_file = os.path.join(
        '/'.join(filename.split('/')[:-1]),
        'cached_{}_{}_{}.pk'.format(
            f'train-with-true-sequence_{env}' if filename.find('train') != -1 else 'dev',
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
                
                if not is_eval:
                    evidence_set = [[sentences[sent2id[(title, int(line_num))]] \
                                        for title, line_num in evi] \
                                            for evi in instance['evidence_set']]
                    # T/F sequences
                    all_sequences = []
                    for evi in evidence_set:
                        if len(evi) > 5: continue
                        sequence = []
                        state = State(label=args.label2id[instance['label']],
                                      pred_label=args.label2id['NOT ENOUGH INFO'],
                                      candidate=[],
                                      evidence_set=evidence_set,
                                      count=0)
                        # actions: 仅限于证据包含的所有句子
                        actions = [Action(sentence=sent.sentence, label=args.label2id[instance['label']]) \
                                   for sent in evi]
                        actions_next = actions
                        for action in actions:
                            state_next, reward, _ = env.step(state, action)
                            actions_next = [action_next for action_next in actions_next \
                                                if action_next.sentence.id != action.sentence.id]
                            sequence.append(Transition(state=state,
                                                       action=action,
                                                       next_state=state_next,
                                                       reward=reward,
                                                       next_actions=actions_next))
                            state = state_next
                        if len(sequence):
                            all_sequences.append(sequence)
                    data.append((claim, args.label2id[instance['label']], evidence_set, sentences, all_sequences))
                else:
                    evidence_set = instance['evidence_set']
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
          acc_loss_trained_in_current_epoch: float=0,
          steps_trained_in_current_epoch: int=0,
          losses_trained_in_current_epoch: List[float]=[]) -> None:
    logger.info('Training')
    env = Env[args.env](args.max_evi_size)
    memory = Memory[args.mem](args.capacity) 
    
    train_ids = list(range(len(train_data)))
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])
    for epoch in train_iterator:
        random.shuffle(train_ids)
        if epochs_trained > 0:
            epochs_trained -= 1
            continue
        epoch_iterator = tqdm([train_ids[i:i + 4] for i in range(0, len(train_ids), 4)],
                              desc='Loss',
                              disable=args.local_rank not in [-1, 0])
        t_loss, t_steps = acc_loss_trained_in_current_epoch, steps_trained_in_current_epoch
        t_losses, losses = losses_trained_in_current_epoch, []
        for step, idxs in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            batch_state, batch_actions = [], []
            for idx in idxs:
                claim, label, evidence_set, sentences, all_sequences = train_data[idx]
                memory.push_sequence(key=label, sequences=all_sequences)
                batch_state.append(
                    State(claim=claim,
                          label=label,
                          evidence_set=evidence_set,
                          pred_label=args.label2id['NOT ENOUGH INFO'],
                          candidate=[],
                          count=0)
                )
                batch_actions.append(
                    [Action(sentence=sent, label='F/T/N') for sent in sentences]
                )
            #pdb.set_trace()
            while True:
                batch_selected_action, _ = agent.select_action(batch_state,
                                                               batch_actions,
                                                               net=agent.q_net,
                                                               is_eval=False)
                batch_state_next, batch_actions_next = [], []
                for state, selected_action, actions in zip(batch_state,
                                                           batch_selected_action,
                                                           batch_actions):
                    state_next, reward, done = env.step(state, selected_action)
                    actions_next = None
                    if not done:
                        actions_next = \
                                list(filter(lambda x: selected_action.sentence.id != x.sentence.id,
                                            actions)) if selected_action.sentence is not None else []
                        if len(actions_next) == 0:
                            actions_next = [Action(sentence=None, label='F/T/N')]
                    memory.push(Transition(state=state,
                                           action=selected_action,
                                           next_state=state_next,
                                           reward=reward,
                                           next_actions=actions_next))
                    if done: continue
                    batch_state_next.append(state_next)
                    batch_actions_next.append(actions_next)
                batch_state = batch_state_next
                batch_actions = batch_actions_next
                # sample batch data and optimize model
                if len(memory) >= args.train_batch_size:
                    if args.mem == 'priority':
                        tree_idx, isweights, batch = memory.sample(args.train_batch_size)
                    else:
                        batch = memory.sample(args.train_batch_size)
                        isweights = None
                    loss = agent.update(batch, isweights)
                    if args.mem == 'priority':
                        memory.batch_update_sumtree(tree_idx, loss.tolist())
                    loss = loss.mean().item()
                    t_loss += loss
                    t_steps += 1
                    losses.append(loss)
                    epoch_iterator.set_description('%.8f' % loss)
                    epoch_iterator.refresh()
                if len(batch_state) == 0: break
            
            if step and step % args.target_update == 0:
                agent.soft_update_of_target_network(args.tau)
            
            if step and step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f'{epoch}-{step + 1}-{t_loss / t_steps}')
                agent.save(save_dir)
                with open(os.path.join(save_dir, 'memory.pk'), 'wb') as fw:
                    pickle.dump(memory, fw)
                with open(os.path.join(save_dir, 'loss.txt'), 'w') as fw:
                    fw.write('\n'.join(list(map(str, losses))))
                t_losses.extend(losses)
                losses = []

        epoch_iterator.close()

        acc_loss_trained_in_current_epoch = 0
        losses_trained_in_current_epoch = []
        
        if steps_trained_in_current_epoch == 0:
            save_dir = os.path.join(args.output_dir, f'{epoch + 1}-0-{t_loss / t_steps}')
            agent.save(save_dir)
            with open(os.path.join(save_dir, 'memory.pk'), 'wb') as fw:
                pickle.dump(memory, fw)
            with open(os.path.join(save_dir, 'loss.txt'), 'w') as fw:
                fw.write('\n'.join(list(map(str, t_losses))))
                
    train_iterator.close()


def evaluate(args: dict, agent: Agent, save_dir: str, dev_data: DataSet=None):
    agent.eval()
    if dev_data is None:
        dev_data = load_and_process_data(args,
                                         os.path.join(args.data_dir, 'dev.jsonl'),
                                         agent.token)
    dev_ids = list(range(len(dev_data)))
    epoch_iterator = tqdm([dev_ids[i:i+12] for i in range(0, len(dev_ids), 12)],
                          disable=args.local_rank not in [-1, 0])
    results = []
    logger.info('Evaluating')
    with torch.no_grad():
        for idxs in epoch_iterator:
            batch_state, batch_actions = [], []
            for idx in idxs:
                claim, label, evidence_set, sentences = dev_data[idx]
                batch_state.append(
                    State(claim=claim,
                          label=label,
                          evidence_set=evidence_set,
                          pred_label=args.label2id['NOT ENOUGH INFO'],
                          candidate=[],
                          count=0)
                )
                batch_actions.append(
                    [Action(sentence=sent, label='F/T/N') for sent in sentences]
                )

            q_value_seq, state_seq = [], []
            for _ in range(args.max_evi_size):
                batch_selected_action, batch_q_value = \
                        agent.select_action(batch_state,
                                            batch_actions,
                                            net=agent.q_net,
                                            is_eval=True)
                
                batch_state_next, batch_actions_next = [], []
                for state, selected_action, actions in zip(batch_state,
                                                           batch_selected_action,
                                                           batch_actions):
                    state_next = BaseEnv.new_state(state, selected_action)
                    actions_next = \
                            list(filter(lambda x: selected_action.sentence.id != x.sentence.id,
                                        actions)) if selected_action.sentence is not None else []
                    if len(actions_next) == 0:
                        actions_next = [Action(sentence=None, label='F/T/N')]
                    
                    batch_state_next.append(state_next)
                    batch_actions_next.append(actions_next)
                
                q_value_seq.append(batch_q_value)
                state_seq.append(batch_state_next)

                batch_state = batch_state_next
                batch_actions = batch_actions_next

            if args.env == 'DuEnv':
                batch_score = list(q_value_seq[-1])
                batch_max_score = batch_score.copy()
                batch_max_t = [-1] * len(state_seq[0])
                for t in range(len(state_seq) - 2, -1, -1):
                    batch_q_now = q_value_seq[t]
                    batch_q_next = q_value_seq[t + 1]
                    for i, (q_now, q_next) in enumerate(zip(batch_q_now, batch_q_next)):
                        batch_score[i] = q_now - args.eps_gamma * q_next + batch_score[i]
                        if batch_max_score[i] < batch_score[i]:
                            batch_max_score[i] = batch_score[i]
                            batch_max_t[i] = t
            else:
                batch_max_t = [-1] * len(state_seq[0])

            assert len(batch_max_t) == len(idxs)
            for i, max_t in enumerate(batch_max_t):
                state = state_seq[max_t][i]
                results.append({
                    'id': state.claim.id,
                    'label': args.id2label[state.label],
                    'evidence': state.evidence_set,
                    'predicted_label': args.id2label[state.pred_label],
                    'predicted_evidence': \
                        reduce(lambda seq1, seq2: seq1 + seq2,
                               map(lambda sent: [list(sent.id)], state.candidate)) \
                            if len(state.candidate) else []
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
        acc_loss_trained_in_current_epoch = 0
        steps_trained_in_current_epoch = 0
        losses_trained_in_current_epoch = []
        if args.checkpoint:
            names = list(filter(lambda x: x != '', args.checkpoint.split('/')))[-1].split('-')
            epochs_trained = int(names[0])
            steps_trained_in_current_epoch = int(names[1])
            acc_loss_trained_in_current_epoch = float('.'.join(names[2].split('.')[:-1])) * steps_trained_in_current_epoch
            agent.load(args.checkpoint)
            with open(os.path.join(args.checkpoint, 'memory.pk'), 'rb') as fr:
                memory = pickle.load(fr)
            with open(os.path.join(args.checkpoint, 'loss.txt'), 'r') as fr:
                losses_trained_in_current_epoch = list(map(float, fr.readlines()))
        train(args,
              agent,
              train_data,
              epochs_trained,
              acc_loss_trained_in_current_epoch,
              steps_trained_in_current_epoch,
              losses_trained_in_current_epoch)
        
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

