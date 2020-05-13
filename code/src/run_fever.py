#!/usr/bin/env python
# coding=utf-8
import logging
import os
import argparse
import json
import pickle
import random
from typing import List, Tuple
from tqdm import tqdm, trange
from time import sleep
from functools import reduce
from collections import defaultdict
import pdb
#from multiprocessing import cpu_count, Pool

import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np

from dqn.bert_dqn import BertDQN, bert_load_and_process_data
from dqn.lstm_dqn import LstmDQN, lstm_load_and_process_data
from dqn.transformer_dqn import TransformerDQN, transformer_load_and_process_data
from environment import BaseEnv, ChenEnv
from replay_memory import ReplayMemory, PrioritizedReplayMemory, ReplayMemoryWithLabel, PrioritizedReplayMemoryWithLabel
from data.structure import *
from data.dataset import collate_fn, FeverDataset
from config import set_com_args, set_dqn_args, set_bert_args
from eval.calc_score import calc_fever_score, truncate_q_values

logger = logging.getLogger(__name__)

#Agent = BertDQN
DQN_MODE = {
    'bert': (BertDQN, bert_load_and_process_data),
    'lstm': (LstmDQN, lstm_load_and_process_data),
    'transformer': (TransformerDQN, transformer_load_and_process_data)
}
#Agent = LstmDQN
#load_and_process_data = lstm_load_and_process_data
Env = ChenEnv
Memory = {
    'random': ReplayMemory,
    'priority': PrioritizedReplayMemory,
    'label_random': ReplayMemoryWithLabel,
    'label_priority': PrioritizedReplayMemoryWithLabel
}


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


def train(args,
          agent,
          train_data: FeverDataset,
          epochs_trained: int=0,
          acc_loss_trained_in_current_epoch: float=0,
          steps_trained_in_current_epoch: int=0,
          losses_trained_in_current_epoch: List[float]=[]) -> None:
    logger.info('Training')
    env = Env(args.max_evi_size)
    if args.mem.find('label') == -1:
        memory = Memory[args.mem](args.capacity)
    else:
        memory = Memory[args.mem](args.capacity, args.num_labels, args.proportion)
    
    data_loader = DataLoader(train_data,
                             num_workers=0,
                             collate_fn=collate_fn,
                             batch_size=args.train_batch_size,
                             shuffle=True)
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])
    for epoch in train_iterator:
        if epochs_trained > 0:
            epochs_trained -= 1
            sleep(0.1)
            continue
        epoch_iterator = tqdm(data_loader,
                              desc='Loss',
                              disable=args.local_rank not in [-1, 0])
        
        log_per_steps = len(epoch_iterator) // 10

        t_loss, t_steps = acc_loss_trained_in_current_epoch, steps_trained_in_current_epoch
        t_losses, losses = losses_trained_in_current_epoch, []
        
        for step, (batch_state, batch_actions) in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
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
                    actions_next = \
                            list(filter(lambda x: selected_action.sentence.id != x.sentence.id,
                                        actions)) if selected_action.sentence is not None else []
                    done = done if len(actions_next) else True
                    #if len(actions_next) == 0:
                        #pdb.set_trace()
                        #actions_next = [Action(sentence=None, label='F/T/N')]
                    
                    data = {'item': Transition(state=state,
                                               action=selected_action,
                                               next_state=state_next,
                                               reward=reward,
                                               next_actions=actions_next,
                                               done=done)}
                    if args.mem.find('label') != -1:
                        data['label'] = state.label
                    memory.push(**data)
                    
                    if done: continue
                    
                    batch_state_next.append(state_next)
                    batch_actions_next.append(actions_next)

                batch_state = batch_state_next
                batch_actions = batch_actions_next
                # sample batch data and optimize model
                if len(memory) >= args.train_batch_size:
                    if args.mem.find('priority') != -1:
                        tree_idx, isweights, batch = memory.sample(args.train_batch_size)
                    else:
                        batch = memory.sample(args.train_batch_size)
                        isweights = None
                    loss = agent.update(batch, isweights,
                                        log=step % log_per_steps == 0 or step == 5)
                    if args.mem.find('priority') != -1:
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
        
        save_dir = os.path.join(args.output_dir, f'{epoch + 1}-0-{t_loss / t_steps}')
        if steps_trained_in_current_epoch == 0:
            agent.save(save_dir)
            with open(os.path.join(save_dir, 'memory.pk'), 'wb') as fw:
                pickle.dump(memory, fw)
            with open(os.path.join(save_dir, 'loss.txt'), 'w') as fw:
                fw.write('\n'.join(list(map(str, t_losses))))
        
        if args.do_eval:
            scores = evaluate(args, agent, save_dir)
            content = f'************ {epoch + 1} ************\nloss={t_loss / t_steps}\n'
            for thred in scores:
                content += f'++++++++++ {thred} ++++++++++\n'
                for label in scores[thred]:
                    content += f'----- {label} -----\n'
                    strict_score, label_accuracy, precision, recall, f1 = scores[thred][label]
                    content += f'FEVER={strict_score}\nLA={label_accuracy}\nPre={precision}\nRecall={recall}\nF1={f1}\n'
            with open(os.path.join(save_dir, 'results.txt'), 'a') as fw:
                fw.write(content)
                
    train_iterator.close()


def evaluate(args: dict, agent, save_dir: str, dev_data: FeverDataset=None):
    agent.eval()
    if dev_data is None:
        dev_data = load_and_process_data(args,
                                         os.path.join(args.data_dir, 'dev.jsonl'),
                                         agent.token,
                                         is_eval=True)
    data_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=1, shuffle=False)
    epoch_iterator = tqdm(data_loader,
                          disable=args.local_rank not in [-1, 0])
    results_of_q_state_seq = []
    results = []
    logger.info('Evaluating')
    with torch.no_grad():
        for batch_state, batch_actions in tqdm(data_loader):
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
                    
                    batch_state_next.append(state_next)
                    
                    if len(actions_next) == 0:
                        #actions_next = [Action(sentence=None, label='F/T/N')]
                        break
                    else:
                        batch_actions_next.append(actions_next)
                
                q_value_seq.append(batch_q_value)
                state_seq.append(batch_state_next)
                
                if len(batch_actions_next) == 0:
                    break

                batch_state = batch_state_next
                batch_actions = batch_actions_next
            
            for i in range(len(batch_state)):
                q_state_values = [[batch_q_value[i], \
                                   (args.id2label[batch_state[i].label], args.id2label[batch_state[i].pred_label]), \
                                   batch_state[i].evidence_set, \
                                   reduce(lambda seq1, seq2: seq1 + seq2,
                                          map(lambda sent: [list(sent.id)],
                                              batch_state[i].candidate)) if len(batch_state[i].candidate) else [], \
                                  ] for batch_q_value, batch_state in zip(q_value_seq, state_seq)]
                idx = state_seq[0][i].claim.id
                results_of_q_state_seq.append([idx, q_state_values])


            batch_max_t = [-1] * len(state_seq[0])

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

    with open(os.path.join(save_dir, 'decision_seq_result.json'), 'w') as fw:
        json.dump(results_of_q_state_seq, fw)
   
    thred_results = defaultdict(dict)
    predicted_list, scores = calc_fever_score(results, args.dev_true_file, logger=None)
    thred_results['scores']['origin'] = scores
    thred_results['predicted_list']['origin'] = predicted_list
    
    for thred in np.arange(0, 1.01, 0.1):
        truncate_results = truncate_q_values(results_of_q_state_seq, thred)
        truncate_predicted_list, truncate_scores = calc_fever_score(truncate_results,
                                                                    args.dev_true_file,
                                                                    logger=None)
        thred_results['scores'][f'{thred}'] = truncate_scores
        thred_results['predicted_list'][f'{thred}'] = truncate_predicted_list
    thred_results = dict(thred_results)
    
    with open(os.path.join(save_dir, 'predicted_result.json'), 'w') as fw:
        json.dump(thred_results, fw)
    logger.info(f'Results are saved in {os.path.join(save_dir, "predicted_result.json")}')

    return thred_results['scores']


def run_dqn(args) -> None:
    Agent, load_and_process_data = DQN_MODE[args.dqn_mode]
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
    args.do_lower_case = bool(args.do_lower_case)
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

