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
from dqn.ggnn_dqn import GGNNDQN, ggnn_load_and_process_data
from dqn.transformer_dqn import TransformerDQN, transformer_load_and_process_data
from environment import BaseEnv, ChenEnv
from replay_memory import ReplayMemory, PrioritizedReplayMemory, ReplayMemoryWithLabel, PrioritizedReplayMemoryWithLabel
from data.structure import *
from data.dataset import collate_fn, FeverDataset
from config import set_com_args, set_dqn_args, set_bert_args
from eval.calc_score import calc_fever_score, truncate_q_values, calc_test_result, calc_fever2_score

logger = logging.getLogger(__name__)

#Agent = BertDQN
DQN_MODE = {
    'bert': (BertDQN, bert_load_and_process_data),
    'lstm': (LstmDQN, lstm_load_and_process_data),
    'ggnn': (GGNNDQN, ggnn_load_and_process_data),
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

    def log_scores(scores, save_dir, title):
        content = ''
        for thred in scores:
            if thred == 'origin': continue
            content += '*' * 20 + ' %0.5f ' % float(thred) + '*' * 20 + '\n'
            for label in scores[thred]:
                #strict_score, label_accuracy, precision, recall, f1 = scores[thred][label]
                content += f'{label}\t{scores[thred][label]}\n'
        with open(os.path.join(save_dir, 'results.txt'), 'a') as fw:
            fw.write(content)
        
        content = title
        for label in scores['origin']:
            content += f'{label}\t{scores["origin"][label]}\n'
        with open(os.path.join(args.output_dir, 'results.txt'), 'a') as fw:
            fw.write(content)

    logger.info('Training')
    env = Env(label2id=args.label2id, K=args.max_evi_size)
    if args.mem.find('label') == -1:
        memory = Memory[args.mem](args.capacity)
    else:
        memory = Memory[args.mem](args.capacity, args.num_labels, args.proportion)
    
    data_loader = DataLoader(train_data,
                             num_workers=1,
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
        
        log_per_steps = len(epoch_iterator) // 5

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
                    weights, rl_loss, sl_loss, loss = \
                            agent.update(batch, isweights,
                                         log=step % log_per_steps == 0 or step == 5)
                    if args.mem.find('priority') != -1:
                        memory.batch_update_sumtree(tree_idx, weights.tolist())
                    t_loss += loss
                    t_steps += 1
                    losses.append(loss)
                    epoch_iterator.set_description('%.4f(%.4f,%.4f)' % (loss, rl_loss, sl_loss))
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
                
                scores = evaluate(args, agent, save_dir)
                log_scores(scores, save_dir,
                           title='*' * 20 + ' %d - %d ' % (epoch + 1, step + 1) + '*' * 20 + f'\nloss={t_loss / t_steps}\n')

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
            log_scores(scores, save_dir,
                       title='*' * 20 + ' %d ' % (epoch + 1) + '*' * 20 + f'\nloss={t_loss / t_steps}\n')
                
    train_iterator.close()


def evaluate(args: dict, agent, save_dir: str, dev_data: FeverDataset=None, is_eval: bool=True):
    agent.eval()
    if dev_data is None:
        _, load_and_process_data = DQN_MODE[args.dqn_mode]
        if args.do_eval:
            filename = 'dev_v6.jsonl'
        elif args.do_test:
            filename = 'test_v6.jsonl'
        elif args.do_fever2:
            filename = 'fever2-dev_v6.jsonl'
        dev_data = load_and_process_data(args,
                                         os.path.join(args.data_dir, filename),
                                         agent.token)
    data_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=1, shuffle=False)
    epoch_iterator = tqdm(data_loader,
                          disable=args.local_rank not in [-1, 0])
    results_of_q_state_seq = []
    results = []
    logger.info('Evaluating')
    with torch.no_grad():
        for batch_state, batch_actions in epoch_iterator:
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
                        continue
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
                                   (args.id2label[batch_state[i].label] if is_eval else None, args.id2label[batch_state[i].pred_label]), \
                                   batch_state[i].evidence_set if is_eval else None, \
                                   reduce(lambda seq1, seq2: seq1 + seq2,
                                          map(lambda sent: [list(sent.id)],
                                              batch_state[i].candidate)) if len(batch_state[i].candidate) else [], \
                                  ] for batch_q_value, batch_state in zip(q_value_seq, state_seq)]
                idx = state_seq[0][i].claim.id
                results_of_q_state_seq.append([idx, q_state_values])

            # 选择末尾Q值最高的那个
            ids = torch.tensor(q_value_seq[-1]).argmax().item()

            state = state_seq[-1][ids]
            results.append({
                'id': state.claim.id,
                'label': args.id2label[state.label] if is_eval else None,
                'evidence': state.evidence_set if is_eval else None,
                'predicted_label': args.id2label[state.pred_label],
                'predicted_evidence': \
                    reduce(lambda seq1, seq2: seq1 + seq2,
                           map(lambda sent: [list(sent.id)], state.candidate)) \
                        if len(state.candidate) else []
            })

    name = 'decision_seq_result.json'
    if not is_eval:
        name = f'test-{name}'
    if args.do_fever2:
        name = f'fever2-{name}'
    with open(os.path.join(save_dir, name), 'w') as fw:
        json.dump({
            'results_of_q_state_seq': results_of_q_state_seq,
            'results_of_last_step': results
        }, fw)

   
    if not is_eval:
        predicted_list = calc_test_result(results, args.test_true_file, logger=None)
        with open(os.path.join(save_dir, 'predictions.jsonl'), 'w') as fw:
            for item in predicted_list:
                fw.write(json.dumps(item) + '\n')
        #for thred in np.arange(0, args.pred_thred + args.pred_thred / 20, args.pred_thred / 20):
        for thred in np.arange(0, 1.1, 1 / 20):
            truncate_results = truncate_q_values(results_of_q_state_seq, thred)
            predicted_list = calc_test_result(truncate_results, args.test_true_file, logger=None)
            with open(os.path.join(save_dir, 'predictions-%.5f.jsonl' % thred), 'w') as fw:
                for item in predicted_list:
                    fw.write(json.dumps(item) + '\n')
        logger.info(f'Testing result is saved in {save_dir}')
        return
    
    if args.do_eval:
        thred_results = defaultdict(dict)
        predicted_list, scores = calc_fever_score(results, args.dev_true_file, logger=None)
        thred_results['scores']['origin'] = scores
        thred_results['predicted_list']['origin'] = predicted_list
        
        #for thred in np.arange(0, args.pred_thred + args.pred_thred / 20, args.pred_thred / 20):
        for thred in np.arange(0, 1.1, 1 / 20):
            truncate_results = truncate_q_values(results_of_q_state_seq, thred)
            truncate_predicted_list, truncate_scores = calc_fever_score(truncate_results,
                                                                        args.dev_true_file,
                                                                        logger=None)
            thred_results['scores'][f'{thred}'] = truncate_scores
            thred_results['predicted_list'][f'{thred}'] = truncate_predicted_list
        thred_results = dict(thred_results)
        
        filename = 'predicted_result.json'
        with open(os.path.join(save_dir, filename), 'w') as fw:
            json.dump(thred_results, fw)
        logger.info(f'Results are saved in {os.path.join(save_dir, filename)}')
        
        return thred_results['scores']

    if args.do_fever2:
        with open(os.path.join(save_dir, 'decision_seq_result.json'), 'r') as fr:
            predicted_fever1_list = json.load(fr)['results_of_last_step']
        predicted_fever2_list, predicted_fever1_list, scores = \
                calc_fever2_score(results,
                                  predicted_fever1_list,
                                  os.path.join(args.data_dir, 'fever2-dev_v6.jsonl'))
        print(scores)
        with open(os.path.join(save_dir, 'fever2-scores.json'), 'w') as fr:
            json.dump({
                'fever2_list': predicted_fever2_list,
                'fever1_list': predicted_fever1_list,
                'scores': scores
            }, fr)
        return scores



def run_dqn(args) -> None:
    Agent, load_and_process_data = DQN_MODE[args.dqn_mode]
    agent = Agent(args)
    agent.to(args.device)
    if args.do_train:
        train_data = load_and_process_data(args,
                                           os.path.join(args.data_dir, 'train_v6.jsonl'),
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
        
    elif args.do_eval or args.do_test or args.do_fever2:
        assert args.checkpoints is not None
        if args.do_eval:
            dev_data = load_and_process_data(args,
                                             os.path.join(args.data_dir, 'dev_v6.jsonl'),
                                             agent.token)
        if args.do_test:
            test_data = load_and_process_data(args,
                                              os.path.join(args.data_dir, 'test_v6.jsonl'),
                                              agent.token)
        if args.do_fever2:
            fever2_data = load_and_process_data(args,
                                                os.path.join(args.data_dir, 'fever2-dev_v6.jsonl'),
                                                agent.token)
        for checkpoint in args.checkpoints:
            agent.load(checkpoint)
            if args.do_eval:
                evaluate(args, agent, checkpoint, dev_data, is_eval=True)
            if args.do_test:
                evaluate(args, agent, checkpoint, test_data, is_eval=False)
            if args.do_fever2:
                evaluate(args, agent, checkpoint, fever2_data, is_eval=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    set_com_args(parser)
    set_dqn_args(parser)
    set_bert_args(parser)
    args = parser.parse_args()
    args.logger = logger
    if args.model_type != 'roberta':
        args.label2id = {
            'NOT ENOUGH INFO': 2,
            'SUPPORTS': 1,
            'REFUTES': 0
        }
        args.id2label = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO']
    else:
        args.label2id = {
            'NOT ENOUGH INFO': 1,
            'SUPPORTS': 2,
            'REFUTES': 0
        }
        args.id2label = ['REFUTES', 'NOT ENOUGH INFO', 'SUPPORTS']
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

