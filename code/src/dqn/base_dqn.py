#!/usr/bin/env python3
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import os

import pdb

from collections import defaultdict
from typing import List, Tuple
from data.structure import Transition, Action, State

class BaseDQN:
    def __init__(self, args) -> None:
        self.q_net = None
        self.t_net = None
        self.optimizer = None
        self.scheduler = None
        
        # discount factor
        self.eps_gamma = args.eps_gamma
        # epsilon greedy
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        # dqn type
        self.dqn_type = args.dqn_type

        self.target_update = args.target_update
        self.steps_done = 0
        self.max_seq_length = args.max_seq_length

        self.device = args.device
        self.logger = args.logger
        self.args = args


    def set_network_untrainable(self, model) -> None:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False


    def to(self, device):
        self.q_net.to(device)
        self.t_net.to(device)
        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.q_net = torch.nn.DataParallel(self.q_net)
            self.t_net = torch.nn.DataParallel(self.t_net)
        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            self.q_net = torch.nn.parallel.DistributedDataParallel(
                self.q_net, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True,
            )
            self.t_net = torch.nn.parallel.DistributedDataParallel(
                self.t_net, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True,
            )


    def update(self, transitions: List[Transition], isweights: List[float]=None) -> float:
        self.q_net.train()
        self.t_net.eval()
        
        batch = Transition(*zip(*transitions))
        
        batch_state_next, batch_actions_next = \
                list(zip(*[(next_state, next_actions) \
                           for next_state, next_actions in zip(batch.next_state,
                                                               batch.next_actions) \
                           if next_state != None]))
        # max_actions, max_q_values: t_net(dqn_type=dqn)/q_net(dqn_type=ddqn)
        batch_max_action, batch_max_q_value = \
            self.select_action(batch_state_next,
                               batch_actions_next,
                               is_eval=False,
                               net=self.q_net if self.dqn_type == 'ddqn' else self.t_net)
        assert len(batch_max_action) == len(batch_state_next)

        # max next state_action value
        no_final_mask = torch.tensor([next_state != None for next_state in batch.next_state],
                                     dtype=torch.bool, device=self.device)
        next_state_values = torch.zeros(no_final_mask.size(), dtype=torch.float, device=self.device)
        if self.args.dqn_type == 'dqn':
            next_state_values[no_final_mask] = \
                torch.tensor(batch_max_q_value, dtype=torch.float, device=self.device)
        elif self.args.dqn_type == 'ddqn':
            max_labels = torch.tensor([action.label for action in batch_max_action],
                                      dtype=torch.long, device=self.device).view(-1, 1)
            next_state_values[no_final_mask] = \
                self.t_net(
                    **self.convert_to_inputs_for_update(batch_state_next, batch_max_action)
                )[0].gather(dim=1, index=max_labels).detach().view(-1)
            del max_labels
        else:
            raise ValueError('dqn_type: dqn/ddqn')
        del batch_max_action, batch_max_q_value, no_final_mask
        
        # rceward
        rewards = torch.tensor(batch.reward, dtype=torch.float, device=self.device)
        
        # expected Q values
        assert next_state_values.size() == rewards.size()
        expected_state_action_values = next_state_values * self.eps_gamma + rewards
        del rewards

        # state_action value of q_net
        labels = torch.tensor([action.label if state_next != None else state.pred_label \
                                for state, action, state_next in zip(batch.state, batch.action, batch.next_state)],
                               dtype=torch.long, device=self.device).view(-1, 1)
        state_action_values = self.q_net(
            **self.convert_to_inputs_for_update(batch.state, batch.action)
        )[0].gather(dim=1, index=labels).view(-1)
        del labels
        
        # compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values,
                                reduction='none')
        if isweights != None:
            isweights = torch.tensor(isweights, dtype=torch.float, device=self.device)
            assert loss.size() == isweights.size()
            mloss = (loss * isweights).mean()
        else:
            mloss = loss.mean()
        
        # optimize model
        mloss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(),
                                       self.args.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.q_net.zero_grad()
        
        self.steps_done += 1

        return loss.detach().cpu().data


    @property
    def epsilon_greedy(self):
        sample = random.random()
        threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        return sample > threshold


    def select_action(self,
                      batch_state: List[State],
                      batch_actions: List[List[Action]],
                      net: nn.Module,
                      is_eval: bool=False) -> Tuple[List[Action], List[float]]:
        assert len(batch_state) == len(batch_actions)
        MAX_SIZE = 256 * 256

        if is_eval: net.eval()

        q_values = None
        with torch.no_grad():
            batch_inputs = self.convert_to_inputs_for_select_action(batch_state, batch_actions)
            K, DIM = list(batch_inputs.values())[0].size()
            INTERVAL = MAX_SIZE // DIM
            q_values = [net(
                            **dict(map(lambda x: (x[0], x[1][i:i + INTERVAL].to(self.device)),
                                       batch_inputs.items()))
                        )[0] for i in range(0, K, INTERVAL)]
            q_values = torch.cat(q_values, dim=0)
            assert q_values.size(0) == K
        
        batch_selected_action, offset = [], 0
        for state, actions in zip(batch_state, batch_actions):
            cur_q_values = q_values[offset:offset + len(actions)]
            if self.epsilon_greedy or is_eval:
                max_action = cur_q_values.argmax().item()
                sent_id = max_action // self.args.num_labels
                label_id = max_action % self.args.num_labels
            else:
                #pdb.set_trace()
                sent_id = random.randint(0, max(0, len(actions) - 1))
                label_id = random.randint(0, self.args.num_labels - 1)
            
            if actions[0].sentence is None:
                assert len(actions) == 1 and cur_q_values.size(0) == 1
                action = Action(sentence=None, label=label_id)
            else:
                action = Action(sentence=actions[sent_id].sentence, label=label_id)

            q = cur_q_values[sent_id, label_id].item()
            batch_selected_action.append((action, q))
            offset += len(actions)

        return tuple(zip(*batch_selected_action))


    def save(self, output_dir: str) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        q_net = self.q_net.module if hasattr(self.q_net, 'module') else self.q_net
        state_dict = {
            'q_net': q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, os.path.join(output_dir, 'model.bin'))
        self.logger.info(f'Saving checkpoint to {output_dir}')


    def load(self, input_dir: str) -> None:
        q_net = self.q_net.module if self.args.n_gpu > 1 else self.q_net
        state_dict = torch.load(os.path.join(input_dir, 'model.bin'),
                                map_location=lambda storage, loc: storage)
        q_net.load_state_dict(state_dict['q_net'])
        self.steps_done = state_dict['steps_done']
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(['scheduler'])
        self.soft_update_of_target_network(1)
        self.logger.info(f'Loading model from {input_dir}')


    def eval(self):
        self.q_net.eval()
        self.t_net.eval()


    def soft_update_of_target_network(self, tau: float=1.) -> None:
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(self.t_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    
    def token(self, text_sequence: str, max_length: int=None) -> Tuple[int]:
        return NotImplementedError()


    def convert_to_inputs_for_select_action(self, state: State, actions: List[Action]) -> List[dict]:
        return NotImplementedError()


    def convert_to_inputs_for_update(self, states: List[State], actions: List[Action]) -> dict:
        return NotImplementedError()

