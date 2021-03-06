#!/usr/bin/python3
# coding: utf-8
import random
import numpy as np
from typing import List, Tuple
from data.structure import Transition


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = [None] * capacity
        self.position = 0
        self.length = 0

    def reset(self) -> None:
        self.position = 0
        self.length = 0

    def push(self, item: Transition) -> None:
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory[:self.length],
                             min(self.length, batch_size))

    def __len__(self) -> int:
        return self.length


class PrioritizedReplayMemory(ReplayMemory):
    epsilon: float = 0.01 # small amount to avoid zero priority
    alpha: float = 0.6 # [0~1] convert the importance of TD error to priority
    beta: float = 0.4 # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error
    
    def __init__(self, capacity: int) -> None:
        super(PrioritizedReplayMemory, self).__init__(capacity)
        # sum_tree
        self.tree = [0.] * (2 * capacity - 1)

    def reset(self) -> None:
        super().reset()
        self.tree = [0.] * (2 * self.capacity - 1)

    def push(self, item: Transition) -> None:
        '''
        对于第一条存储的数据，我们认为它的优先级P是最大的，同时，
        对于新来的数据，我们也认为它的优先级与当前树中优先级最大的经验相同
        '''
        idx = self.position + self.capacity - 1
        super().push(item)
        priority = max(self.tree[-self.capacity:])
        if priority == 0:
            priority = self.abs_err_upper
        self.update_sumtree(idx, priority, is_error=False)

    def sample(self, batch_size: int) -> Tuple[List[int], List[float], List[Transition]]:
        idxs, isweights, batch = [], [], []
        segment = self.tree[0] / batch_size
        
        self.beta = min(1., self.beta + self.beta_increment_per_sampling)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, priority = self.get_from_sumtree(s)
            isweights.append(priority / self.tree[0])
            batch.append(self.memory[idx + 1 - self.capacity])
            idxs.append(idx)

        isweights = np.power(np.asarray(isweights) / max(min(isweights), self._get_priority(0.)),
                             -self.beta).tolist()

        return idxs, isweights, batch

    def update_sumtree(self, idx: int, value: float, is_error: bool=True) -> None:
        priority = self._get_priority(value) if is_error else value
        change = priority - self.tree[idx]
        self.tree[idx] = priority

        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def batch_update_sumtree(self, batch_idx: List[int], batch_value: List[float], is_error: bool=True) -> None:
        for idx, value in zip(batch_idx, batch_value):
            self.update_sumtree(idx, value, is_error)

    def get_from_sumtree(self, x: float) -> Tuple[int, float]:
        cur = 0
        while 2 * cur + 1 < len(self.tree):
            left = 2 * cur + 1
            right = left + 1
            if self.tree[left] >= x:
                cur = left
            else:
                x -= self.tree[left]
                cur = right
        return cur, self.tree[cur]

    def _get_priority(self, error):
        return min(abs(error) + self.epsilon, self.abs_err_upper) ** self.alpha
