#!/usr/bin/python3
# coding: utf-8
import random
import math
import numpy as np
from collections import defaultdict
from typing import List, Tuple
from data.structure import Transition


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = [None] * capacity
        self.position = 0
        self.length = 0
    
    def epsilon_greedy(self) -> bool:
        # epsilon greedy
        sample = random.random()
        threshold = 0.05 + (1. - 0.05) * math.exp(-1. * self.eps_count / 1000)
        self.eps_count += 1
        return sample < threshold

    def reset(self) -> None:
        self.position = 0
        self.length = 0
        self.sequences = {}
        self.eps_count = 0

    def push(self, item: Transition) -> None:
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def push_sequence(self, key, sequences: List[List[Transition]]):
        if not len(sequences): return
        if key not in self.sequences:
            self.sequences[key] = []
        self.sequences[key].extend(sequences)

    def sample(self, batch_size: int) -> List[Transition]:
        batch = []
        if self.epsilon_greedy:
            batch += self.sample_from_sequences()[:batch_size]
        if len(batch) < batch_size:
            batch += random.sample(self.memory[:self.length],
                                   min(batch_size - len(data), self.length))
        return batch

    def sample_from_sequences(self):
        data = []
        for sequences in self.sequences.values():
            data += random.choice(sequences)
        random.shuffle(data)
        return data

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

    def sample(self, batch_size: int) -> Tuple[List[int], List[Transition]]:
        idxs, batch = [], []
        segment = self.tree[0] / batch_size
        
        self.beta = min(1., self.beta + self.beta_increment_per_sampling)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, priority = self.get_from_sumtree(s)
            batch.append(self.memory[idx + 1 - self.capacity])
            idxs.append(idx)

        return idxs, batch

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
            if idx == -1: continue
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


class ReplayMemoryWithLabel:
    def __init__(self, capacity: int, num_labels: int=3, proportion: List[float]=[2., 2., 1.]) -> None:
        self.proportion = np.asarray(proportion) / sum(proportion)
        self.capacity_per_label = list(map(int, self.proportion * capacity))
        print('capacity_per_label: ', self.capacity_per_label)
        self.replay_memories = [ReplayMemory(capacity) for capacity in self.capacity_per_label]

    def reset(self):
        for replay_memory in self.replay_memories:
            replay_memory.reset()

    def push(self, label: int, item: Transition) -> None:
        self.replay_memories[label].push(item)

    def sample(self, batch_size: int) -> List[Transition]:
        sizes = list(map(int, batch_size * self.proportion))
        sizes[1] = batch_size - sizes[0] - sizes[2]
        assert sum(sizes) == batch_size
        
        batch = []
        for replay_memory, size in zip(self.replay_memories, sizes):
            batch += replay_memory.sample(size)
        random.shuffle(batch)

        return batch
    
    def __len__(self):
        return min([len(replay_memory) for replay_memory in self.replay_memories])


class PrioritizedReplayMemoryWithLabel(ReplayMemoryWithLabel):
    def __init__(self, capacity: int, num_labels: int=3, proportion: List[float]=[2., 2., 1.]) -> None:
        super(PrioritizedReplayMemoryWithLabel, self).__init__(capacity, num_labels, proportion)
        self.replay_memories = [PrioritizedReplayMemory(capacity) for capacity in self.capacity_per_label]

    def sample(self, batch_size: int) -> Tuple[List[Tuple[int, int]], List[float], List[Transition]]:
        sizes = list(map(int, batch_size * self.proportion))
        sizes[1] = batch_size - sizes[0] - sizes[2]
        assert sum(sizes) == batch_size
        
        idxs, isweights, batch = [], [], []
        for label, (replay_memory, size) in enumerate(zip(self.replay_memories, sizes)):
            _idxs, _isweights, _batch = replay_memory.sample(size)
            idxs += list(zip([label] * len(_idxs), _idxs))
            isweights += _isweights
            batch += _batch
        data = list(zip(idxs, isweights, batch))
        random.shuffle(data)
        
        return tuple(zip(*data))
    
    def batch_update_sumtree(self, batch_idx: List[Tuple[int, int]], batch_value: List[float], is_error: bool=True) -> None:
        for (label, idx), value in zip(batch_idx, batch_value):
            self.replay_memories[label].update_sumtree(idx, value, is_error)
