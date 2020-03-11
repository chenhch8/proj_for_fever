#!/usr/bin/python3
# coding: utf-8
import random
from typing import List

from data.structure import Transition

class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = [0.] * capacity
        self.position = 0
        self.length = 0

    def reset(self) -> None:
        self.position = 0
        self.length = 0

    def push(self, item: Transition) -> None:
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size) -> List[Transition]:
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self) -> int:
        return self.length


class PrioritizedReplayMemory(ReplayMemory):
    e: float = 0.01
    alpha: float = 0.6
    belta: float = 0.4
    beta_increment_per_sampling = 0.001
    
    def __init__(self, capacity: int) -> None:
        super(PrioritizedReplayMemory, self).__init__(capacity)
        # sum_tree
        self.tree = [0.] * (2 * capacity - 1)

    def reset(self) -> None:
        super().reset()
        self.tree = [0.] * (self.capacity - 1)

    def push(self, priority: float, item: Transition) -> None:
        pass

    def sample(self, batch_size: int) -> Tuple[List[float], List[Transition]]:
        pass

    def update_sumtree(self, idx: int, priority: float) -> None:
        pass

    def sample_from_tree(self, x: List[float]) -> None:
        pass
