#!/usr/bin/python3
# coding: utf-8
import random
from typing import List

from data_structure import Transition

class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def reset(self) -> None:
        self.position = 0

    def push(self, item) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> List[Transition]:
        return random.sample(self.memory, min(len(self.memory), self.capacity))

    def __len__(self) -> int:
        return len(self.memory)
