#!/usr/bin/env python3
# coding=utf-8
from data.structure import State, Evidence, Action
from typing import Tuple

class BaseEnv:
    def __init__(self, K=5):
        self.K = K
    
    def jaccard(self, e1: Evidence, e2: Evidence) -> float:
        sents1 = set(map(lambda sent: tuple(sent.id), e1))
        sents2 = set(map(lambda sent: tuple(sent.id), e2))
        return (len(sents1 & sents2) + 1.0) / (len(sents1 | sents2) + 1.0)

    def score(self, state: State) -> float:
        return NotImplementedError()

    def reward(self, state_now: State, state_next: State) -> float:
        return NotImplementedError()

    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        return NotImplementedError()


class DuEnv(BaseEnv):
    def __init__(self, K=5):
        super(DuEnv, self).__init__(K)

    def score(self, state: State) -> float:
        I = 1 if state.label == state.pred_label else -1
        if len(state.evidence_set):
            max_jaccard = max([self.jaccard(evi, state.candidate) for evi in state.evidence_set])
        else:
            max_jaccard = self.jaccard([], state.candidate)
        return I * max_jaccard

    def reward(self, state_now: State, state_next: State) -> float:
        if len(state_now.candidate) == self.K:
            return self.score(state_now)
        else:
            return self.score(state_now) - self.score(state_next)
    
    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        done = len(state.candidate) >= self.K
        state_next = State(claim=state.claim,
                           label=state.label,
                           evidence_set=state.evidence_set,
                           candidate=state.candidate + [action.sentence],
                           pred_label=action.label) if not done else None
        return state_next, self.reward(state, state_next), done

