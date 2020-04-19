#!/usr/bin/env python3
# coding=utf-8
from data.structure import State, Evidence, Action
from typing import Tuple, Set

def get_id_from_evidence(e: Evidence) -> Set[Tuple[str, int]]:
    return set(map(lambda sent: tuple(sent.id), e))

class BaseEnv:
    def __init__(self, K=5):
        self.K = K
    
    def jaccard(self, e1: Evidence, e2: Evidence) -> float:
        sents1 = get_id_from_evidence(e1)
        sents2 = get_id_from_evidence(e2)
        return (len(sents1 & sents2) + 1.0) / (len(sents1 | sents2) + 1.0)

    @classmethod
    def new_state(cls, state: State, action: Action) -> State:
        return State(claim=state.claim,
                     label=state.label,
                     evidence_set=state.evidence_set,
                     candidate=state.candidate + [action.sentence] \
                                if action.sentence is not None else state.candidate,
                     pred_label=action.label,
                     count=state.count + 1)
    
    def is_done(self, state: State) -> bool:
        return state.count == self.K

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
        if self.is_done(state_now):
            return self.score(state_now)
        else:
            return self.score(state_now) - self.score(state_next)
    
    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        done = self.is_done(state)
        state_next = BaseEnv.new_state(state, action) if not done else None
        return state_next, self.reward(state, state_next), done


class ChenEnv(BaseEnv):
    def __init__(self, K=5):
        super(ChenEnv, self).__init__(K)

    def reward(self, state: State, action: Action) -> float:
        if self.is_done(state) or action.sentence is None:
            if self.is_done(state):
                cond1 = state.pred_label == state.label
            else:
                cond1 = state.label == action.label
            candidate = get_id_from_evidence(state.candidate)
            cond2 = any([len(get_id_from_evidence(evi) - candidate) == 0 \
                            for evi in state.evidence_set])
        else:
            cond1 = state.label == action.label
            cond2 = any([action.sentence in evi for evi in state.evidence_set])

        if state.label == 2: # N
            return 1. if cond1 else -1.
        else: # T/F
            if cond1 and cond2:
                return 1.
            elif cond1 and not cond2:
                return 0.
            elif not cond1 and cond2:
                return -1.
            elif not (cond1 or cond2):
                return -2.
            else:
                return ValueError('condition error')

    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        done = self.is_done(state)
        state_next = BaseEnv.new_state(state, action) if not done else None
        return state_next, self.reward(state, action), done

