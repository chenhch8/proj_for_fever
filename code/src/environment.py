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
                     candidate=state.candidate + [action.sentence],
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


class ChenEnv(BaseEnv):
    def __init__(self, label2id, K=5):
        super(ChenEnv, self).__init__(K)
        self.label2id = label2id

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

        if state.label == self.label2id['NOT ENOUGH INFO']: # N
            if cond1: return 1
            elif self.is_done(state): return -1
            else: return 0
        else: # T/F
            #return 1. if cond1 and cond2 else -1
            if cond1 and cond2:
                return 1.
            elif cond1 and not cond2:
                return 0.
            elif self.is_done(state):
                return -1.
            else:
                return 0.

    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        state_next = BaseEnv.new_state(state, action)
        done = self.is_done(state_next)
        return state_next, self.reward(state_next, action), done

