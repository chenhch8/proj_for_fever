#coding: utf8 
from collections import namedtuple
from typing import List

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_actions'))
Action = namedtuple('Action', ('label', 'sentence'))
State = namedtuple('State', ('claim', 'label', 'evidence_set', 'candidate', 'pred_label'))
Sentence = namedtuple('Sentence', ('id', 'tokens', 'str'))
Claim = namedtuple('Claim', ('id', 'str', 'tokens'))
Evidence = List[Sentence]
EvidenceSet = List[Evidence]

Sentence.__new__.__defaults__ = (None, None)
