#!/usr/bin/env python
# coding=utf-8
import torch
from torch import nn
from transformers import BertModel

class BertForFEVER(nn.Module):
    def __init__(self, BERT_MODEL='bert-base-uncased', num_labels=3):
        super(BertForFEVER, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        config = self.bert.config
        self.fc = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_labels, bias=True),
#             nn.Softmax(dim=1)
        )
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
#             elif isinstance(module, BERTLayerNorm):
#                 module.beta.data.normal_(mean=0.0, std=config.initializer_range)
#                 module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.fc.apply(init_weights)
    def forward(self, input_ids, token_type_ids, attention_mask, position_ids):
        #all_hidden_state, pooler_output
        _, pooler_output = self.bert(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     position_ids=position_ids)
        logits = self.fc(pooler_output)
        scores = torch.softmax(logits, dim=1)
        return scores, logits
