#!/usr/bin/env python3
# coding=utf-8
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel

from itertools import chain

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_layers_of_classifier = config.num_layers_of_classifier

        self.bert = BertModel(config)
        self.classifier = nn.Sequential(
            chain(
                *[[nn.Dropout(config.hidden_dropout_prob), \
                   nn.Linear(config.hidden_size,
                             config.hidden_size if i < self.num_layers_of_classifier - 1 else self.num_labels)] \
                  + [nn.ReLU()] if i < self.num_layers_of_classifier - 1 else []
                for i in range(self.num_layers_of_classifier)]
            )
        )

        self.init_weights()

    def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
