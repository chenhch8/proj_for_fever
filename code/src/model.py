#!/usr/bin/env python
# coding=utf-8
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel
import numpy as np

#import pdb

# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    #mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


class RELU(nn.Module):
    def __repr__(self):
        return 'RELU()'
    def forward(self, x):
        """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
            For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


class BertForFEVER(nn.Module):
    def __init__(self, BERT_MODEL='bert-base-uncased', num_labels=3, insert_k_layer=8):
        super(BertForFEVER, self).__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.config = self.bert.config
        self._insert_k_layer = min(insert_k_layer, self.config.num_hidden_layers - 1)
        self.proj_sentence = nn.Sequential(
            nn.Linear(self.config.hidden_size * 4,
                      self.config.hidden_size, bias=True),
            RELU(),
            nn.Dropout(self.config.hidden_dropout_prob),
        )
        self.ranking_score = nn.Sequential(
            nn.Linear(self.config.hidden_size, 1, bias=True),
            nn.Dropout(self.config.hidden_dropout_prob)
        )
        self.classification = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size * 4,
                      self.config.hidden_size,
                      bias=True),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, num_labels, bias=True)
        )
        self.attn_dropout = nn.Dropout(self.config.attention_probs_dropout_prob)

        # 重写forward函数
        self.bert.encoder.forward = self.__bert_encoder_forward_modified__
        self.bert.forward = self.__bert_forward_modified__
        
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.bias.data.zero_()
        
        self.proj_sentence.apply(init_weights)
        self.ranking_score.apply(init_weights)
        self.classification.apply(init_weights)

    #def isnan(self, tensor):
    #    return torch.isnan(tensor).sum() > 0

    def __bert_encoder_forward_modified__(self,
                                          hidden_states,
                                          attention_mask=None,
                                          head_mask=None,
                                          P_mask=None,
                                          H_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        k_hidden_states = None
        
        for i, layer_module in enumerate(self.bert.encoder.layer):
            if self.config.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # modify here
            if i == self._insert_k_layer:
                k_hidden_states = hidden_states
                hidden_states = self.__alignment_layer__(hidden_states, P_mask, H_mask)
            
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.config.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.config.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.config.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs, k_hidden_states


    def __bert_forward_modified__(self,
                                  input_ids,
                                  attention_mask=None,
                                  token_type_ids=None,
                                  position_ids=None,
                                  head_mask=None,
                                  P_mask=None,
                                  H_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.bert.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # modify here
        embedding_output = self.bert.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs, k_sequence_output = \
                self.bert.encoder(embedding_output,
                                  attention_mask=extended_attention_mask,
                                  head_mask=head_mask,
                                  P_mask=P_mask,
                                  H_mask=H_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(k_sequence_output) # modify here

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


    def __alignment_layer__(self, hidden_states, P_mask, H_mask):
        batch, N, d = hidden_states.size()
        # (batch, N, d)
        P_attn, _ = self.__attention_layer__(query=hidden_states,
                                             key=hidden_states.permute([0, 2, 1]),
                                             value=hidden_states,
                                             mask=P_mask.matmul(H_mask.permute([0, 2, 1])))
        H_attn, _ = self.__attention_layer__(query=hidden_states,
                                             key=hidden_states.permute([0, 2, 1]),
                                             value=hidden_states,
                                             mask=H_mask.matmul(P_mask.permute([0, 2, 1])))
        # (batch, N, d)
        HP = hidden_states
        HP_attn = H_attn * H_mask + P_attn * P_mask
        assert HP_attn.size() == torch.Size([batch, N, d])
        
        proj_HP = self.proj_sentence(torch.cat([HP,
                                                HP_attn,
                                                HP - HP_attn,
                                                HP * HP_attn], dim=-1))  # (batch, N, d)
        assert proj_HP.size() == torch.Size([batch, N, d])
        proj_HP = proj_HP + hidden_states
        
        return proj_HP


    def __attention_layer__(self, query, key, value, mask):
        '''
        Args:
        	query: [B, L_q, D_q]
        	key: [B, D_q, L_v]
        	value: [B, L_v, D_v]，一般来说就是k
        	mask: [B, L_q, L_v]

        Returns:
        	上下文张量和attetention张量
        '''
        mask = (1.0 - mask) * -10000.0
        mask_attn_scores = query.matmul(key).div(np.sqrt(key.size(-1))) + mask
        mask_attn_probs = torch.softmax(mask_attn_scores, dim=-1)
        mask_attn_probs = self.attn_dropout(mask_attn_probs)
        attn_query = mask_attn_probs.matmul(value)
        return attn_query, mask_attn_probs


    def forward(self, input_ids, token_type_ids, attention_mask, position_ids=None):
        segment_mask = token_type_ids[:, :, None].type_as(input_ids).float()  # (batch, N, 1)
        mask = attention_mask[:, :, None].type_as(input_ids).float() # (batch. N, 1)
        
        P_mask = segment_mask  # evidence, (batch, N, 1)
        H_mask = (1 - segment_mask) * mask  # claim, (batch, N, 1)
        H_mask[:, 0] = 0  # 去除 [CLC]
        
        # (batch, N, d), (batch, d)
        last_hidden_state, pooler_output = self.bert(input_ids=input_ids,
                                                     token_type_ids=token_type_ids,
                                                     attention_mask=attention_mask,
                                                     position_ids=position_ids,
                                                     P_mask=P_mask,
                                                     H_mask=H_mask)
        batch, N, d = last_hidden_state.size()

        premise = last_hidden_state * P_mask  # evidence, (batch, N, d)
        hypothesis = last_hidden_state * H_mask # claim, (batch, N, d)
        assert premise.size() == hypothesis.size() and premise.size() == last_hidden_state.size()


        avg_premise = premise.sum(dim=1).div(P_mask.sum(dim=1))  # (batch, d)
        #del new_P, P_mask
        avg_hypothesis = hypothesis.sum(dim=1).div(H_mask.sum(dim=1))  # (batch, d)
        #del new_H, H_mask
        assert avg_premise.size() == torch.Size([batch, d]) and avg_hypothesis.size() == avg_premise.size()

        max_premise, _ = replace_masked(premise, P_mask, -1e7).max(dim=1)
        max_hypothesis, _ = replace_masked(hypothesis, H_mask, -1e7).max(dim=1)
        assert max_premise.size() == torch.Size([batch, d]) and max_hypothesis.size() == max_premise.size()

        cat_premise_hypothesis = torch.cat([avg_premise, max_premise, avg_hypothesis, max_hypothesis], dim=1)

        logits_clc = self.classification(cat_premise_hypothesis)
        scores_clc = torch.softmax(logits_clc, dim=1)  # (batch, 3)
        
        logits_rank = self.ranking_score(pooler_output) # (batch, 1)

        return scores_clc, logits_clc, logits_rank


if __name__ == '__main__':
    from transformers import BertTokenizer
    from processor import convert_example_to_feature
    from feverdataset import Example

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    model = BertForFEVER('./data/bert-base-uncased').to('cuda:1')
    sentences = [['Hello, dog', 'my dog is cute'],
                 ['my dog is quite cute', 'simple test']]
    features = [convert_example_to_feature((Example(i, sent[0], sent[1]),
                                           11, tokenizer)) for i, sent in enumerate(sentences)]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to('cuda:1')
    all_input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long).to('cuda:1')
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to('cuda:1')

    print(all_input_ids.size(), all_input_masks.size(), all_segment_ids.size())
    print(all_input_ids)
    print(all_input_masks)
    print(all_segment_ids)

    outputs = model(input_ids=all_input_ids,
                    attention_mask=all_input_masks,
                    token_type_ids=all_segment_ids)
    scores_clc, scores_rel, logits_clc, logits_rank = outputs
    print(scores_clc.size(), logits_clc.size())
    print(scores_clc, logits_clc)
    print(scores_rel.size(), logits_rank.size())
    print(scores_rel, logits_rank)
