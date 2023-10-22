import torch.nn as nn
import torch
import numpy as np

import collections
import utils

class Model_SentAvg(nn.Module):
    """ 
    class for simple baseline, averaging all sentences, each sentence consists of averaged all tokens
    """
    def __init__(self, encoder, output_size, activation_fn="gelu", dr_rate=0.1):
        super().__init__()

        ## encoder (should be moved out as a class)
        self.encoder = encoder
        # self.encoder_out_size = encoder.config.hidden_size
        self.encoder_out_size = 768  # test

        ##
        self.activation_fn = utils.get_activation_fn(activation_fn)

        ## linear layers
        # print(self.encoder_out_size)
        # print(ewkfjlwef)

        linear_1_out = self.encoder_out_size // 2
        self.linear_1 = nn.Linear(self.encoder_out_size, linear_1_out)
        nn.init.xavier_uniform_(self.linear_1.weight)

        self.linear_out = nn.Linear(linear_1_out, output_size)
        nn.init.xavier_uniform_(self.linear_out.weight)

        self.dropout = nn.Dropout(dr_rate)

    def forward(self, batch):

        # input and meta data
        input_ids = batch["input_ids"]  # zero padded as the longest length in the batch
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.size(0)

        sent_num = batch["sent_num"]
        len_sents = batch["len_sents"]
        max_sent_num_batch = max(sent_num)

        ## Stage1: encode all sentences        
        # iterate each sentence to encode (due to memory shortage, for efficient modeling)
        encoded_sents = []
        attn_sents = []
        for idx_sent in range(max_sent_num_batch):
            cur_sent_ids = input_ids[:, idx_sent, :]
            cur_attn_mask = attention_mask[:, idx_sent, :]

            # encoding
            encoder_out = self.encoder(cur_sent_ids, cur_attn_mask, output_attentions=True, output_hidden_states=True)
            repr_encoded = encoder_out["last_hidden_state"]  # (batch_size, max_sent_len, dim)
            attn_layers = encoder_out["attentions"]  # (batch, layer_num, max_sent_len, max_sent_len)

            # sentence repr by averaging
            cur_sent_lens = len_sents[:, idx_sent]
            cur_sent_lens = cur_sent_lens + 1e-9 # prevent zero division
            repr_encoded = repr_encoded * cur_attn_mask.unsqueeze(2)  # to make sure, might not be necessary since used in encoder
            cur_avg_repr = torch.div(torch.sum(repr_encoded, dim=1), cur_sent_lens.unsqueeze(1))  # (batch_size, dim)  for i-th sent

            # attention scores, averaging all layers
            attn_last_layer = attn_layers[-1]  # the last layer of mh attentions  
            attn_last_avg = torch.div(torch.sum(attn_last_layer, dim=1), attn_last_layer.shape[1])  # average all mh
            cur_attn_last_avg = attn_last_avg * cur_attn_mask.unsqueeze(2)  # masking for actual input

            # print(cur_avg_repr.shape)  # (batch, dim)
            # print(attn_last_avg.shape)  # (batch, max_sent_len, max_sent_len)

            encoded_sents.append(cur_avg_repr)
            attn_sents.append(cur_attn_last_avg)

        encoded_sents = torch.stack(encoded_sents, dim=0).transpose(1, 0)
        attn_sents = torch.stack(attn_sents, dim=0).transpose(1, 0)

        ## Stage 2: averaging all sent reprs 
        ilc_vec = torch.div(torch.sum(encoded_sents, dim=1), sent_num.unsqueeze(1))

        ## Stage 3: FC
        fc_out = self.linear_1(ilc_vec)
        fc_out = self.activation_fn(fc_out)
        fc_out = self.dropout(fc_out)

        fc_out = self.linear_out(fc_out)

        return fc_out