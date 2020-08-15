#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Poject: zgwangNLP 
# AUTHOR : Zhiguo Wang (zhiguow@amazon.com)
# Created by zhiguow at 2/17/20


class Seq2SeqState(object):
    def __init__(
            self,
            prev_state=None,
            encoder_padding_mask=None, cached_encoder_keys=None, cached_encoder_values=None,
            batch_id=0):
        self.scores = []
        self.token_ids = []
        if prev_state is not None:
            self.batch_id = prev_state.batch_id
            self.scores.extend(prev_state.scores)
            self.token_ids.extend(prev_state.token_ids)
            self.encoder_padding_mask = prev_state.encoder_padding_mask
            self.cached_encoder_keys = prev_state.cached_encoder_keys
            self.cached_encoder_values = prev_state.cached_encoder_values
            if prev_state.cached_decoder_keys is not None:
                num_decoder_layer = len(prev_state.cached_decoder_keys)
                self.cached_decoder_keys = [[] for _ in range(num_decoder_layer)]
                self.cached_decoder_values = [[] for _ in range(num_decoder_layer)]
                for i in range(num_decoder_layer):
                    self.cached_decoder_keys[i].extend(prev_state.cached_decoder_keys[i])
                    self.cached_decoder_values[i].extend(prev_state.cached_decoder_values[i])
        else:
            self.batch_id = batch_id
            self.scores = [0.0]
            self.token_ids = [0]  # <s>
            self.encoder_padding_mask = encoder_padding_mask  # [src_len]
            self.cached_encoder_keys = cached_encoder_keys  # list of [src_len, model_dim]
            self.cached_encoder_values = cached_encoder_values  # list of [src_len, model_dim]
            self.cached_decoder_keys = None  # list of list [1, model_dim]
            self.cached_decoder_values = None  # list of list [1, model_dim]

    def is_terminal(self, eos_id=2):
        if len(self.token_ids) > 0 and self.token_ids[-1] == eos_id:
            return True
        return False

    def get_score(self):
        if len(self.scores) == 0:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def cache_decoder_key_value(self, idx, keys, values):
        '''

        :param idx:
        :param keys: list of [1, batch_size, dim]
        :param values: list of [1, batch_size, dim]
        :return:
        '''
        num_decoder_layer = len(keys)
        if self.cached_decoder_keys is None:
            self.cached_decoder_keys = [[] for _ in range(num_decoder_layer)]
            self.cached_decoder_values = [[] for _ in range(num_decoder_layer)]

        for i in range(num_decoder_layer):
            self.cached_decoder_keys[i].append(keys[i][:, idx, :])  # 2nd dim is for batch_size
            self.cached_decoder_values[i].append(values[i][:, idx, :])  # 2nd dim is for batch_size

    def apply_action(self, log_prob, tok_id):
        self.scores.append(log_prob)
        self.token_ids.append(tok_id)


