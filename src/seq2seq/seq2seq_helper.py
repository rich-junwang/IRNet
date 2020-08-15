#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Poject: zgwangNLP 
# AUTHOR : Zhiguo Wang (zhiguow@amazon.com)
# Created by zhiguow at 2/18/20

import os
from task_router_utils import TaskHelper
from preprocesser_utils import BasePreprocesser
import pad_utils
import json
from seq2seq import gpt2_tokenizaton_utils


class Seq2SeqHelper(TaskHelper):
    def __init__(self, config):
        super(Seq2SeqHelper, self).__init__(config)
        self.seq2seq_preprocesser = Seq2SeqPreprocesser(config)
        self.preprocessers = [self.seq2seq_preprocesser]

    def enhance_config(self, config):
        config = super().enhance_config(config)
        if 'max_src_len' not in config.keys():
            config['max_src_len'] = 1024

        if 'max_tgt_len' not in config.keys():
            config['max_tgt_len'] = 1024

        if 'use_pointwise_accuracy' not in config.keys():
            config['use_pointwise_accuracy'] = True

        if 'beam_size' not in config.keys():
            config['beam_size'] = 5

        return config

    def read_all_instances(self, in_path):
        return super().read_all_instances(in_path)

    def dump_prediction(self, batch_data, predictions, out_file=None):
        hypothesises = predictions[-1]
        examples = batch_data['examples']
        for i, example in enumerate(examples):
            hypos = [hypo for hypo in hypothesises if hypo.batch_id == i]
            if len(hypos) == 0:
                continue
            results = []
            for hypo in hypos:
                tgt_text = self.seq2seq_preprocesser.tokenizer.decode(hypo.token_ids)
                score = hypo.get_score()
                results.append({'text': tgt_text, 'score': score})
            example['predictions'] = results
            example.pop('src_ids', None)
            example.pop('src_position_ids', None)
            if out_file is not None:
                out_file.write(json.dumps(example) + "\n")
        return examples


class Seq2SeqPreprocesser(BasePreprocesser):

    def __init__(self, config):
        super(Seq2SeqPreprocesser, self).__init__()
        self.config = config

        # load tokenizer
        vocab_path = os.path.join(self.config.pretrained_path, "roberta-large-vocab.json")
        merges_path = os.path.join(self.config.pretrained_path, "roberta-large-merges.txt")
        self.tokenizer = gpt2_tokenizaton_utils.GPT2Tokenizer(vocab_path, merges_path)

    def preprocess(self, instance, is_training=True):
        '''
            input format
            instance = {"id": "abc", "src": "xxxxxxxxx", "tgt": "xxxxxxxxx", }
        '''
        if self.config.use_pointwise_accuracy:
            is_training = True
        if "src_list" in instance.keys():
            src_ids = []
            src_position_ids = []
            for i, src_text in enumerate(instance['src_list']):
                _, src_token_ids = self.tokenizer.encode(src_text, max_len=self.config.max_src_len)
                if i != 0:
                    src_token_ids.pop(0) # remove the first <s> token
                src_ids.extend(src_token_ids)
                src_position_ids.extend(list(range(len(src_token_ids))))
            instance['src_ids'] = src_ids
            instance['src_position_ids'] = src_position_ids
        else:
            _, src_token_ids = self.tokenizer.encode(instance["src"], max_len=self.config.max_src_len)
            instance['src_ids'] = src_token_ids
            # instance['src_position_ids'] = list(range(len(src_token_ids)))
        if is_training and 'tgt' in instance.keys():
            _, tgt_token_ids = self.tokenizer.encode(instance["tgt"], max_len=self.config.max_tgt_len)
            instance['tgt_ids'] = tgt_token_ids
        return instance

    def batchify_func(self, examples, result=None, is_training=True):
        src_ids = []  # [batch_size, src_len]
        src_position_ids = []  # [batch_size, src_len]
        for example in examples:
            src_ids.append(example['src_ids'])
            if "src_position_ids" in example.keys():
                src_position_ids.append(example['src_position_ids'])
        max_src_len = max([len(src) for src in src_ids])
        src_ids = pad_utils.pad_2d_vals(src_ids, dim2_size=max_src_len, dvalue=self.tokenizer.pad_token_id)

        result = {
            'examples': examples,
            'src_ids': src_ids,
        }

        if len(src_position_ids) > 0:
            src_position_ids = pad_utils.pad_2d_vals(src_position_ids, dim2_size=max_src_len, dvalue=self.tokenizer.pad_token_id)
            result['src_position_ids'] = src_position_ids

        if self.config.use_pointwise_accuracy:
            is_training = True
        if is_training:
            tgt_ids = []  # [batch_size, tgt_len]
            for example in examples:
                tgt_ids.append(example['tgt_ids'])
            max_tgt_len = max([len(tgt) for tgt in tgt_ids])
            tgt_ids = pad_utils.pad_2d_vals(tgt_ids, dim2_size=max_tgt_len, dvalue=self.tokenizer.pad_token_id)
            result['tgt_ids'] = tgt_ids
        return result

    def prepare_batch(self, batch_data, result=None, is_training=True):
        result = {
            "input_ids": batch_data["src_ids"],
        }
        if "src_position_ids" in batch_data.keys():
            result['position_ids'] = batch_data['src_position_ids']
        if self.config.use_pointwise_accuracy:
            is_training = True
        if is_training:
            result['target_ids'] = batch_data['tgt_ids']
        return result

