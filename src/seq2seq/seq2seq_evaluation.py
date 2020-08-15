#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Poject: zgwangNLP 
# AUTHOR : Zhiguo Wang (zhiguow@amazon.com)
# Created by zhiguow at 2/24/20
import json
import argparse
import os


def evaluate(in_path):
    # prepare a reference file and prediction file
    import spacy
    import preprocess_utils
    nlp = spacy.load('en_core_web_sm', disable=['parser'])
    ref_file = open(in_path + ".ref", 'wt', encoding='utf-8')
    pred_file = open(in_path + ".pred", 'wt', encoding='utf-8')
    with open(in_path, 'rt', encoding='utf-8') as in_file:
        for line in in_file:
            instance = json.loads(line.strip())
            ref_words = preprocess_utils.word_tokenize(instance['tgt'], nlp)
            ref_file.write(" ".join(ref_words) + "\n")

            pred_words = preprocess_utils.word_tokenize(instance['predictions'][0]['text'], nlp)
            pred_file.write(" ".join(pred_words) + "\n")
    ref_file.close()
    pred_file.close()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(base_dir, "multi-bleu.perl")

    os.system('%s %s.ref < %s.pred' % (eval_script, in_path, in_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_path', type=str)

    args, unparsed = parser.parse_known_args()
    print('Evaluating on ', args.in_path)
    evaluate(args.in_path)
    print('DONE!')


