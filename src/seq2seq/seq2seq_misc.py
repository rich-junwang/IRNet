#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Poject: zgwangNLP 
# AUTHOR : Zhiguo Wang (zhiguow@amazon.com)
# Created by zhiguow at 2/18/20

import json
import argparse
import sagemaker_utils
import os

def convert_RC_to_QG(in_path, out_path):
    out_file = open(out_path, 'wt', encoding='utf-8')
    with open(in_path, 'rt', encoding='utf-8') as in_file:
        for line in in_file:
            instance = json.loads(line.strip())
            context = instance['context']
            answer_text = instance['answers'][0]['text']
            answer_start = instance['answers'][0]['answer_start']
            answer_end = answer_start + len(answer_text)
            src_text = context[:answer_start] + "<A> " + answer_text + " </A>" + context[answer_end:]
            new_instance = {
                'id': instance['qid'],
                'src': src_text,
                'tgt': instance['question']
            }
            out_file.write(json.dumps(new_instance) + "\n")
    out_file.close()


def launch_training_job(local=False):
    instance_count = 1
    init_path = "s3://zhiguow-data/pretrained_model/bart_large_model/"
    instance_type = 'ml.p3.2xlarge'
    folder_name = 'exp20200219'
    job_name = "question-generation-" + folder_name
    prefix = "s3://zhiguow-data/question_generation/" + folder_name
    config_path = prefix + "/config/"
    train_path = prefix + "/train/"
    test_path = prefix + "/val/"
    if local:
        instance_type = 'local'
        train_path = "file:///home/ubuntu/zhiguow_EBS/exps/seq2seq/QG/"
        test_path = "file:///home/ubuntu/zhiguow_EBS/exps/seq2seq/QG/"
        config_path = "file:///home/ubuntu/zhiguow_EBS/exps/seq2seq/QG/"
        init_path = "file:///home/ubuntu/zhiguow_EBS/exps/zgwangNLP/models/bart_large_model"

    sagemaker_utils.launch_training_job(
        job_name,
        train_path,
        test_path,
        config_path,
        init_path,
        instance_type,
        instance_count)


def launch_a_serving_job(local=False):
    trained_model_location = "s3://sagemaker-us-east-2-510951828445/question-generation-exp20200219-2020-02-19-15-02-07-771/output/model.tar.gz"
    instance_type = 'ml.p3.2xlarge'
    if local:
        instance_type = 'local'
        trained_model_location = "file:///home/ubuntu/zhiguow_EBS/exps/seq2seq/QG/models/model.tar.gz"
    predictor = sagemaker_utils.launch_serving_job(trained_model_location, instance_type)
    return predictor


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
        pass
    ref_file.close()
    pred_file.close()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(base_dir, "multi-bleu.perl")

    os.system('%s %s.ref < %s.pred' % (eval_script, in_path, in_path))


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--in_path', type=str)
    parser.add_argument('--out_path', type=str)

    args, unparsed = parser.parse_known_args()
    # in_path = "/Users/zhiguow/Downloads/SQuAD_dev.jsonl"
    # out_path = "/Users/zhiguow/Downloads/val.json"

    print('Converting for ', args.in_path)
    convert_RC_to_QG(args.in_path, args.out_path)
    '''

    launch_training_job(local=False)

    # predictor = launch_a_serving_job(local=True)

    # in_path = "/home/ubuntu/zhiguow_EBS/exps/seq2seq/QG/prediction.1"
    # evaluate(in_path)
    print('DONE!')
