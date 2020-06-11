#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from infer import get_infer
from eval_megaface import EvalMegaFace
from utils.io import _merge_results
from eval_ijbc import EvalIJBC
import time
import requests

import signal

eval_sets_dict = {
    'megaface': 'EvalMegaFace',
    'ijbc': 'EvalIJBC',
    }


model_type_dict = {
    'mxnet_fp32': 'mxnet_fp32',
    'mxnet_fp16': 'mxnet_fp16',
    }


def _load_model(args):
    print("Loading model: %s" %(args.model_path))
    net = get_infer(args)
    model_type = args.model_type
    if net is None:
        print("ERROR: Uknown model type: ", model_type)
        print("Please use the supported types: ", model_type_dict.keys())
        sys.exit(-1)
    #
    return net

def start_eval_process(eval_info):
    # append eval task
    r = requests.post("http://127.0.0.1:%s/eval"%(eval_info["args"].port), data=eval_info).json()
    if r["success"]:
        print("eval set %s evaluate process append success!"%(eval_info['eval_set']))
        print("current evaluate tasks:")
        print(r["tasks"])

    else:
        print("eval set %s evaluate process append failed, please restart manally using command [ python -c \"from argparse import Namespace; import requests; r=requests.post('http://127.0.0.1:%d/eval', data=%s).json(); print('success:', r['success']);\" ]"%(eval_info['eval_set'], eval_info['args'].port, eval_info))


def start_merge_process(merge_info):
    # append eval task
    r = requests.post("http://127.0.0.1:%d/merge"%(merge_info["args"].port), data=merge_info).json()
    if r["success"]:
        print("start merge process success %s"%(merge_info['eval_set']))
    else:
        print("merge %s process falied, please restart manally using command [ python -c \"from argparse import Namespace; import requests; r=requests.post('http://127.0.0.1:%d/merge', data=%s).json(); print('success:', r['success']);\" ]"%(merge_info['eval_set'], merge_info["args"].port, merge_info))



def _main(args):
    print("args: ", args)
    start_all = time.time()

    # load model.
    start = time.time()
    net = _load_model(args)
    dura = (time.time() - start)
    print("Loading model time cost: %f seconds."%dura)

    # laod datasets and infer.
    eval_sets = [it.strip() for it in args.eval_sets.strip().split(',')]

    merge_sets = []
    # start extract embeddings
    for ieval_set in eval_sets:
        if ieval_set not in eval_sets_dict:
            print("unknown eval set %s, pass"%(ieval_set))
            continue

        print("\nExtract on %s..."%(ieval_set))
        start = time.time()
        # initialize
        evaluator = eval(eval_sets_dict[ieval_set])(args)
        # extract
        ret = evaluator.extract_embedding(net)
        dura = (time.time() - start) / 60.0
        print("\nExtract on %s done! Time cost: %fm"%(ieval_set, dura))

        if not ret:
            print("extract failed")
            continue

        # TODO: start CPU Evaluate
        eval_info = {
            "eval_set": ieval_set,
            "args": args,
        }
        start_eval_process(eval_info)

        merge_sets.append(ieval_set)

        # Extract done

    if len(merge_sets) > 0:
        merge_info = {
            "eval_set": ','.join(merge_sets),
            "args": args,
            "net_scale_type": net._net_scale_type
        }
        start_merge_process(merge_info)
   #
    dura_all = (time.time() - start_all) / 60.0
    print("Total extract time cost: %fm"%dura_all)
#


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help='Input model file path, e.g., ...')
    parser.add_argument(
        '--model_type',
        type=str,
        default='mxnet_fp32',
        help='Model types: mxnet_fp32, mxnet_fp16, ...')
    parser.add_argument(
        '--eval_sets',
        type=str,
        default='megaface,lfw',
        help='Evaluation on benchmarks: megaface, lfw, ...')
    parser.add_argument(
        '--image_size',
        type=str,
        default='3,112,112',
        help='Input image size of network.')
    parser.add_argument(
        '--gpus',
        type=str,
        default='0,1,2,3,4,5,6,7,8,9',
        help='Set gpu id.')
    parser.add_argument(
        '--port',
        type=int,
        default=7776,
        help='evaluate server port.')
    parser.add_argument(
        '--clear',
        type=bool,
        default=True,
        help='Whether to clear the feature files.')
    parser.add_argument(
        '--net_scale',
        type=str,
        default=None,
        help='net scale, large or small, need by pytorch')
    parser.add_argument(
        '--flip',
        type=bool,
        default=True,
        help='Whether to flip.')
    parser.add_argument(
        '--save_badcase',
        type=bool,
        default=False,
        help='Whether to save badcase for analysis.')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    #


    _main(args)


