# -*- coding: utf-8 -*-
#
# train.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------
# only support pytorch backend
from ogb.lsc import WikiKG90Mv2Evaluator
from train_pytorch import test, test_mp
from train_pytorch import load_model
import torch.multiprocessing as mp
import numpy as np
import torch
import argparse
import json
import os
import logging
import time
import torch as th

# from .dataloader import EvalDataset, TrainDataset, NewBidirectionalOneShotIterator
# from .dataloader import get_dataset

# from .utils import get_compatible_batch_size, save_model, CommonArgParser
from dataloader import EvalDataset
from dataloader import get_dataset

from utils import get_compatible_batch_size, CommonArgParser

backend = os.environ.get('DGLBACKEND', 'pytorch')
assert backend.lower() == 'pytorch'
# from .train_pytorch import load_model, load_model_from_checkpoint
# from .train_pytorch import train, train_mp
# from .train_pytorch import test, test_mp


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--emb_save_path', type=str, default="emb/",
                          help='path to save useful embedding')
        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='A list of gpu ids, e.g. 0 1 2 4')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='Training a knowledge graph embedding model with both CPUs and GPUs.'
                               'The embeddings are stored in CPU memory and the training is performed in GPUs.'
                               'This is usually used for training a large knowledge graph embeddings.')
        self.add_argument('--valid', action='store_true',
                          help='Evaluate the model on the validation set in the training.')
        self.add_argument('--rel_part', action='store_true',
                          help='Enable relation partitioning for multi-GPU training.')
        self.add_argument('--async_update', action='store_true',
                          help='Allow asynchronous update on node embedding for multi-GPU training.'
                               'This overlaps CPU and GPU computation to speed up.')
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.'
                               'The positive score will be adjusted '
                               'as pos_score = pos_score * edge_importance')

        self.add_argument('--print_on_screen', action='store_true')
        self.add_argument('--encoder_model_name', type=str, default='shallow',
                          help='shallow or roberta or concat')
        self.add_argument('--mlp_lr', type=float, default=0.0001,
                          help='The learning rate of optimizing mlp')
        self.add_argument('--seed', type=int, default=0,
                          help='random seed')

        self.add_argument('--eval_sample_num', type=int, default=0,
                          help='when eval, how mush data must be abandon')

        self.add_argument('--is_eval', type=int, default=1,
                          help='eval model')

        self.add_argument('--dtype', type=int, default=16,
                          help='eval')

        self.add_argument('--use_mmap', type=int, default=1,
                          help='mode param not be init')

        self.add_argument('--use_valid_train', type=int, default=0,
                          help='mode param not be init')

        self.add_argument('--use_lr_decay', type=int, default=0,
                          help='todo')

        self.add_argument('--use_relation_weight', type=int, default=0,
                          help='todo')

        self.add_argument('--use_2_layer_mlp', type=int, default=1,
                          help='todo')

        # 有啥用？
        self.add_argument('--save_entity_emb', type=int, default=0,
                          help='mode param not be init')

        self.add_argument('--save_rel_emb', type=int, default=0,
                          help='mode param not be init')

        self.add_argument('--save_mlp', type=int, default=1,
                          help='mode param not be init')


class JsonArg(argparse.ArgumentParser):
    def __init__(self):
        super(JsonArg, self).__init__()

    def get_parser(self):
        self.add_argument('--load_json', type=str, default="config.json",
                          help='Load settings from file in json format. Command line options override values in file.')
        self.add_argument('--root_save_path', type=str, default="/data/xu/data_nips/complex_c/ComplEx_wikikg90m-v2_concat_d_512_g_10.0_lr_0.1_seed_0_4_mrr_0.20194695889949799_step_1499999/",
                          help='model parameter save path.')   
        self.add_argument('--emb_save_path', type=str, default= "/data/xu/data_nips/complex_c",
                          help='model parameter save path.')   
        args = self.parse_args()
        root_save_path = args.root_save_path
        emb_save_path = args.emb_save_path
        root_data_path = "/data/xu/smore/dataset/"
        
        args.load_json = os.path.join(root_save_path, args.load_json)
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = self.parse_args(namespace=t_args)
        args.data_path = root_data_path
        args.save_path = root_save_path
        # args.save_path = os.path.join(root_save_path, args.save_path)
        args.emb_save_path = emb_save_path
        args.eval = False
        args.test = False
        args.num_test_proc = 1
        args.num_proc = 1
        args.num_thread = 8
        args.gpu = [-1]
        return args


def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    folder = '{}_{}_{}_d_{}_g_{}'.format(args.model_name, args.dataset, args.encoder_model_name, args.hidden_dim,
                                         args.gamma)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


def set_logger(args):
    '''
    Write logs to console and log file
    '''
    os.makedirs(args.save_path, exist_ok=True)
    log_file = os.path.join(args.save_path, 'train.log')
    # logging.basicConfig(
    #     format='%(asctime)s %(levelname)-8s %(message)s',
    #     level=logging.INFO,
    #     datefmt='%Y-%m-%d %H:%M:%S',
    #     filename=log_file,
    #     filemode='a+'
    # )
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    filehandler = logging.FileHandler(log_file)
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)

    logger.addHandler(filehandler)
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    args.log = logger
    f = open(os.path.join(args.save_path, 'mrr.txt'), 'a+')
    args.mrr_log = f


def combine_rst(rst_list, has_t=False):
    rst_list = list(rst_list)
    rst_list.sort(key=lambda node: node['h,r->t']['rank'])
    if has_t:
        rst = {
            "t_pred_score": rst_list[0]['h,r->t']["t_pred_score"],
            't_pred_top10': rst_list[0]['h,r->t']["t_pred_top10"],
            'h,r': rst_list[0]['h,r->t']["h,r"],
            't': rst_list[0]['h,r->t']["t"],
        }
        rank = 1
        for candidate_score in rst_list[1:]:
            assert candidate_score['h,r->t']["rank"] == rank
            rank += 1
            rst["t_pred_score"] = th.cat([rst["t_pred_score"], candidate_score['h,r->t']["t_pred_score"]], axis=0)
            rst["t_pred_top10"] = th.cat([rst["t_pred_top10"], candidate_score['h,r->t']["t_pred_top10"]], axis=0)
            rst["h,r"] = th.cat((rst["h,r"], candidate_score['h,r->t']["h,r"]), axis=0)
            rst["t"] = th.cat((rst["t"], candidate_score['h,r->t']["t"]), axis=0)
    else:
        rst = {
            "t_pred_score": rst_list[0]['h,r->t']["t_pred_score"],
            't_pred_top10': rst_list[0]['h,r->t']["t_pred_top10"],
            'h,r': rst_list[0]['h,r->t']["h,r"],
        }
        rank = 1
        for candidate_score in rst_list[1:]:
            assert candidate_score['h,r->t']["rank"] == rank
            rank += 1
            rst["t_pred_score"] = th.cat([rst["t_pred_score"], candidate_score['h,r->t']["t_pred_score"]], axis=0)
            rst["t_pred_top10"] = th.cat([rst["t_pred_top10"], candidate_score['h,r->t']["t_pred_top10"]], axis=0)
            rst["h,r"] = th.cat((rst["h,r"], candidate_score['h,r->t']["h,r"]), axis=0)
    return rst


def get_mrr(rst_dict):
    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': rst_dict['t_pred_top10'],
                            't': rst_dict['t']}
    evaluator = WikiKG90Mv2Evaluator()
    ret = evaluator.eval(input_dict)
    return ret['mrr']


def set_evalCan_testCan(dataset, eval_can_path, test_can_path):
    dataset.valid_dict['h,r->t']['t_candidate'] = np.load(eval_can_path)
    dataset.test_dict['h,r->t']['t_candidate'] = np.load(test_can_path)


def main():

    args = JsonArg().get_parser()
    # args = ArgParser().parse_args()
    assert args.dataset == 'wikikg90m-v2'
    set_global_seed(args.seed)

    init_time_start = time.time()
    # load dataset and samplers
    dataset = get_dataset(args.data_path,
                          args.dataset,
                          args.format,
                          args.delimiter,
                          args.data_files,
                          args.has_edge_importance)
    # set_evalCan_testCan(dataset, os.path.join(args.save_path, "eval_can.npy"), os.path.join(args.save_path, "test_can.npy"))

    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = dataset.n_entities
    args.batch_size = get_compatible_batch_size(args.batch_size, args.neg_sample_size)
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)
    # We should turn on mix CPU-GPU training for multi-GPU training.
    if len(args.gpu) > 1:
        args.mix_cpu_gpu = True
        if args.num_proc < len(args.gpu):
            args.num_proc = len(args.gpu)
    # We need to ensure that the number of processes should match the number of GPUs.
    if len(args.gpu) > 1 and args.num_proc > 1:
        assert args.num_proc % len(args.gpu) == 0, \
            'The number of processes needs to be divisible by the number of GPUs'
    # For multiprocessing training, we need to ensure that training processes are synchronized periodically.
    if args.num_proc > 1:
        args.force_sync_interval = 1000

    args.eval_filter = not args.no_eval_filter
    if args.neg_deg_sample_eval:
        assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

    args.soft_rel_part = False
    # print ("To build training dataset")
    # t1 = time.time()
    # train_data = TrainDataset(dataset, args, ranks=args.num_proc, has_importance=args.has_edge_importance)
    # print ("Training dataset built, it takes %d seconds"%(time.time()-t1))
    # if there is no cross partition relaiton, we fall back to strict_rel_part
    args.strict_rel_part = False
    args.num_workers = 32  # fix num_worker to 8
    set_logger(args)
    print(vars(args))

    if args.valid or args.test:
        if len(args.gpu) > 1:
            args.num_test_proc = args.num_proc if args.num_proc < len(args.gpu) else len(args.gpu)
        else:
            args.num_test_proc = args.num_proc

        print("To create eval_dataset")
        t1 = time.time()
        eval_dataset = EvalDataset(dataset, args)
        print("eval_dataset created, it takes %d seconds" % (time.time() - t1))

    if args.valid:
        if args.num_proc > 1:
            # valid_sampler_heads = []
            valid_sampler_tails = []
            for i in range(args.num_proc):
                print("creating valid sampler for proc %d" % i)
                t1 = time.time()
                # valid_sampler_head = eval_dataset.create_sampler('valid', args.batch_size_eval,
                #                                                   args.neg_sample_size_eval,
                #                                                   args.neg_sample_size_eval,
                #                                                   args.eval_filter,
                #                                                   mode='head',
                #                                                   num_workers=args.num_workers,
                #                                                   rank=i, ranks=args.num_proc)
                valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.eval_filter,
                                                                 mode='tail',
                                                                 num_workers=args.num_workers,
                                                                 rank=i, ranks=args.num_proc)
                # valid_sampler_heads.append(valid_sampler_head)
                valid_sampler_tails.append(valid_sampler_tail)
                print("Valid sampler for proc %d created, it takes %s seconds" % (i, time.time() - t1))
        else:  # This is used for debug
            # valid_sampler_head = eval_dataset.create_sampler('valid', args.batch_size_eval,
            #                                                  args.neg_sample_size_eval,
            #                                                  1,
            #                                                  args.eval_filter,
            #                                                  mode='head',
            #                                                  num_workers=args.num_workers,
            #                                                  rank=0, ranks=1)
            valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                             args.neg_sample_size_eval,
                                                             1,
                                                             args.eval_filter,
                                                             mode='tail',
                                                             num_workers=args.num_workers,
                                                             rank=0, ranks=1)
    if args.test:
        if args.num_test_proc > 1:
            test_sampler_tails = []
            # test_sampler_heads = []
            for i in range(args.num_test_proc):
                print("creating test sampler for proc %d" % i)
                t1 = time.time()
                # test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
                #                                                  args.neg_sample_size_eval,
                #                                                  args.neg_sample_size_eval,
                #                                                  args.eval_filter,
                #                                                  mode='head',
                #                                                  num_workers=args.num_workers,
                #                                                  rank=i, ranks=args.num_test_proc)
                test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.eval_filter,
                                                                mode='tail',
                                                                num_workers=args.num_workers,
                                                                rank=i, ranks=args.num_test_proc)
                # test_sampler_heads.append(test_sampler_head)
                test_sampler_tails.append(test_sampler_tail)
                print("Test sampler for proc %d created, it takes %s seconds" % (i, time.time() - t1))
        else:
            # test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
            #                                                 args.neg_sample_size_eval,
            #                                                 1,
            #                                                 args.eval_filter,
            #                                                 mode='head',
            #                                                 num_workers=args.num_workers,
            #                                                 rank=0, ranks=1)
            test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_eval,
                                                            1,
                                                            args.eval_filter,
                                                            mode='tail',
                                                            num_workers=args.num_workers,
                                                            rank=0, ranks=1)
    # pdb.set_trace()

    
    # load model
    print("To create model")
    t1 = time.time()
    # model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1], dataset.relation_feat.shape[1])

    model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1],
                       dataset.relation_feat.shape[1])
    # LOAD Embedding and MLP
    model.load_emb(args.save_path, args.dataset)

    # LOAD Features
    if args.encoder_model_name in ['roberta', 'concat']:
        model.entity_feat.emb = dataset.entity_feat
        model.relation_feat.emb = dataset.relation_feat

    # LOAD Mlp 在load_emb()中已经加载
    # if args.encoder_model_name in ['concat', 'roberta', 'shallow_net']:
    #     model.transform_net.load_state_dict(th.load(os.path.join(args.save_path, args.dataset+"_"+args.model_name+"_mlp"))['transform_state_dict'])

    print("Model created, it takes %s seconds" % (time.time() - t1))
    if args.model_name in ['RotatE']:
        args.hidden_dim *= 2
    model.save_useful_emb(args.emb_save_path, args.dataset, args.hidden_dim)
    print("emb save successfully!")




if __name__ == '__main__':
    # input()
    main()
