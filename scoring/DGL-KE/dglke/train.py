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

import os
import logging
import time
import random
import json
import torch as th
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# from .dataloader import EvalDataset, TrainDataset, NewBidirectionalOneShotIterator
# from .dataloader import get_dataset
# from .train_pytorch import load_model, load_model_from_checkpoint
# from .train_pytorch import train, train_mp
# from .train_pytorch import test, test_mp
# from .utils import get_compatible_batch_size, save_model, CommonArgParser
from dataloader import EvalDataset, TrainDataset, NewBidirectionalOneShotIterator
from dataloader import get_dataset
from train_pytorch import load_model, load_model_from_checkpoint
from train_pytorch import train, train_mp
from train_pytorch import test, test_mp
from utils import get_compatible_batch_size, save_model, CommonArgParser

backend = os.environ.get('DGLBACKEND', 'pytorch')
assert backend.lower() == 'pytorch'
import torch
import numpy as np
import torch.multiprocessing as mp

from ogb.lsc import WikiKG90MDataset, WikiKG90Mv2Evaluator


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='A list of gpu ids, e.g. 0 1 2 4')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='Training a knowledge graph embedding model with both CPUs and GPUs.' \
                               'The embeddings are stored in CPU memory and the training is performed in GPUs.' \
                               'This is usually used for training a large knowledge graph embeddings.')
        self.add_argument('--valid', action='store_true',
                          help='Evaluate the model on the validation set in the training.')
        self.add_argument('--rel_part', action='store_true',
                          help='Enable relation partitioning for multi-GPU training.')
        self.add_argument('--async_update', action='store_true',
                          help='Allow asynchronous update on node embedding for multi-GPU training.' \
                               'This overlaps CPU and GPU computation to speed up.')
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.' \
                               'The positive score will be adjusted ' \
                               'as pos_score = pos_score * edge_importance')

        self.add_argument('--print_on_screen', action='store_true')
        self.add_argument('--encoder_model_name', type=str, default='shallow',
                          help='shallow or roberta or concat')
        self.add_argument('--mlp_lr', type=float, default=0.0001,
                          help='The learning rate of optimizing mlp')
        self.add_argument('--seed', type=int, default=0,
                          help='random seed')

        self.add_argument('--eval_sample_num', type=int, default=10,
                          help='when eval, how mush data must be abandon')

        self.add_argument('--dtype', type=int, default=32,
                          help='load entity feature as float16')

        self.add_argument('--ckpts', type=str, default='',
                          help='checkpoint model be used continue train')

        self.add_argument('--is_eval', type=int, default=0,
                          help='mode param not be init')

        self.add_argument('--save_float16', type=int, default=0,
                          help='mode param not be init')

        self.add_argument('--save_entity_emb', type=int, default=0,
                          help='mode param not be init')

        self.add_argument('--save_rel_emb', type=int, default=0,
                          help='mode param not be init')

        self.add_argument('--save_mlp', type=int, default=1,
                          help='mode param not be init')

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

        self.add_argument('--threshold', type=float, default=0.17,
                    help='todo')

        # self.add_argument('--lr_decay_rate', type=float, default=None,
        #             help='todo')
        
        # self.add_argument('--lr_decay_interval', type=float, default=None,
        #             help='todo')

def prepare_save_path(args):
    args.predefine_save_path = args.save_path
    if not os.path.exists(args.save_path): #ckpts/
        os.makedirs(args.save_path)

    folder = '{}_{}_{}_d_{}_g_{}_lr_{}_seed_{}'.format(args.model_name, args.dataset, args.encoder_model_name, args.hidden_dim,
                                         args.gamma, args.lr, args.seed)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += "_" + str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    conf_file = os.path.join(args.save_path, 'config.json')
    dict = {}
    config = args
    dict.update(vars(config))
    with open(conf_file, 'w') as outfile:
        print('conf_file in:', conf_file)
        json.dump(dict, outfile, indent=4)

def set_logger(args):
    '''
    Write logs to console and log file
    '''
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

def main():
    args = ArgParser().parse_args()
    prepare_save_path(args)
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
        if "force_sync_interval" not in vars(args):
            args.force_sync_interval = 1000

    args.eval_filter = not args.no_eval_filter
    if args.neg_deg_sample_eval:
        assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

    args.soft_rel_part = args.mix_cpu_gpu and args.rel_part
    print("To build training dataset")
    t1 = time.time()
    train_data = TrainDataset(dataset, args, ranks=args.num_proc, has_importance=args.has_edge_importance)
    print("Training dataset built, it takes %d seconds" % (time.time() - t1))
    # if there is no cross partition relaiton, we fall back to strict_rel_part
    args.strict_rel_part = False # args.mix_cpu_gpu and (train_data.cross_part == False)
    args.num_workers = 32  # fix num_worker to 8
    set_logger(args)
    print(vars(args))
    with open(os.path.join(args.save_path, args.encoder_model_name), 'w') as f:
        f.write(args.encoder_model_name)
    if args.num_proc > 1:
        lock = mp.Lock()
        manager = mp.Manager()
        shared_dict = manager.list()
        shared_dict.append({})
        shared_dict.append({"best_step": 0, "best_mrr": 0.})
        train_samplers = []
        for i in range(args.num_proc):
            args.log.info("Building training sampler for proc %d" % i)
            t1 = time.time()
            # for each GPU, allocate num_proc // num_GPU processes
            train_sampler_head = train_data.create_sampler(args.batch_size,
                                                           args.neg_sample_size,
                                                           args.neg_sample_size,
                                                           mode='head',
                                                           num_workers=args.num_workers,
                                                           shuffle=True,
                                                           exclude_positive=False,
                                                           rank=i)
            train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                           args.neg_sample_size,
                                                           args.neg_sample_size,
                                                           mode='tail',
                                                           num_workers=args.num_workers,
                                                           shuffle=True,
                                                           exclude_positive=False,
                                                           rank=i)
            train_samplers.append(NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                                  args.neg_sample_size, args.neg_sample_size,
                                                                  True, dataset.n_entities,
                                                                  args.has_edge_importance))
            args.log.info("Training sampler for proc %d created, it takes %s seconds" % (i, time.time() - t1))

        train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                        args.neg_sample_size, args.neg_sample_size,
                                                        True, dataset.n_entities,
                                                        args.has_edge_importance)
    else:  # This is used for debug
        train_sampler_head = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='head',
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False)
        train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='tail',
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False)
        train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                        args.neg_sample_size, args.neg_sample_size,
                                                        True, dataset.n_entities,
                                                        args.has_edge_importance)

    if args.valid or args.test:
        if len(args.gpu) > 1:
            args.num_test_proc = args.num_proc if args.num_proc < len(args.gpu) else len(args.gpu)
        else:
            args.num_test_proc = args.num_proc
        args.log.info("To create eval_dataset")
        t1 = time.time()
        eval_dataset = EvalDataset(dataset, args)
        args.log.info("eval_dataset created, it takes %d seconds" % (time.time() - t1))

    if args.valid:
        if args.num_proc > 1:
            # valid_sampler_heads = []
            valid_sampler_tails = []
            for i in range(args.num_proc):
                args.log.info("creating valid sampler for proc %d" % i)
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
                args.log.info("Valid sampler for proc %d created, it takes %s seconds" % (i, time.time() - t1))
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
                args.log.info("creating test sampler for proc %d" % i)
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
                args.log.info("Test sampler for proc %d created, it takes %s seconds" % (i, time.time() - t1))
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
    
    # load model
    args.log.info("To create model")
    t1 = time.time()
    if args.ckpts == '':
        args.log.info('====build new model===')
        model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1],
                           dataset.relation_feat.shape[1])
    else:
        args.log.info('===load pre model continue train===')
        model = load_model_from_checkpoint(args, dataset.n_entities, dataset.n_relations, args.ckpts, ent_feat_dim=768,
                                           rel_feat_dim=768)

    if args.encoder_model_name in ['roberta', 'concat']:
        model.entity_feat.emb = dataset.entity_feat
        model.relation_feat.emb = dataset.relation_feat

    args.log.info("Model created, it takes %s seconds" % (time.time() - t1))
    model.evaluator = WikiKG90Mv2Evaluator()

    if args.num_proc > 1 or args.async_update:
        model.share_memory()

    emap_file = dataset.emap_fname
    rmap_file = dataset.rmap_fname
    # We need to free all memory referenced by dataset.
    eval_dataset = None
    dataset = None

    args.log.info('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))

    # train
    start = time.time()
    rel_parts = train_data.rel_parts if args.strict_rel_part or args.soft_rel_part else None
    cross_rels = train_data.cross_rels if args.soft_rel_part else None

    for arg in vars(args):
        if arg != 'log' or arg != 'mrr_log':
            args.log.info('{:20}:{}'.format(arg, getattr(args, arg)))
    if args.num_proc > 1:
        procs = []
        barrier = mp.Barrier(args.num_proc)
        # check proc
        # check_proc = mp.Process(target=file_save_and_check, args=(args,))
        # procs.append(check_proc)
        # check_proc.start()
        for i in range(args.num_proc):
            # valid_sampler = [valid_sampler_heads[i], valid_sampler_tails[i]] if args.valid else None
            # test_sampler = [test_sampler_heads[i], test_sampler_tails[i]] if args.test else None
            valid_sampler = [valid_sampler_tails[i]] if args.valid else None
            test_sampler = [test_sampler_tails[i]] if args.test else None
            proc = mp.Process(target=train_mp, args=(args,
                                                     model,
                                                     train_samplers[i],
                                                     valid_sampler,
                                                     test_sampler,
                                                     i,
                                                     rel_parts,
                                                     cross_rels,
                                                     barrier,
                                                     shared_dict,
                                                     lock,
                                                     ))
            procs.append(proc)
            proc.start()
        
        for proc in procs:
            proc.join()
    else:
        valid_samplers = [valid_sampler_tail] if args.valid else None
        test_samplers = [test_sampler_tail] if args.test else None
        # valid_samplers = [valid_sampler_head, valid_sampler_tail] if args.valid else None
        # test_samplers = [test_sampler_head, test_sampler_tail] if args.test else None
        train(args, model, train_sampler, valid_samplers, test_samplers, rel_parts=rel_parts)

    args.log.info('training takes {} seconds'.format(time.time() - start))

if __name__ == '__main__':
    main()
