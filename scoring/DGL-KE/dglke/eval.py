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
from strategy.grid_search_weight_M import aggregate_score_on_can
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

from dataloader import EvalDataset
from dataloader import get_dataset

from utils import get_compatible_batch_size

backend = os.environ.get('DGLBACKEND', 'pytorch')
assert backend.lower() == 'pytorch'


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class JsonArg(argparse.ArgumentParser):
    def __init__(self):
        super(JsonArg, self).__init__()

    def get_parser(self):
        self.add_argument('--load_json', type=str, default="dglke_path/transe_a/TransE_l2_wikikg90m-v2_shallow_d_768_g_10.0_lr_0.2_seed_0_33_mrr_0.23064033687114716_step_9949999/config.json",
                          help='path of the config.json which is config file for training.')
        self.add_argument('--valid_can_path', type=str, default="",
                          help='path of the validation candidate')
        self.add_argument('--test_can_path', type=str, default="",
                          help='path of the test candidate')
        self.add_argument('--rst_prename', type=str, default="",
                          help='prefix of the infer result')
        args = self.parse_args()
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = self.parse_args(namespace=t_args)

        args.eval = True
        args.test = True
        args.num_test_proc = 30
        args.num_proc = 30
        args.num_thread = 1
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
            rst["t_pred_score"] = th.cat([rst["t_pred_score"], candidate_score['h,r->t']["t_pred_score"]], 0)
            rst["t_pred_top10"] = th.cat([rst["t_pred_top10"], candidate_score['h,r->t']["t_pred_top10"]], 0)
            rst["h,r"] = th.cat((rst["h,r"], candidate_score['h,r->t']["h,r"]), 0)
            rst["t"] = th.cat((rst["t"], candidate_score['h,r->t']["t"]), 0)
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
            rst["t_pred_score"] = th.cat([rst["t_pred_score"], candidate_score['h,r->t']["t_pred_score"]], 0)
            rst["t_pred_top10"] = th.cat([rst["t_pred_top10"], candidate_score['h,r->t']["t_pred_top10"]], 0)
            rst["h,r"] = th.cat((rst["h,r"], candidate_score['h,r->t']["h,r"]), 0)
    return rst


def get_mrr(rst_dict):
    input_dict = {}
    t_candidate, t_pred_score = aggregate_score_on_can(rst_dict['t_candidate'], rst_dict['t_pred_score'])
    ensemble_score_matirx_sort_index = np.argsort(-t_pred_score, axis=1)
    ensemble_score_matirx_top10_index = ensemble_score_matirx_sort_index[:, :10]
    t_pred_top10 = np.take_along_axis(t_candidate, ensemble_score_matirx_top10_index, axis=1)
    input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10,
                            't': rst_dict['t']}
    evaluator = WikiKG90Mv2Evaluator()
    ret = evaluator.eval(input_dict)
    return ret['mrr']


def set_evalCan_testCan(rst_dict, valid_can_path, test_can_path):
    if valid_can_path is not None:
        rst_dict['t_candidate'] = np.load(valid_can_path)
    if test_can_path is not None:
        rst_dict['t_candidate'] = np.load(test_can_path)


def main():
    args = JsonArg().get_parser()
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
    set_evalCan_testCan(dataset.valid_dict['h,r->t'], args.valid_can_path, None)
    set_evalCan_testCan(dataset.test_dict['h,r->t'], None, args.test_can_path)

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
    model.evaluator = WikiKG90Mv2Evaluator()

    if args.num_proc > 1 or args.async_update:
        model.share_memory()

    emap_file = dataset.emap_fname
    rmap_file = dataset.rmap_fname
    # We need to free all memory referenced by dataset.
    eval_dataset = None
    dataset = None

    print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))

    # train
    start = time.time()
    # rel_parts = train_data.rel_parts if args.strict_rel_part or args.soft_rel_part else None
    # cross_rels = train_data.cross_rels if args.soft_rel_part else None

    if args.num_proc > 1:
        eval_mrr = 0
        if args.eval:
            procs = []
            rst_list = mp.Manager().list()
            # barrier = mp.Barrier(args.num_proc)
            for i in range(args.num_proc):
                # valid_sampler = [valid_sampler_heads[i], valid_sampler_tails[i]] if args.valid else None
                # test_sampler = [test_sampler_heads[i], test_sampler_tails[i]] if args.test else None
                valid_sampler = [valid_sampler_tails[i]] if args.valid else None
                proc = mp.Process(target=test_mp, args=(args,
                                                        model,
                                                        valid_sampler,
                                                        i,
                                                        'Val',
                                                        rst_list,))
                procs.append(proc)
                proc.start()
            for proc in procs:
                proc.join()
            print("eval over")
            eval_rst = combine_rst(rst_list, has_t=True)
            set_evalCan_testCan(eval_rst, args.valid_can_path, None)
            eval_mrr = get_mrr(eval_rst)
            os.makedirs(os.path.dirname(os.path.join(args.save_path, args.rst_prename,
                        "valid_candidate_mrr_{}.pkl".format(eval_mrr))), exist_ok=True)
            th.save(eval_rst, os.path.join(args.save_path, args.rst_prename,
                    "valid_candidate_mrr_{}.pkl".format(eval_mrr)), pickle_protocol=4)
            print("eval_rst save in "+os.path.join(args.save_path, args.rst_prename, "valid_candidate_mrr_{}.pkl".format(eval_mrr)))
        if args.test:
            procs = []
            rst_list = mp.Manager().list()
            # barrier = mp.Barrier(args.num_proc)
            for i in range(args.num_proc):
                # valid_sampler = [valid_sampler_heads[i], valid_sampler_tails[i]] if args.valid else None
                # test_sampler = [test_sampler_heads[i], test_sampler_tails[i]] if args.test else None
                test_sampler = [test_sampler_tails[i]] if args.test else None
                proc = mp.Process(target=test_mp, args=(args,
                                                        model,
                                                        test_sampler,
                                                        i,
                                                        'Test',
                                                        rst_list,))
                procs.append(proc)
                proc.start()
            for proc in procs:
                proc.join()
            test_rst = combine_rst(rst_list, has_t=False)
            set_evalCan_testCan(test_rst, None, args.test_can_path)
            os.makedirs(os.path.dirname(os.path.join(args.save_path,  args.rst_prename,
                        "test_candidate_mrr_{}.pkl".format(eval_mrr))), exist_ok=True)
            th.save(test_rst, os.path.join(args.save_path,  args.rst_prename,
                    "test_candidate_mrr_{}.pkl".format(eval_mrr)), pickle_protocol=4)
            print("test_rst save in " + os.path.join(args.save_path,  args.rst_prename, "test_candidate_mrr_{}.pkl".format(eval_mrr)))
    else:
        if args.eval:
            valid_samplers = [valid_sampler_tail] if args.valid else None
            # valid_samplers = [valid_sampler_head, valid_sampler_tail] if args.valid else None
            # test_samplers = [test_sampler_head, test_sampler_tail] if args.test else None
            input_dict_val, candidate_score_val, ret_val = test(args, model, valid_samplers, step=0, rank=0, mode='Val')
            # args.save_path = args.save_path.split("_mrr")[0] + '_mrr_%s_step_%s' % (ret['mrr'], step)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            # evaluator = WikiKG90Mv2Evaluator()
            # ret_val = evaluator.eval(input_dict_val)
            th.save(input_dict_val, os.path.join(args.save_path, "valid_{}_{}_mrr_{}.pkl".format('cur', 0, ret_val['mrr'])))
            th.save(candidate_score_val,
                    os.path.join(args.save_path, "valid_candidate_score_{}_{}_mrr_{}.pkl".format('cur', 0, ret_val['mrr'])))
            print('val MRR: %s' % ret_val['mrr'])

        if args.test:
            test_samplers = [test_sampler_tail] if args.test else None
            candidate_score_test = test(args, model, test_samplers, step=0, rank=0, mode='Test')
            # args.save_path = args.save_path.split("_mrr")[0] + '_mrr_%s_step_%s' % (ret['mrr'], step)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            # evaluator = WikiKG90Mv2Evaluator()
            # ret_test = evaluator.eval(input_dict_test)
            # th.save(input_dict_test,
            #         os.path.join(args.save_path, "test_{}_{}.pkl".format('cur', 0)))
            th.save(candidate_score_test, os.path.join(args.save_path,
                                                       "test_candidate_score_{}_{}.pkl".format('cur', 0)))
            print('test finished')

    print('eval takes {} seconds'.format(time.time() - start))


if __name__ == '__main__':
    # input()
    main()
