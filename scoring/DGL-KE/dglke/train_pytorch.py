# -*- coding: utf-8 -*-
#
# train_pytorch.py
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

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th
import logging
import random
import os
import json
import shutil

# logging.basicConfig(format='%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: ' \
#                            '%(message)s', level=logging.INFO)

from distutils.version import LooseVersion

TH_VERSION = LooseVersion(th.__version__)
if TH_VERSION.version[0] == 1 and TH_VERSION.version[1] < 2:
    raise Exception("DGL-ke has to work with Pytorch version >= 1.2")
# from .models.pytorch.tensor_models import thread_wrapped_func
# from .models import KEModel
# from .utils import save_model, get_compatible_batch_size, load_model_config
# from .dataloader import EvalDataset
# from .dataloader import get_dataset
# from .analyse_val_error_case import get_realtion_result
from models.pytorch.tensor_models import thread_wrapped_func
from models import KEModel
from utils import save_model, get_compatible_batch_size, load_model_config
from dataloader import EvalDataset
from dataloader import get_dataset
from analyse_val_error_case import get_realtion_result
import os
import logging
import time
from functools import wraps

import dgl
from dgl.contrib import KVClient
import dgl.backend as F


import pdb
from collections import defaultdict
from ogb.lsc import WikiKG90Mv2Dataset, WikiKG90Mv2Evaluator
from tqdm import tqdm
import pickle
from math import ceil


g_cur_path = os.path.dirname(os.path.abspath(__file__)) + '/'

def load_model(args, n_entities, n_relations, ent_feat_dim, rel_feat_dim, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim,
                    args.gamma,
                    double_entity_emb=args.double_ent,
                    double_relation_emb=args.double_rel,
                    ent_feat_dim=ent_feat_dim,
                    rel_feat_dim=rel_feat_dim)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model


def load_model_from_checkpoint(args, n_entities, n_relations, ckpt_path, ent_feat_dim, rel_feat_dim):
    model = load_model(args, n_entities, n_relations, ent_feat_dim, rel_feat_dim)
    model.load_emb(ckpt_path, args.dataset)
    return model


def train(args, model, train_sampler, valid_samplers=None, test_samplers=None, rank=0, rel_parts=None, cross_rels=None,
          barrier=None, shared_dict=None, lock=None, client=None):
    logs = []
    tor = 1000000

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.async_update:
        model.create_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.prepare_relation(th.device('cuda:' + str(gpu_id)))
    if args.soft_rel_part:
        model.prepare_cross_rels(cross_rels)

    if args.encoder_model_name in ['roberta', 'concat', 'shallow_net']:
        # print('train---gpu_id:', gpu_id)
        if int(gpu_id) >= 0:
            model.transform_net = model.transform_net.to(th.device('cuda:' + str(gpu_id)))
        args.log.info('---optimizer: Adam')
        optimizer = th.optim.Adam(model.transform_net.parameters(), args.mlp_lr)
    else:
        # optimizer = th.optim.Adam(model.entity_emb.emb)
        optimizer = None

    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0
    best_step = 0
    max_mrr = 0
    # max_mrr_path = ['', '']
    max_mrr_path = ['']
    # rel_statis = {}
    if args.use_valid_train:
        args.log.info('---use_valid_train---')
    # import pdb
    # pdb.set_trace()
    for step in range(0, args.max_step):
        # import pdb
        # pdb.set_trace()
        start1 = time.time()
        # if not args.use_valid_train:
        pos_g, neg_g = next(train_sampler)

        # rel_list = pos_g.edata['id'].numpy().tolist()    
        # for rel in rel_list:
        #     if rel not in rel_statis:
        #         rel_statis[rel] = 0
        #     rel_statis[rel] += 1

        # print(pos_g.ndata)
        sample_time += time.time() - start1
        # print("[proc ", rank, '] next sampler time:', time.time() - start1)

        if client is not None:
            model.pull_model(client, pos_g, neg_g)

        start1 = time.time()
        if optimizer is not None:
            optimizer.zero_grad()
        loss, log = model.forward(pos_g, neg_g, gpu_id, rank=rank)
        # print("[proc ", rank, "] start step: ", step, 'loss:', loss.cpu().detach().numpy().tolist())
        # print('---loss: ', loss, 'step: ', step)

        forward_time += time.time() - start1

        start1 = time.time()
        loss.backward()
        backward_time += time.time() - start1

        start1 = time.time()
        if client is not None:
            model.push_gradient(client)
        else:
            # print('update embeding')
            model.update(gpu_id)

        if optimizer is not None:
            optimizer.step()

        update_time += time.time() - start1
        logs.append(log)

        # force synchronize embedding across processes every X steps
        if args.force_sync_interval > 0 and \
                (step + 1) % args.force_sync_interval == 0 and barrier is not None:
            barrier.wait()

        if (step + 1) % args.log_interval == 0:
            if (client is not None) and (client.get_machine_id() != 0):
                pass
            else:
                output_string = "[proc {}][Train]({}/{}) average ".format(rank, (step + 1), args.max_step)
                for k in logs[0].keys():
                    v = sum(l[k] for l in logs) / len(logs)
                    output_string = output_string + "{}: {} ".format(k, v)
                    # args.log.info('[proc  {} ] [Train]({}/{}) average {}: {}'.format(rank, (step + 1), args.max_step, k, v))
                args.log.info(output_string)
                logs = []
                args.log.info('[proc {}][Train]{} steps take {:.3f} seconds, sample:{:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(rank, args.log_interval,
                                                                                time.time() - start, sample_time, forward_time, backward_time, update_time))
                # args.log.info('[proc  {} ] sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                #     rank, sample_time, forward_time, backward_time, update_time))
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0
                start = time.time()

            
            # # 开始部分超参数手动更新
            # if args.num_proc > 1:
            #     param_path = args.save_path.split("_mrr")[0]
            #     # param_path = args.save_path.split("_rank")[0]
            # else:
            #     param_path = args.save_path.split("_mrr")[0]

            if args.lr_decay_rate is not None and (step + 1
                                               ) % args.lr_decay_interval == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * args.lr_decay_rate
            # param_dict = json.load(open(param_path + '/config.json'))
            # if param_dict['lr'] != args.lr:
            #     args.log.info('update lr from %s to %s' % (args.lr, param_dict['lr']))
            #     args.lr = param_dict['lr']
            # else:
            #     args.log.info('not update lr: {:.3f}'.format(args.lr))

            # param_dict = None

        # if True:
        if args.valid and (step + 1) % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            valid_start = time.time()
            if args.strict_rel_part or args.soft_rel_part:
                model.writeback_relation(rank, rel_parts)
            # forced sync for validation
            args.log.info('[proc {}] start barrier wait at step {}'.format(rank, step))
            if barrier is not None:
                barrier.wait()
            args.log.info('[proc {}] barrier wait in validation take {:.3f} seconds:'.format(rank, time.time() - valid_start))
            # rel_sort = sorted(rel_statis.items(), key=lambda x: x[1], reverse=True)
            # print('rel_statis: %s' % json.dumps(rel_sort))
            # print('rel_count:', len(rel_sort))
            # rel_statis = {}
            valid_start = time.time()
            if valid_samplers is not None:
                valid_input_dict, candidate_score = test(args, model, valid_samplers, step, rank, mode='Valid')
                evaluator = WikiKG90Mv2Evaluator()
                ret = evaluator.eval(valid_input_dict)
                
                args.log.info('valid rank: %s, step: %s, MRR: %s nsamples %s' % (rank, step, ret['mrr'], len(valid_input_dict['h,r->t']['t'])))
                
                if args.num_proc > 1:
                    # mrr_record_path = args.save_path.split("_mrr")[0] + '_step_%s' % (ret['mrr'], step)
                    # with lock:
                    #     if not os.path.exists(mrr_record_path):
                    #         os.makedirs(mrr_record_path)
                    # file_name = "mrr_%s_samples_%s_rank_%s" % (ret['mrr'], valid_input_dict['h,r->t']['t_pred_top10'].shape[0], step)
                    # f = open(os.path.join(mrr_record_path, file_name), 'wb')
                    # f.close()
                    
                    with lock:
                        save_dict = shared_dict[0]
                        if step not in save_dict:
                            save_dict[step] = {}
                        save_dict[step][rank] = (len(valid_input_dict['h,r->t']['t']), ret['mrr']) 
                        shared_dict[0] = save_dict
                    # print("[{}] shared_dict {}".format(rank, shared_dict))
                    args.save_path = args.save_path.split("_mrr")[0] 
                    # args.save_path = args.save_path.split("_rank")[0] + '_rank_%s_nsample_%s_mrr_%s_step_%s' % (rank, valid_input_dict['h,r->t']['t_pred_top10'].shape[0], ret['mrr'], step)
                else:
                    
                    args.save_path = args.save_path.split("_mrr")[0] + '_mrr_%s_step_%s' % (ret['mrr'], step)
                if args.use_valid_train:
                    args.save_path = args.save_path + '_model_valid'
               
                # candidate set持续变化, 此处不再起作用
                # if ret['mrr'] > 0.05:
                #     if not os.path.exists(args.save_path):
                #         os.makedirs(args.save_path)
                #     th.save(valid_input_dict, os.path.join(args.save_path, "valid_{}_{}_mrr_{}.pkl".format(rank, step, ret['mrr'])))
                #     th.save(candidate_score, os.path.join(args.save_path, "valid_candidate_score_{}_{}_mrr_{}.pkl".format(rank, step, ret['mrr'])))

                # relation mrr statis analyse
                # print('get_realtion_result_error--')
                # get_realtion_result(candidate_score, args) 
                
                # original saving:
                if args.num_proc > 1:
                    
                    if barrier is not None:
                        barrier.wait()
                    if rank == 0 and not args.no_save_emb:## and max_mrr < ret['mrr']:
                        save_dict = shared_dict[0]
                        cur_mrr = 0.
                        sample_cnt = 0
                        for i in range(args.num_proc):
                        # for file in os.listdir(mrr_record_path):
                        #     file_ = file.split('_')
                        #     mrr, step, 
                            # print(step,i)
                            # print(save_dict[step])
                            # print(save_dict[step][i])
                            # print(save_dict[step][i][0])
                            # args.log.write("step %s rank %s nsamples %s mrr %s\n" % (step, i, save_dict[step][i][0], save_dict[step][i][1]))
                            sample_cnt += save_dict[step][i][0]
                            cur_mrr = cur_mrr + save_dict[step][i][1] * save_dict[step][i][0]
                        cur_mrr /= sample_cnt
                        args.mrr_log.write('rank %s step %s MRR %s nsample %s\n' % (rank, step, cur_mrr, sample_cnt))
                        args.mrr_log.flush()
                        args.log.info("step % overall mrr %" % (step, cur_mrr))
                        max_mrr = shared_dict[1]["best_mrr"]
                        best_step = shared_dict[1]["best_step"]
                        if max_mrr < cur_mrr: 
                            best_step = step
                            max_mrr = cur_mrr
                            shared_dict[1] = {"best_step": best_step, "best_mrr": max_mrr}
                            args.save_path = args.save_path + '_mrr_%s_step_%s' % (cur_mrr, step)
                            if max_mrr > args.threshold:
                                if not os.path.exists(args.save_path):
                                    os.makedirs(args.save_path)
                                # time.sleep(10)
                                
                                args.log.info('proc {} after barrier'.format(rank))

                                if args.async_update:
                                    model.finish_async_update()
                                args.log.info('proc {} finish async update'.format(rank))

                                if args.strict_rel_part or args.soft_rel_part:
                                    model.writeback_relation(rank, rel_parts)
                                args.log.info('proc {} return'.format(rank))

                                save_model(args, model, None, None) #保存2个最佳的模型
                                if os.path.exists(max_mrr_path[0]):
                                    args.log.info('delete_path:'.format(max_mrr_path[0]))
                                    cmd = 'rm -rf ' + max_mrr_path[0] 
                                    # args.log.info('delete_path:'.format(g_cur_path + max_mrr_path[0]))
                                    # cmd = 'rm -rf ' + g_cur_path + max_mrr_path[0] 
                                    args.log.info('cmd: {}'.format(cmd))
                                    status = os.system(cmd)
                                    args.log.info('cmd status:{}'.format(str(status)))
                                max_mrr_path = [args.save_path]
                                # max_mrr_path = max_mrr_path[1:] + [args.save_path]
                                args.log.info('proc {} model saved'.format(rank))
                        # if os.path.exists(max_mrr_path[0]):
                        #     args.log.info('delete_path: {}'.format(g_cur_path + max_mrr_path[0]))
                        #     cmd = 'rm -rf ' + g_cur_path + max_mrr_path[0]
                        #     args.log.info('cmd: {}'.format(cmd))
                        #     status = os.system(cmd)
                        #     args.log.info('cmd status: {}'.format(status))
                        #     if os.path.exists(g_cur_path + max_mrr_path[0]):
                        #         os.removedirs(g_cur_path + max_mrr_path[0])

                        # max_mrr_path = max_mrr_path[1:] + [args.save_path]
                        # args.log.info('proc {} model saved'.format(rank))
                    if barrier is not None:
                        barrier.wait()
                # new saving:
                else:
                    args.mrr_log.write('rank %s step %s MRR %s\n' % (rank, step, ret['mrr']))
                    args.mrr_log.flush()
                    if max_mrr < ret['mrr']:
                        best_step = step
                        max_mrr = ret['mrr']
                        if rank == 0 and not args.no_save_emb and ret['mrr'] > args.threshold:
                            if not os.path.exists(args.save_path):
                                os.makedirs(args.save_path)
                            get_realtion_result(candidate_score, args)
                            time.sleep(10)
                            if barrier is not None:
                                barrier.wait()
                            args.log.info('proc {} after barrier'.format(rank))

                            if args.async_update:
                                model.finish_async_update()
                            args.log.info('proc {} finish async update'.format(rank))

                            if args.strict_rel_part or args.soft_rel_part:
                                model.writeback_relation(rank, rel_parts)
                            args.log.info('proc {} return'.format(rank))

                            save_model(args, model, None, None) #保存2个最佳的模型
                            if os.path.exists(max_mrr_path[0]):
                                args.log.info('delete_path:'.format(max_mrr_path[0]))
                                cmd = 'rm -rf ' + max_mrr_path[0] 
                                # args.log.info('delete_path:'.format(g_cur_path + max_mrr_path[0]))
                                # cmd = 'rm -rf ' + g_cur_path + max_mrr_path[0] 
                                args.log.info('cmd: {}'.format(cmd))
                                status = os.system(cmd)
                                args.log.info('cmd status:{}'.format(str(status)))
                            max_mrr_path = [args.save_path]
                            # max_mrr_path = max_mrr_path[1:] + [args.save_path]
                            args.log.info('proc {} model saved'.format(rank))

            # if test_samplers is not None:
            #     test_input_dict = test(args, model, test_samplers, step, rank, mode='Test')
            #     evaluator = WikiKG90MEvaluator()
            #     ret = evaluator.eval(test_input_dict)
            #     print('test rank: %s, step: %s, MRR: %s' % (rank, step, ret['mrr']))
            #     th.save(test_input_dict, os.path.join(args.save_path, "test_{}_{}.pkl".format(rank, step)))
            args.log.info('[proc  {} ] validation and test take {:.3f} seconds:'.format(rank, time.time() - valid_start))
            if args.soft_rel_part:
                model.prepare_cross_rels(cross_rels)
            if barrier is not None:
                barrier.wait()

            #开始部分超参数手动更新
            if args.num_proc > 1:
                param_path = args.save_path.split("_mrr")[0]
                # param_path = args.save_path.split("_rank")[0]
            else:
                param_path = args.save_path.split("_mrr")[0]

            
            # param_dict = json.load(open(param_path + '/config.json'))
            # if param_dict['lr'] != args.lr:
            #     args.log.info('update lr from %s to %s' % (args.lr, param_dict['lr']))
            #     args.lr = param_dict['lr']
            # else:
            #     args.log.info('not update lr: {:.3f}'.format(args.lr))

            # param_dict = None

            # if param_dict['lr'] != args.lr:
            #     print('update lr from %s to %s' % (args.lr, param_dict['lr']))
            #     args.lr = param_dict['lr']
        if args.num_proc > 1:
            if barrier is not None:
                barrier.wait()
            bst_dict = shared_dict[1]
            if step - bst_dict["best_step"] >= tor:
                args.log.info("early stopping, rank %s break" % rank)
                break
        else:
            if abs(step - best_step) >= tor:
                args.log.info("early stopping")
                break
    args.log.info('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
    time.sleep(5)
    # print("args.no_save_emb:", args.no_save_emb)
    if rank == 0 and not args.no_save_emb:
        args.save_path = args.save_path.split('_rank')[0] + '_mrr_end'
        save_model(args, model, None, None)
        args.log.info('proc {} model saved'.format(rank))
        if args.encoder_model_name == 'concat':
            args.save_path = args.save_path.split('_rank')[0] + '_emb_path'
            model.save_useful_emb(args.save_path, args.dataset)
    if barrier is not None:
        barrier.wait()
    args.log.info('proc {} after barrier'.format(rank))

    if args.async_update:
        model.finish_async_update()
    args.log.info('proc {} finish async update'.format(rank))

    if args.strict_rel_part or args.soft_rel_part:
        model.writeback_relation(rank, rel_parts)
    args.log.info('proc {} return'.format(rank))
    if rank == 0:
        args.log.info('training finish!')
    args.mrr_log.close()
    dim = args.hidden_dim * 2 if args.model_name in ['RotatE'] else args.hidden_dim
    model.save_useful_emb(args.save_path, args.dataset, dim)
    print("emb save successfully!")


def test(args, model, test_samplers, step, rank=0, mode='Test', sample_num=10):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    # print (test_samplers)
    # pdb.set_trace()
    sample_num = args.eval_sample_num
    with th.no_grad():
        logs = defaultdict(list)
        answers = defaultdict(list)
        candidate_score = defaultdict(list)
        querys = defaultdict(list)
        count = 0
        for sampler in test_samplers:
            # print("[rank:", rank, " ] sampler.num_edges:", sampler.num_edges, " sampler.batch_size:",
            #       sampler.batch_size)
            # quert: hr ans: t candidate:t_candidate
            for query, ans, candidate in sampler:# tqdm(sampler, disable=not args.print_on_screen,
                                              #total=ceil(sampler.num_edges / sampler.batch_size)):
                # wikikg90m-v1:仅用部分样本做test                                             
                # if sampler.num_edges > 50 and random.randint(0, max(0, sample_num)) != 0:
                #     continue
                # log: 前10can， score：完整的无排序can score
                # import pdb 
                # pdb.set_trace()
                log, scores = model.forward_test_wikikg(query, ans, candidate, sampler.mode, gpu_id)
                # print('--log.shape: ', log.shape)
                # print('--mode: ', sampler.mode)
                logs[sampler.mode].append(log)
                answers[sampler.mode].append(ans)
                candidate_score[sampler.mode].append(scores)
                querys[sampler.mode].append(query)

                count += 1
        args.log.info("[{}] finished {} forward, predict_num: {}".format(rank, mode, count))

        input_dict = {}
        assert len(answers) == 1
        assert 'h,r->t' in answers
        if 'h,r->t' in answers:
            assert 'h,r->t' in logs, "h,r->t not in logs"
            # wikikg90m-v2
            input_dict['h,r->t'] = {'t_pred_top10': th.cat(logs['h,r->t'], 0),
                                    't': th.cat(answers['h,r->t'], 0)}
            # wikikg90m-v1
            # input_dict['h,r->t'] = {'t_correct_index': th.cat(answers['h,r->t'], 0),
            #                         't_pred_top10': th.cat(logs['h,r->t'], 0)}
        # if 't,r->h' in answers:
        #     assert 't,r->h' in logs, "t,r->h not in logs"
        #     input_dict['t,r->h'] = {'h_correct_index': th.cat(answers['t,r->h'], 0), 'h_pred_top10': th.cat(logs['t,r->h'], 0)}
    for i in range(len(test_samplers)):
        test_samplers[i] = test_samplers[i].reset()
    # test_samplers[0] = test_samplers[0].reset()
    # test_samplers[1] = test_samplers[1].reset()
    evaluator = WikiKG90Mv2Evaluator()
    
    ret = evaluator.eval(input_dict)
    args.log.info('mode: %s rank: %s, step: %s, MRR: %s' % (mode, rank, step, ret['mrr']))
    candidate_score[sampler.mode] = {
        "t_pred_score": th.cat(candidate_score[sampler.mode], 0),
        't': th.cat(answers['h,r->t'], 0),
        't_pred_top10': th.cat(logs['h,r->t'], 0),
        'h,r': th.cat(querys[sampler.mode], 0),
    }
    # print(mode, '---candidate_score.t_pred_score.shape: ', candidate_score[sampler.mode]['t_pred_score'].shape)
    # print(mode, '---candidate_score.t_pred_top10.shape: ', candidate_score[sampler.mode]['t_pred_top10'].shape)
    # print(mode, '---candidate_score.h,r.shape: ', candidate_score[sampler.mode]['h,r'].shape)
    # print(mode, '---candidate_score.t.shape: ', candidate_score[sampler.mode]['t'].shape)
    # wikikg90m-v1
    # candidate_score[sampler.mode] = {
    #     "t_pred_score": th.cat(candidate_score[sampler.mode], 0),
    #     't_correct_index': th.cat(answers['h,r->t'], 0),
    #     't_pred_top10': th.cat(logs['h,r->t'], 0),
    #     'h,r': th.cat(querys[sampler.mode], 0),
    # }
    # print(mode, '---candidate_score.t_pred_score.shape: ', candidate_score[sampler.mode]['t_pred_score'].shape)
    # print(mode, '---candidate_score.t_pred_top10.shape: ', candidate_score[sampler.mode]['t_pred_top10'].shape)
    # print(mode, '---candidate_score.h,r.shape: ', candidate_score[sampler.mode]['h,r'].shape)
    # print(mode, '---candidate_score.t_correct_index.shape: ', candidate_score[sampler.mode]['t_correct_index'].shape)

    return input_dict, candidate_score


@thread_wrapped_func
def train_mp(args, model, train_sampler, valid_samplers=None, test_samplers=None, rank=0, rel_parts=None,
             cross_rels=None, barrier=None, shared_dict=None, lock=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    train(args, model, train_sampler, valid_samplers, test_samplers, rank, rel_parts, cross_rels, barrier, shared_dict, lock)


@thread_wrapped_func
def test_mp(args, model, test_samplers, rank=0, mode='Test'):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test(args, model, test_samplers, 0, rank, mode)


def file_save_and_check(args):
    max_val = [0.0, 0.0]
    max_val_path = ['', '']
    counter = 1
    def remove(path):
        if os.path.exists(path):
            for i in os.listdir(path):
                os.remove(os.path.join(path, i))
    bst_model_path = os.path.join(args.predefine_save_path, "save_model")
    if not os.path.exists(bst_model_path):
        os.makedirs(bst_model_path)
    while True:
        if args.eval_interval * counter - 1 == 1:
            counter += 1
        step = min(args.eval_interval * counter - 1, 20) 
        cur_model_path = args.predefine_save_path
        mrr_record = []
        n_samples_record = []
        for file in os.listdir(cur_model_path):
            if file == 'save_model':
                continue
            file_token = file.split('_')
            if len(file_token) >= 4 and file_token[-4] == 'mrr':
                cur_step = file_token[-1]
                if int(cur_step) == step:
                    mrr_record.append(float(file_token[-3]))
                    n_samples_record.append(int(file_token[-5]))
                    if file.split("_")[-7] == '0':
                        cur_model_name = file
        
            if len(mrr_record) == args.num_proc:
                mrr_val = 0.0
                for i in range(len(mrr_record)):
                    mrr_val += (mrr_record[i] * n_samples_record[i])
                mrr_val /= sum(n_samples_record)
                if mrr_val < max_val[1]:
                    remove(os.path.join(cur_model_path, cur_model_name))
                else:
                    bst_model_name = cur_model_name.split("_rank")[0] + "_mrr_{}_step_".format(mrr_val) + cur_model_name.split("_step_")[1]
                    bst_path = os.path.join(bst_model_path, bst_model_name)
                    if not os.path.exists(bst_path):
                        os.makedirs(bst_path)
                    args.log.info("cur path: {}".format(os.path.join(cur_model_path, cur_model_name)))
                    for subfile in os.listdir(os.path.join(cur_model_path, cur_model_name)):
                        args.log.info("move {} to {}".format(os.path.join(cur_model_path, cur_model_name, subfile), bst_path))
                        shutil.move(os.path.join(cur_model_path, cur_model_name, subfile), \
                            os.path.join(bst_path, subfile))
                    
                    if mrr_val > max_val[0]:
                        if max_val[0] != 0.0:
                            if os.path.exists(max_val_path[1]):
                                args.log.info("remove: {}".format(max_val_path[1]))
                                shutil.rmtree(max_val_path[1])
                            max_val[1] = max_val[0]
                            max_val_path[1] = max_val_path[0]
                        max_val[0] = mrr_val
                        max_val_path[0] = bst_path
                    else:
                        if os.path.exists(max_val_path[1]):
                            args.log.info("remove: {}".format(max_val_path[1]))
                            shutil.rmtree(max_val_path[1])
                        max_val[1] = mrr_val
                        max_val_path[1] = bst_path
                counter += 1
                args.log.info("step:{}".format(step))
                break
        if  step > args.max_step:
            args.log.info("remove: {}".format(max_val_path[1]))
            break  
        time.sleep(5)
    args.log.info("check proc return!")