import torch
import numpy as np
import argparse
import nni
import os
import time
from scipy.special import log_softmax
import pdb
import logging
from typing import Optional, Union, Dict

import shutil
import os.path as osp
import pandas as pd

from utils import * 
from ogb.utils.url import decide_download, download_url, extract_zip, makedirs


class WikiKG90Mv2Evaluator:
    def eval(self, input_dict):
        '''
            Format of input_dict:
            - 'h,r->t'
                - t_pred_top10: np.ndarray of shape (num_eval_triplets, 10)
                    (i,j) represents the j-th prediction for i-th triplet
                    Only top10 prediction is taken into account
                - t: np.ndarray of shape (num_eval_triplets,)
        '''
        assert 'h,r->t' in input_dict
        assert ('t_pred_top10' in input_dict['h,r->t']) and ('t' in input_dict['h,r->t'])

        # h,r->t
        t_pred_top10 = input_dict['h,r->t']['t_pred_top10']
        t = input_dict['h,r->t']['t']
        if not isinstance(t_pred_top10, torch.Tensor):
            t_pred_top10 = torch.from_numpy(t_pred_top10)
        if not isinstance(t, torch.Tensor):
            t = torch.from_numpy(t)

        assert t_pred_top10.shape[1] == 10 and len(t_pred_top10) == len(t)

        # verifying that there is no duplicated prediction for each triplet
        duplicated = False
        for i in range(len(t_pred_top10)):
            if len(torch.unique(t_pred_top10[i][t_pred_top10[i] >= 0])) != len(t_pred_top10[i][t_pred_top10[i] >= 0]):
                duplicated = True
                break

        if duplicated:
            print('Found duplicated tail prediction for some triplets! MRR is automatically set to 0.')
            mrr = 0
        else:
            mrr = self._calculate_mrr(t.to(t_pred_top10.device), t_pred_top10)

        return {'mrr': mrr}

    def _calculate_mrr(self, t, t_pred_top10):
        '''
            - t: shape (num_eval_triplets, )
            - t_pred_top10: shape (num_eval_triplets, 10)
        '''
        tmp = torch.nonzero(t.view(-1,1) == t_pred_top10, as_tuple=False)

        # reciprocal rank
        # if rank is larger than 10, then set the reciprocal rank to 0.
        rr = torch.zeros(len(t)).to(tmp.device)
        rr[tmp[:,0]] = 1./(tmp[:,1].float() + 1.)

        # mean reciprocal rank
        return float(rr.mean().item())

    def save_test_submission(self, input_dict: Dict, dir_path: str, mode: str):
        assert 'h,r->t' in input_dict
        assert 't_pred_top10' in input_dict['h,r->t']
        assert mode in ['test-dev', 'test-challenge']

        t_pred_top10 = input_dict['h,r->t']['t_pred_top10']
        
        for i in range(len(t_pred_top10)):
            assert len(pd.unique(t_pred_top10[i][t_pred_top10[i] >= 0])) == len(t_pred_top10[i][t_pred_top10[i] >= 0]), 'Found duplicated tail prediction for some triplets!'

        if mode == 'test-dev':
            assert t_pred_top10.shape == (15000, 10)
            filename = osp.join(dir_path, 't_pred_wikikg90m-v2_test-dev')
        elif mode == 'test-challenge':
            assert t_pred_top10.shape == (10000, 10)
            filename = osp.join(dir_path, 't_pred_wikikg90m-v2_test-challenge')

        makedirs(dir_path)

        if isinstance(t_pred_top10, torch.Tensor):
            t_pred_top10 = t_pred_top10.cpu().numpy()
        t_pred_top10 = t_pred_top10.astype(np.int32)

        np.savez_compressed(filename, t_pred_top10=t_pred_top10)

def _get_args():
    def get_weight_tuple(s):
        weight_tuple = [0.1, 0.2]
        return weight_tuple
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--nni', action='store_true')
    parser.add_argument('--canweight_search', action='store_true')
    parser.add_argument('--weight_tuple', type=get_weight_tuple)

    parser.add_argument('--log_file', type=str, default="../output/result.log")
    parser.add_argument('--manual_feature_path', default="/manual_feature/smore-rule")
    parser.add_argument('--infer_val_path_list', action='append')
    parser.add_argument('--infer_test_path_list', action='append')
    parser.add_argument('--infer_dev_path_list', action='append')

    args = parser.parse_args()
    return args


def aggregate_score_on_can(can, score_matrix, args):
    can_unique = -1 * np.ones_like(can, dtype=np.int32)
    
    score_matrix_unique = -np.inf * np.ones_like(score_matrix, dtype=np.float32)
    for i in range(can.shape[0]):
        unique_elements, unique_elements_index, unique_elements_count = np.unique(
            can[i, :], return_index=True, return_counts=True)
        if unique_elements[0] == -1:
            unique_elements = unique_elements[1:]
            unique_elements_index = unique_elements_index[1:]
            unique_elements_count = unique_elements_count[1:]
        unique_elements_score = score_matrix[i, unique_elements_index]*unique_elements_count
        can_unique[i, 0:len(unique_elements)] = unique_elements
        score_matrix_unique[i, 0:len(unique_elements)] = unique_elements_score
    return can_unique, score_matrix_unique



def aggregate_score_on_can_with_weight(can, score_matrix, can_model_index, model_weight, args):
    """
    can: num_query * num_candidates, node index of candidates (including candidates from all the recall models), np.int32

    score_matrix: num_query * num_candidates, node score corresponding to parameter can, np.float32, will be updated to save memory

    can_model_index: num_query * num_candidates, which recall model the candidate from (model index start from 0, value range [0, num_recall_model) ), np.int32, set 0 for candidate ID -1

    model_weight: numpy vector with num_recall_model values, np.float32
    """
    can_ret = -1 * np.ones_like(can, dtype=np.int32)
    score_matrix_ret = np.zeros_like(score_matrix, dtype=np.float32)
    
    
    #update score_matrix with model weights
    np.multiply(score_matrix, model_weight[can_model_index], out = score_matrix)
    
    for i in range(can.shape[0]):
        
        sorted_can = np.sort(can[i, :])
        
        argsorted_can = np.argsort(can[i, :])
        
        unq_can = np.unique(can[i, :])
        
        unq_can_reverse = {v:k for k, v in enumerate(unq_can)}
        
        norm_sorted_can = np.array([unq_can_reverse[item] for item in sorted_can], dtype=np.int32)
        
        np.add.at(score_matrix_ret[i, :], norm_sorted_can, score_matrix[i, argsorted_can])
        
        len_unq = len(unq_can)
        if unq_can[0] == -1:
            can_ret[i, 0:len_unq-1] = unq_can[1:len_unq]
            score_matrix_ret[i, 0:len_unq-1] = score_matrix_ret[i, 1:len_unq]
            score_matrix_ret[i, len_unq-1:] = -1000000000000
        else:
            can_ret[i, 0:len_unq] = unq_can
            score_matrix_ret[i, len_unq:] = -1000000000000
        
    return can_ret, score_matrix_ret



def get_t_pred_top10(candidate, score_matrix_list, weight_list, args):
    ensemble_score_matirx = 0
    for score_matix, weight in zip(score_matrix_list, weight_list):  
        ensemble_score_matirx += score_matix * weight
    # ensemble_score_matirx_sort_index = np.argsort(-ensemble_score_matirx, axis=1)
    
    ensemble_score_matirx_sort_index = np.argsort(-ensemble_score_matirx, axis=1)
    
    ensemble_score_matirx_top10_index = ensemble_score_matirx_sort_index[:, :10]
    t_pred_top10 = np.take_along_axis(candidate, ensemble_score_matirx_top10_index, axis=1)
    return t_pred_top10
    # ensemble_score_matirx[candidate==-1]

def normal_score_matrix(candidate, score_matrix, min, max):
    score_matrix_max = max #score_matrix[candidate != -1].max()
    score_matrix_min = min #score_matrix[candidate != -1].min()
    score_matrix = (score_matrix-score_matrix_min)/(score_matrix_max-score_matrix_min)
    return score_matrix, (score_matrix_max, score_matrix_min)


def preprocessed_can_score_matrix_list(candidate, score_matrix_list, param_list, args, norm_value):

    st = time.time()
    
    for i in range(len(score_matrix_list)):
        arr = np.array(norm_value[i])
        min, max = arr.min(), arr.max()
        if args.logsoftmax:
            score_matrix_list[i], tmp = normal_logSoftmax_score_matrix(candidate, score_matrix_list[i])
        else:
            score_matrix_list[i], tmp = normal_score_matrix(candidate, score_matrix_list[i], min, max)
        param_list.append(tmp)
   
    can_unique = None
    for i in range(len(score_matrix_list)):
        can_unique, score_matrix_list[i] = aggregate_score_on_can(candidate, score_matrix_list[i], args)
    logger.info(f"processing use:{time.time() - st}")
    return can_unique, score_matrix_list, param_list

def get_mrr(val_t, val_can, score_matrix_list, weight_tuple, args):
    t_pred_top10 = get_t_pred_top10(val_can, score_matrix_list, weight_tuple, args)
    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10, 't': val_t}
    evaluator = WikiKG90Mv2Evaluator()
    ret = evaluator.eval(input_dict)
    return ret['mrr']


def get_test_submit_dict(candidate, test_score_matrix_list, weight_list):
    t_pred_top10 = get_t_pred_top10(candidate, test_score_matrix_list, weight_list)
    test_submit_dict = {}
    test_submit_dict['h,r->t'] = {'t_pred_top10': t_pred_top10}
    return test_submit_dict


if __name__ == "__main__":
    args = _get_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    num_models = len(model_test_path)+10
    if args.nni:
        params = nni.get_next_parameter()
    else:
        params = { "w_0": 0.426219640165791, "w_1": 0.8616968835925444, "w_2": 0.6481807827486468, "w_3": 0.9993742946865175, "w_4": 0.5013158923090032, "w_5": 0.9992923913191991, "w_6": 0.9367668316373489, "w_7": 0.8223118196167226, "w_8": 0.49592170864584373, "w_9": 0.3096707166189418, "w_10": 0.30634803747490114, "w_11": 0.061835772126218116, "w_12": 0.9539844364710732, "w_13": 0.271609320837523, "w_14": 0.643873636920946, "w_15": 0.13550934370760676, "w_16": 0.020741256623141826, "w_17": 0.399658646356691, "w_18": 0.8278764362759322, "w_19": 0.43604295414754074, "w_20": 0.14156634171546809, "w_21": 0.13258352327257736, "w_22": 0.6070021054063779, "w_23": 0.5441432919360004 }
    weight_tuple = []
    for i in range(num_models):
        weight_tuple.append(params['w_'+str(i)])

    vars(args)["weight_tuple"] = weight_tuple
    # init weight_search_range
    weight_tuple = args.weight_tuple
    logger.info(f"weight {weight_tuple}")
    print(args)
    model_test_path = args.infer_test_path_list
    model_val_path = args.infer_val_path_list

   
    # load all data
    if args.preprocess:
        tmp_test_infer_rst = torch.load(model_test_path[0])
        test_can = tmp_test_infer_rst['t_candidate']
        logger.info(f"test_can.shape is { test_can.shape}")

        test_score_matrix_list = []
        order_name = {}
        
        # get test_score_matrix_list
        cnt = 0
        for test_score_matrix_path in model_test_path:
            name = test_score_matrix_path.split('/')[2]
            start = torch.load(test_score_matrix_path)
            if 'h,r->t' in start:
                start = start['h,r->t']['t_pred_score'].numpy()
            else:
                start = start['t_pred_score']
                if type(start) != np.ndarray:
                    start = start.numpy()
            test_score_matrix_list.append(start)
            logger.debug(f"test score_matrix_path: {test_score_matrix_path} \n shape: {start.shape}")
            logger.info(f"load number {cnt} file done!")
            order_name[cnt] = name
            cnt += 1
            if len(test_score_matrix_list) == num_models: break
        logger.debug(f'test score matrix load done, matrix_num:{test_score_matrix_list}')
  
        for feat_name in feat_names:
            path = f"{args.manual_feature_path}/test_feats/{feat_name}"
            logger.info(f'load file {path}')
            feat = np.load(path)
            test_score_matrix_list.append(feat)
        
        logger.info(f'test score matrix load done, matrix_num:{len(test_score_matrix_list)}')

        # preprocessed matrix
        params_list = []
        save_path = args.save_path
        
        norm_value_test_path = "../norm_value_test.npy"
        if os.path.exists(norm_value_test_path) == False:    
            norm_value = all_featnorm(args,logger)
        else:
            norm_value=np.load(norm_value_test_path)
        
        test_can = test_can.astype(np.int32)
        test_can, test_score_matrix_list, params_list = preprocessed_can_score_matrix_list(test_can, test_score_matrix_list, params_list, args, norm_value)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'test_can.npy'), test_can)
        for i in range(len(test_score_matrix_list)): np.savez(os.path.join(save_path, 'model_'+str(i)+'_test_score_matrix.npz'), score_matrix=test_score_matrix_list[i], max=params_list[i][0], min=params_list[i][1])
    else: #'model_0_val_score_matrix.npz'
        data_path = args.data_path
        test_can = np.load(os.path.join(data_path, 'test_can.npy'))
        test_score_matrix_list = []
        for i in range(num_models):
            data = np.load(os.path.join(data_path, 'model_'+str(i)+'_test_score_matrix.npz'))['score_matrix']
            test_score_matrix_list.append(data)

    t_pred_top10 = get_t_pred_top10(test_can, test_score_matrix_list, weight_tuple, args)
    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10}
    evaluator = WikiKG90Mv2Evaluator()
    evaluator.save_test_submission( input_dict=input_dict, dir_path="test_prediction_reault", mode="test-challenge")