import torch
import numpy as np
import argparse
import numpy as np
from ogb.lsc import WikiKG90Mv2Evaluator
import nni
import os
import time
from scipy.special import log_softmax
import pdb
import logging
from utils import * 

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
    parser.add_argument('--log_file', type=str, default="../output/rule.log")
    parser.add_argument('--manual_feature_path', default="/data/manual_feature/smore-rule")
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
    
    ensemble_score_matirx_sort_index = np.argsort(-ensemble_score_matirx, axis=1)
    
    ensemble_score_matirx_top10_index = ensemble_score_matirx_sort_index[:, :10]
    t_pred_top10 = np.take_along_axis(candidate, ensemble_score_matirx_top10_index, axis=1)
    return t_pred_top10


def normal_score_matrix(candidate, score_matrix, min, max):
    score_matrix_max = max 
    score_matrix_min = min 
    score_matrix = (score_matrix-score_matrix_min)/(score_matrix_max-score_matrix_min)
    return score_matrix, (score_matrix_max, score_matrix_min)



def preprocessed_can_score_matrix_list(candidate, score_matrix_list, param_list, args, norm_value):
    st = time.time()
    
    for i in range(len(score_matrix_list)):
        arr = np.array(norm_value[i])
        min, max = arr.min(), arr.max()
        
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
    model_test_path = args.infer_test_path_list
    model_val_path = args.infer_val_path_list

    num_models = len(model_val_path)+10
    if args.nni:
        params = nni.get_next_parameter()
    else:
        params = {'w_0': 0.1, 'w_1':0.1, 'w_2': 0.1, 'w_3':0.1, 'w_4': 0.1, 'w_5':0.1, 'w_6': 0.1, 'w_7':0.1, 'w_8': 0.1, 'w_9':0.1, 'w_10': 0.1, 'w_11':0.1, 'w_12':0.1, 'w_13': 0.1, 'w_14':0.1, 'w_15':0.1,'w_16':0.1, 'w_17':0.1, 'w_18': 0.1, 'w_19':0.1, 'w_20': 0.1, 'w_21':0.1, 'w_22':0.1}
    weight_tuple = []
    for i in range(num_models):
        weight_tuple.append(params['w_'+str(i)])

    vars(args)["weight_tuple"] = weight_tuple
    # init weight_search_range
    weight_tuple = args.weight_tuple
    logger.info(f"weight {weight_tuple}")
    print(args)
   
    # load all data
    if args.preprocess:
        tmp_val_infer_rst = torch.load(model_val_path[0])

        val_t = tmp_val_infer_rst['t']
        logger.info(f"val_t.shape:{val_t.shape}" )
        val_can = tmp_val_infer_rst['t_candidate']
        logger.info(f"val_can.shape is { val_can.shape}")

        val_score_matrix_list = []
        order_name = {}
        
        # get val_score_matrix_list
        cnt = 0
        for val_score_matrix_path in model_val_path:
            name = val_score_matrix_path.split('/')[2]
            start = torch.load(val_score_matrix_path)
            if 'h,r->t' in start:
                start = start['h,r->t']['t_pred_score'].numpy()
            else:
                start = start['t_pred_score']
                if type(start) != np.ndarray:
                    start = start.numpy()
            val_score_matrix_list.append(start)
            logger.debug(f"valid score_matrix_path: {val_score_matrix_path} \n shape: {start.shape}")
            logger.info(f"load number {cnt} file done!")
            order_name[cnt] = name
            cnt += 1
            if len(val_score_matrix_list) == num_models: break
        logger.debug(f'valid score matrix load done, matrix_num:{val_score_matrix_list}')
  
        for feat_name in feat_names:
            path = f"{args.manual_feature_path}/valid_feats/{feat_name}"
            logger.info(f'load file {path}')
            feat = np.load(path)
            val_score_matrix_list.append(feat)
        
        logger.info(f'valid score matrix load done, matrix_num:{len(val_score_matrix_list)}')

        # preprocessed matrix
        params_list = []
        save_path = args.save_path
        
        norm_value_path = "../norm_value.npy"
        if os.path.exists(norm_value_path) == False:    
            norm_value = featnorm(args,logger)
        else:
            norm_value=np.load(norm_value_path)
        val_can = val_can.astype(np.int32)
        val_t = val_t.numpy().astype(np.int32)
        val_can, val_score_matrix_list, params_list = preprocessed_can_score_matrix_list(val_can, val_score_matrix_list, params_list, args, norm_value)
        # for i in range(len(val_score_matrix_list)): print(score_matrix=val_score_matrix_list[i].min(), score_matrix=val_score_matrix_list[i].max())
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'val_can.npy'), val_can)
        np.save(os.path.join(save_path, 'val_t.npy'), val_t)
        for i in range(len(val_score_matrix_list)): np.savez(os.path.join(save_path, 'model_'+str(i)+'_val_score_matrix.npz'), score_matrix=val_score_matrix_list[i], max=params_list[i][0], min=params_list[i][1])
    else: #'model_0_val_score_matrix.npz'
        data_path = args.data_path
        val_can = np.load(os.path.join(data_path, 'val_can.npy'))
        val_t = np.load(os.path.join(data_path, 'val_t.npy'))
        val_score_matrix_list = []
        for i in range(num_models):
            data = np.load(os.path.join(data_path, 'model_'+str(i)+'_val_score_matrix.npz'))['score_matrix']
            val_score_matrix_list.append(data)
    
    mrr = get_mrr(val_t, val_can, val_score_matrix_list, weight_tuple, args)
    logger.info(f"mrr is {mrr}")
    if args.nni:
        nni.report_final_result(mrr)