import numpy as np
import torch as th
import sys
from ogb.lsc import WikiKG90MEvaluator

evaluator = WikiKG90MEvaluator()

#input_dict = th.load("test_top10_index_by_train_add_val.pkl")
#input_dict = th.load("test_top10_index_by_train_by_model_mrr_0.88.pkl")
#input_dict = th.load("test_top10_index_by_ensemble_model_compIEx_mrr_0.885_and_transE_mrr_0.881.pkl")
path = sys.argv[1]
print('path:', path)
input_dict = th.load(path)
print(type(input_dict['h,r->t']['t_pred_top10']))
#input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10}
print(input_dict['h,r->t'].keys())
if 't_correct_index' in input_dict['h,r->t']:
    del input_dict['h,r->t']['t_correct_index']

print(input_dict['h,r->t'].keys())
print("test.shape:", input_dict['h,r->t']['t_pred_top10'].shape)
print("test.dtype", type(input_dict['h,r->t']['t_pred_top10']))
input_dict['h,r->t']['t_pred_top10'] = th.from_numpy(input_dict['h,r->t']['t_pred_top10'])
print("test.dtype", type(input_dict['h,r->t']['t_pred_top10']))
evaluator.save_test_submission(input_dict = input_dict, dir_path = './')