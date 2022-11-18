import torch
import numpy as np
import time
import random
import sys
import json
import os

class WikiKG90MEvaluator:

    def eval(self, input_dict):
        '''
            Format of input_dict:
            - 'h,r->t'
                - t_pred_top10: np.ndarray of shape (num_eval_triplets, 10)
                    each element must < 1001
                    (i,j) represents the j-th prediction for i-th triplet
                    Only top10 prediction is taken into account
                - t_correct_index: np.ndarray of shape (num_eval_triplets,)

        '''
        assert 'h,r->t' in input_dict
        assert ('t_pred_top10' in input_dict['h,r->t']) and ('t_correct_index' in input_dict['h,r->t'])

        # h,r->t
        t_pred_top10 = input_dict['h,r->t']['t_pred_top10']
        t_correct_index = input_dict['h,r->t']['t_correct_index']
        if not isinstance(t_pred_top10, torch.Tensor):
            t_pred_top10 = torch.from_numpy(t_pred_top10)
        if not isinstance(t_correct_index, torch.Tensor):
            t_correct_index = torch.from_numpy(t_correct_index)

        assert t_pred_top10.shape[1] == 10 and len(t_pred_top10) == len(t_correct_index)
        assert (0 <= t_pred_top10).all() and (t_pred_top10 < 1001).all()
        assert (0 <= t_correct_index).all() and (t_correct_index < 1001).all()

        mrr = self._calculate_mrr(t_correct_index.to(t_pred_top10.device), t_pred_top10)

        return {'mrr': mrr}

    def _calculate_mrr(self, correct_index, pred_top10):
        '''
            - correct_index: shape (num_eval_triplets, )
            - pred_top10: shape (num_eval_triplets, 10)
        '''
        # extract indices where correct_index is within top10
        tmp = torch.nonzero(correct_index.view(-1,1) == pred_top10)

        # reciprocal rank
        # if rank is larger than 10, then set the reciprocal rank to 0.
        rr = torch.zeros(len(correct_index)).to(tmp.device)
        rr[tmp[:,0]] = 1./(tmp[:,1].float() + 1.)

        # mean reciprocal rank
        return float(rr.mean().item())

    def save_test_submission(self, input_dict, dir_path):
        assert 'h,r->t' in input_dict
        assert 't_pred_top10' in input_dict['h,r->t']

        t_pred_top10 = input_dict['h,r->t']['t_pred_top10']

        assert t_pred_top10.shape == (1359303, 10) and (0 <= t_pred_top10).all() and (t_pred_top10 < 1001).all()

        if isinstance(t_pred_top10, torch.Tensor):
            t_pred_top10 = t_pred_top10.cpu().numpy()
        t_pred_top10 = t_pred_top10.astype(np.int16)

        makedirs(dir_path)
        filename = osp.join(dir_path, 't_pred_wikikg90m')
        np.savez_compressed(filename, t_pred_top10=t_pred_top10)

t = time.time()
st = time.time()

mode_path_list = [
("TransE_l2_wikikg90m_shallow_d_768_g_10.015_mrr_0.8819052577018738_step_2000000/valid_candidate_score_cur_0_mrr_0.8819052577018738.pkl", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
("ComplEx_wikikg90m_concat_d_512_g_10.022_mrr_0.885155200958252_step_349999/valid_candidate_score_cur_0_mrr_0.88497394323349.pkl", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
("ComplEx_wikikg90m_concat_d_512_g_10.027_lr_0.1_seed_1_0_mrr_0.8735669255256653_step_149999/valid_candidate_score_cur_0_mrr_0.8734618425369263.pkl", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
("ComplEx_wikikg90m_concat_d_512_g_10.046_mrr_0.8777506351470947_step_224999/valid_candidate_score_cur_0_mrr_0.8758164644241333.pkl", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
("ComplEx_wikikg90m_concat_d_512_g_10.0_lr_0.1_seed_77_2_mrr_0.8805294036865234_step_299999/valid_candidate_score_cur_0_mrr_0.8785282969474792.pkl", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
("DistMult_wikikg90m_concat_d_512_g_10.0_lr_0.1_seed_13_0_mrr_0.8845107555389404_step_474999/valid_candidate_score_cur_0_mrr_0.8840998411178589.pkl", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
("DistMult_wikikg90m_concat_d_512_g_20.0_lr_0.1_seed_47_0_hinge_mrr_0.8706580400466919_step_649999/valid_candidate_score_cur_0_mrr_0.8699035048484802.pkl", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
("SimplE_wikikg90m_concat_d_512_g_10.0_lr_0.1_seed_77_2_mrr_0.8838344812393188_step_399999/valid_candidate_score_cur_0_mrr_0.8838178515434265.pkl", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
("TransE_l2_wikikg90m_shallow_d_768_g_10.0_lr_0.1_seed_1_0_mrr_0.8686023354530334_step_5349999/valid_candidate_score_cur_0_mrr_0.8678067922592163.pkl", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
]

mode_result_list = []
assert len(mode_path_list) > 0

#correct_index = np.load("val_t_correct_index.npy")
val_hr = np.load("val_hr.npy")
print("val_hr.shape:", val_hr.shape)
correct_index = np.load("val_t_correct_index.npy")
print("correct_index.shape:", correct_index.shape)

model_list = []
weight_list = []

for p, w in mode_path_list:
    start = torch.load(p)['h,r->t']['t_pred_score'].numpy()
    model_list.append(start)
    weight_list.append(w)
    print('model_path:', p, 'weight:', w, 'shpae:', start.shape)
    sys.stdout.flush()

print('model load done, model_num:', len(model_list))

record_list = set()
if os.path.exists('log.grid_search'):
    used_w = json.load(open('log.grid_search'))
    record_list = set(used_w)
print('used weight:', len(record_list))
sys.stdout.flush()

flag = 0
for i in range(0, 10000000):
    weight = []
    st = ''
    start_time = time.time()
    for w in weight_list:
        ind = random.randint(0, len(w) - 1)
        time.sleep(0.01)
        weight.append(w[ind])
        st += "_" + str(w[ind])
    assert len(weight) == len(model_list)
    while st in record_list:
        print('----->st:', st, "already used")
        sys.stdout.flush()
        weight = []
        st = ''
        for w in weight_list:
            time.sleep(0.01)
            ind = random.randint(0, len(model_list) - 1)
            weight.append(w[ind])
            st += "_" + str(w[ind])
        assert len(weight) == len(model_list)

    record_list.add(st)
    print('----->st: ', st, 'new candidata')
    print('----->weight:', weight)
    sys.stdout.flush()
    if len(set(weight)) == 1 and flag == 1:
        continue
    if len(set(weight)) == 1 and flag == 0:
        flag = 1

    ensemble_score = 0
    for t, m in enumerate(model_list):
        ensemble_score += m * weight[t]

    top10=[]
    top10_entity = []
    top10_score = []

    ensemble_score_index = np.argsort(-ensemble_score, axis=1)
    # print(ensemble_score_index.shape)
    ensemble_score_index = ensemble_score_index[:, :10]
    print(ensemble_score_index.shape)

    input_dict = {}
    input_dict['h,r->t'] = {'t_correct_index': correct_index, 't_pred_top10': ensemble_score_index}

    evaluator = WikiKG90MEvaluator()
    ret = evaluator.eval(input_dict)
    print('weight:', weight)
    print("mrr:", ret['mrr'])
    print('------spend time:', time.time() - start_time, '---------\n')
    sys.stdout.flush()