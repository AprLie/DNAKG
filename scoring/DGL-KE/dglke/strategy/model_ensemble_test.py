import numpy as np
import torch
import numpy as np
import time
import random
import sys

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
entity_freq = torch.load("../test_always_correct_entity_gt_5.pkl")
print('entity_relation_num:', len(entity_freq))
print('load test_always_correct_entity_gt_5.pkl speed time:', time.time() - st)

mode_path_list = [
("ensemble_ckpts/TransE_l2_wikikg90m_shallow_d_768_g_10.015_mrr_0.8819052577018738_step_2000000/test_candidate_score_cur_0_mrr_0.0029543458949774504.pkl", 1),
("ensemble_ckpts/ComplEx_wikikg90m_concat_d_512_g_10.022_mrr_0.885155200958252_step_349999/test_candidate_score_cur_0_mrr_0.002948673442006111.pkl", 0.3),
("ensemble_ckpts/ComplEx_wikikg90m_concat_d_512_g_10.046_mrr_0.8777506351470947_step_224999/test_candidate_score_cur_0_mrr_0.002946344204246998.pkl", 0.3),
("ensemble_ckpts/ComplEx_wikikg90m_concat_d_512_g_10.0_lr_0.1_seed_77_2_mrr_0.8805294036865234_step_299999/test_candidate_score_cur_0_mrr_0.0029231279622763395.pkl", 0.1),
("ensemble_ckpts/ComplEx_wikikg90m_concat_d_512_g_10.027_lr_0.1_seed_1_0_mrr_0.8735669255256653_step_149999/test_candidate_score_cur_0_mrr_0.0029489020816981792.pkl", 0.3),
("ensemble_ckpts/TransE_l2_wikikg90m_shallow_d_768_g_10.0_lr_0.1_seed_1_0_mrr_0.8686023354530334_step_5349999/test_candidate_score_cur_0_mrr_0.002975531853735447.pkl", 0.4),
("ensemble_ckpts/SimplE_wikikg90m_concat_d_512_g_10.0_lr_0.1_seed_77_2_mrr_0.8838344812393188_step_399999/test_candidate_score_cur_0_mrr_0.002915985183790326.pkl", 0.1),
("ensemble_ckpts/DistMult_wikikg90m_concat_d_512_g_10.0_lr_0.1_seed_13_0_mrr_0.8845107555389404_step_474999/test_candidate_score_cur_0_mrr_0.002936682663857937.pkl", 0.3),
("ensemble_ckpts/DistMult_wikikg90m_concat_d_512_g_20.0_lr_0.1_seed_47_0_hinge_mrr_0.8706580400466919_step_649999/test_candidate_score_cur_0_mrr_0.0029708545189350843.pkl", 0.8)
]

mode_result_list = []
assert len(mode_path_list) > 0
for (path, sc) in mode_path_list:
    print('model_path:', path, 'weight:', sc)

test_hr = np.load("test_hr.npy")
print("test_hr.shape:", test_hr.shape)

test_t_candidate = np.load("test_t_candidate.npy", mmap_mode='r')
print('test_t_candidate.shape:', test_t_candidate.shape)

start = torch.load(mode_path_list[0][0])['h,r->t']['t_pred_score'].numpy()
print('model_path:', mode_path_list[0][0], 'weight:', mode_path_list[0][1], 'shape:', start.shape)
ensemble_score = start * mode_path_list[0][1]
start = None

for i, (path, score) in enumerate(mode_path_list):
    if i == 0: continue
    start = torch.load(path)['h,r->t']['t_pred_score'].numpy()
    print('model_path:', path, 'weight:', score, 'shape:', start.shape)
    sys.stdout.flush()
    ensemble_score += start * mode_path_list[i][1]
    start = None

top10=[]
top10_entity = []
top10_score = []

max_freq = 5
max_correct_candidate_num = 700

print("max_freq:", max_freq)
st = time.time()
select_relation_entity_num = []
for r, v in entity_freq.items():
    always_correct_entity = v['always_correct_entity']
    always_correct_entity_dict = {}
    for (e, freq) in always_correct_entity:
            if freq <= max_freq: continue
            always_correct_entity_dict[e] = freq
            if len(always_correct_entity_dict) >= max_correct_candidate_num:
                print("relation:", r, 'candidate:', e, 'max_correct_candidate_num:', max_correct_candidate_num)
                break

    v['always_correct_entity_dict'] = always_correct_entity_dict
    print("relation:", r, "always_correct_entity_num:", len(always_correct_entity_dict))
    sys.stdout.flush()
    if len(always_correct_entity_dict) > 0:
        select_relation_entity_num.append([r, len(always_correct_entity_dict)])

select_relation_entity_num = sorted(select_relation_entity_num, key=lambda x:x[1], reverse=1)
print('---select_relation_entity_num----')
for (r, n) in select_relation_entity_num:
    print('relation:', r, 'always_correct_entity_num:', n)

print('build always_correct_entity_dict spend time: ', time.time() - st)
sys.stdout.flush()

count = -1
for pre in ensemble_score:
    count += 1
    if count % 10000 == 0: print('count:', count)
    assert len(pre) == 1001
    if test_hr[count][1] in entity_freq:
        candidate = test_t_candidate[count].tolist()
        candidate_set = set(candidate)
        index_list = []
        always_correct_entity_dict = entity_freq[test_hr[count][1]]['always_correct_entity_dict']
        keys = set(list(always_correct_entity_dict.keys())) & candidate_set
        tmp = []
        for key in keys:
            tmp.append([key, always_correct_entity_dict[key]])
        tmp = sorted(tmp, key=lambda x:x[1], reverse=1)
        for k, v in tmp:
            if k in candidate_set:
                index = candidate.index(k)
                index_list.append(index)
            if len(index_list) >= 10: break
        index_list = index_list[:10]

    argsort = np.argsort(-pre)[:10]
    if test_hr[count][1] in entity_freq:
        for arg in argsort:
            if len(index_list) < 10 and arg not in index_list:
                index_list.append(arg)
        argsort = np.array(index_list)
    assert len(set(argsort.tolist())) == 10
    top10.append(argsort)

assert count == len(ensemble_score) - 1
top10 = np.array(top10)
input_dict = {}
input_dict['h,r->t'] = {'t_pred_top10': top10}
print('spend time:', time.time() - t)
torch.save(input_dict, "test_score_cur_0_max_freq_%s_max_correct_candidate_num_%s.pkl" % (max_freq, max_correct_candidate_num))