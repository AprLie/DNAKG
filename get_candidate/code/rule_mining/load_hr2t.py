import pickle
import numpy as np
from collections import defaultdict
from ogb.lsc import WikiKG90Mv2Dataset
from tqdm import *
import torch



dataset = WikiKG90Mv2Dataset(root='../dataset')
num_entities = dataset.num_entities
num_relations = dataset.num_relations
num_feat_dims = dataset.num_feat_dims
print('num_entities', num_entities)
print('num_relations', num_relations)
print('num_feat_dims', num_feat_dims)


valid_task = dataset.valid_dict['h,r->t'] # get a dictionary storing the h,r->t task.
v_hr = valid_task['hr']
len_v = v_hr.shape[0]
v_t = valid_task['t'].reshape(len_v, -1)
v_hrt = np.concatenate((v_hr, v_t), axis=1)
test_challenge = dataset.test_dict(mode='test-challenge')['h,r->t']
tch_hr = test_challenge['hr']
test_dev = dataset.test_dict(mode='test-dev')['h,r->t']
tdv_hr = test_dev['hr']

print(tdv_hr)

pca = ['80']
threshold = [50]
type = ['valid', 'test-challenge', 'test-dev']



for d in type:
    for p in pca:
        with open(d+'_hr2candidate_'+p+'.pkl', 'rb') as f:
            hr2t = pickle.load(f)
        for t in threshold:
            candidate = []
            head = []
            relation = []
            if d == 'valid':
                for h, r, _t in tqdm(v_hrt):
                    head.append(h)
                    relation.append(r)
                    key = str(h)+'_'+str(r)
                    temp = hr2t[key]
                    temp = temp[:min(len(temp), t)]
                    for i in range(t-len(temp)):
                        temp.append(-1)
                    candidate.append(temp)
            elif d =='test-challenge':
                for h, r in tqdm(tch_hr):
                    head.append(h)
                    relation.append(r)
                    key = str(h)+'_'+str(r)
                    temp = hr2t[key]
                    temp = temp[:min(len(temp), t)]
                    for i in range(t-len(temp)):
                        temp.append(-1)
                    candidate.append(temp)
            else:
                for h, r in tqdm(tdv_hr):
                    head.append(h)
                    relation.append(r)
                    key = str(h) + '_' + str(r)
                    temp = hr2t[key]
                    temp = temp[:min(len(temp), t)]
                    for i in range(t - len(temp)):
                        temp.append(-1)
                    candidate.append(temp)
            candidate = np.array(candidate)
            print(candidate)
            if d == 'valid':
                idx = 0
                cnt = 0
                for h, r, _t in tqdm(v_hrt):
                    for j in range(t):
                        if candidate[idx][j] == _t:
                            cnt += 1
                            break
                    idx += 1
                print(d, p, t, cnt, cnt/15000)

            if t == 50 and p == '80':
                # new = dict()
                # # tail_neg = torch.tensor(candidate)
                # new['head'] = np.array(head)
                # new['relation'] = np.array(relation)
                # new['tail_neg'] = tail_neg
                # print(new)
                np.save("rulemining_"+d+".npy", candidate)
                # torch.save(new, '../rule_mining_candidate_50/'
                #            + d + '.pt')

            with open(d+'_pca_'+p+'candidate_'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(candidate, f)
