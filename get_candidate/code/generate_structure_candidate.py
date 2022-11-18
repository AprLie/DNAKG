import torch
import numpy
import pickle
from ogb.lsc import WikiKG90Mv2Dataset
import numpy as np
from collections import defaultdict
from tqdm import *

dataset_path = 'dataset'

dataset = WikiKG90Mv2Dataset(root='dataset')
num_entities = dataset.num_entities
num_relations = dataset.num_relations
num_feat_dims = dataset.num_feat_dims
print('num_entities', num_entities)
print('num_relations', num_relations)
print('num_feat_dims', num_feat_dims)

train_hrt = dataset.train_hrt
print(len(train_hrt))
valid_task = dataset.valid_dict['h,r->t']
v_hr = valid_task['hr']
len_v = v_hr.shape[0]
v_t = valid_task['t'].reshape(len_v, -1)
v_hrt = np.concatenate((v_hr, v_t), axis=1)
test_challenge = dataset.test_dict(mode='test-challenge')['h,r->t']
tch_hr = test_challenge['hr']

'''
calculate the in-degrees of the entities and the generate the co-occurred tails of the relations
'''
def generate_entity2indegree_and_r2tail(mode):
    entity2indegree = defaultdict(int)
    r2tail = defaultdict(set)
    for h, r, t in tqdm(train_hrt):
        entity2indegree[t] += 1
        r2tail[r].add(t)
    if mode == 'test-challenge':
        for h, r, t in v_hrt:
            entity2indegree[t] += 1
            r2tail[r].add(t)
    with open('final_r2tail_'+mode+'.pkl', 'wb') as f:
        pickle.dump(r2tail, f)
    with open('final_entity2indegree_'+mode+'.pkl', 'wb') as f:
        pickle.dump(entity2indegree, f)


'''
sort the co-occurred tails according to their in-degrees for each relation
'''
def sort_r2tail(mode):
    with open('final_r2tail_'+mode+'.pkl', 'rb') as f:
        r2tail = pickle.load(f)
    with open('final_entity2indegree_'+mode+'.pkl', 'rb') as f:
        ent2ind = pickle.load(f)
    r2sorted_tail = defaultdict(list)
    for r in tqdm(r2tail):
        tail_list = list(r2tail[r])
        tail_list.sort(key=lambda x: ent2ind[x], reverse=True)
        r2sorted_tail[r] = tail_list[:]
    with open('final_r2sorted_tail_'+mode+'.pkl', 'wb') as f:
        pickle.dump(r2sorted_tail, f)

'''
preserve the top 20000 candidates for each query
'''
def generate_smore_candidate(mode):
    new = dict()
    head = []
    relation = []
    tail_neg = []
    with open('final_r2sorted_tail_'+mode+'.pkl', 'rb') as f:
        r2sorted_tail = pickle.load(f)
    if mode == 'test-challenge':
        hr = tch_hr
    else:
        hr = v_hr
    tot_cnt = 0
    for h, r in hr:
        origin = r2sorted_tail[r]
        can_len = len(origin)
        candidate = origin[:min(can_len, 20000)]
        candidate = candidate + [-1 for i in range(20000-min(can_len, 20000))]
        tail_neg.append(candidate)
        head.append(h)
        relation.append(r)
    print(tot_cnt)

    tail_neg = np.array(tail_neg)
    # tail_neg = torch.tensor(tail_neg)

    new['head'] = np.array(head)
    new['relation'] = np.array(relation)
    new['tail_neg'] = tail_neg
    print(new)
    np.save('smore_'+mode+'.npy', tail_neg)



for mode in ['valid', 'test-challenge']:
    generate_entity2indegree_and_r2tail(mode)
    sort_r2tail(mode)
    generate_smore_candidate(mode)
