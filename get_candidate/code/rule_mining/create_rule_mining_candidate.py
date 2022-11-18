from ogb.lsc import WikiKG90Mv2Dataset
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os


root_path = r'../amie/pool_files_pca080/'

from ogb.lsc import WikiKG90Mv2Dataset
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle


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
test_dev = dataset.test_dict(mode='test-dev')['h,r->t']
tch_hr = test_challenge['hr']
tdv_hr = test_dev['hr']

r2h = defaultdict(list)
hr2candidate = dict()


for h, r in tdv_hr:
    r2h[r].append(h)
    hr2candidate[str(h)+'_'+str(r)] = []

check_relation = 0
print(len(r2h))

for r in r2h:
    check_relation += 1
    if check_relation % 10 == 0:
        print(check_relation)
    file_name = 'pca080_rel_'+str(r)+'_ruleNewTriples.npy'
    if os.path.exists(root_path + file_name):
        temp = np.load(root_path + file_name).tolist()
        for h, _r, t in temp:
            key = str(h)+'_'+str(_r)
            if key in hr2candidate:
                hr2candidate[key].append(t)

#print(hr2candidate)
with open('test-dev_hr2candidate_80.pkl', 'wb') as f:
    pickle.dump(hr2candidate, f)

