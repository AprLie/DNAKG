import numpy as np
import sys
import torch as th

val_hr = np.load("test_hr.npy")
val_t_candidate = np.load("test_t_candidate.npy", mmap_mode='r')
# val_t_correct_index = np.load("val_t_correct_index.npy")

relation = {}
count = 0
for i, hr in enumerate(val_hr):
    count += 1
    if count % 10000 == 0:
        print('count:', count)
        sys.stdout.flush()
    r = hr[1]
    #if r != 814: continue
    if r not in relation:
        relation[r] = {'candidate_entity': {}, 'candidate_entity_count': 0}

    for j, candidate in enumerate(val_t_candidate[i]):

        if candidate not in relation[r]['candidate_entity']:
            relation[r]['candidate_entity'][candidate] = 0

        relation[r]['candidate_entity'][candidate] += 1
        relation[r]['candidate_entity_count'] += 1


print('relation_num:', len(relation))
print('candidate_entity_814:', len(relation[814]['candidate_entity']))
print('candidate_entity_814_count:', relation[814]['candidate_entity_count'])
sys.stdout.flush()

tmp = {}
cor_min_freq = 5

for r, info in relation.items():

    candidate_entity = info['candidate_entity']

    candidate_entity = sorted(candidate_entity.items(), key=lambda x:x[1], reverse=1)

    index = 0
    count = 0
    for (e, f) in candidate_entity:
        if f > cor_min_freq:
            index += 1
            count += f
        else:
            break

    print('relation:', r, 'candidate_entity:', len(candidate_entity), 'cor_min_freq:', cor_min_freq, 'maybe_correct_entity_num:', index, 'maybe_correct_entity_count:', count)
    sys.stdout.flush()

    # info['error_max_freq'] = cor_min_freq
    # info['always_correct_entity'] = candidate_entity[:index]
    tmp[r] = {}
    tmp[r]['always_correct_entity'] = candidate_entity[:index]
    tmp[r]['error_max_freq'] = cor_min_freq
    tmp[r]['relation'] = r

th.save(tmp, '../test_always_correct_entity_gt_%s.pkl' % cor_min_freq)
#th.save(relation, 'val_candidate_entity_info.pkl')