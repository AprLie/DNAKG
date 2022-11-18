import numpy as np
import sys
import torch as th

val_hr = np.load("val_hr.npy")
val_t_candidate = np.load("val_t_candidate.npy", mmap_mode='r')
val_t_correct_index = np.load("val_t_correct_index.npy")

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
        relation[r] = {'correct_entity': {},'correct_entity_count': 0, 'error_entity': {}, 'error_entity_count': 0, 'candidate_entity': {}, 'candidate_entity_count': 0}
    cor_index = val_t_correct_index[i]
    cor_entity = val_t_candidate[i][cor_index]
    if cor_entity not in relation[r]['correct_entity']:
         relation[r]['correct_entity'][cor_entity] = 0
    relation[r]['correct_entity'][cor_entity] += 1
    relation[r]['correct_entity_count'] += 1

    for j, candidate in enumerate(val_t_candidate[i]):

        if candidate not in relation[r]['candidate_entity']:
            relation[r]['candidate_entity'][candidate] = 0
        relation[r]['candidate_entity'][candidate] += 1
        relation[r]['candidate_entity_count'] += 1

        if candidate == cor_entity: continue

        if candidate not in relation[r]['error_entity']:
             relation[r]['error_entity'][candidate] = 0
        relation[r]['error_entity'][candidate] += 1
        relation[r]['error_entity_count'] += 1


print('relation_num:', len(relation))

print('correct_entity_814:', len(relation[814]['correct_entity']))
print('correct_entity_814_count:', relation[814]['correct_entity_count'])

print('error_entity_814:', len(relation[814]['error_entity']))
print('error_entity_814_count:', relation[814]['error_entity_count'])

print('candidate_entity_814:', len(relation[814]['candidate_entity']))
print('candidate_entity_814_count:', relation[814]['candidate_entity_count'])
sys.stdout.flush()

tmp = {}
for r, info in relation.items():
    correct_entity = info['correct_entity']
    error_entity = info['error_entity']
    candidate_entity = info['candidate_entity']

    #print('relation:', r, 'correct_entity:', len(correct_entity), 'error_entity:', len(error_entity), 'candidate_entity:', len(candidate_entity))

    #correct_entity = sorted(correct_entity, key=lambda x:x[1], reverse=1)
    candidate_entity = sorted(candidate_entity.items(), key=lambda x:x[1], reverse=1)
    error_entity = sorted(error_entity.items(), key=lambda x:x[1], reverse=1)
    error_max_freq = error_entity[0][1]
    #print('relation:', r, 'correct_entity:', len(correct_entity), 'error_entity:', len(error_entity), 'candidate_entity:', len(candidate_entity), 'error_max_freq:', error_max_freq)
    index = 0
    count = 0
    for (e, f) in candidate_entity:
        if f > 5:
            index += 1
            count += f
        else: break

    print('relation:', r, 'correct_entity:', len(correct_entity), 'error_entity:', len(error_entity), 'candidate_entity:', len(candidate_entity), 'error_max_freq:', error_max_freq, 'always_correct_entity_num:', index, 'always_correct_entity_count:', count)
    sys.stdout.flush()

    info['error_max_freq'] = 5
    info['always_correct_entity'] = candidate_entity[:index]
    tmp[r] = {}
    tmp[r]['error_max_freq'] = error_max_freq
    tmp[r]['always_correct_entity'] = candidate_entity[:index]
    tmp[r]['relation'] = r

th.save(tmp, '../val_always_correct_entity_gt_5.pkl')
#th.save(relation, 'val_candidate_entity_info.pkl')