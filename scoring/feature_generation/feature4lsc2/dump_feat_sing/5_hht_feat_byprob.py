
from utils import *

h2t_prob = pickle.load(open(prob_dir + "/h2t_prob.pkl", "rb"))
t2h_prob = pickle.load(open(prob_dir + "/t2h_prob.pkl", "rb"))

print("load data done")

hh_byt_prob = dict()
for h in tqdm(h2t_prob):
    if len(h2t_prob[h]) > 10:
        continue
    for t in h2t_prob[h]:
        prob = h2t_prob[h][t]
        if len(t2h_prob[t]) > 10:
            continue
        for h2 in t2h_prob[t]:
            prob2 = t2h_prob[t][h2]
            if h not in hh_byt_prob:
                hh_byt_prob[h] = dict()
            if h2 not in hh_byt_prob[h]:
                hh_byt_prob[h][h2] = prob * prob2
            else:
                hh_byt_prob[h][h2] += prob * prob2


def get_hht_feat(t_candidate, hr, path):
    hht_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        h1 = hr[i, 0]
        if h1 not in hh_byt_prob:
            continue
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in t2h_prob:
                for h2 in t2h_prob[tail]:
                    if h2 not in hh_byt_prob[h1]:
                        continue
                    prob = hh_byt_prob[h1][h2] * h2t_prob[h2][tail]
                    hht_feat[i, j] += prob
    np.save(path, hht_feat)
    return hht_feat

val_save_path = f"{args.save_path}/valid_feats"
if os.path.exists(val_save_path) == False: 
    os.makedirs(val_save_path)
get_hht_feat(val_t_candidate, val_hr, val_save_path+"/hht_feat.npy")
print(f"valid done, save to {val_save_path}")

test_save_path = f"{args.save_path}/test_feats"
if os.path.exists(test_save_path) == False: 
    os.makedirs(test_save_path)
get_hht_feat(test_t_candidate, test_hr, test_save_path+"/hht_feat.npy")
print(f"test done, save to {test_save_path}")