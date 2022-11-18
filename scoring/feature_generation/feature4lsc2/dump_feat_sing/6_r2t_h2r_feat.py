
from utils import *

h2r_prob = pickle.load(open(prob_dir + "/h2r_prob2.pkl", "rb"))
r2t_prob = pickle.load(open(prob_dir + "/r2t_prob.pkl", "rb"))
r2h_prob = pickle.load(open(prob_dir + "/r2h_prob.pkl", "rb"))
print("load done")

r2t_h2r = np.zeros((relation_num, relation_num), dtype=np.float16)
for i in tqdm(range(relation_num)):
    for t in r2t_prob[i]:
        prob = r2t_prob[i][t]
        if t not in h2r_prob:
            continue
        for r in h2r_prob[t]:
            prob2 = h2r_prob[t][r]
            r2t_h2r[i, r] += prob * prob2


def get_r2t_h2r_feat(t_candidate, hr, path):
    r2t_h2r_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r1 = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in h2r_prob:
                for r2 in h2r_prob[tail]:
                    prob = r2t_h2r[r1, r2] * r2h_prob[r2][tail]
                    r2t_h2r_feat[i, j] += prob
    np.save(path, r2t_h2r_feat)
    return r2t_h2r_feat



val_save_path = f"{args.save_path}/valid_feats"
if os.path.exists(val_save_path) == False: 
    os.makedirs(val_save_path)
get_r2t_h2r_feat(val_t_candidate, val_hr, val_save_path+"/r2t_h2r_feat.npy")
print(f"valid done, save to {val_save_path}")

test_save_path = f"{args.save_path}/test_feats"
if os.path.exists(test_save_path) == False: 
    os.makedirs(test_save_path)
get_r2t_h2r_feat(test_t_candidate, test_hr, test_save_path+"/r2t_h2r_feat.npy")
print(f"test done, save to {test_save_path}")