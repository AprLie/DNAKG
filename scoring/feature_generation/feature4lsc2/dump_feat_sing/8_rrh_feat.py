
from utils import *


# 8. rrh_feat r2h h2r

h2r_prob = pickle.load(open(prob_dir + "/h2r_prob2.pkl", "rb"))
r2h_prob = pickle.load(open(prob_dir + "/r2h_prob.pkl", "rb"))

print("load data done")

rrh = np.zeros((relation_num, relation_num))
for i in tqdm(range(relation_num)):
    for h in r2h_prob[i]:
        prob = r2h_prob[i][h]
        for r in h2r_prob[h]:
            prob2 = h2r_prob[h][r]
            rrh[i, r] += prob * prob2

r2t_prob = pickle.load(open(prob_dir + "/r2t_prob.pkl", "rb"))
t2r_prob = pickle.load(open(prob_dir + "/t2r_prob.pkl", "rb"))


def get_rrh_feat(t_candidate, hr, path):
    rrh_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r1 = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in t2r_prob:
                for r2 in t2r_prob[tail]:
                    if tail not in r2t_prob[r2]:
                        print(r1, r2, tail)
                        exit()
                    prob = rrh[r1, r2] * r2t_prob[r2][tail]
                    rrh_feat[i, j] += prob
    np.save(path, rrh_feat)


val_save_path = f"{args.save_path}/valid_feats"
if os.path.exists(val_save_path) == False: 
    os.makedirs(val_save_path)
get_rrh_feat(val_t_candidate, val_hr, val_save_path+"/rrh_feat.npy")
print(f"valid done, save to {val_save_path}")

test_save_path = f"{args.save_path}/test_feats"
if os.path.exists(test_save_path) == False: 
    os.makedirs(test_save_path)
get_rrh_feat(test_t_candidate, test_hr, test_save_path+"/rrh_feat.npy")
print(f"test done, save to {test_save_path}")