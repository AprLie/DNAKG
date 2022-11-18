from utils import *

r2t_prob = pickle.load(open(prob_dir + "/r2t_prob.pkl", "rb"))
t2r_prob = pickle.load(open(prob_dir + "/t2r_prob.pkl", "rb"))

print("load data done")
rrt = np.zeros((relation_num, relation_num))
for i in tqdm(range(relation_num)):
    for t in r2t_prob[i]:
        prob = r2t_prob[i][t]
        for r in t2r_prob[t]:
            prob2 = t2r_prob[t][r]
            rrt[i, r] += prob * prob2


def get_rrt_feat(t_candidate, hr, path):
    rrt_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r1 = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail in t2r_prob:
                for r2 in t2r_prob[tail]:
                    prob = rrt[r1, r2] * r2t_prob[r2][tail]
                    rrt_feat[i, j] += prob
    np.save(path, rrt_feat)


val_save_path = f"{args.save_path}/valid_feats"
if os.path.exists(val_save_path) == False: 
    os.makedirs(val_save_path)
get_rrt_feat(val_t_candidate, val_hr, val_save_path+"/rrt_feat.npy")
print(f"valid done, save to {val_save_path}")

test_save_path = f"{args.save_path}/test_feats"
if os.path.exists(test_save_path) == False: 
    os.makedirs(test_save_path)
get_rrt_feat(test_t_candidate, test_hr,
             test_save_path+"/rrt_feat.npy")
print(f"test done, save to {test_save_path}")
