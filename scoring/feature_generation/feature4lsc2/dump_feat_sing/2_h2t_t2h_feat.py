from utils import *

h2t_prob = pickle.load(open(prob_dir + "/h2t_prob.pkl", "rb"))
t2h_prob = pickle.load(open(prob_dir + "/t2h_prob.pkl", "rb"))

print("load data done")

# 2. h2t_t2h_feat h2t t2h
def get_h2t_t2h_feat(t_candidate, hr, path):
    h2t_t2h_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        h = hr[i, 0]
        if h not in h2t_prob:
            continue
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail not in h2t_prob:
                continue
            for e in h2t_prob[h]:
                if e not in h2t_prob[tail]:
                    continue
                prob = h2t_prob[h][e] * t2h_prob[e][tail]
                h2t_t2h_feat[i][j] += prob
    np.save(path, h2t_t2h_feat)
    return h2t_t2h_feat


val_save_path = f"{args.save_path}/valid_feats"
if os.path.exists(val_save_path) == False: 
    os.makedirs(val_save_path)
get_h2t_t2h_feat(val_t_candidate, val_hr,
                 val_save_path+"/h2t_t2h_feat.npy")
print(f"valid done, save to {val_save_path}")

test_save_path = f"{args.save_path}/test_feats"
if os.path.exists(test_save_path) == False: 
    os.makedirs(test_save_path)
get_h2t_t2h_feat(test_t_candidate, test_hr,
                 test_save_path+"/h2t_t2h_feat.npy")
print(f"test done, save to {test_save_path}")