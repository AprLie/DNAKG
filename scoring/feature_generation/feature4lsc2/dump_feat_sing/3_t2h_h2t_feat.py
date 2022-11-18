from utils import *

h2t_prob = pickle.load(open(prob_dir + "/h2t_prob.pkl", "rb"))
t2h_prob = pickle.load(open(prob_dir + "/t2h_prob.pkl", "rb"))

print("load data done")


def get_t2h_h2t_feat(t_candidate, hr, path):
    t2h_h2t_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        h = hr[i, 0]
        if h not in t2h_prob:
            continue
        for j in range(t_candidate.shape[1]):
            tail = t_candidate[i, j]
            if tail not in t2h_prob:
                continue
            for e in t2h_prob[h]:
                if e not in t2h_prob[tail]:
                    continue
                prob = t2h_prob[h][e] * h2t_prob[e][tail]
                t2h_h2t_feat[i][j] += prob
    np.save(path, t2h_h2t_feat)
    return t2h_h2t_feat

val_save_path = f"{args.save_path}/valid_feats"
if os.path.exists(val_save_path) == False: 
    os.makedirs(val_save_path)
get_t2h_h2t_feat(val_t_candidate, val_hr,
                 val_save_path+ "/t2h_h2t_feat.npy")
print(f"valid done, save to {val_save_path}")

test_save_path = f"{args.save_path}/test_feats"
if os.path.exists(test_save_path) == False: 
    os.makedirs(test_save_path)
get_t2h_h2t_feat(test_t_candidate, test_hr,
                 test_save_path+ "/t2h_h2t_feat.npy")
print(f"test done, save to {test_save_path}")