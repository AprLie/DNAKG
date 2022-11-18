
from utils import *


r2t_prob = pickle.load(open(prob_dir + "/r2t_prob.pkl", "rb"))

print("load data done")


# 7. r2t_feat r2t 
def get_r2t_feat(t_candidate, hr, path):
    r2t_feat = np.zeros(t_candidate.shape, dtype=np.float16)
    for i in tqdm(range(t_candidate.shape[0])):
        r = hr[i, 1]
        for j in range(t_candidate.shape[1]):
            t = t_candidate[i, j]
            if t in r2t_prob[r]:
                r2t_feat[i, j] = r2t_prob[r][t]
    np.save(path, r2t_feat)
    return r2t_feat


val_save_path = f"{args.save_path}/valid_feats"
if os.path.exists(val_save_path) == False: 
    os.makedirs(val_save_path)
get_r2t_feat(val_t_candidate, val_hr, val_save_path+"/r2t_feat.npy")
print(f"valid done, save to {val_save_path}")

test_save_path = f"{args.save_path}/test_feats"
if os.path.exists(test_save_path) == False: 
    os.makedirs(test_save_path)
get_r2t_feat(test_t_candidate, test_hr, test_save_path+"/r2t_feat.npy")
print(f"test done, save to {test_save_path}")