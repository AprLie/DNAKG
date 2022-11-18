import numpy as np
import pickle
from tqdm import tqdm
import os

train_hrt = np.load("/mnt/usscv100data/wikikg90m-v2/processed/train_hrt.npy", mmap_mode="r")

data = dict()

for h, r, t in tqdm(train_hrt):
    if h not in data:
        data[h] = dict()
    if not r in data[h]:
        data[h][r] = 1
    else:
        data[h][r] += 1

del train_hrt

for h in data:
    h_sum = sum(data[h].values())
    for r in data[h]:
        data[h][r] /= h_sum

folder = "/mnt/xj_output/feature_output"
if os.path.exists(folder) == False:
    os.mkdir(folder)
pickle.dump(data, open("/mnt/xj_output/feature_output/h2r_prob2.pkl", "wb"))
