import numpy as np
import torch

path = 'dataset/wikikg90m-v2/'
val = torch.load(path + 'eval-original/valid.pt')
val_save = val['tail_neg'].numpy()
np.save(path + 'processed/val_candidate.npy', val_save)