from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from doctest import testfile

import sys
import argparse
import logging
import os
import numpy as np
import pickle
from scipy.sparse import lil_matrix
from tqdm import tqdm
import math
from multiprocessing import Pool


class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--candidate_path',type=str, required=True)
        self.add_argument('--candidate_name', type=str, default="")
        self.add_argument('--save_path', type=str, default = "./save_features")
        self.add_argument('--prob_path', type=str, requited=True)
        self.add_argument('--wikidata_path', type=str, requited=True)


args = CommonArgParser().parse_args()
test_num = 10000
relation_num = 1387
prob_dir = args.prob_path 
base_dir = args.wikidata_path

val_hr = np.load(base_dir + "/processed/val_hr.npy", mmap_mode="r")
test_hr = np.load(base_dir + "/processed/test-challenge_hr.npy",mmap_mode="r")[:test_num]

val_file = f"{args.candidate_path}/combine_valid_can_{args.candidate_name}.npy"
val_t_candidate = np.load(val_file, mmap_mode='r')
print(f"loaded valid candidate file: {val_file}")
test_file = f"{args.candidate_path}/combine_test_can_{args.candidate_name}.npy"
test_t_candidate = np.load(test_file, mmap_mode="r")[:test_num]
print(f"loaded test candidate file: {val_file}")