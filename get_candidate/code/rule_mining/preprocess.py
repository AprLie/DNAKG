import numpy as np
from ogb.lsc import WikiKG90MDataset
from tqdm import tqdm
#%%
# the data path of the numpy file of train_hrt
train_hrt_data_path = 'dataset/wikikg90m-v2/processed/train_hrt.npy'
# the save directory of files
file_save_dir = './rules_input_file/'
#%%
# convert the numpy array of triples into txt file, to match the format of the input file of AMIE 3 (a rule miner)
def convert_to_file(triples, save_path):
    fp = open(save_path, 'w')
    for triple in tqdm(triples):
        fp.write('%d\t%d\t%d\n'%(triple[0], triple[1], triple[2]))
    fp.close()
#%%
# load the numpy array of train_hrt
train_hrt = np.load(train_hrt_data_path)

# split the data
train_hrt_0 = train_hrt[0:2_0000_0000]
train_hrt_1 = train_hrt[2_0000_0000:4_0000_0000]
train_hrt_2 = np.vstack((train_hrt[4_0000_0000:], train_hrt[:1_0000_0000]))
train_hrt_3 = train_hrt[1_0000_0000:3_0000_0000]
train_hrt_4 = train_hrt[3_0000_0000:]

# convert the numpy array to file
convert_to_file(train_hrt_0, file_save_dir + 'train_hrt_0.txt')
convert_to_file(train_hrt_1, file_save_dir + 'train_hrt_1.txt')
convert_to_file(train_hrt_2, file_save_dir + 'train_hrt_2.txt')
convert_to_file(train_hrt_3, file_save_dir + 'train_hrt_3.txt')
convert_to_file(train_hrt_4, file_save_dir + 'train_hrt_4.txt')
