# Code of DNA-KG for NeurIPS'22 OGB-LSC

## Track: WikiKG90Mv2

This is the repo of Team DNAKG for NeurIPS'22 OGB-LSC challenge.  

Team member: Xu Chen, Xiaojun Ma, Lun Du, Yue Fan, Jiayi Liao, Chongjian Yue, Qiang Fu, Shi Han.

## Installation Requirements
We use Python=3.7, Pytorch=1.12 in our experiments.

```sh
    pip install dgl==0.4.3 ogb nni==2.8
```


## 0. Data Preparation

Download data from ogb and save to `dataset/`

<!-- 2. Preprocess data for DGL-KE training. 
    ```shell
    python scoring/DGL-KE/data_prepare.py
    ``` -->

## 1. Candidate Generation
### 1.1 Structure-based Strategy
To get structure-based candidates, please run
```shell
    sh get_candidate/scripts/generate_structure_candidate.sh
```

The result is saved as a `npy` file.
### 1.2 Rule Mining
Please go to `get_candidate/code/rule_mining` for more details.

### 1.3 Combine Candidates
To combine above candidates, please run
```shell
get_candidate/code/combine_candidate.py
```
Remember to revise your path saving candidate: `your_path_to_structure_candidate` and `your_path_to_rulemining_candidate`.
For all the candidate set, it remove the candidate tail $t_i$ which appear in the training set for each $(h,r,[t_1,t_2,...,t_n])$. Then concatenate these candidate sets.

**arguments**:

```python
--path_train_hrt  			path of the train_hrt.npy
--path_val_hr				path of the val_hr.npy
--path_test_hr				path of the test_hr.npy
--candidate_type			candidate type, choice in ['test', 'valid']
--candidate_path_list		used to append the path of the candidate
--combine_can_path			path of the processed and concatenated candidate
--folder_prev_check			folder used to store some temporary files
```

easy-use scripts:

```
    sh get_candidate/scripts/combine_candidate.sh
```

## 2. Scoring Methods
### 2.1 KGE Method 
#### 2.1.1 Training
To train 13 KGE methods, please
```shell
    cd scoring/DGL-KE/training_scripts
```
then run each scripts. For example, to train the Transe_a, run
```shell
    sh transe_a.sh
```
The trained model is saved in `dglke_path/`.
Note that valdation candidate should be generated to evaluate model performance, and pay attention to the path of validation candidate.
#### 2.1.2 Scoring
To generate scores for candidates, plsase

```shell
    cd DGL-KE/scoring_scripts
    sh infer_all.sh
```
The Python script `DGL-KE/dglke/eval.py` is used to evaluate the model on the validation set and test set. You can run it to infer a new models instead of inferring all trained models.

**arguments**:

```sh
    --load_json		path of the config.json which is config file for training
    --valid_can_path	path of the validation candidate
    --test_can_path		path of the test candidate
    --rst_prename		prefix of the infer result
```

Note that we fixed some parameters in the `eval.py` as:
```sh
    args.eval = True
    args.test = True
    args.num_test_proc = 30
    args.num_proc = 30
    args.num_thread = 1
    args.gpu = [-1]
```

### 2.2 Feature-based Scoring
The code references the solution of [《NOTE: Solution for KDD-Cup 2021 WikiKG90M-LSC》](https://ogb.stanford.edu/paper/kddcup2021/wikikg90m_BD-PGL.pdf)

To get the features, you should first generate the candidate and save it in `your_candidate_path`. Then, please run
```shell
    cd scoring/feature_generation
    python ./feature4lsc2/walk_probability/h2r.py
    python ./feature4lsc2/walk_probability/h2t.py
    python ./feature4lsc2/walk_probability/r2t.py
    python ./feature4lsc2/walk_probability/r2h.py
    python ./feature4lsc2/walk_probability/t2r.py
    python ./feature4lsc2/walk_probability/t2h.py

    python {py_filename} --candidate_path {your_candidate_path} --save_path {your_save_path}
    # run py files in ./feature4lsc2/dump_feat/
```
This will give you the manual feature score.


## 3. Ensemble
1. To speed up hyperparameter searching, we preprocess the data with 

```shell
    cd ensemble/manual_feature_nni
    python3 ../code/manualfeature_getmrr.py --preprocess --save_path '../testmanual_smore_rule/'
```
and search the best weights with

```shell
    python3 ../code/manualfeature_getmrr.py --nni --save_path '../testmanual_smore_rule/'
```

2. use the best weights to generate the test results for submission
```bash
    python3 ../code/test_mrr.py --preprocess --save_path '../final_manual_smore_rule/' --data_path '../final_manual_smore_rule/'
```

## Acknowledgement
Our implementation is based on the winning solutions LSC@KDD Cup 2021. Please refer to https://ogb.stanford.edu/kddcup2021/results/#awardees_wikikg90m for more details.