from socket import getnameinfo
import torch 
import pdb 
import numpy as np 

valid_can_path = "/data/combine_valid_can_smore-rule.npy"
test_can_path = "/data/combine_test_can_smore-rule.npy"
dev_can_path ="/data/combine_dev_can_smore-rule.npy"
feat_names=["h2t_h2t_feat.npy", "hht_feat.npy", "r2t_feat.npy", "rrh_feat.npy", "rt_feat.npy","h2t_t2h_feat.npy", "ht_feat.npy", "r2t_h2r_feat.npy", "rrt_feat.npy", "t2h_h2t_feat.npy"]




def featnorm(args, logger, test_mode="challenge"):
    assert test_mode in ["challenge","dev"]
    # aims to norm all the prediction score from 13 base models, 2 ote models and 10 manual feats from PGL
    model_test_path = args.infer_test_path_list
    model_val_path = args.infer_val_path_list
    model_dev_path = args.infer_dev_path_list

    norm_score = []
    # load candidate, which shape should be #query * #candidate_set
    val_can = np.load(valid_can_path)
    if test_mode == "challenge":
        test_can = np.load(test_can_path)
    else:
        test_can = np.load(dev_can_path)

    for i in range(len(model_val_path)):
        name = model_val_path[i].split('/')[8]
        val_ = torch.load(model_val_path[i])
        val_pred_score = val_['t_pred_score']

        if test_mode == "challenge":
            test_ = torch.load(model_test_path[i])
        else:
            test_ = torch.load(model_dev_path[i])
        
        test_pred_score = test_['t_pred_score']
        norm_score.append([val_pred_score[val_can != -1].min().item(), val_pred_score[val_can != -1].max().item(), test_pred_score[test_can != -1].min().item(), test_pred_score[test_can != -1].max().item()])
        logger.info(f"model {name} val min {val_pred_score[val_can != -1].min().item()} max {val_pred_score[val_can != -1].max().item()} test min {test_pred_score[test_can != -1].min().item()} max {test_pred_score[test_can != -1].max().item()}")

    # load manual feats for valid and test-challenge
    for feat_name in feat_names:
        valid_path = f"{args.manual_feature_path}/valid_feats/{feat_name}"
        logger.info(f'load valid feat file: {valid_path}')
        val_pred_score = np.load(valid_path)

        if test_mode == "challenge": munual_test_path = f"{args.manual_feature_path}/test_feats/{feat_name}"
        else: munual_test_path = f"{args.manual_feature_path}/dev_feats/{feat_name}"
        
        test_pred_score = np.load(munual_test_path)
        logger.info(f'load test feat file: {munual_test_path}')

        norm_score.append([val_pred_score[val_can != -1].min().item(), val_pred_score[val_can != -1].max().item(), test_pred_score[test_can != -1].min().item(), test_pred_score[test_can != -1].max().item()])
    return norm_score



if __name__ == '__main__':
    print("test")