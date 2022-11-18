### smore+rule
## test

path_train_hrt="dataset/wikikg90m-v2/processed/train_hrt.npy"
path_val_hr="dataset/wikikg90m-v2/processed/val_hr.npy"
path_test_hr="dataset/wikikg90m-v2/processed/test-challenge_hr.npy"
folder_prev_check="data/"

/opt/conda/bin/python code/combine_candidate.py \
--candidate_type "test" \
--path_train_hrt ${path_train_hrt} \
--path_val_hr ${path_val_hr} \
--path_test_hr ${path_test_hr} \
--folder_prev_check ${folder_prev_check} \
--combine_can_path data/combine_test_can_smore-rule-tmp.npy \
--candidate_path_list {your_path_to_structure_candidate}/smore_test-challenge.npy \
--candidate_path_list {your_path_to_rulemining_candidate}/rulemining_test-challenge.npy \
>> combine_candidate.log

## valid
/opt/conda/bin/python code/combine_candidate.py \
--candidate_type "valid" \
--path_train_hrt ${path_train_hrt} \
--path_val_hr ${path_val_hr} \
--path_test_hr ${path_test_hr} \
--folder_prev_check ${folder_prev_check} \
--combine_can_path data/combine_valid_can_smore-rule-tmp.npy \
--candidate_path_list {your_path_to_structure_candidate}/smore_test-challenge.npy \
--candidate_path_list {your_path_to_rulemining_candidate}/rulemining_test-challenge.npy \
>> combine_candidate.log