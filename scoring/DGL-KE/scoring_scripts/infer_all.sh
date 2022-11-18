export valid_can_path="data/combine_valid_can_smore-rule-tmp.npy"
export test_can_path="data/combine_test_can_smore-rule-tmp.npy"
export rst_prename="test"

echo "scripts/autosf_concat.sh"
python code/eval.py  --load_json \
dglke_path/autosf/AutoSF_wikikg90m-v2_shallow_d_768_g_50.0_lr_0.15_seed_0_4_mrr_0.1819125860929489_step_6874999/config.json \
--valid_can_path ${valid_can_path} \
--test_can_path ${test_can_path} \
--rst_prename ${rst_prename} \


# echo "scripts/complex_shallow.sh"
# python code/eval.py  --load_json \
# dglke_path/complex_shallow/ComplEx_wikikg90m-v2_shallow_d_512_g_10.0_lr_0.1_seed_77_0_mrr_0.17148838564753532_step_849999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \


# echo "scripts/distmult_shallow.sh"
# python code/eval.py  --load_json \
# dglke_path/distmult_shallow/DistMult_wikikg90m-v2_shallow_d_512_g_10.0_lr_0.1_seed_13_1_mrr_0.17049912363290787_step_2599999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \



# echo "scripts/rotate_concat.sh"
# python code/eval.py  --load_json \
# dglke_path/rotate_concat/RotatE_wikikg90m-v2_concat_d_100_g_8.0_lr_0.1_seed_0_0_mrr_0.17069180309772491_step_1599999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \



# echo "scripts/rotate_shallow.sh"
# python code/eval.py  --load_json \
# dglke_path/rotate_shallow/RotatE_wikikg90m-v2_shallow_d_256_g_12.0_lr_0.1_seed_0_3_mrr_0.2024185173213482_step_7849999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \



# echo "scripts/autosf.sh"
# python code/eval.py  --load_json \
# dglke_path/autosf/AutoSF_wikikg90m-v2_shallow_d_768_g_50.0_lr_0.15_seed_0_4_mrr_0.1819125860929489_step_6874999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \



# echo "scripts/complex_c.sh"
# python code/eval.py  --load_json \
# dglke_path/complex_c/ComplEx_wikikg90m-v2_concat_d_512_g_10.0_lr_0.1_seed_0_4_mrr_0.20194695889949799_step_1499999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \



# echo "scripts/complex_d.sh"
# python code/eval.py  --load_json \
# dglke_path/complex_d/ComplEx_wikikg90m-v2_concat_d_512_g_10.0_lr_0.1_seed_1_12_mrr_0.1941165328025818_step_974999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \



# echo "scripts/complex_e.sh"
# python code/eval.py  --load_json \
# dglke_path/complex_e/ComplEx_wikikg90m-v2_concat_d_512_g_10.0_lr_0.1_seed_9_0_mrr_0.19627264142036438_step_999999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \


# echo "scripts/complex_f.sh"
# python code/eval.py  --load_json \
# dglke_path/complex_f/ComplEx_wikikg90m-v2_concat_d_512_g_10.0_lr_0.1_seed_77_1_mrr_0.2085658460855484_step_2874999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \



# echo "scripts/distmult_g.sh"
# python code/eval.py  --load_json \
# dglke_path/distmult_g/DistMult_wikikg90m-v2_concat_d_512_g_10.0_lr_0.1_seed_13_0_mrr_0.19861449301242828_step_974999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \



# echo "scripts/simple_i.sh"
# python code/eval.py  --load_json \
# dglke_path/simple_i/SimplE_wikikg90m-v2_concat_d_512_g_10.0_lr_0.1_seed_77_0_mrr_0.18862368166446686_step_999999_mrr_end/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \



# echo "scripts/transe_a.sh"
# python code/eval.py  --load_json \
# dglke_path/transe_a/TransE_l2_wikikg90m-v2_shallow_d_768_g_10.0_lr_0.2_seed_0_33_mrr_0.23064033687114716_step_9949999/config.json \
# --valid_can_path ${valid_can_path} \
# --test_can_path ${test_can_path} \
# --rst_prename ${rst_prename} \



echo "info over"
