import argparse
import os
import pickle
import numpy as np
import mkl
mkl.get_max_threads()  # type: ignore


def generate_prev_dict(dict_type, folder_prev_check, path_train_hrt, path_val_hr, path_test_hr):

    path_prev_check_test = os.path.join(folder_prev_check, 'prev_check_test_dict.pkl')
    path_prev_check_valid = os.path.join(folder_prev_check, 'prev_check_valid_dict.pkl')
    # check exist
    if os.path.exists(path_prev_check_test) and os.path.exists(path_prev_check_valid):
        rst = None
        if dict_type == "valid":
            with open(path_prev_check_valid, 'rb') as f:
                rst = pickle.load(f)
        elif dict_type == "test":
            with open(path_prev_check_test, 'rb') as f:
                rst = pickle.load(f)
        return rst
    # 生成val和test的hr构成的query对应的历史tail
    train_hrt = np.load(path_train_hrt)
    val_hr = np.load(path_val_hr)
    test_hr = np.load(path_test_hr)

    prev_check_valid_dict = {}
    prev_check_test_dict = {}
    for i in range(val_hr.shape[0]):
        if val_hr[i, 0] not in prev_check_valid_dict:
            prev_check_valid_dict[val_hr[i, 0]] = {}
        prev_check_valid_dict[val_hr[i, 0]][val_hr[i, 1]] = []
    for i in range(test_hr.shape[0]):
        if test_hr[i, 0] not in prev_check_test_dict:
            prev_check_test_dict[test_hr[i, 0]] = {}
        prev_check_test_dict[test_hr[i, 0]][test_hr[i, 1]] = []

    def add_dict():
        for i in range(train_hrt.shape[0]):
            key1, key2 = train_hrt[i, 0], train_hrt[i, 1]
            val = train_hrt[i, 2]
            if key1 in prev_check_valid_dict:
                if key2 in prev_check_valid_dict[key1]:
                    prev_check_valid_dict[key1][key2].append(val)
            if key1 in prev_check_test_dict:
                if key2 in prev_check_test_dict[key1]:
                    prev_check_test_dict[key1][key2].append(val)
    add_dict()
    with open(path_prev_check_test, 'wb') as f:
        pickle.dump(prev_check_test_dict, f)
    with open(path_prev_check_valid, 'wb') as f:
        pickle.dump(prev_check_valid_dict, f)

    # get rst
    if os.path.exists(path_prev_check_test) and os.path.exists(path_prev_check_valid):
        rst = None
        if dict_type == "valid":
            with open(path_prev_check_valid, 'rb') as f:
                rst = pickle.load(f)
        elif dict_type == "test":
            with open(path_prev_check_test, 'rb') as f:
                rst = pickle.load(f)
        return rst


# 去历史结果
def prev_check(hr_np, can_np, prev_check_dict):
    val_can_unique = -1 * np.ones_like(can_np, dtype=np.int64)
    used_count = 0
    for i in range(can_np.shape[0]):
        key1, key2 = hr_np[i, 0], hr_np[i, 1]
        tmp_set = set(can_np[i, :])  # 自带去重

        if len(prev_check_dict[key1][key2]) != 0:
            tmp_set = tmp_set - set(prev_check_dict[key1][key2])  # 去除掉used t
        tmp_set.discard(-1)  # 去除掉空
        used_count += can_np.shape[1]-len(tmp_set)
        val_can_unique[i, 0:len(tmp_set)] = np.asarray(list(tmp_set))
    print(f"used count: {used_count}")
    return val_can_unique


def get_candidate_list(path_list):
    can_list = []
    for path in path_list:
        print(f"load {path}")
        tmp = np.load(path)
        can_list.append(tmp)
    return can_list


def concatenate_checkedCan(checkd_can_list):
    print("concatenate_checkedCan")
    checked_can = np.concatenate(checkd_can_list, axis=1)
    return checked_can


class JsonArg(argparse.ArgumentParser):
    def __init__(self):
        super(JsonArg, self).__init__()
        self.add_argument('--path_train_hrt', type=str)
        self.add_argument('--path_val_hr', type=str)
        self.add_argument('--path_test_hr', type=str)
        self.add_argument('--candidate_type', type=str, choices=['test', 'valid'])
        self.add_argument('--candidate_path_list', action='append')
        self.add_argument('--combine_can_path', type=str,)
        self.add_argument('--folder_prev_check', type=str, )


if __name__ == "__main__":
    args = JsonArg().parse_args()

    prev_check_dict = generate_prev_dict(args.candidate_type, args.folder_prev_check,
                                         args.path_train_hrt, args.path_val_hr, args.path_test_hr)
    if args.candidate_type == "test":
        eval_hr = np.load(args.path_test_hr)
    else:
        assert args.candidate_type == "valid"
        eval_hr = np.load(args.path_val_hr)
    eval_can_list = get_candidate_list(args.candidate_path_list)
    for i in range(len(eval_can_list)):
        print(f"remove used node for can {i}")
        eval_can_list[i] = prev_check(eval_hr,  eval_can_list[i], prev_check_dict)
    combine_can = concatenate_checkedCan(eval_can_list)
    np.save(args.combine_can_path, combine_can)
    print(f"can save in {args.combine_can_path}")
