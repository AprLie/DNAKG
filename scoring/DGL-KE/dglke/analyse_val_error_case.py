import json
import sys
import numpy as np
import torch as th
import os

def get_realtion_result(candidate_score, args=None):
    """
    """
    # candidate_score = {'h,r->t': candidate_score}

    hr = candidate_score['h,r->t']['h,r']

    candidate = candidate_score

    predict = candidate_score

    print('hr.shape:', hr.shape)
    print('t_correct_index shape:', predict['h,r->t']['t'].shape)
    print('t_pred_top10 shape:', predict['h,r->t']['t_pred_top10'].shape)

    predict['h,r->t']['t'] = predict['h,r->t']['t'].tolist()
    predict['h,r->t']['t_pred_top10'] = predict['h,r->t']['t_pred_top10'].tolist()

    tt = {}

    for i, a in enumerate(hr):
        if predict['h,r->t']['t'][i] in predict['h,r->t']['t_pred_top10'][i]:
            index = predict['h,r->t']['t_pred_top10'][i].index(predict['h,r->t']['t'][i])
        else:
            index = -1
            #print('---===============')
        #print(candidate[i])
        # cor_num = candidate[i][predict['h,r->t']['t_correct_index'][i]]
        # top10_num = []
        # for id in predict['h,r->t']['t_pred_top10'][i]:
        #     top10_num.append(candidate[i][id])

        # print('%s\t%s' % (a.tolist()[1], index + 1))
        #if i > 100: break
        r = str(a.tolist()[1])
        position = str(index + 1)
        if r not in tt:
          tt[r] = {'0':0.000001, '1':0.000001, '2':0.000001, '3':0.000001, '4':0.000001, '5':0.000001, '6':0.000001, '7':0.000001, '8':0.000001, '9':0.000001, '10':0.000002}

        tt[r][position] += 1

    aa = sorted(tt.items(), key=lambda x:int(x[0]))

    path = (args.save_path + '/' + 'predict_result_analyse.csv') if args is not None else 'predict_result_analyse.csv'
    if args is not None and not os.path.exists(args.save_path): #ckpts/
        os.makedirs(args.save_path)
    fp = open(path, 'w+')
    print('predict_result_analyse_path:', path)
    tmp = []
    for r, i in aa:
        mrr = 0
        count = 0
        for t in range(0, 11):
            count += i[str(t)]
            if t == 0:
                i[str(t)] = int(i[str(t)])
                continue
            mrr += i[str(t)] * 1 / t
            i[str(t)] = int(i[str(t)])
        mrr = mrr / count
        count = int(count)
        des = r + '\t' + str(i['0']) + '\t' + str(i['1']) + '\t' + str(i['2']) + '\t' + str(i['3']) + '\t' + str(i['4']) + '\t' + str(i['5']) + '\t' + str(i['6']) + '\t' + str(i['7']) + '\t' + str(i['8']) + '\t' + str(i['9']) + '\t' + str(i['10']) + '\t' + str(mrr) + '\t' + str(count)
        tmp.append([des, (1 - mrr) * count * 5])
        # fp.write(des + '\n')

    tmp = sorted(tmp, key=lambda x:x[1], reverse=True)

    for t in tmp:
        fp.write(t[0] + '\n')
    fp.close()