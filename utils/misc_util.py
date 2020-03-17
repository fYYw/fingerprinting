import os
import copy
import json
from sklearn.metrics import f1_score


def get_mse(pred: list, tar: list):
    se = []
    for p, t in zip(pred, tar):
        se.append((p - t) ** 2)
    return 1. * sum(se) / len(se)


def distance_to_label(pred: list, label: list):
    r = []
    for p in pred:
        dis = [(p - l) ** 2 for l in label]
        idx = dis.index(min(dis))
        r.append(label[idx])
    return r


def report_result(result, y_names, save_file):
    logs = []
    packed_result = zip(result['author'],
                        result['vader'][0], result['vader'][1],
                        result['flair'][0], result['flair'][1],
                        result['blob_sentiment'][0], result['blob_sentiment'][1],
                        result['blob_subjective'][0], result['blob_subjective'][1])
    perf_log = ''
    for y_name in y_names:
        [preds, tars] = result[y_name]
        f1_ = f1_score(tars, preds, labels=[0, 2], average='macro')
        mse_ = get_mse(preds, tars)
        perf_log += "{0} f1: {1:.4f}, mse: {2:.4f}; ".format(y_name, f1_, mse_)
    logs.append(perf_log + '\n')
    print(perf_log)
    logs.append('Author\tVP\tVT\tFP\tFT\tSP\tST\tBP\tBT\n')
    for au, vp, vt, fp, ft, sp, st, bp, bt in packed_result:
        logs.append('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(au, vp, vt,
                                                                  fp, ft, sp, st, bp, bt))
    if save_file:
        with open(save_file, 'w') as f:
            for log in logs:
                f.write(log)
