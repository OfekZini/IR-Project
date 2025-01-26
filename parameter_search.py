import numpy as np
import pandas as pd
import json
import csv
import os
from time import time, sleep
from sklearn.model_selection import ParameterGrid
from backend import BackendClass


param_ranges = {
    'text_k1': np.arange(1.2, 1.21, 0.25),
    'text_b': np.arange(0.5, 0.51, 0.25),
    'text_w': np.arange(0.6, 0.85, 0.25),
    'title_w': np.arange(0.1, 0.35, 0.25),
    'anchor_w': np.arange(0, 0.75, 0.25),
    'pr_w': np.arange(0.2, 0.6, 0.2),
    'pv_w': np.arange(0.2, 0.6, 0.2)
}

with open('queries_train.json', 'rt') as f:
    queries = json.load(f)


def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


def precision_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(predicted_list) == 0:
        return 0.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(predicted_list), 3)


def recall_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(true_set) < 1:
        return 1.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(true_set), 3)


def f1_at_k(true_list, predicted_list, k):
    p = precision_at_k(true_list, predicted_list, k)
    r = recall_at_k(true_list, predicted_list, k)
    if p == 0.0 or r == 0.0:
        return 0.0
    return round(2.0 / (1.0 / p + 1.0 / r), 3)


def results_quality(true_list, predicted_list):
    p5 = precision_at_k(true_list, predicted_list, 10)
    f1_30 = f1_at_k(true_list, predicted_list, 30)
    if p5 == 0.0 or f1_30 == 0.0:
        return 0.0
    return round(2.0 / (1.0 / p5 + 1.0 / f1_30), 3)


param_grid = ParameterGrid(param_ranges)

backend = BackendClass()
# Define parameter ranges (you already have these)


results_file = 'detailed_results.csv'

# Check if file exists to determine header and write mode
file_exists = os.path.isfile(results_file)

with open(results_file, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    if not file_exists:
        header = list(param_grid[0].keys()) + ['query', 'duration', 'quality']
        csvwriter.writerow(header)
# results = {}
    for params in param_grid:

        text_k1 = params['text_k1']
        text_b = params['text_b']
        text_w = params['text_w']
        title_w = params['title_w']
        anchor_w = params['anchor_w']
        pr_w = params['pr_w']
        pv_w = params['pv_w']

        # qs_res = []
        sleep(120)
        for q, true_wids in queries.items():
            duration, ap = None, None
            sleep(1)
            t_start = time()
            res = backend.search_prm(q,
                                      text_w,
                                      title_w,
                                      anchor_w,
                                      pr_w,
                                      pv_w,
                                     text_k1,
                                     text_b
                                      )
            duration = time() - t_start
            pred_wids, _ = zip(*res)
            rq = results_quality(true_wids, pred_wids)

            # qs_res.append((q, duration, rq))
            # Write row for each query
            csvwriter.writerow([
                text_k1, text_b, text_w, title_w, anchor_w, pr_w, pv_w,
                q, duration, rq
            ])

#     results[str(params)] = (
#         sum(result for _, _, result in qs_res) / len(qs_res),
#         sum(dur for _, dur, _ in qs_res) / len(qs_res),
#         max(dur for _, dur, _ in qs_res)
#     )
#
# print(results)

