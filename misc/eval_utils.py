from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def score_consist_f1(score_list, predict_list, print_switch=False):
    '''
    9 f1-scores times corresponding consistency rate
    '''
    consist_score = 0
    max_consist_score = 0
    for score_idx in range(1, len(score_list)):
        consistent_rate = 1.-(np.sum(np.logical_xor(predict_list[score_idx-1], predict_list[score_idx]))\
                          /predict_list[score_idx].shape[0])
        consist_score += score_list[score_idx-1]*consistent_rate*np.sqrt(score_idx)
        max_consist_score += np.sqrt(score_idx)

    if print_switch:
        print('Consistent Score : {}'.format(consist_score/max_consist_score))
    return consist_score/max_consist_score

def score_early_f1(score_list, print_switch=False):
    '''
    10 f1-scores divided by root of score idx
    '''
    early_score = 0.
    max_early_score = 0.
    for score_idx, score in enumerate(score_list):
        early_score += score/np.sqrt(score_idx+1)
        max_early_score += 1./np.sqrt(score_idx+1)

    if print_switch:
        print('Early Score : {}'.format(early_score / max_early_score))
    return early_score / max_early_score

def metric_evaluation(labels, predict_labels):
    # 1. Processing Prediction Result
    pred_one_value_labels = np.array([float(i) for i in (predict_labels > 0.5)])
    ground_one_value_labels = np.array([float(i) for i in (labels > 0.5)])

    # 2. Calculate Metrics
    metric_res = {}
    metric_res['Accuracy'] = accuracy_score(ground_one_value_labels, pred_one_value_labels)
    metric_res['Precision'] = precision_score(ground_one_value_labels, pred_one_value_labels, zero_division=0)
    metric_res['Recall'] = recall_score(ground_one_value_labels, pred_one_value_labels)
    metric_res['F1'] = f1_score(ground_one_value_labels, pred_one_value_labels)
    metric_res['AUC'] = roc_auc_score(ground_one_value_labels, pred_one_value_labels)
    return metric_res


def eval_split(dp_lw_model, loader, split = 'val'):
    bsz_for_test = {'Hack':31, 'Ransomware':323, 'Darknet':584}

    dp_lw_model.eval()
    loader.reset_iterator(split)
    loader.used_ratio_list = loader.test_ratio_list
    ED_split_number = loader.split_number
    loader.batch_size = bsz_for_test[loader.illicit_name]

    n = 0
    loss_sum = 0
    val_metrics_score = {}
    true_labels = torch.zeros([0, 1]).cuda()
    predictions = torch.zeros([0, ED_split_number]).cuda()
    while True:
        data = loader.get_batch_all_tag(split)
        AF_feat = data['AF_feat']
        fr_path_feat = data['fr_path_feat']
        bk_path_feat = data['bk_path_feat']
        fr_intersect_edge_batch = data['fr_intersect_edge_batch']
        bk_intersect_edge_batch = data['bk_intersect_edge_batch']
        fr_intersect_feat_batch = data['fr_intersect_feat_batch']
        bk_intersect_feat_batch = data['bk_intersect_feat_batch']

        labels = torch.reshape(torch.tensor(data['labels'], dtype=torch.float32),[-1, 1]).cuda()
        loss_amplify = torch.reshape(torch.tensor(data['loss_amplify'], dtype=torch.float32),[-1, 1]).cuda()
        bsz_used_downside = len(AF_feat[0]); n = n + bsz_used_downside
        dp_lw_model.model_core.split='test'

        with torch.no_grad():
            out, each_split_prediction, ground_labels = \
                dp_lw_model([AF_feat, fr_path_feat, bk_path_feat,
                             fr_intersect_edge_batch, bk_intersect_edge_batch,
                             fr_intersect_feat_batch, bk_intersect_feat_batch],
                             labels, loss_amplify, return_pred_switch=True)

        loss = out['total_loss']*bsz_used_downside
        loss_sum = loss_sum + loss

        true_labels = torch.cat((true_labels, ground_labels), dim=0)
        predictions = torch.cat((predictions, each_split_prediction), dim=0)
        if n>=0.1*loader.total_address_num: print('{} {} samples'.format(n, split)); break

    # Calculate metrics for each split
    f1_score_history = []
    print('Evaluation loss is {}'.format(loss_sum / n))
    for split_index in range(ED_split_number):
        metric_res = metric_evaluation(true_labels, predictions[:, split_index])
        val_metrics_score['Split_{}_Result'.format(split_index)] = metric_res
        f1_score_history.append(metric_res['F1'])

    # Calculate Consistency Score
    predictions = np.reshape(predictions.detach().cpu().numpy(), [ED_split_number, -1])
    consist_f1 = score_consist_f1(f1_score_history, predictions)
    early_f1 = score_early_f1(f1_score_history)
    val_metrics_score['Consistent_F1'] = consist_f1
    val_metrics_score['Early_weighted_F1'] = early_f1

    # Print can caluculate Early weighted F1
    print('---------  Dataset Split {} ---------  '.format(split))
    avg_acc = 0; avg_prec = 0; avg_rec = 0
    for split_index in range(ED_split_number):
        each_split_metric_res = val_metrics_score['Split_{}_Result'.format(split_index)]
        print(f'-- Hour {split_index+1}', end=' : ')
        for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']: # , 'Confidence'
            metric_score = each_split_metric_res[metric_name]
            print('{}: {} ;'.format(metric_name, round(metric_score, 4)), end=' ')
            val_metrics_score['Split_{}_Result'.format(split_index)][metric_name] = round(metric_score, 4)
            avg_acc += metric_score if metric_name == 'Accuracy' else 0
            avg_prec += metric_score if metric_name == 'Precision' else 0
            avg_rec += metric_score if metric_name == 'Recall' else 0
        print('')
    print('Avg Accuracy is {}'.format(round(avg_acc / ED_split_number, 4)))
    print('Avg Precision is {}'.format(round(avg_prec / ED_split_number, 4)))
    print('Avg Recall is {}'.format(round(avg_rec / ED_split_number, 4)))
    print('Early Weighed F1 is {} --------'.format(val_metrics_score['Early_weighted_F1']))
    print('Consistency Score is {} --------'.format(val_metrics_score['Consistent_F1']))

    loader.batch_size = loader.opt.batch_size
    dp_lw_model.train()
    dp_lw_model.model_core.split = 'train'
    loader.used_ratio_list = loader.train_ratio_list
    return loss_sum/n, val_metrics_score
