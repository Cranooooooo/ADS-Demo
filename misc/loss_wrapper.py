import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
color_list = ['#8B1A1A', '#FF6A6A', '#EEE9E9', '#66CD00']

import warnings
warnings.filterwarnings('always')

def plotting(hz_rate):
    # plt.xticks(rotation=0, )
    # plt.figure(figsize=(12,4))
    label = [str(i + 1) for i in range(23)]

    rect_hz_rate = np.zeros([23, 5])
    for i in range(23): rect_hz_rate[i, :] = np.sum(hz_rate[:i + 1, :], axis=0)

    AF = rect_hz_rate[:, 0]
    fr_path = rect_hz_rate[:, 1]
    bk_path = rect_hz_rate[:, 2]
    fr_graph = rect_hz_rate[:, 3]
    bk_graph = rect_hz_rate[:, 4]

    plt.plot(label, AF, '-.', color='#4DA24E', alpha=.8, linewidth=2, label="AF B.R", markersize=8, marker='*') #
    plt.plot(label, fr_path, '-.', color='#FFB227', alpha=.8, linewidth=2, label="Fr Path B.R", markersize=8, marker='o') #
    plt.plot(label, bk_path, '-.', color='#FFB227', alpha=.8, linewidth=2, label="Bk Path B.R", markersize=8, marker='^') #
    plt.plot(label, fr_graph, '-.', color='#EE4545', alpha=.8, linewidth=2, label="Fr Graph B.R", markersize=8, marker='o') #
    plt.plot(label, bk_graph, '-.', color='#EE4545', alpha=.8, linewidth=2, label="Bk Graph B.R", markersize=8, marker='^') #

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.legend(frameon=False)
    plt.yticks(size=14)
    plt.xticks(size=14)
    plt.show()
    plt.close()

def score_consist_f1(score_list, predict_list, print_switch=False):
    '''
    9 f1-scores times corresponding consistency rate
    '''
    consist_score = 0
    max_consist_score = 0
    for score_idx in range(1, len(score_list)):
        pre_predict = np.array([float(i) for i in (predict_list[score_idx-1] > 0.5)])
        this_predict = np.array([float(i) for i in (predict_list[score_idx] > 0.5)])
        consistent_rate = 1.-(np.sum(np.logical_xor(pre_predict, this_predict))\
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

def score_reliable_f1(score_list, predict_list):
    '''
    1. Find the time when the prediction after that point will not change (reliable point --> r_p)
    2. reward early r_p, punish bad performance

    return a list rather than a score
    each element: reliable_rate * f1_score
    reliable_rate: how many addresses' predictions won't change after that point
    '''
    reliable_f1_list = []
    predict_list = np.array(predict_list)
    sample_num = predict_list.shape[1]
    time_split_num = len(score_list)

    for score_idx in range(time_split_num):
        reliable_recd = np.zeros([sample_num])

        for left_idx in range(score_idx+1, time_split_num):
            reliable_recd += (1.-np.sign(predict_list[score_idx, :]*predict_list[left_idx, :]))

        reliable_recd = np.ceil(reliable_recd/(reliable_recd+1.))
        reliable_rate = (sample_num - np.sum(reliable_recd))/sample_num
        reliable_f1_list.append(reliable_rate*score_list[score_idx])

    return reliable_f1_list



class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.crit = nn.L1Loss()
        self.hidden_dim = opt.hidden_dim
        self.drop_prob_lm = opt.drop_prob_lm
        self.model_core = model

    def metric_evaluation(self, labels, predict_labels, eval_switch=True):
        # 1. Processing Prediction Result
        pred_one_value_labels = np.array([float(i) for i in (predict_labels > 0.5)])
        ground_one_value_labels = np.array([float(i) for i in (labels > 0.5)])

        # 2. Calculate Metrics
        metric_res = {}
        if eval_switch:
            metric_res['Accuracy'] = accuracy_score(ground_one_value_labels, pred_one_value_labels)
            metric_res['Precision'] = precision_score(ground_one_value_labels, pred_one_value_labels, zero_division=0)
            metric_res['Recall'] = recall_score(ground_one_value_labels, pred_one_value_labels)
            metric_res['F1'] = f1_score(ground_one_value_labels, pred_one_value_labels)
            metric_res['AUC'] = roc_auc_score(ground_one_value_labels, pred_one_value_labels)
        else:
            metric_res['Accuracy'] = 0.
            metric_res['Precision'] = 0.
            metric_res['Recall'] = 0.
            metric_res['F1'] = 0.
            metric_res['AUC'] = 0.

        return metric_res

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, self.hidden_dim), weight.new_zeros(bsz, self.hidden_dim))


    def forward(self, feats_list, labels, loss_amplify, return_pred_switch=False, epoch=1, iter=0, eval_switch=True):
        # 1. Set Loss
        out = {}
        if return_pred_switch: self.split = 'test'
        else: self.split = 'train'
        ED_split_number = len(feats_list[0])
        bsz = len(feats_list[0][0])

        total_loss = 0.
        predict_inver_loss = 0.
        predict_inver_amplifier = 0.1*max(epoch-4, 0)
        former_hazard_list = torch.zeros([ED_split_number, bsz, 5]).cuda()
        survival_predict_history = torch.zeros([bsz, 0]).cuda()
        f1_score_history = []

        AF_hidden = self.init_hidden(bsz)
        fr_Path_hidden = self.init_hidden(bsz)
        bk_Path_hidden = self.init_hidden(bsz)
        fr_Path_GCN_hidden = self.init_hidden(bsz)
        bk_Path_GCN_hidden = self.init_hidden(bsz)

        # 2. Loop for every life split
        prev_pred = None
        skip_list = []
        for hour_index in range(ED_split_number):
            if hour_index/5. > np.random.rand() and hour_index>6 and self.split=='train': pass #
            else:
                # For different Models use different features
                this_split_AF_feat = torch.tensor(np.array(feats_list[0][hour_index], dtype=np.float32)).cuda()
                this_split_fr_path_feat = torch.tensor(np.array(feats_list[1][hour_index], dtype=np.float32)).squeeze(dim=1).cuda()
                this_split_bk_path_feat = torch.tensor(np.array(feats_list[2][hour_index], dtype=np.float32)).squeeze(dim=1).cuda()
                this_split_fr_itsc_edge = feats_list[3][hour_index]
                this_split_bk_itsc_edge = feats_list[4][hour_index]
                this_split_fr_itsc_feat = feats_list[5][hour_index]
                this_split_bk_itsc_feat = feats_list[6][hour_index]

                # input into model
                former_hazard_list, predict_labels, AF_hidden, fr_Path_hidden, bk_Path_hidden, fr_Path_GCN_hidden, bk_Path_GCN_hidden, skip_list = \
                    self.model_core(this_split_AF_feat, AF_hidden,
                                    this_split_fr_path_feat, this_split_bk_path_feat, fr_Path_hidden, bk_Path_hidden,
                                    this_split_fr_itsc_edge, this_split_bk_itsc_edge, fr_Path_GCN_hidden, bk_Path_GCN_hidden,
                                    this_split_fr_itsc_feat, this_split_bk_itsc_feat,
                                    split_index=hour_index, former_hazard_list=former_hazard_list, skip_list=skip_list, prev_pred=prev_pred)
                prev_pred = predict_labels

            # Processing Metrics
            predict_labels = torch.reshape(predict_labels, [-1])
            hour_idx_weight = np.sqrt(hour_index)
            if hour_index != 0:
                pre_predict = survival_predict_history[:, -1]
                predict_inver_loss += torch.mean(torch.abs(torch.sign(pre_predict * predict_labels) - 1) / 2.) * hour_idx_weight

            this_loss = torch.mean((loss_amplify * (torch.reshape(predict_labels, [-1, 1])-labels))**2)
            total_loss += this_loss/(hour_index+1)
            out['ED_split_number_{}_loss'.format(hour_index)] = this_loss
            metric_res = self.metric_evaluation(labels, predict_labels, eval_switch=eval_switch)
            out['ED_split_number_{}_metrics'.format(hour_index)] = metric_res
            f1_score_history.append(metric_res['F1'])
            survival_predict_history = torch.cat((survival_predict_history, predict_labels.unsqueeze(-1)), dim=-1)


        # Hz diff sign loss
        hazard_change_loss = 0
        for split_idx in range(ED_split_number-1):
            hazard_change_loss += split_idx * torch.sum((1. - torch.sign(torch.sum(former_hazard_list[split_idx, :]) *\
                                                                         torch.sum(former_hazard_list[split_idx+1, :])))) / bsz

        # Processing hazard change loss
        out['Prediction_Loss'] = total_loss
        out['Hazard_Diff_Loss'] = predict_inver_amplifier * hazard_change_loss
        out['Predic_Diff_Loss'] = predict_inver_amplifier * predict_inver_loss/bsz
        out['total_loss'] = total_loss + out['Hazard_Diff_Loss'] + out['Predic_Diff_Loss']

        # Processing Score
        if eval_switch:
            all_predict = np.reshape(survival_predict_history.detach().cpu().numpy(), [ED_split_number, -1])
            consist_f1 = score_consist_f1(f1_score_history, all_predict)
            early_f1 = score_early_f1(f1_score_history)
            reliable_f1 = score_reliable_f1(f1_score_history, all_predict)
            out['Consistent_F1'] = consist_f1
            out['Early_weighted_F1'] = early_f1
            out['Reliable_F1'] = reliable_f1

        else:
            out['Consistent_F1'] = 0.
            out['Early_weighted_F1'] = 0.
            out['Reliable_F1'] = 0.

        # Visualization Processing
        plotting(former_hazard_list.view(23, 5).detach().cpu().numpy())

        if return_pred_switch: return out, survival_predict_history, labels
        else: return out



