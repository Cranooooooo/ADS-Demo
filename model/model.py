####################
#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
sota_0: Our model
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import time
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
from .misc_model.Evo_Path_GCN import Evo_Path_GNN


class E_Path_Tracer(nn.Module):
    def __init__(self, opt):
        super(E_Path_Tracer, self).__init__()
        # Module Switch
        self.opt = opt
        self.split = 'train'
        self.thres_weight = {'Hack':0.8, 'Ransomware':0.3, 'Darknet':0.6}

        # Input Feature Dimension
        self.bsz = opt.batch_size
        self.analysis_span = opt.split_number
        self.feat_AF_dim = opt.AF_dim
        self.hidden_dim = opt.hidden_dim
        self.drop_prob_lm = opt.drop_prob_lm

        # Model Pipeline -- AF
        self.feat_AF_BN = nn.BatchNorm1d(self.feat_AF_dim)
        self.feat_AF_fc = nn.Sequential(nn.Linear(self.feat_AF_dim, self.hidden_dim), nn.Tanh(),
                                        nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh())
        self.AF_global_lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.feat_AF_fc.apply(self.init_weights)
        self.AF_global_lstm.apply(self.init_weights)

        # Model Pipeline -- Path and Graph
        self.path_related_module_build()
        self.graph_related_module_build()

        # Model Pipeline -- Final Prediction
        self.predictor_module_build()


    def predictor_module_build(self):
        self.prob_predictor = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Sigmoid(),
                                            nn.Linear(self.hidden_dim, 1), nn.Sigmoid())

        self.AF_rate = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
                                     nn.Linear(self.hidden_dim, 1), nn.Tanh())
        self.fr_path_rate = nn.Sequential(nn.Linear(2*self.hidden_dim, self.hidden_dim), nn.Tanh(),
                                          nn.Linear(self.hidden_dim, 1), nn.Tanh())
        self.bk_path_rate = nn.Sequential(nn.Linear(2*self.hidden_dim, self.hidden_dim), nn.Tanh(),
                                          nn.Linear(self.hidden_dim, 1), nn.Tanh())
        self.fr_graph_rate = nn.Sequential(nn.Linear(3*self.hidden_dim, self.hidden_dim), nn.Tanh(),
                                           nn.Linear(self.hidden_dim, 1), nn.Tanh())
        self.bk_graph_rate = nn.Sequential(nn.Linear(3*self.hidden_dim, self.hidden_dim), nn.Tanh(),
                                           nn.Linear(self.hidden_dim, 1), nn.Tanh())

        self.rate_weight = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
                                         nn.Linear(self.hidden_dim, 5), nn.Tanh()) # nn.Softmax(dim=-1))#

    def path_related_module_build(self):
        # Necessary Module
        self.head_num = 4
        self.path_num = 10
        self.path_zoom_len = self.opt.path_zoom_len
        self.feat_Path_dim = self.opt.path_feat_dim

        self.feat_fr_Path_BN = nn.BatchNorm2d(self.feat_Path_dim)  # Input: (N->bsz, C->channel, H->Height, W->Width)
        self.feat_bk_Path_BN = nn.BatchNorm2d(self.feat_Path_dim)  # Input: (N->bsz, C->channel, H->Height, W->Width)

        self.feat_fr_Path_fc = nn.Sequential(nn.Linear(self.feat_Path_dim, self.hidden_dim), nn.Tanh(),
                                             nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh())
        self.feat_bk_Path_fc = nn.Sequential(nn.Linear(self.feat_Path_dim, self.hidden_dim), nn.Tanh(),
                                             nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh())

        for head_idx in range(self.head_num):
            exec(f'self.feat_fr_Path_pre_att_{head_idx} = nn.Linear(self.hidden_dim, int(self.hidden_dim/self.head_num))')
            exec(f'self.feat_bk_Path_pre_att_{head_idx} = nn.Linear(self.hidden_dim, int(self.hidden_dim/self.head_num))')
            exec(f'self.fr_Path_att_{head_idx} = nn.Sequential(nn.Linear(int(2*self.hidden_dim), 1), nn.Softmax(dim=1))')
            exec(f'self.bk_Path_att_{head_idx} = nn.Sequential(nn.Linear(int(2*self.hidden_dim), 1), nn.Softmax(dim=1))')
        self.fr_path_global_lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.bk_path_global_lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        # init
        self.feat_fr_Path_fc.apply(self.init_weights)
        self.feat_bk_Path_fc.apply(self.init_weights)
        self.fr_path_global_lstm .apply(self.init_weights)
        self.bk_path_global_lstm.apply(self.init_weights)

        for head_idx in range(self.head_num):
            exec(f'self.feat_fr_Path_pre_att_{head_idx}.apply(self.init_weights)')
            exec(f'self.feat_bk_Path_pre_att_{head_idx}.apply(self.init_weights)')
            exec(f'self.fr_Path_att_{head_idx}.apply(self.init_weights)')
            exec(f'self.bk_Path_att_{head_idx}.apply(self.init_weights)')


        # Evolve Path-Encoding weight
        self.AF_to_W_ii = nn.Sequential(nn.Linear(2 * self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.AF_to_W_hi = nn.Sequential(nn.Linear(2 * self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.AF_to_W_if = nn.Sequential(nn.Linear(2 * self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.AF_to_W_hf = nn.Sequential(nn.Linear(2 * self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.AF_to_W_ig = nn.Sequential(nn.Linear(2 * self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.AF_to_W_hg = nn.Sequential(nn.Linear(2 * self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.AF_to_W_io = nn.Sequential(nn.Linear(2 * self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.AF_to_W_ho = nn.Sequential(nn.Linear(2 * self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())

        self.AF_to_b_i = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Tanh())
        self.AF_to_b_f = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Tanh())
        self.AF_to_b_g = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Tanh())
        self.AF_to_b_o = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Tanh())

        self.AF_to_W_ii.apply(self.init_weights)
        self.AF_to_W_hi.apply(self.init_weights)
        self.AF_to_W_if.apply(self.init_weights)
        self.AF_to_W_hf.apply(self.init_weights)
        self.AF_to_W_ig.apply(self.init_weights)
        self.AF_to_W_hg.apply(self.init_weights)
        self.AF_to_W_io.apply(self.init_weights)
        self.AF_to_W_ho.apply(self.init_weights)

        self.AF_to_b_i.apply(self.init_weights)
        self.AF_to_b_f.apply(self.init_weights)
        self.AF_to_b_g.apply(self.init_weights)
        self.AF_to_b_o.apply(self.init_weights)



    def graph_related_module_build(self):
        # Necessary Module
        self.feat_Intersec_dim = self.opt.intersect_dim
        self.fr_intsc_feat_BN = nn.BatchNorm1d(self.feat_Intersec_dim)
        self.bk_intsc_feat_BN = nn.BatchNorm1d(self.feat_Intersec_dim)

        self.fr_intsc_feat_fc = nn.Sequential(nn.Linear(2 * self.hidden_dim, int(self.hidden_dim * self.feat_Intersec_dim)), nn.Tanh())
        self.bk_intsc_feat_fc = nn.Sequential(nn.Linear(2 * self.hidden_dim, int(self.hidden_dim * self.feat_Intersec_dim)), nn.Tanh())

        for head_idx in range(self.head_num):
            exec(f'self.feat_fr_Path_intsc_pre_att_{head_idx} = nn.Linear(self.hidden_dim, int(self.hidden_dim/self.head_num))')
            exec(f'self.feat_bk_Path_intsc_pre_att_{head_idx} = nn.Linear(self.hidden_dim, int(self.hidden_dim/self.head_num))')

            exec(f'self.fr_Path_intsc_att_{head_idx} = nn.Sequential(nn.Linear(int(2*self.hidden_dim), 1), nn.Softmax(dim=1))')
            exec(f'self.bk_Path_intsc_att_{head_idx} = nn.Sequential(nn.Linear(int(2*self.hidden_dim), 1), nn.Softmax(dim=1))')
        self.fr_graph_global_lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.bk_graph_global_lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.fr_intsc_feat_fc.apply(self.init_weights)
        self.bk_intsc_feat_fc.apply(self.init_weights)
        self.fr_graph_global_lstm.apply(self.init_weights)
        self.bk_graph_global_lstm.apply(self.init_weights)

        for head_idx in range(self.head_num):
            exec(f'self.feat_fr_Path_intsc_pre_att_{head_idx}.apply(self.init_weights)')
            exec(f'self.feat_bk_Path_intsc_pre_att_{head_idx}.apply(self.init_weights)')
            exec(f'self.fr_Path_intsc_att_{head_idx}.apply(self.init_weights)')
            exec(f'self.bk_Path_intsc_att_{head_idx}.apply(self.init_weights)')


        # Evolve Graph-Encoding weight
        self.AF_to_fr_messageNN = nn.Sequential(nn.Linear(2*self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.AF_to_fr_updateNN = nn.Sequential(nn.Linear(2*self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.AF_to_bk_messageNN = nn.Sequential(nn.Linear(2*self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.AF_to_bk_updateNN = nn.Sequential(nn.Linear(2*self.hidden_dim, int(self.hidden_dim * self.hidden_dim)), nn.Tanh())
        self.fr_Path_GCN = Evo_Path_GNN(1, self.hidden_dim, aggr='mean')
        self.bk_Path_GCN = Evo_Path_GNN(1, self.hidden_dim, aggr='mean')

        self.AF_to_fr_messageNN.apply(self.init_weights)
        self.AF_to_fr_updateNN.apply(self.init_weights)
        self.AF_to_bk_messageNN.apply(self.init_weights)
        self.AF_to_bk_updateNN.apply(self.init_weights)

    def init_hidden(self):
        weight = next(self.parameters())
        return (weight.new_zeros(self.bsz, self.hidden_dim),
                weight.new_zeros(self.bsz, self.hidden_dim))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.01)

    def path_process(self, bsz, h_global_AF, fr_path_feat, bk_path_feat, fr_Path_hidden, bk_Path_hidden):
        # 0. Pre-Processing Path Feature
        fr_path_feat = self.feat_fr_Path_BN(
            fr_path_feat.reshape([bsz, self.feat_Path_dim, self.path_num, self.path_zoom_len]))
        bk_path_feat = self.feat_bk_Path_BN(
            bk_path_feat.reshape([bsz, self.feat_Path_dim, self.path_num, self.path_zoom_len]))

        fr_path_feat = fr_path_feat.reshape([bsz, self.path_num, self.path_zoom_len, self.feat_Path_dim])
        bk_path_feat = bk_path_feat.reshape([bsz, self.path_num, self.path_zoom_len, self.feat_Path_dim])

        fr_path_feat = self.feat_fr_Path_fc(fr_path_feat)
        bk_path_feat = self.feat_bk_Path_fc(bk_path_feat)

        # 1. Encoding
        # 1.1 Process weights
        fr_w_kernel = torch.relu(torch.sign(torch.cat((h_global_AF, fr_Path_hidden[0]), dim=-1)))
        bk_w_kernel = torch.relu(torch.sign(torch.cat((h_global_AF, bk_Path_hidden[0]), dim=-1)))

        fr_W_ii = torch.reshape(self.AF_to_W_ii(fr_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        fr_W_hi = torch.reshape(self.AF_to_W_hi(fr_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        fr_W_if = torch.reshape(self.AF_to_W_if(fr_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        fr_W_hf = torch.reshape(self.AF_to_W_hf(fr_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        fr_W_ig = torch.reshape(self.AF_to_W_ig(fr_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        fr_W_hg = torch.reshape(self.AF_to_W_hg(fr_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        fr_W_io = torch.reshape(self.AF_to_W_io(fr_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        fr_W_ho = torch.reshape(self.AF_to_W_ho(fr_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))

        bk_W_ii = torch.reshape(self.AF_to_W_ii(bk_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        bk_W_hi = torch.reshape(self.AF_to_W_hi(bk_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        bk_W_if = torch.reshape(self.AF_to_W_if(bk_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        bk_W_hf = torch.reshape(self.AF_to_W_hf(bk_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        bk_W_ig = torch.reshape(self.AF_to_W_ig(bk_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        bk_W_hg = torch.reshape(self.AF_to_W_hg(bk_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        bk_W_io = torch.reshape(self.AF_to_W_io(bk_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))
        bk_W_ho = torch.reshape(self.AF_to_W_ho(bk_w_kernel), (bsz, self.hidden_dim, self.hidden_dim))

        fr_b_i = self.AF_to_b_i(fr_w_kernel)
        fr_b_f = self.AF_to_b_f(fr_w_kernel)
        fr_b_g = self.AF_to_b_g(fr_w_kernel)
        fr_b_o = self.AF_to_b_o(fr_w_kernel)

        bk_b_i = self.AF_to_b_i(bk_w_kernel)
        bk_b_f = self.AF_to_b_f(bk_w_kernel)
        bk_b_g = self.AF_to_b_g(bk_w_kernel)
        bk_b_o = self.AF_to_b_o(bk_w_kernel)

        # 1.2 Encoding
        sigmoid = nn.Sigmoid();
        tanh = nn.Tanh()
        fr_h_t = torch.zeros([bsz, self.path_num, self.hidden_dim]).cuda()
        fr_c_t = torch.zeros([bsz, self.path_num, self.hidden_dim]).cuda()
        bk_h_t = torch.zeros([bsz, self.path_num, self.hidden_dim]).cuda()
        bk_c_t = torch.zeros([bsz, self.path_num, self.hidden_dim]).cuda()

        for path_node_idx in range(self.path_zoom_len):
            # i - Calculation
            fr_i_t = sigmoid(torch.matmul(fr_path_feat[:, :, path_node_idx, :], fr_W_ii) + torch.matmul(fr_h_t, fr_W_hi) + fr_b_i.unsqueeze(1))
            bk_i_t = sigmoid(torch.matmul(bk_path_feat[:, :, path_node_idx, :], bk_W_ii) + torch.matmul(bk_h_t, bk_W_hi) + bk_b_i.unsqueeze(1))

            # f - Calculation
            fr_f_t = sigmoid(torch.matmul(fr_path_feat[:, :, path_node_idx, :], fr_W_if) + torch.matmul(fr_h_t, fr_W_hf) + fr_b_f.unsqueeze(1))
            bk_f_t = sigmoid(torch.matmul(bk_path_feat[:, :, path_node_idx, :], bk_W_if) + torch.matmul(bk_h_t, bk_W_hf) + bk_b_f.unsqueeze(1))

            # g - Calculation
            fr_g_t = sigmoid(torch.matmul(fr_path_feat[:, :, path_node_idx, :], fr_W_ig) + torch.matmul(fr_h_t, fr_W_hg) + fr_b_g.unsqueeze(1))
            bk_g_t = sigmoid(torch.matmul(bk_path_feat[:, :, path_node_idx, :], bk_W_ig) + torch.matmul(bk_h_t, bk_W_hg) + bk_b_g.unsqueeze(1))

            # o - Calculation
            fr_o_t = sigmoid(torch.matmul(fr_path_feat[:, :, path_node_idx, :], fr_W_io) + torch.matmul(fr_h_t, fr_W_ho) + fr_b_o.unsqueeze(1))
            bk_o_t = sigmoid(torch.matmul(bk_path_feat[:, :, path_node_idx, :], bk_W_io) + torch.matmul(bk_h_t, bk_W_ho) + bk_b_o.unsqueeze(1))

            # update c
            fr_c_t = fr_f_t * fr_c_t + fr_i_t * fr_g_t
            bk_c_t = bk_f_t * bk_c_t + bk_i_t * bk_g_t

            # update h
            fr_h_t = fr_o_t * tanh(fr_c_t)
            bk_h_t = bk_o_t * tanh(bk_c_t)

    

        # 2. Multi-Head Attention
        fr_att_feat_list = []
        bk_att_feat_list = []
        fr_att_cal_kernel = torch.cat((fr_h_t, h_global_AF.unsqueeze(dim=1).repeat([1, self.path_num, 1])), dim=-1)
        bk_att_cal_kernel = torch.cat((bk_h_t, h_global_AF.unsqueeze(dim=1).repeat([1, self.path_num, 1])), dim=-1)

        for head_idx in range(self.head_num):
            exec(f'fr_Path_att = self.fr_Path_att_{head_idx}(fr_att_cal_kernel)')
            exec(f'bk_Path_att = self.bk_Path_att_{head_idx}(bk_att_cal_kernel)')

            exec(f'fr_att_feat_list.append(torch.sum(fr_Path_att * self.feat_fr_Path_pre_att_{head_idx}(fr_h_t), dim=1))')
            exec(f'bk_att_feat_list.append(torch.sum(bk_Path_att * self.feat_bk_Path_pre_att_{head_idx}(bk_h_t), dim=1))')

        fr_path_final = torch.cat(fr_att_feat_list, dim=-1)
        bk_path_final = torch.cat(bk_att_feat_list, dim=-1)

        h_global_fr_path, c_global_fr_path = self.fr_path_global_lstm(fr_path_final, (fr_Path_hidden[0], fr_Path_hidden[1]))
        h_global_bk_path, c_global_bk_path = self.bk_path_global_lstm(bk_path_final, (bk_Path_hidden[0], bk_Path_hidden[1]))

        fr_Path_hidden = (h_global_fr_path, c_global_fr_path)
        bk_Path_hidden = (h_global_bk_path, c_global_bk_path)

        return fr_h_t, bk_h_t, h_global_fr_path, h_global_bk_path, fr_Path_hidden, bk_Path_hidden

    def graph_process(self, bsz, h_global_AF, fr_h_t, bk_h_t, fr_itsc_feat, bk_itsc_feat, fr_itsc_edge, bk_itsc_edge,
                      fr_Graph_hidden, bk_Graph_hidden, former_hazard_list, split_index, skip_list=[], prev_predict=None):

        fr_h_t_insect = torch.clone(fr_h_t)
        bk_h_t_insect = torch.clone(bk_h_t)
        skip_count= 0
        skip_thres_1 = self.opt.skip_thres_1*self.thres_weight[self.opt.illicit_name]
        skip_thres_2 = self.opt.skip_thres_2


        # node-feat, edge-feat, edge-link
        for addr_idx in range(bsz):
            # Accelerate if using rate prediction
            skip_flag = False
            if split_index >= 0:# and self.split=='train':
                second_hz_rate = torch.sum(former_hazard_list[:split_index, addr_idx, 3])
                third_hz_rate = torch.sum(former_hazard_list[:split_index, addr_idx, 4])
                if second_hz_rate > skip_thres_1 and third_hz_rate > skip_thres_1: skip_flag=True
                if skip_flag: skip_count += 1; skip_list.append(addr_idx); continue


            fr_w_kernel = torch.relu(torch.sign(torch.cat((h_global_AF, fr_Graph_hidden[0]), dim=-1)))
            bk_w_kernel = torch.relu(torch.sign(torch.cat((h_global_AF, bk_Graph_hidden[0]), dim=-1)))

            # 3.1 Processing Path-GCN --> For loop in batch size
            fr_intsc_feat_fc = torch.reshape(self.fr_intsc_feat_fc(fr_w_kernel),
                                             (-1, self.feat_Intersec_dim, self.hidden_dim))
            fr_messageNN = torch.reshape(self.AF_to_fr_messageNN(fr_w_kernel),
                                         (-1, self.hidden_dim, self.hidden_dim))
            fr_updateNN = torch.reshape(self.AF_to_fr_updateNN(fr_w_kernel),
                                        (-1, self.hidden_dim, self.hidden_dim))

            bk_intsc_feat_fc = torch.reshape(self.bk_intsc_feat_fc(bk_w_kernel),
                                             (-1, self.feat_Intersec_dim, self.hidden_dim))
            bk_messageNN = torch.reshape(self.AF_to_bk_messageNN(bk_w_kernel),
                                         (-1, self.hidden_dim, self.hidden_dim))
            bk_updateNN = torch.reshape(self.AF_to_bk_updateNN(bk_w_kernel),
                                        (-1, self.hidden_dim, self.hidden_dim))

            # Forward Path-GCN
            this_fr_itsc_feat = torch.tensor(np.array(list(fr_itsc_feat[addr_idx]), dtype=np.float32)).cuda()
            this_fr_itsc_edge = torch.tensor(np.array(list(fr_itsc_edge[addr_idx]), dtype=int)).cuda()

            if this_fr_itsc_feat.shape[0]==0: continue
            else:
                intsced_fr_path = self.fr_Path_GCN(fr_h_t[addr_idx, :, :], this_fr_itsc_feat, this_fr_itsc_edge,
                                                   fr_intsc_feat_fc[addr_idx, :, :], fr_messageNN[addr_idx, :, :], fr_updateNN[addr_idx, :, :])
            fr_h_t_insect[addr_idx, :, :] = intsced_fr_path


            # Backward Path-GCN
            this_bk_itsc_feat = torch.tensor(np.array(list(bk_itsc_feat[addr_idx]), dtype=np.float32)).cuda()
            this_bk_itsc_edge = torch.tensor(np.array(list(bk_itsc_edge[addr_idx]), dtype=int)).cuda()

            if this_bk_itsc_feat.shape[0] == 0: continue
            else:
                intsced_bk_path = self.bk_Path_GCN(bk_h_t[addr_idx, :, :], this_bk_itsc_feat, this_bk_itsc_edge,
                                                   bk_intsc_feat_fc[addr_idx, :, :], bk_messageNN[addr_idx, :, :], bk_updateNN[addr_idx, :, :])
            bk_h_t_insect[addr_idx, :, :] = intsced_bk_path



        # 3.2 Attention
        fr_att_feat_list = []
        bk_att_feat_list = []
        fr_att_cal_kernel = torch.cat((fr_h_t_insect, h_global_AF.unsqueeze(dim=1).repeat([1, self.path_num, 1])), dim=-1)
        bk_att_cal_kernel = torch.cat((bk_h_t_insect, h_global_AF.unsqueeze(dim=1).repeat([1, self.path_num, 1])), dim=-1)

        for head_idx in range(self.head_num):
            exec(f'fr_Path_insect_att = self.fr_Path_intsc_att_{head_idx}(fr_att_cal_kernel)')
            exec(f'bk_Path_insect_att = self.bk_Path_intsc_att_{head_idx}(bk_att_cal_kernel)')

            exec(f'fr_att_feat_list.append(torch.sum(fr_Path_insect_att * self.feat_fr_Path_intsc_pre_att_{head_idx}(fr_h_t_insect), dim=1))')
            exec(f'bk_att_feat_list.append(torch.sum(bk_Path_insect_att * self.feat_bk_Path_intsc_pre_att_{head_idx}(bk_h_t_insect), dim=1))')

        fr_graph_final = torch.cat(fr_att_feat_list, dim=-1)
        bk_graph_final = torch.cat(bk_att_feat_list, dim=-1)

        h_global_fr_graph, c_global_fr_graph = self.fr_graph_global_lstm(fr_graph_final, (fr_Graph_hidden[0], fr_Graph_hidden[1]))
        h_global_bk_graph, c_global_bk_graph = self.bk_graph_global_lstm(bk_graph_final, (bk_Graph_hidden[0], bk_Graph_hidden[1]))

        fr_Graph_hidden = (h_global_fr_graph, c_global_fr_graph)
        bk_Graph_hidden = (h_global_bk_graph, c_global_bk_graph)

        return h_global_fr_graph, h_global_bk_graph, fr_Graph_hidden, bk_Graph_hidden, skip_list

    def forward(self, AF_feat, AF_hidden,
                fr_path_feat, bk_path_feat, fr_Path_hidden, bk_Path_hidden,
                fr_itsc_edge, bk_itsc_edge, fr_Graph_hidden, bk_Graph_hidden,
                fr_itsc_feat, bk_itsc_feat, split_index, former_hazard_list, skip_list=[], prev_pred=None):

        # 1. Processing AF
        bsz = AF_feat.shape[0]
        feat_AF = self.feat_AF_BN(AF_feat)
        feat_AF = self.feat_AF_fc(feat_AF)
        h_global_AF, c_global_AF = self.AF_global_lstm(feat_AF, (AF_hidden[0], AF_hidden[1]))
        AF_hidden = (h_global_AF, c_global_AF)

        # 2. Processing Path
        fr_h_t, bk_h_t, h_global_fr_path, h_global_bk_path, fr_Path_hidden, bk_Path_hidden = \
            self.path_process(bsz, h_global_AF, fr_path_feat, bk_path_feat, fr_Path_hidden, bk_Path_hidden)

        # 3. Path-GCN
        h_global_fr_graph, h_global_bk_graph, fr_Graph_hidden, bk_Graph_hidden, skip_list = \
            self.graph_process(bsz, h_global_AF, fr_h_t, bk_h_t, fr_itsc_feat, bk_itsc_feat,
                               fr_itsc_edge, bk_itsc_edge, fr_Graph_hidden, bk_Graph_hidden,
                               former_hazard_list, split_index, skip_list)

        # 4. Prediction
        # 4.1 Calculate each rate
        rate_weight = self.rate_weight(h_global_fr_path)
        former_hazard_list[split_index, :, 0] = self.AF_rate(h_global_AF).squeeze(-1)*rate_weight[:, 0]
        former_hazard_list[split_index, :, 1] = self.fr_path_rate(torch.cat((h_global_AF, h_global_fr_path), dim=-1)).squeeze(-1)*rate_weight[:, 1]
        former_hazard_list[split_index, :, 2] = self.fr_path_rate(torch.cat((h_global_AF, h_global_bk_path), dim=-1)).squeeze(-1)*rate_weight[:, 2]
        former_hazard_list[split_index, :, 3] = self.fr_graph_rate(torch.cat((h_global_AF, h_global_fr_path, h_global_fr_graph), dim=-1)).squeeze(-1)*rate_weight[:, 3]
        former_hazard_list[split_index, :, 4] = self.fr_graph_rate(torch.cat((h_global_AF, h_global_bk_path, h_global_bk_graph), dim=-1)).squeeze(-1)*rate_weight[:, 4]


        sp = torch.nn.Softplus()
        rate_sum = torch.sum(former_hazard_list[:split_index + 1, :, :], dim=0)
        survival_predict = torch.exp(-1*sp(torch.sum(rate_sum, dim=-1)))+0.


        if prev_pred!=None:
            for i in skip_list: survival_predict[i] = prev_pred[i]

        return former_hazard_list, survival_predict, AF_hidden, fr_Path_hidden, bk_Path_hidden, fr_Graph_hidden, bk_Graph_hidden, skip_list

