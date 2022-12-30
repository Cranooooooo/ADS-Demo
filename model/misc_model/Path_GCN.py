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
from .basic_encoders import Encoder
from torch_geometric.nn import GCNConv
import numpy as np


class Path_GCN(nn.Module):
    def __init__(self, opt):
        super(Path_GCN, self).__init__()
        # Input Feature Dimension
        self.feat_AF_dim = opt.feat_AF_dim
        self.feat_Path_dim = opt.feat_Path_dim
        self.feat_Intersec_dim = opt.feat_Intersec_dim

        # Model Dimension
        self.bsz = opt.batch_size
        self.analysis_span = opt.analysis_span
        self.path_num = opt.path_num
        self.path_zoom_len = opt.path_zoom_len
        self.hidden_dim = opt.hidden_dim
        self.drop_prob_lm = opt.drop_prob_lm

        # Model Pipeline -- AF
        self.feat_AF_BN = nn.BatchNorm1d(self.feat_AF_dim)
        self.feat_AF_fc = nn.Sequential(nn.Linear(self.feat_AF_dim, self.hidden_size), nn.Tanh(),
                                        nn.Linear(self.hidden_size, self.hidden_size), nn.Sigmoid())
        self.feat_AF_lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.feat_AF_hz_rate = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
                                        nn.Linear(self.hidden_size, 1), nn.Softplus())

        # Model Pipeline -- Path
        self.feat_Path_BN = nn.BatchNorm2d(self.feat_Path_dim) # Input: (N->bsz, C->channel, H->Height, W->Width)
        self.feat_Path_fc = nn.Sequential(nn.Linear(self.feat_Path_dim, self.hidden_size), nn.Tanh(),
                                          nn.Linear(self.hidden_size, self.hidden_size), nn.Sigmoid())
        self.feat_Path_encoder = Encoder(d_model=self.hidden_dim, N=3, heads=2)
        self.feat_Path_att = nn.Sequential(nn.Linear(int(2*self.hidden_size), self.hidden_size), nn.Tanh(),
                                           nn.Linear(self.hidden_size, 1), nn.Sigmoid(), nn.Softmax(dim=-1))
        self.feat_Path_lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.feat_Path_hz_rate = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
                                        nn.Linear(self.hidden_size, 1), nn.Softplus())

        # Model Pipeline -- GCN
        self.feat_Intersec_BN = nn.BatchNorm1d(self.feat_Intersec_dim)
        self.feat_Intersec_fc = nn.Sequential(nn.Linear(self.feat_Intersec_dim, self.hidden_size), nn.Tanh(),
                                              nn.Linear(self.hidden_size, self.hidden_size), nn.Sigmoid())
        self.conv1 = nn.Sequential(GCNConv(self.hidden_dim, self.hidden_dim), nn.Sigmoid())
        self.conv2 = nn.Sequential(GCNConv(self.hidden_dim, self.hidden_dim), nn.Sigmoid())
        self.feat_Intersec_att = nn.Sequential(nn.Linear(int(2*self.hidden_size), self.hidden_size), nn.Tanh(),
                                               nn.Linear(self.hidden_size, 1), nn.Sigmoid(), nn.Softmax(dim=-1))
        self.feat_Intersec_lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.feat_Intersec_hz_rate = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
                                        nn.Linear(self.hidden_size, 1), nn.Softplus())

        self._init_parameters_seg()

    def init_hidden(self):
        weight = next(self.parameters())
        return (weight.new_zeros(self.bsz, self.hidden_size),
                weight.new_zeros(self.bsz, self.hidden_size))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.01)

    def _init_parameters_seg(self):
        # Model Pipeline -- AF
        self.feat_AF_fc.apply(self.init_weights)
        self.feat_AF_lstm.apply(self.init_weights)
        self.feat_AF_hz_rate.apply(self.init_weights)

        # Model Pipeline -- Path
        self.feat_Path_fc.apply(self.init_weights)
        self.feat_Path_encoder.apply(self.init_weights)
        self.feat_Path_att.apply(self.init_weights)
        self.feat_Path_lstm.apply(self.init_weights)
        self.feat_Path_hz_rate.apply(self.init_weights)

        # Model Pipeline -- GCN
        self.feat_Intersec_fc.apply(self.init_weights)
        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.feat_Intersec_att.apply(self.init_weights)
        self.feat_Intersec_lstm.apply(self.init_weights)
        self.feat_Intersec_hz_rate.apply(self.init_weights)

    def forward(self, AF_feat, Path_feat, Intersect_feat, AF_lstm_hidden, Path_lstm_hidden, Intersect_lstm_hidden):
        # 1. Processing AF
        feat_AF = self.feat_AF_BN(AF_feat)
        feat_AF = self.feat_AF_fc(feat_AF)
        h_AF_tmp, c_AF_tmp = self.feat_lstm(feat_AF, (AF_lstm_hidden[0], AF_lstm_hidden[1]))
        AF_lstm_didden = (h_AF_tmp, c_AF_tmp)
        AF_hz_rate = self.feat_AF_hz_rate(h_AF_tmp)


        # 2. Processing Path
        # 2.1 Path Encoding
        feat_Path = self.feat_Path_BN(Path_feat.reshape([self.bsz, self.path_num, self.path_zoom_len, self.feat_Path_dim]))
        feat_Path = self.feat_Path_fc(feat_Path.reshape([-1, self.feat_Path_dim])).\
                    reshape([self.bsz*self.path_num, self.path_zoom_len, self.hidden_dim])
        feat_Path = torch.mean(self.feat_Path_encoder(feat_Path), dim=1).reshape([self.bsz, self.path_num, self.hidden_dim])

        # 2.2 Path Attention
        atten_score = self.feat_Path_att(torch.concatenate([h_AF_tmp.unsqueeze(1).repeat(1,self.path_num,1),
                                                            feat_Path], dim=-1))
        atten_feat_Path = torch.sum(atten_score.repeat(1,1,self.hidden_dim)*feat_Path, dim=1)
        assert atten_feat_Path.shape == torch.Size([self.bsz, self.hidden_dim])

        # 2.3 Path LSTM
        h_Path_tmp, c_Path_tmp = self.feat_Path_lstm(atten_feat_Path, (Path_lstm_hidden[0], Path_lstm_hidden[1]))
        Path_lstm_didden = (h_Path_tmp, c_Path_tmp)
        Path_hz_rate = self.feat_Path_hz_rate(h_Path_tmp)


        # 3. Processing Intersect


        # 2.4 Original Prediction
        survival_predict = self.label_predict(torch.cat((h_feat_tmp, h_state_tmp, h_inten_tmp), dim=-1)).squeeze(-1)

        hazard_rate = torch.zeros([addr_num]).cuda()
        return fibo_inter_feat_hidden, fibo_inter_segm_hidden, fibo_inter_stat_hidden, hazard_rate, survival_predict

