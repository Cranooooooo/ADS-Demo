from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
import numpy as np
import time

class NN_MessagePassingLayer(MessagePassing):
    def __init__(self, hidden_dim, aggr='mean'):
        super(NN_MessagePassingLayer, self).__init__()
        self.aggr = aggr
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_weight, edge_index, messageNN_weight, updateNN_weight):
        assert list(messageNN_weight.shape[-2:]) == [self.hidden_dim * 2, self.hidden_dim]
        assert list(updateNN_weight.shape[-2:]) == [self.hidden_dim * 2, self.hidden_dim]
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, messageNN_weight=messageNN_weight, updateNN_weight=updateNN_weight)

    def message(self, x_i, x_j, edge_index, messageNN_weight, updateNN_weight):
        return torch.matmul(torch.cat((x_i, x_j), dim=-1), messageNN_weight)

    def update(self, aggr_out, x, edge_index, messageNN_weight, updateNN_weight):
        return torch.matmul(torch.cat((x, aggr_out), dim=-1), updateNN_weight)


class Evo_Path_GNN(torch.nn.Module):
    def __init__(self, layer_num, hidden_dim, aggr='mean', **kwargs):
        super(Evo_Path_GNN, self).__init__()
        self.encoder = nn.Linear(hidden_dim, hidden_dim)
        self.mp_layer = NN_MessagePassingLayer(hidden_dim=hidden_dim, aggr=aggr)
        self.decoder = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_feat, edge_feat, edge_list, intsc_feat_fc, messageNN, updateNN):
        # Preparation
        update_node_feat = torch.clone(node_feat)
        edge_num = len(edge_list[0])
        edge_feat = torch.matmul(edge_feat, intsc_feat_fc)

        # Degree calculation
        degree_list = [0]*10
        for edge_idx in range(edge_num):
            source = edge_list[0][edge_idx]
            sink = edge_list[1][edge_idx]
            degree_list[source] = degree_list[source] + 1
            degree_list[sink] = degree_list[sink] + 1

        # Update
        for edge_idx in range(edge_num):
            source = edge_list[0][edge_idx]
            sink = edge_list[1][edge_idx]
            source_degree = degree_list[source]
            sink_degree = degree_list[sink]

            # Update Source
            update_message = 1./source_degree*torch.matmul(messageNN, edge_feat[edge_idx,:])*node_feat[sink, :]
            update_node_feat[source, :] = torch.matmul(update_node_feat[source, :] + update_message, updateNN)

            # Update Sink
            update_message = 1./sink_degree*torch.matmul(messageNN, edge_feat[edge_idx,:])*node_feat[source, :]
            update_node_feat[sink, :] = torch.matmul(update_node_feat[sink, :] + update_message, updateNN)

        return node_feat