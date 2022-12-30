#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
import numpy as np


def path_preprocessing(split_path_info):
    # 1. Cut each path to collect time_split path info
    feat_batch = [[] for i in range(23)]
    for i in range(23):  # Loop for each split
        this_split_path_info = split_path_info[i]
        path_num = len(this_split_path_info);
        assert path_num <= 10

        path_feat = np.zeros([10, 6, 16])
        for path_idx, each_path in enumerate(this_split_path_info):
            step_size = each_path.shape[0] / float(6)
            for path_node_idx in range(6):
                sample_zone = each_path[int(path_node_idx * step_size):max(int((path_node_idx + 1) * step_size),
                                                                           int((path_node_idx + 1) * step_size) + 1),
                              1:]  # feature-0 is the tx-index, so we skip
                path_feat[path_idx, path_node_idx, :] = np.mean(sample_zone, axis=0)

        feat_batch[i] = [path_feat]
    return feat_batch


def intersect_preprocessing(split_intersect_info):
    # 1. Cut each path to collect time_split path info
    edge_batch = [[[], []] for i in range(23)]
    feat_batch = [[] for i in range(23)]

    for i in range(23):  # Loop for each split
        this_split_path_info = split_intersect_info[i]  # All Intersection info at this moment

        if len(this_split_path_info) == 0:
            continue
        else:
            edge_index = [[], []]
            for link_idx, link_info in enumerate(this_split_path_info):  # link = intersection
                edge_source = link_info[0]
                edge_sink = link_info[1]
                edge_feat = link_info[2]
                edge_index[0] = edge_index[0] + [edge_source]
                edge_index[1] = edge_index[1] + [edge_sink]
                feat_batch[i] = feat_batch[i] + [edge_feat]
            edge_batch[i] = edge_index
    return edge_batch, feat_batch


def feat_process_for_predict(save_path, picked_node_id):
    # Account Feature
    AF_feat_split = [[] for i in range(23)]
    AF_path_feat = np.load(os.path.join(save_path, f'address_feature_{picked_node_id}.npy'))
    life_len = AF_path_feat.shape[0]; assert life_len >= 1
    AF_dim = AF_path_feat.shape[1]; assert AF_dim == 16
    for hour_idx in range(23):
        if hour_idx < life_len: AF_feat_split[hour_idx] = AF_path_feat[hour_idx, :]
        else: AF_feat_split[hour_idx] = AF_path_feat[life_len - 1, :]

    # Path Feature, pre-process to fix length
    raw_bk_path_feat_file = os.path.join(save_path, f'backward_{picked_node_id}_feat.npz')
    bk_path_feat = np.load(raw_bk_path_feat_file, allow_pickle=True)['arr_0']
    bk_split_path_feat = path_preprocessing(bk_path_feat)

    raw_fr_path_feat_file = os.path.join(save_path, f'forward_{picked_node_id}_feat.npz')
    fr_path_feat = np.load(raw_fr_path_feat_file, allow_pickle=True)['arr_0']
    fr_split_path_feat = path_preprocessing(fr_path_feat)

    # Intersection Feature, pre-process
    raw_fr_intersect_file = os.path.join(save_path, f'forward_{picked_node_id}_intsct_feat.npz')
    raw_bk_intersect_file = os.path.join(save_path, f'backward_{picked_node_id}_intsct_feat.npz')
    fr_intersect = np.load(raw_fr_intersect_file, allow_pickle=True)['arr_0']
    bk_intersect = np.load(raw_bk_intersect_file, allow_pickle=True)['arr_0']
    fr_split_intersect_edge, fr_split_intersect_feat = intersect_preprocessing(fr_intersect)
    bk_split_intersect_edge, bk_split_intersect_feat = intersect_preprocessing(bk_intersect)

    # Assemble
    AF_feat_batch = [AF_feat_split[hour_idx] for hour_idx in range(23)]
    fr_path_feat = [fr_split_path_feat[hour_idx] for hour_idx in range(23)]
    bk_path_feat = [bk_split_path_feat[hour_idx] for hour_idx in range(23)]
    fr_intersect_edge = [fr_split_intersect_edge[hour_idx] for hour_idx in range(23)]
    bk_intersect_edge = [bk_split_intersect_edge[hour_idx] for hour_idx in range(23)]
    fr_intersect_feat = [fr_split_intersect_feat[hour_idx] for hour_idx in range(23)]
    bk_intersect_feat = [bk_split_intersect_feat[hour_idx] for hour_idx in range(23)]

    r_AF_feat_batch = [[0]] * 23
    r_fr_feat_batch = [[0]] * 23
    r_bk_feat_batch = [[0]] * 23
    r_fr_intersect_edge_batch = [[0]] * 23
    r_bk_intersect_edge_batch = [[0]] * 23
    r_fr_intersect_feat_batch = [[0]] * 23
    r_bk_intersect_feat_batch = [[0]] * 23

    for i in range(23):
        r_AF_feat_batch[i][0] = np.array(AF_feat_batch[i])
        r_fr_feat_batch[i][0] = np.array(fr_path_feat[i])
        r_bk_feat_batch[i][0] = np.array(bk_path_feat[i])
        r_fr_intersect_edge_batch[i][0] = np.array(fr_intersect_edge[i])
        r_bk_intersect_edge_batch[i][0] = np.array(bk_intersect_edge[i])
        r_fr_intersect_feat_batch[i][0] = np.array(fr_intersect_feat[i])
        r_bk_intersect_feat_batch[i][0] = np.array(bk_intersect_feat[i])


    return r_AF_feat_batch, r_fr_feat_batch, r_bk_feat_batch, \
           r_fr_intersect_edge_batch, r_bk_intersect_edge_batch, r_fr_intersect_feat_batch, r_bk_intersect_feat_batch