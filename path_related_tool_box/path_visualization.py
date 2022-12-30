#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from igraph import *
import igraph as ig
from colour import Color
import json
import matplotlib.pyplot as plt



red = Color("red")
Color_list = list(red.range_to(Color("green"),104))
color_list = [str(i).split(' ')[-1] for i in Color_list][1:-1]


def graph_building_igraph(modified_edge_list, source_node_tx_list, num_vtx, direction):
    '''
    :param btc_data:
    :param num_vertex:
    :return:
    '''

    # 1. Build graph
    print('node number is {}'.format(num_vtx))
    bitcoin_graph = Graph(directed=True)
    bitcoin_graph.add_vertices(num_vtx)

    for edge_counter, edge_item in enumerate(modified_edge_list):
        # if edge_counter%5000==0: print('Edge adding {} / {}'.format(edge_counter, len(modified_edge_list)))
        source_index = int(edge_item[0])
        destin_index = int(edge_item[1])
        activation_value = round(float(edge_item[2]),4) if direction=='forward' else round(float(edge_item[3]),4)
        hop_index = int(edge_item[3]) + 11

        # 1.2 build edge
        if direction=='forward':
            bitcoin_graph.add_edges([(source_index, destin_index)])
            edge_id = bitcoin_graph.get_eid(source_index, destin_index)
        else:
            bitcoin_graph.add_edges([(destin_index, source_index)])
            edge_id = bitcoin_graph.get_eid(destin_index, source_index)


        if source_index not in source_node_tx_list:
            bitcoin_graph.vs[source_index]["color"] = color_list[-1]
            bitcoin_graph.vs[source_index]["size"] = 5
            bitcoin_graph.vs[source_index]["label"] = str(activation_value)# + '/Tx id : {}'.format(edge_item[-1])
        else:
            bitcoin_graph.vs[source_index]["size"] = 5

        if destin_index not in source_node_tx_list:
            bitcoin_graph.vs[destin_index]["color"] = color_list[-1]
            bitcoin_graph.vs[destin_index]["size"] = 5
            bitcoin_graph.vs[destin_index]["label"] = str(activation_value)# + '/Tx id : {}'.format(edge_item[-1])
        else:
            bitcoin_graph.vs[destin_index]["size"] = 5

        bitcoin_graph.es[edge_id]['weight'] = activation_value
        bitcoin_graph.es[edge_id]["color"] = color_list[hop_index]
        bitcoin_graph.es[edge_id]['width'] = 2.*activation_value

    bitcoin_graph.vs.select(_degree=0).delete()
    if direction == 'forward':
        for vx in bitcoin_graph.vs.select(_indegree=0):
            vx["size"] = 10; vx["color"] = color_list[0]; vx["label"] = 'Spend Txs'
    if direction == 'backward':
        for vx in bitcoin_graph.vs.select(_outdegree=0):
            vx["size"] = 10; vx["color"] = color_list[0]; vx["label"] = 'Receive Txs'

    print('Node num is {}'.format(len(bitcoin_graph.vs)))
    return bitcoin_graph


def path_plot(direction, file_path):
    # 1. Load data
    with open(file_path, 'r') as f: path_raw_data = json.load(f)

    # 2. Get overall parameters
    if direction == 'forward': max_hop_num = max([int(key.split('_')[0]) for key in path_raw_data.keys()])
    if direction == 'backward': max_hop_num = min([int(key.split('_')[0]) for key in path_raw_data.keys()])
    if direction == 'forward': start_hop=1; interval=1
    else: start_hop=-1; interval=-1

    # 3. Loop into every node pair and igraph node idx mapping
    modified_edge_list = []; node_name_mapping = {}
    part_source_list = path_raw_data['0_hop']
    forward_part_source_plot_list = [i['cascade_txs_index'] for i in part_source_list]
    assert len(forward_part_source_plot_list) == len(set(forward_part_source_plot_list))
    for source_node in forward_part_source_plot_list:
        node_name_mapping[source_node] = len(list(node_name_mapping.keys()))

    for hop_index in range(start_hop, max_hop_num, interval):
        for index, edge_item in enumerate(path_raw_data['{}_hop'.format(hop_index)]):
            source_node, destin_node = edge_item['source_node'], edge_item['cascade_txs_index']
            path_source_height = edge_item['path_source_height']
            [trust_score, influ_score, asset_score] = edge_item['activate_score']
            if source_node not in node_name_mapping.keys(): node_name_mapping[source_node] = len(list(node_name_mapping.keys()))
            if destin_node not in node_name_mapping.keys(): node_name_mapping[destin_node] = len(list(node_name_mapping.keys()))

            # 4.1 Add edges
            modified_edge_list.append([node_name_mapping[source_node],
                                       node_name_mapping[destin_node],
                                       trust_score, influ_score, asset_score,
                                       path_source_height, hop_index, destin_node])

    # 5. Plot Asset transition path
    print('------------- Building Graph -------------')
    part_source_plot_list = list(set(forward_part_source_plot_list))
    print('Source Node num is {}'.format(len(part_source_plot_list)))
    btc_graph = graph_building_igraph(modified_edge_list,
                                      source_node_tx_list=[node_name_mapping[source_node] for source_node in part_source_plot_list],
                                      num_vtx = len(node_name_mapping.keys()),
                                      direction=direction)
    # layout = btc_graph.layout('kk')
    # layout=layout,
    fig, ax = plt.subplots(figsize=(15,25))
    ig.plot(btc_graph, target=ax, bbox=(400, 400, 400, 400), edge_arrow_size=5.,
            vertex_label=btc_graph.vs['label'], vertex_color=btc_graph.vs['color'], vertex_size=btc_graph.vs['size'])
    print('------------ Plotting Done ------------')