#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json
import numpy as np
import copy
import blocksci

analyze_span=24
addr_type_dict = {'0': blocksci.address_type.nonstandard,
                  '1': blocksci.address_type.pubkey,
                  '2': blocksci.address_type.pubkeyhash,
                  '3': blocksci.address_type.multisig_pubkey,
                  '4': blocksci.address_type.scripthash,
                  '5': blocksci.address_type.multisig,
                  '6': blocksci.address_type.nulldata,
                  '7': blocksci.address_type.witness_pubkeyhash,
                  '8': blocksci.address_type.witness_scripthash}



def format_change(early_path_dict):
    rec_early_path_dict = {}
    rec_early_path_dict['forward'] = []
    rec_early_path_dict['backward'] = []

    for direct in ['forward', 'backward']:
        this_direct = early_path_dict[direct][()]
        for time_idx, hour_time in enumerate(range(6, int(analyze_span*6), 6)):
            rec_early_path_dict[direct].append(this_direct[str(hour_time)])

    return rec_early_path_dict


def prepare_path_feature(main_blk_chain, path_item):
    '''
    1. log(TX_id / Global_min_TX_id)
    2. Trust or Influ scores

    3. log(Input total amount)
    4. Input Tx num
    5. Input Unique Address num
    6. Input Tx max amount
    7. Input Tx min amount
    8. Input Tx avg amount
    9. Input Tx amount variance

    10. log(Output total amount)
    11. Output Tx num
    12. Output Tx max amount
    13. Output Tx min amount
    14. Output Tx avg amount
    15. Output Tx amount variance
    16. Path node Output Tx amount

    17. Fee amount
    :param node_info:
    :return:
    '''
    path_len = len(path_item)
    path_feat = np.zeros([path_len, 17])
    for idx, path_node_idx in enumerate(path_item):
        blksci_tx_node = main_blk_chain.tx_with_index(int(path_node_idx))
        # 1. Feature 1 - 2
        path_feat[idx, 0] = blksci_tx_node.index
        path_feat[idx, 1] = path_casc_score_record[f'{path_node_idx}_{path_item[idx+1]}']\
        if idx != path_len-1 else 1.

        # 2. Feature 3 - 9
        address_num_list = blksci_tx_node.inputs.address.address_num
        path_feat[idx, 2] = np.log10(1. + blksci_tx_node.input_value / 1000.)
        path_feat[idx, 3] = len(address_num_list)
        path_feat[idx, 4] = len(set(address_num_list))
        spent_value_list = blksci_tx_node.ins.spent_output.value
        if len(spent_value_list)==0: spent_value_list=[0]
        path_feat[idx, 5] = np.log10(1. + np.max(spent_value_list) / 1000.)
        path_feat[idx, 6] = np.log10(1. + np.min(spent_value_list) / 1000.)
        path_feat[idx, 7] = np.log10(1. + np.mean(spent_value_list) / 1000.)
        path_feat[idx, 8] = np.log10(1. + np.var(spent_value_list) / 1000.)

        # 3. Feature 10 - 16
        address_num_list = blksci_tx_node.outputs.address.address_num
        path_feat[idx, 9] = np.log10(1. + blksci_tx_node.output_value / 1000.)
        path_feat[idx, 10] = len(address_num_list)
        path_feat[idx, 11] = len(set(address_num_list))
        spend_value_list = blksci_tx_node.outs.spending_input.value
        if len(spend_value_list) == 0: spend_value_list = [0]
        path_feat[idx, 12] = np.log10(1. + np.max(spend_value_list) / 1000.)
        path_feat[idx, 13] = np.log10(1. + np.min(spend_value_list) / 1000.)
        path_feat[idx, 14] = np.log10(1. + np.mean(spend_value_list) / 1000.)
        path_feat[idx, 15] = np.log10(1. + np.var(spend_value_list) / 1000.)

        # 4. Feature 17
        path_feat[idx, 16] = np.log10(1. + blksci_tx_node.fee / 1000.)

    return path_feat

def path_builder_iterator(path_list, edge_sink_dict_struct,
                          hop_index, direction, max_path_num_each_hop = 10):
    # 1. Find all current paths which have further destination
    cannot_extend = 1
    extended_path_list = []
    current_path_num = len(path_list)
    if current_path_num > max_path_num_each_hop:
        if direction == 'forward':
            path_list.sort(key=lambda x: main_blk_chain_.tx_with_index(int(x[-1])).output_value, reverse=True)
        else:
            path_list.sort(key=lambda x: main_blk_chain_.tx_with_index(int(x[0])).input_value, reverse=True)

    for path_idx in range(current_path_num):
        destin_list = edge_sink_dict_struct.get(path_list[path_idx][-1])
        if destin_list!=None:
            for destin_candidate in destin_list:
                extended_path_list.append(path_list[path_idx] + [destin_candidate])
            if extended_path_list != []: cannot_extend*=0

        else: extended_path_list.append(path_list[path_idx]) # If cannot extend, just copy it

    if cannot_extend==1 or hop_index>100: return path_list[:max_path_num_each_hop]
    hop_index+=1
    complete_path_list = path_builder_iterator(extended_path_list[:max_path_num_each_hop],
                                               edge_sink_dict_struct, hop_index, direction=direction)

    return complete_path_list[:max_path_num_each_hop]


def graph_building(main_blk_chain, modified_edge_list, addr_birth_height, direction):
    '''
    :param btc_data:
    :param num_vertex:
    :return:
    '''
    # 1. Build graph
    edge_source_height_diff_to_birth = {}
    edge_destin_height_diff_to_birth = {}
    edge_source_dict_struct = {}
    edge_sink_dict_struct = {}
    global main_blk_chain_
    main_blk_chain_ = main_blk_chain

    # 1.1 Get the most edge feature information first,
    #     since 100 hours contains 1 hours information, no need to calculate again
    for edge_counter, edge_item in enumerate(modified_edge_list):
        source_index = str(edge_item[0])
        destin_index = str(edge_item[1])

        former_tx = main_blk_chain.tx_with_index(edge_item[0])
        later_tx = main_blk_chain.tx_with_index(edge_item[1])
        edge_source_height_diff_to_birth[source_index] = former_tx.block_height - addr_birth_height
        edge_destin_height_diff_to_birth[destin_index] = later_tx.block_height - addr_birth_height

        # Only analyze [-24, +24]
        if np.abs(edge_source_height_diff_to_birth[source_index]) > 24*6 or \
           np.abs(edge_destin_height_diff_to_birth[destin_index]) > 24*6: continue

        # Build Source Destin relation
        if edge_source_dict_struct.get(destin_index) != None: edge_source_dict_struct[destin_index].append(source_index)
        else: edge_source_dict_struct[destin_index] = [source_index]
        edge_source_dict_struct[source_index] = edge_source_dict_struct.get(source_index)

        if edge_sink_dict_struct.get(source_index) != None: edge_sink_dict_struct[source_index].append(destin_index)
        else: edge_sink_dict_struct[source_index] = [destin_index]
        edge_sink_dict_struct[destin_index] = edge_sink_dict_struct.get(destin_index)


    # 3.1. Get path list
    path_end, path_start = [], []
    path_start_block_height = []
    for key, values in edge_source_dict_struct.items():
        if values == None:
            path_start.append(key)
            path_start_block_height.append(main_blk_chain.tx_with_index(int(key)).block_height)

    path_list = [[str(path_source)] for path_source in path_start] # Initial Path seed list
    path_list = path_builder_iterator(path_list, edge_sink_dict_struct,
                                      hop_index=0, direction=direction)
    if direction == 'forward': path_list.sort(key=lambda x: main_blk_chain.tx_with_index(int(x[0])).output_value, reverse=True)
    else: path_list.sort(key=lambda x: main_blk_chain.tx_with_index(int(x[-1])).input_value, reverse=True)


    # 3.2. Get Path at split at different time
    path_diff_time_feat = [[] for i in range(6, int(analyze_span*6), 6)]
    for split_idx, max_height_diff in enumerate(range(6, int(analyze_span*6), 6)):

        for path_idx, path_item in enumerate(path_list):
            # 2.3.0 Skip tx out of max_height_diff
            if direction =='backward':
                if edge_destin_height_diff_to_birth[str(path_item[-1])] > max_height_diff: continue
            if direction == 'forward':
                # if the first tx is put of research period
                if edge_source_height_diff_to_birth[str(path_item[0])] > max_height_diff: continue
                else:
                    # if last tx is still less than max-height, then no need to cut
                    if edge_destin_height_diff_to_birth[str(path_item[-1])] < max_height_diff: pass
                    # need to cut
                    else:
                        for tx_idx_in_path, tx_in_this_path in enumerate(path_item[:-1]):
                            if edge_source_height_diff_to_birth[str(tx_in_this_path)] > max_height_diff:
                                path_item = path_item[:tx_idx_in_path]; break
            path_diff_time_feat[split_idx].append(prepare_path_feature(main_blk_chain, path_item))
    return path_diff_time_feat




def early_path_processor(main_blk_chain, file_path, addr_object, direction, addr_birth_height):
    # 1. Load Raw path file
    try:
        with open(file_path, 'r') as f: path_raw_data = json.load(f)
    except: return None

    # 2. Set Result save object
    if direction == 'forward': max_hop_num = max([int(key.split('_')[0]) for key in path_raw_data.keys()])
    if direction == 'backward': max_hop_num = min([int(key.split('_')[0]) for key in path_raw_data.keys()])
    modified_edge_list = []
    global path_casc_score_record; path_casc_score_record = {}


    # 3. Early Path cut-off tx idx
    # 3.1. Forward Processing
    if direction == 'forward':
        for hop_index in range(1, max_hop_num):
            # modify this hop's node[
            for index, edge_item in enumerate(path_raw_data['{}_hop'.format(hop_index)]):
                source_node, destin_node = edge_item['source_node'], edge_item['cascade_txs_index']
                path_source_height = edge_item['path_source_height']
                [trust_score, influ_score, asset_score] = edge_item['activate_score']
                modified_edge_list.append([source_node, destin_node,
                                           trust_score, influ_score, asset_score,
                                           path_source_height, hop_index])
                path_casc_score_record[f'{source_node}_{destin_node}'] = trust_score


    # 3.2. Backward Processing
    if direction == 'backward':
        for hop_index in range(-1, max_hop_num, -1):
            # modify this hop's node
            for index, edge_item in enumerate(path_raw_data['{}_hop'.format(hop_index)]):
                source_node, destin_node = edge_item['source_node'], edge_item['cascade_txs_index']
                path_source_height = edge_item['path_source_height']
                # if addr_birth_height - path_source_height > 1200: continue # skip tx larger than max_height
                # Can't decide backward here
                [trust_score, influ_score, asset_score] = edge_item['activate_score']
                modified_edge_list.append([destin_node, source_node,
                                           trust_score, influ_score, asset_score,
                                           path_source_height, hop_index])
                path_casc_score_record[f'{destin_node}_{source_node}'] = influ_score

    # 4. Build the graph and calculate independent cascade
    path_diff_time_list = graph_building(main_blk_chain, modified_edge_list, addr_birth_height, direction)
    return path_diff_time_list



def fetch_intersect_addr_feat(main_blk_chain, addr_cl_name, inter_sect_height):
    '''
    1.  Balance of Address 1
    2.  Number of Input Transactions (full history) 1
    3.  Number of Output Transactions (full history) 1
    4.  Amount ratio of input transactions to output transactions (full history) 1
    5.  The lifetime of the address expressed in number of hours(6 blocks) 1
    6.  The sum of all the values transferred to (resp. from) the address. 2
    7.  The average (resp. standard deviation) of the values transferred to/from the address. 4
    :param addr_cl_name:
    :param inter_sect_height:
    :return:
    '''
    feat_idx = 0
    intersect_feat = np.zeros([10])
    addr_object = main_blk_chain.address_from_index(addr_cl_name // 10, addr_type_dict[str(addr_cl_name % 10)])
    input_idx_bf_intersect_height = np.where(addr_object.input_txes.block_height <= inter_sect_height)[0]
    if len(input_idx_bf_intersect_height) == 0:input_tx_amt=0
    else: input_tx_amt = addr_object.input_txes.input_value[:input_idx_bf_intersect_height[-1]+1]

    output_idx_bf_intersect_height = np.where(addr_object.output_txes.block_height <= inter_sect_height)[0]
    assert len(output_idx_bf_intersect_height) > 0
    output_tx_amt = addr_object.output_txes.output_value[:output_idx_bf_intersect_height[-1]+1]
    start_block_height = addr_object.first_tx.block_height

    # intersect_feat[feat_idx] = addr_object.balance(inter_sect_height); feat_idx+=1
    intersect_feat[feat_idx] = len(input_idx_bf_intersect_height); feat_idx+=1
    intersect_feat[feat_idx] = len(output_idx_bf_intersect_height); feat_idx+=1
    intersect_feat[feat_idx] = intersect_feat[feat_idx-2]/float(intersect_feat[feat_idx-1]); feat_idx+=1
    intersect_feat[feat_idx] = (inter_sect_height-start_block_height)/6.; feat_idx+=1 # in hours

    intersect_feat[feat_idx] = np.log10(1.+np.sum(input_tx_amt)/1000.); feat_idx+=1
    intersect_feat[feat_idx] = np.log10(1.+np.sum(output_tx_amt)/1000.); feat_idx+=1

    intersect_feat[feat_idx] = np.log10(1.+np.sum(input_tx_amt)/(intersect_feat[1]+1.)/1000.); feat_idx+=1
    intersect_feat[feat_idx] = np.log10(1.+np.sum(output_tx_amt)/(intersect_feat[2]+1.)/1000.); feat_idx+=1

    intersect_feat[feat_idx] = np.log10(1.+np.std(input_tx_amt)/1000.); feat_idx+=1
    intersect_feat[feat_idx] = np.log10(1.+np.std(output_tx_amt)/1000.); feat_idx+=1

    return intersect_feat

def intersection_prepare(main_blk_chain, path_evolve_list, direction, min_height, max_height, analysis_span=23):
    '''
    interact_info_at_diff_hour
    [ -> Different Hour
      [ --> Different Interact Item
        [Path-i, Path-j,
           [
            Interact_Addr_Feat_1,
            Interact_Addr_Feat_2,
            ...
            ]
        ]
      ]
    ]

    :param path_evolve_list:
    :param direction:
    :param analysis_span:
    :return:
    '''
    # 0. Many duplicate paths, store pre-calculated for acceleration
    addr_height_map_feature_dict = {}
    bad_flag = False

    # 1. Loop into every hour
    assert len(path_evolve_list) == analysis_span
    interact_info_at_diff_hour = [[]]*analysis_span
    for hour_idx in range(analysis_span):
        # print(f'Hour: {hour_idx+1}', end= ' / ')
        # 1.1 Set tmp_save object
        this_hour_all_path = path_evolve_list[hour_idx]
        path_bond_addr_list = []

        # 1.2 Loop into every path
        # print(f'Path number is {len(this_hour_all_path)}', end=' / ')
        for this_hour_path_idx in range(len(this_hour_all_path)):
            # Only the source(bk) or the sink(fr) is the same, we call intersect
            if direction == 'forward': this_path_tx_node = this_hour_all_path[this_hour_path_idx][-1]
            if direction == 'backward': this_path_tx_node = this_hour_all_path[this_hour_path_idx][0]

            this_path_tx_node = int(this_path_tx_node)
            blksci_tx = main_blk_chain.tx_with_index(this_path_tx_node)
            blksci_tx_height = blksci_tx.block_height
            if blksci_tx.is_coinbase:
                bad_flag = True
                return 0, bad_flag
            if blksci_tx_height < min_height: continue
            if blksci_tx_height > max_height: break

            # Forward --> check its input, then get the bound address
            # Backward --> check its output, then get the bound address
            if direction == 'forward': related_tx = blksci_tx.inputs[0].tx.outputs # input_tx_output_list
            if direction == 'backward': related_tx = blksci_tx.outputs[0].tx.inputs # output_tx_input_list
            this_node_idx = related_tx.tx_index.tolist().index(this_path_tx_node)
            bound_addr_num = related_tx.address.address_num[this_node_idx]
            bound_addr_type = related_tx.address.raw_type[this_node_idx]
            bound_addr_cl_name = int(bound_addr_num*10+bound_addr_type)
            path_bond_addr_list.append(f'{bound_addr_cl_name}_{blksci_tx_height}')

        # 1.3 Find the interaction
        this_hour_inter_sect = []
        for obj_path_idx in range(len(this_hour_all_path)-1):
            for sbj_path_idx in range(obj_path_idx+1, len(this_hour_all_path)):
                obj_addr_node = path_bond_addr_list[obj_path_idx]
                sbj_addr_node = path_bond_addr_list[sbj_path_idx]

                # Check if have common [addr_node, height]
                if obj_addr_node == sbj_addr_node: intersect_i = obj_addr_node
                else: continue
                bound_addr_cl_name = int(intersect_i.split('_')[0])
                blksci_tx_height = int(intersect_i.split('_')[1])

                # Check if we have prepared
                if type(addr_height_map_feature_dict.get(f'{bound_addr_cl_name}_{blksci_tx_height}')) == np.ndarray:
                    intersect_feat = addr_height_map_feature_dict[f'{bound_addr_cl_name}_{blksci_tx_height}']
                else:
                    intersect_feat = fetch_intersect_addr_feat(main_blk_chain, bound_addr_cl_name, blksci_tx_height)
                    addr_height_map_feature_dict[f'{bound_addr_cl_name}_{blksci_tx_height}'] = intersect_feat

                this_hour_inter_sect.append([obj_path_idx, sbj_path_idx, intersect_feat])
        interact_info_at_diff_hour[hour_idx] = copy.deepcopy(this_hour_inter_sect)
        # print(f'intersect_count: {len(interact_info_at_diff_hour[hour_idx])}')

    return interact_info_at_diff_hour, bad_flag
