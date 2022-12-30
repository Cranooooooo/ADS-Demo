#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json
import time

height_diff_thres = 6*24
def forward_cas_fetch(main_blk_chain, forward_path, source_node_info, unique_source_list, research_end_height):
    # 1. Initialize output objetcs
    forward_cascade_dict = {}
    forward_cascade_dict['0_hop'] = source_node_info

    # 2. Loop into every relation and hop
    max_hop = 0; hop_index = 0; stop_flag = False
    while not stop_flag:
        path_end_flag_list = []
        for each_hop_info in forward_cascade_dict['{}_hop'.format(hop_index)]:
            tx_index = each_hop_info['cascade_txs_index']
            path_source_height = each_hop_info['path_source_height']
            inhibit_trust = each_hop_info['activate_score'][0]
            if hop_index != 0 and tx_index in unique_source_list: continue  # Prevent forever loop
            this_tx = main_blk_chain.tx_with_index(int(tx_index))
            if this_tx.input_value == 0.0: continue


            # Deep into those TXs with rela requirements
            spend_value_list = this_tx.outs.spending_input.value
            spend_tx_dedup_dict = {str(tx_idx): 0.0 for tx_idx in set(list(this_tx.outs.spending_tx_index))}
            for spend_tx_count, spend_tx_indx in enumerate(list(this_tx.outs.spending_tx_index)):
                spend_tx_dedup_dict[str(spend_tx_indx)] += spend_value_list[spend_tx_count]
            assert (sum(spend_tx_dedup_dict.values()) == sum(spend_value_list))


            # Collect all asset score, Trust score above average can cascade
            if len(spend_tx_dedup_dict.keys()) == 0: continue
            current_thres = 0.01
            for candidate_idx, cascade_candidate in enumerate(list(spend_tx_dedup_dict.keys())):
                trust_output_ratio = inhibit_trust*spend_tx_dedup_dict[cascade_candidate] / float(this_tx.output_value)

                if trust_output_ratio > current_thres:
                    candidate_tx = main_blk_chain.tx_with_index(int(cascade_candidate))
                    candidate_height = candidate_tx.block_height
                    if (candidate_height-path_source_height)<=height_diff_thres and \
                        (candidate_height<research_end_height):
                        influ_output_ratio = spend_tx_dedup_dict[cascade_candidate] / candidate_tx.input_value
                        asset_output_ratio = (trust_output_ratio + influ_output_ratio) / 2.
                        max_hop = hop_index+1

                        # update the related txs and nodes
                        if forward_cascade_dict.get('{}_hop'.format(hop_index + 1)) == None:
                            forward_cascade_dict['{}_hop'.format(hop_index + 1)] = []
                        forward_cascade_dict['{}_hop'.format(hop_index + 1)] += [
                            {'source_node': int(tx_index),
                             'activate_score': [trust_output_ratio, influ_output_ratio, asset_output_ratio],
                             'cascade_txs_index': int(cascade_candidate),
                             'path_source_height':path_source_height}]
                        path_end_flag_list += [True]

        hop_index += 1
        stop_flag = (True not in path_end_flag_list)

    # 3. Save results for further use
    print('Forward Max {} hops  '.format(max_hop))
    # for i in forward_cascade_dict.keys(): print(i, forward_cascade_dict[i])
    with open(forward_path, 'w') as f: json.dump(forward_cascade_dict, f)
    return 0


def backward_cas_fetch(main_blk_chain, backward_path, source_node_info, unique_source_list, research_end_height):
    # 1. Initialize output objetcs
    backward_cascade_dict = {}
    backward_cascade_dict['0_hop'] = source_node_info

    # 2. Loop into every relation and hop
    max_hop = 0; hop_index = 0; stop_flag = False
    while not stop_flag:
        path_end_flag_list = []
        for each_hop_info in backward_cascade_dict['{}_hop'.format(hop_index)]:
            tx_index = each_hop_info['cascade_txs_index']
            path_source_height = each_hop_info['path_source_height']
            inhibit_influ = each_hop_info['activate_score'][1]
            if hop_index != 0 and tx_index in unique_source_list: continue  # Prevent forever loop
            this_tx = main_blk_chain.tx_with_index(int(tx_index))
            if this_tx.output_value == 0.0: continue


            # Deep into those TXs with rela requirements
            spent_value_list = this_tx.ins.spent_output.value
            spent_tx_dedup_dict = {str(tx_idx): 0.0 for tx_idx in set(list(this_tx.ins.spent_tx_index))}
            for spent_tx_count, spent_tx_indx in enumerate(list(this_tx.ins.spent_tx_index)):
                spent_tx_dedup_dict[str(spent_tx_indx)] += spent_value_list[spent_tx_count]
            assert (sum(spent_tx_dedup_dict.values()) == sum(spent_value_list))

            # Collect all asset score, Influence score above average can cascade
            if len(spent_tx_dedup_dict.keys()) == 0: continue
            current_thres = 0.01
            for candidate_idx, cascade_candidate in enumerate(list(spent_tx_dedup_dict.keys())):
                influ_output_ratio = inhibit_influ*spent_tx_dedup_dict[cascade_candidate] / float(this_tx.input_value)

                if influ_output_ratio > current_thres:
                    candidate_tx = main_blk_chain.tx_with_index(int(cascade_candidate))
                    candidate_height = candidate_tx.block_height
                    if (path_source_height-candidate_height)<=height_diff_thres:
                        max_hop = hop_index-1
                        trust_output_ratio = spent_tx_dedup_dict[cascade_candidate] / candidate_tx.output_value
                        asset_output_ratio = (trust_output_ratio + influ_output_ratio) / 2.

                        # update the related txs and nodes
                        if backward_cascade_dict.get('{}_hop'.format(hop_index - 1)) == None:
                            backward_cascade_dict['{}_hop'.format(hop_index - 1)] = []
                        backward_cascade_dict['{}_hop'.format(hop_index - 1)] += [
                            {'source_node': int(tx_index),
                             'activate_score': [trust_output_ratio, influ_output_ratio, asset_output_ratio],
                             'cascade_txs_index': int(cascade_candidate),
                             'path_source_height':path_source_height}]
                        path_end_flag_list += [True]

        hop_index -= 1
        stop_flag = (True not in path_end_flag_list)

    # 3. Save results for further use
    print('Backward Max {} hops'.format(max_hop))
    # for i in backward_cascade_dict.keys(): print(i, backward_cascade_dict[i])
    with open(backward_path, 'w') as f: json.dump(backward_cascade_dict, f)
    return 0



def path_prepare(main_blk_chain, picked_addr, research_end_height, forward_path, backward_path):
    # 3.1 Forward Processing
    start_time = time.time()
    raw_forward_tx_list = []
    forward_unique_source_list = []
    forward_unique_source_height = []
    for input_tx in picked_addr.input_txes: raw_forward_tx_list.append(input_tx)
    raw_forward_tx_list.sort(key=lambda x: x.block_height, reverse=False)

    for input_tx in raw_forward_tx_list:
        if input_tx.block_height > research_end_height: continue
        forward_unique_source_list.append(input_tx.index)
        forward_unique_source_height.append(input_tx.block_height)
    forward_source_node_info = [{'source_node': 0, 'activate_score': [1., 1., 1.],
                                 'cascade_txs_index': forward_unique_source_list[i],
                                 'path_source_height': forward_unique_source_height[i]}
                                for i in range(len(forward_unique_source_list[:1000]))]
    print('Forward tx num : {}'.format(len(forward_unique_source_list[:1000])))
    forward_cas_fetch(main_blk_chain, forward_path, forward_source_node_info, forward_unique_source_list, research_end_height)
    print('Forward Fetching cost {} s'.format(round(time.time() - start_time, 2)))
    print('')
    print('=='*20)

    # 3.2 Backward Processing
    start_time = time.time()
    raw_backward_tx_list = []
    backward_unique_source_list = []
    backward_unique_source_height = []
    for output_tx in picked_addr.output_txes: raw_backward_tx_list.append(output_tx)
    raw_backward_tx_list.sort(key=lambda x: x.block_height, reverse=False)

    for output_tx in raw_backward_tx_list:
        if output_tx.block_height > research_end_height: continue
        backward_unique_source_list.append(output_tx.index)
        backward_unique_source_height.append(output_tx.block_height)
    backward_source_node_info = [{'source_node': 0, 'activate_score': [1., 1., 1.],
                                  'cascade_txs_index': backward_unique_source_list[i],
                                  'path_source_height': backward_unique_source_height[i]}
                                 for i in range(len(backward_unique_source_list[:1000]))]
    print('Backward tx num : {}'.format(len(backward_unique_source_list[:1000])))
    backward_cas_fetch(main_blk_chain, backward_path, backward_source_node_info, backward_unique_source_list, research_end_height)
    print('Backward Fetching cost {} s'.format(round(time.time() - start_time, 2)))
    print('')
