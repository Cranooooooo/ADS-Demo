#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import blocksci
import numpy as np
import os


def feature_prepare(blksci_source_node, node_id, save_path, research_end_height=610637):
    '''
    Each feature should be prepared with time evolution for later analysis.
    Currently, Split whole life every 6 or 144 blocks.
    Since we can't know the life-span before it ends.
    Tip: Use 6 and 144 blocks (6 blocks/h * 24 = 144) to represent one hour and one day

    1.  Balance of Address
    2.  Number of Input Transactions (full history & up-to-date split)
    3.  Number of Output Transactions (full history & up-to-date split)
    4.  Amount ratio of input transactions to output transactions (full history & up to date split)
    5.  The lifetime of the address expressed in number of hours/days(144 blocks)
    6.  The activity hours/days(144 blocks).
    7.  The maximum number of hourly/daily transactions to/from the address.
    8.  The sum of all the values transferred to (resp. from) the address.
    9.  The average (resp. standard deviation) of the values transferred to/from the address.
    10. The number of different addresses which have transferred money to the address, and subsequently received money from it.
    11. The minimum (resp. maximum, average) delay between the time when the address has received some bitcoins, and the time it has sent some others.
    12. The maximum difference between the balance of the address in two consecutive days.
    '''
    # 0.0 Pre checking
    feat_dim=11

    # 0.1. Pre-process: Get Start-block and End-block / Set research period
    split_unit = 6. # 6: One Hour, 72: Half Day, 144: One Day
    input_tx_block_height = blksci_source_node.input_txes.block_height
    output_tx_block_height = blksci_source_node.output_txes.block_height
    start_block_height = np.min(output_tx_block_height)
    research_end_height = min(research_end_height, start_block_height+6000)

    if len(input_tx_block_height) != 0:
        end_block_height = min(max(np.max(input_tx_block_height), np.max(output_tx_block_height)), research_end_height)
    else:
        end_block_height = np.max(output_tx_block_height)
    sample_number = int(np.ceil((end_block_height-start_block_height)/split_unit))
    if sample_number<0: print(node_id); print(''); return 0

    # 0.2. Pre-process: Balance Dict construction
    balance_change_block_height = set(input_tx_block_height).union(set(output_tx_block_height), set([0]))
    dedup_balance_change_block_height = list(balance_change_block_height)
    dedup_balance_change_block_height.sort()
    dedup_balance_change_block_height.append(dedup_balance_change_block_height[-1]+split_unit)

    # 1. Loop for every sample index and prepare corresponding features
    feature_pad_map = np.zeros([sample_number, feat_dim])
    active_counter = 0
    max_hour_input_tx_num = 0
    max_hour_output_tx_num = 0

    all_input_tx_index_list = [input_tx.index for input_tx in blksci_source_node.input_txes]
    all_output_tx_index_list = [output_tx.index for output_tx in blksci_source_node.output_txes]
    all_input_tx_index_list.sort()
    all_output_tx_index_list.sort()

    # print('Sample Size : {}'.format(sample_number), end=' --- ')
    for sample_idx in range(sample_number):
        this_record_height = int(start_block_height + sample_idx*split_unit)
        pre_record_height = 0 if sample_idx == 0 else max(int(start_block_height + (sample_idx-1) * split_unit), start_block_height)

        # Skip some calculation if not tx happened in this period
        for height_item in dedup_balance_change_block_height:
            if height_item > pre_record_height and height_item <= this_record_height:
                no_change_flag = False
                break
            else: no_change_flag = True


        # AF1: Balance of Address
        if no_change_flag: feature_pad_map[sample_idx, 0] = feature_pad_map[sample_idx-1, 0]
        else: feature_pad_map[sample_idx, 0] = np.log10(1.+blksci_source_node.balance(this_record_height)/1000.)


        # AF2: Number of Input Transactions (full history & up to date split)
        if no_change_flag:
            feature_pad_map[sample_idx, 1] = feature_pad_map[sample_idx-1, 1]
            feature_pad_map[sample_idx, 2] = 0

        else:
            feature_pad_map[sample_idx, 1] = len(np.where(input_tx_block_height <= this_record_height)[0]) # Full history
            feature_pad_map[sample_idx, 2] = len(np.intersect1d(np.where(input_tx_block_height <= this_record_height)[0],
                                                                np.where(input_tx_block_height > pre_record_height)[0])) # up to date split
            if feature_pad_map[sample_idx, 2]>max_hour_input_tx_num: max_hour_input_tx_num = int(feature_pad_map[sample_idx, 2])


        # AF3: Number of Output Transactions (full history & up to date split)
        if no_change_flag:
            feature_pad_map[sample_idx, 3] = feature_pad_map[sample_idx-1, 3]
            feature_pad_map[sample_idx, 4] = 0
        else:
            feature_pad_map[sample_idx, 3] = len(np.where(output_tx_block_height <= this_record_height)[0]) # Full history
            feature_pad_map[sample_idx, 4] = len(np.intersect1d(np.where(output_tx_block_height <= this_record_height)[0],
                                                                np.where(output_tx_block_height > pre_record_height)[0])) # up to date split
            if feature_pad_map[sample_idx, 4]>max_hour_output_tx_num: max_hour_output_tx_num = int(feature_pad_map[sample_idx, 4])


        # AF4: Amount ratio of input transactions to output transactions (full history & up to date split)
        if no_change_flag:
            feature_pad_map[sample_idx, 5] = feature_pad_map[sample_idx-1, 5]
            feature_pad_map[sample_idx, 6] = 0
        else:
            feature_pad_map[sample_idx, 5] = feature_pad_map[sample_idx, 3] / (feature_pad_map[sample_idx, 1] + 1.)
            feature_pad_map[sample_idx, 6] = feature_pad_map[sample_idx, 4] / (feature_pad_map[sample_idx, 2] + 1.)


        # AF5: The lifetime of the address expressed in number of hours/days(144 blocks)
        if no_change_flag:
            feature_pad_map[sample_idx, 7] = feature_pad_map[sample_idx-1, 7]
        else:
            most_recent_height_idx = np.where(np.array(dedup_balance_change_block_height) <= this_record_height)[0][-1]
            most_recent_height = dedup_balance_change_block_height[most_recent_height_idx]
            feature_pad_map[sample_idx, 7] = ((most_recent_height - start_block_height) / float(split_unit))


        # AF6: The activity Ratio # hours/days(144 blocks).
        if feature_pad_map[sample_idx, 2] != 0 or feature_pad_map[sample_idx, 4] != 0:
            active_counter+=1
        feature_pad_map[sample_idx, 8] = active_counter/(sample_idx + 1.)

        # AF7: The maximum number of hourly/daily transactions to/from the address.
        feature_pad_map[sample_idx, 9] = max_hour_input_tx_num
        feature_pad_map[sample_idx, 10] = max_hour_output_tx_num




    # 3. Special Case, disposable address
    if feature_pad_map.shape[0] == 0:
        feature_pad_map = np.zeros([1, feat_dim])
        feature_pad_map[0, 0] = 0.0
        feature_pad_map[0, 1] = len(input_tx_block_height)
        feature_pad_map[0, 2] = len(input_tx_block_height)
        feature_pad_map[0, 3] = len(output_tx_block_height)
        feature_pad_map[0, 4] = len(output_tx_block_height)
        feature_pad_map[0, 5] = len(output_tx_block_height) / (len(input_tx_block_height) + 1)
        feature_pad_map[0, 6] = len(output_tx_block_height) / (len(input_tx_block_height) + 1)
        feature_pad_map[0, 7] = 0.0
        feature_pad_map[0, 8] = 1.0
        feature_pad_map[0, 9] = len(input_tx_block_height)
        feature_pad_map[0, 10] = len(output_tx_block_height)

    # 4. Save Result
    assert feature_pad_map.shape[0] >= 1
    assert feature_pad_map.shape[1] == 11
    np.save(save_path, feature_pad_map)


def feature_prepare_complement(main_blk_chain, blksci_source_node, load_path):
    '''
    Each feature should be prepared with time evolution for later analysis.
    Currently, Split whole life every 6 or 144 blocks.
    Since we can't know the life-span before it ends.
    Tip: Use 6 and 144 blocks (6 blocks/h * 24 = 144) to represent one hour and one day

    1.  Balance of Address
    2.  Number of Input Transactions (full history & up-to-date split)
    3.  Number of Output Transactions (full history & up-to-date split)
    4.  Amount ratio of input transactions to output transactions (full history & up-to-date split)
    5.  The lifetime of the address expressed in number of hours/days(144 blocks)
    6.  The activity hours/days(144 blocks).
    7.  The maximum number of hourly/daily transactions to/from the address.
    8.  The sum of all the values transferred to (resp. from) the address.
    9.  The average (resp. standard deviation) of the values transferred to/from the address.
    10. The number of different addresses which have transferred money to the address, and subsequently received money from it.
    11. The minimum (resp. maximum, average) delay between the time when the address has received some bitcoins, and the time it has sent some others.
    12. The maximum difference between the balance of the address in two consecutive days.
    '''
    # 0.0 Pre checking
    pre_account_feat = np.load(load_path)
    sample_number = pre_account_feat.shape[0]
    feat_dim = 5

    # 0.1. Pre-process: Get Start-block and End-block
    split_unit = 6. # 6: One Hour, 72: Half Day, 144: One Day
    input_tx_block_height = blksci_source_node.input_txes.block_height
    output_tx_block_height = blksci_source_node.output_txes.block_height
    start_block_height = np.min(output_tx_block_height)

    # 0.2. Pre-process: Balance Dict construction
    balance_change_block_height = set(input_tx_block_height).union(set(output_tx_block_height), set([0]))
    dedup_balance_change_block_height = list(balance_change_block_height)
    dedup_balance_change_block_height.sort()
    dedup_balance_change_block_height.append(dedup_balance_change_block_height[-1]+split_unit)

    # 1. Loop for every sample index and prepare corresponding features
    feature_pad_map = np.zeros([sample_number, feat_dim])

    all_input_tx_index_list = [input_tx.index for input_tx in blksci_source_node.input_txes]
    all_output_tx_index_list = [output_tx.index for output_tx in blksci_source_node.output_txes]
    all_input_tx_index_list.sort()
    all_output_tx_index_list.sort()

    AF_1_pin = 0
    AF_2_pin = 0
    AF_3_pin = 0
    AF_4_pin = 0
    Max_input_amt = 0
    Max_output_amt = 0

    print('Sample Size : {}'.format(sample_number), end=' / Feature Shape: ')
    for sample_idx in range(sample_number):
        this_record_height = int(start_block_height + sample_idx*split_unit)
        pre_record_height = 0 if sample_idx == 0 else max(int(start_block_height + (sample_idx-1) * split_unit), start_block_height)

        # Skip some calculation if not tx happened in this period
        for height_item in dedup_balance_change_block_height:
            if height_item > pre_record_height and height_item <= this_record_height:
                no_change_flag = False
                break
            else: no_change_flag = True


        # AF1,2: How many input/output tx with 0 amount till now
        if no_change_flag:
            feature_pad_map[sample_idx, 0] = feature_pad_map[sample_idx - 1, 0]
            feature_pad_map[sample_idx, 1] = feature_pad_map[sample_idx - 1, 1]
        else:
            # Count zero input tx
            for input_tx_idx in all_input_tx_index_list[AF_1_pin:]:
                this_input_tx = main_blk_chain.tx_with_index(input_tx_idx)
                if this_input_tx.block_height<=this_record_height:
                    if this_input_tx.input_value == 0: feature_pad_map[sample_idx, 0]+=1
                    AF_1_pin+=1
                else: break

            # Count zero output tx
            for output_tx_idx in all_output_tx_index_list[AF_2_pin:]:
                this_output_tx = main_blk_chain.tx_with_index(output_tx_idx)
                if this_output_tx.block_height<=this_record_height:
                    if this_output_tx.output_value == 0: feature_pad_map[sample_idx, 1]+=1
                    AF_2_pin+=1
                else: break

        # AF3,4,5: Max Input/Ouput amount Tx time and the time difference
        if no_change_flag:
            feature_pad_map[sample_idx, 2] = feature_pad_map[sample_idx - 1, 2]
            feature_pad_map[sample_idx, 3] = feature_pad_map[sample_idx - 1, 3]
            feature_pad_map[sample_idx, 4] = feature_pad_map[sample_idx - 1, 4]
        else:
            # Get the max input amt till now
            for input_tx_idx in all_input_tx_index_list[AF_3_pin:]:
                this_input_tx = main_blk_chain.tx_with_index(input_tx_idx)
                if this_input_tx.block_height<=this_record_height:
                    if this_input_tx.input_value > Max_input_amt: feature_pad_map[sample_idx, 2]=sample_idx+1
                    AF_3_pin+=1
                else: break

            # Get the max Output amt till now
            for output_tx_idx in all_output_tx_index_list[AF_4_pin:]:
                this_output_tx = main_blk_chain.tx_with_index(output_tx_idx)
                if this_output_tx.block_height<=this_record_height:
                    if this_output_tx.output_value > Max_output_amt: feature_pad_map[sample_idx, 3]=sample_idx+1
                    AF_4_pin+=1
                else: break

            feature_pad_map[sample_idx, 4] = feature_pad_map[sample_idx, 2] - feature_pad_map[sample_idx, 3]

    assert pre_account_feat.shape[1] == 11
    assert feature_pad_map.shape[1] == 5
    os.remove(load_path)
    complete_feat = np.concatenate([pre_account_feat, feature_pad_map], axis=-1)
    np.save(load_path.replace('bf_', ''), complete_feat)
    print(complete_feat.shape)



def address_feature_prepare(main_blk_chain, blksci_source_node, node_id, save_folder):
    save_path = save_folder + 'bf_address_feature_{}.npy'.format(node_id)
    feature_prepare(blksci_source_node, node_id, save_path, research_end_height=610637)
    feature_prepare_complement(main_blk_chain, blksci_source_node, save_path)