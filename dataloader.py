from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import random
import torch
import torch.utils.data as data
import atexit

class DataLoader(data.Dataset):
    def __init__(self, opt):
        # Set Dimension
        self.opt = opt
        self.batch_size = opt.batch_size
        self.split_number = opt.split_number
        self.AF_dim = opt.AF_dim
        self.path_feat_dim = opt.path_feat_dim
        self.intersect_dim = opt.intersect_dim

        # Setting information -- Illicit Type and Corresponding Address idx
        self.illicit_name = opt.illicit_name
        self.analysis_span = opt.split_number
        self.prepare_data_path()
        self.prepare_name_list()
        self.prepare_addr_split()

        # Positive / Negative ratio
        self.loss_amplify = 5
        self.train_ratio_list = [1, 5]
        self.test_ratio_list = [1, int(self.nega_num/self.posi_num)]
        self.used_ratio_list = self.train_ratio_list

    def prepare_data_path(self):
        # 1. Address Label info
        self.addr_name_and_tag_path = './misc/addr_name_and_tag.json'
        self.this_illicit_posi_addr = f'./misc/{self.illicit_name}_posi_list.npy'
        self.this_illicit_nega_addr = f'./misc/{self.illicit_name}_nega_list.npy'

        # 2. Pre-Processing Save Folder --> prsd means after pre-processing
        self.pre_process_data_path = './pre_process_data/'
        self.prsd_AF_folder = self.pre_process_data_path + 'Account_feature/'
        self.prsd_path_feature_folder = self.pre_process_data_path + f'Path_feature/'
        self.prsd_path_intersect_folder = self.pre_process_data_path + f'Intersect_feature/'


    def prepare_name_list(self):
        self.posi_addr_idx = np.load(self.this_illicit_posi_addr).tolist()
        self.nega_addr_idx = np.load(self.this_illicit_nega_addr).tolist()
        self.posi_num = len(self.posi_addr_idx); random.shuffle(self.posi_addr_idx)
        self.nega_num = len(self.nega_addr_idx); random.shuffle(self.nega_addr_idx)
        print(f'Type: {self.illicit_name} --- Positive: {self.posi_num} / Negative: {self.nega_num}')

    def prepare_addr_split(self):
        # Separate out indexes for each of the provided splits
        self.ix_to_label = {}
        for i in self.posi_addr_idx: self.ix_to_label[str(i)] = 1.
        for i in self.nega_addr_idx: self.ix_to_label[str(i)] = 0.
        self.split_ix = {'posi_train':[], 'posi_val':[], 'posi_test':[],
                         'nega_train':[], 'nega_val':[], 'nega_test':[]}
        self.total_address_num = self.posi_num + self.nega_num

        # Positive Split --- Train / Val / Test
        train_posi_cut = int(0.8*self.posi_num)
        val_posi_cut = train_posi_cut + int(0.1*self.posi_num)
        self.split_ix['posi_train'] += self.posi_addr_idx[:train_posi_cut]
        self.split_ix['posi_val'] += self.posi_addr_idx[train_posi_cut:val_posi_cut]
        self.split_ix['posi_test'] += self.posi_addr_idx[val_posi_cut:]

        # Negative Split --- Train / Val / Test
        train_nega_cut = int(0.8*self.nega_num)
        val_nega_cut = train_nega_cut + int(0.1*self.nega_num)
        self.split_ix['nega_train'] += self.nega_addr_idx[:train_nega_cut]
        self.split_ix['nega_val'] += self.nega_addr_idx[train_nega_cut:val_nega_cut]
        self.split_ix['nega_test'] += self.nega_addr_idx[val_nega_cut:]

        # Split Assignment
        self.iterators = {}
        self._prefetch_process = {}
        for split in ['posi_train', 'posi_val', 'posi_test', 'nega_train', 'nega_val', 'nega_test']:
            self.iterators[split] = 0
            self._prefetch_process[split] = BlobFetcher(split, self, 'train' in split)

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys(): del self._prefetch_process[split]
        atexit.register(cleanup)


    def get_batch_for_ED(self, split, rect_bsz):
        label_batch = []
        loss_amplify_batch = []
        AF_feat_batch = [[0]*rect_bsz]*self.split_number
        fr_feat_batch = [[0]*rect_bsz]*self.split_number
        bk_feat_batch = [[0]*rect_bsz]*self.split_number
        fr_intersect_edge_batch = [[0]*rect_bsz]*self.split_number
        bk_intersect_edge_batch = [[0]*rect_bsz]*self.split_number
        fr_intersect_feat_batch = [[0]*rect_bsz]*self.split_number
        bk_intersect_feat_batch = [[0]*rect_bsz]*self.split_number
        wrapped = False

        for batch_idx in range(rect_bsz):
            # fetch path feature
            tmp_AF_path_feat, tmp_fr_path_feat, tmp_bk_path_feat, \
            tmp_fr_intersect_edge, tmp_bk_intersect_edge, tmp_fr_intersect_feat, tmp_bk_intersect_feat, \
            ix, tmp_wrapped = self._prefetch_process[split].get()

            for i in range(self.split_number):
                AF_feat_batch[i][batch_idx] = np.array(tmp_AF_path_feat[i])
                fr_feat_batch[i][batch_idx] = np.array(tmp_fr_path_feat[i])
                bk_feat_batch[i][batch_idx] = np.array(tmp_bk_path_feat[i])
                fr_intersect_edge_batch[i][batch_idx] = np.array(tmp_fr_intersect_edge[i])
                bk_intersect_edge_batch[i][batch_idx] = np.array(tmp_bk_intersect_edge[i])
                fr_intersect_feat_batch[i][batch_idx] = np.array(tmp_fr_intersect_feat[i])
                bk_intersect_feat_batch[i][batch_idx] = np.array(tmp_bk_intersect_feat[i])


            # label and Loss amplifier
            label_batch.append(self.ix_to_label[str(ix)])
            if int(self.ix_to_label[str(ix)]) == 1: loss_amplify_batch.append(self.loss_amplify)
            else: loss_amplify_batch.append(1.)
            if tmp_wrapped: wrapped = True

        data = {}
        data['AF_feat'] = AF_feat_batch
        data['fr_path_feat'] = fr_feat_batch
        data['bk_path_feat'] = bk_feat_batch
        data['fr_intersect_edge_batch'] = fr_intersect_edge_batch
        data['bk_intersect_edge_batch'] = bk_intersect_edge_batch
        data['fr_intersect_feat_batch'] = fr_intersect_feat_batch
        data['bk_intersect_feat_batch'] = bk_intersect_feat_batch
        data['labels'] = label_batch
        data['loss_amplify'] = loss_amplify_batch

        return data, wrapped


    def get_batch_all_tag(self, macro_split):
        '''
        :param macro_split: train, val, test
        :param batch_size:
        :return:
        '''
        # Case 1. Keep the ratio
        # if keep_ratio:
        label_batch = []
        loss_amplify_batch = []
        AF_feat_batch = [[]] * self.split_number
        fr_feat_batch = [[]] * self.split_number
        bk_feat_batch = [[]] * self.split_number
        fr_intersect_edge_batch = [[]] * self.split_number
        bk_intersect_edge_batch = [[]] * self.split_number
        fr_intersect_feat_batch = [[]] * self.split_number
        bk_intersect_feat_batch = [[]] * self.split_number

        it_pos_now = {}
        it_max = {}
        wrapped_for_each_split = {}

        for tag_item in ['posi', 'nega']:
            micro_split = '{}_{}'.format(tag_item, macro_split)
            it_pos_now[micro_split] = self.iterators[micro_split]
            it_max[micro_split] = self.iterators[micro_split]
            wrapped_for_each_split[micro_split] = False

            amp_ratio = self.used_ratio_list[0] if tag_item == 'posi' else self.used_ratio_list[1]
            rect_bsz = int(amp_ratio * self.batch_size)
            this_data, wrapped = self.get_batch_for_ED(micro_split, rect_bsz)
            wrapped_for_each_split[micro_split] = wrapped

            # add to whole data batch
            label_batch += this_data['labels']
            loss_amplify_batch += this_data['loss_amplify']
            for i in range(self.split_number):
                AF_feat_batch[i] = AF_feat_batch[i] + this_data['AF_feat'][i]
                fr_feat_batch[i] = fr_feat_batch[i] + this_data['fr_path_feat'][i]
                bk_feat_batch[i] = bk_feat_batch[i] + this_data['bk_path_feat'][i]
                fr_intersect_edge_batch[i] = fr_intersect_edge_batch[i] + this_data['fr_intersect_edge_batch'][i]
                bk_intersect_edge_batch[i] = bk_intersect_edge_batch[i] + this_data['bk_intersect_edge_batch'][i]
                fr_intersect_feat_batch[i] = fr_intersect_feat_batch[i] + this_data['fr_intersect_feat_batch'][i]
                bk_intersect_feat_batch[i] = bk_intersect_feat_batch[i] + this_data['bk_intersect_feat_batch'][i]

            # Check
            assert len(label_batch) == len(AF_feat_batch[0]) == len(fr_feat_batch[0]) == len(bk_feat_batch[0]) ==\
                   len(fr_intersect_feat_batch[0]) == len(bk_intersect_feat_batch[0])

        return {'AF_feat': AF_feat_batch, 'fr_path_feat': fr_feat_batch, 'bk_path_feat': bk_feat_batch,
                'fr_intersect_edge_batch': fr_intersect_edge_batch, 'fr_intersect_feat_batch': fr_intersect_feat_batch,
                'bk_intersect_edge_batch': bk_intersect_edge_batch, 'bk_intersect_feat_batch': bk_intersect_feat_batch,
                'labels': label_batch, 'loss_amplify': loss_amplify_batch,
                'bounds':{'it_pos_now': it_pos_now, 'it_max': it_max, 'wrapped': wrapped_for_each_split}}

    def __getitem__(self, idx):
        """This function returns a tuple that is further passed to collate_fn
        """
        file_name = f'{idx}.npy'
        file_name_list = [self.prsd_AF_folder + file_name,
                          self.prsd_path_feature_folder + file_name.replace('.npy', '_fr.npy'),
                          self.prsd_path_feature_folder + file_name.replace('.npy', '_bk.npy'),
                          self.prsd_path_intersect_folder + file_name.replace('.npy', '_fr_edge.npy'),
                          self.prsd_path_intersect_folder + file_name.replace('.npy', '_bk_edge.npy'),
                          self.prsd_path_intersect_folder + file_name.replace('.npy', '_fr_feat.npy'),
                          self.prsd_path_intersect_folder + file_name.replace('.npy', '_bk_feat.npy')]


        AF_feat_batch = np.load(file_name_list[0])
        fr_path_feat = np.load(file_name_list[1])
        bk_path_feat = np.load(file_name_list[2])

        fr_intersect_edge = np.load(file_name_list[3], allow_pickle=True)
        bk_intersect_edge = np.load(file_name_list[4], allow_pickle=True)
        fr_intersect_feat = np.load(file_name_list[5], allow_pickle=True)
        bk_intersect_feat = np.load(file_name_list[6], allow_pickle=True)

        return AF_feat_batch, fr_path_feat, bk_path_feat, \
               fr_intersect_edge, bk_intersect_edge, fr_intersect_feat, bk_intersect_feat, idx

    def __len__(self):
        return len(self.num_address)

    def reset_iterator(self, macro_split):
        for split in self.iterators.keys():
            if macro_split in split:
                del self._prefetch_process[split]
                self._prefetch_process[split] = \
                    BlobFetcher(split, self, split in ['{}_train'.format(tag) for tag in ['posi', 'nega']])
                self.iterators[split] = 0


class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=4, # 4 is usually enough
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            # print(self.split, 'Next Epoch')
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped: self.reset()

        # SIMAO
        assert tmp[-1] == ix, "ix not equal"
        return (tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[-1], wrapped)

