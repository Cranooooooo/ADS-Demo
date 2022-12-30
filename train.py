#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torch
import json
import os
import opts
import model
import time
from misc.loss_wrapper import LossWrapper
from dataloader import DataLoader
from misc import utils
from misc import eval_utils

import warnings
warnings.filterwarnings('always')
try: import tensorboardX as tb
except ImportError: print("tensorboardX is not installed"); tb = None


def add_summary_value(writer, key, value, iteration):
    if writer: writer.add_scalar(key, value, iteration)

def save_checkpoint(model, infos, optimizer, histories=None, model_name='model_1'):
    # if checkpoint_path doesn't exist
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    checkpoint_path = os.path.join(opt.checkpoint_path, 'model_{}.pth'.format(model_name))
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer_{}.pth'.format(model_name))
    torch.save(optimizer.state_dict(), optimizer_path)
    with open(os.path.join(opt.checkpoint_path, 'infos_{}.pkl'.format(model_name)), 'wb') as f:
        utils.pickle_dump(infos, f)
    if histories:
        with open(os.path.join(opt.checkpoint_path, 'histories_{}.pkl'.format(model_name)), 'wb') as f:
            utils.pickle_dump(histories, f)

def train(opt):
    # 1. Load former Info and check all necessary things
    split_loader = DataLoader(opt)
    infos = {}
    histories = {}

    infos['iter'] = 0
    infos['epoch'] = 0
    infos['iterators'] = split_loader.iterators
    infos['split_ix'] = split_loader.split_ix
    infos['opt'] = opt
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    split_loader.iterators = infos.get('iterators', split_loader.iterators)
    split_loader.split_ix = infos.get('split_ix', split_loader.split_ix)
    if opt.load_best_score == 1:
        best_val_test_score = infos.get('best_val_test_score', None)


    # 2. Model and Optimizer Setting
    model_core = model.setup(opt)
    dp_lw_model = LossWrapper(model_core, opt)
    dp_lw_model.load_state_dict(torch.load(os.path.join(opt.start_from, f'{opt.illicit_name}.pth')))
    dp_lw_model = dp_lw_model.cuda()

    epoch_done = True
    dp_lw_model.train() # Assure in training mode
    optimizer = utils.build_optimizer(dp_lw_model.parameters(), opt)
    optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    if opt.start_from != None:
        prev_optmizer_path = opt.start_from+f'optimizer_{opt.id}_best.pth'
        if os.path.isfile(prev_optmizer_path): print('Load Done')
        else: print(opt.start_from, prev_optmizer_path, os.path.isfile(prev_optmizer_path))

    # 3. Training Start
    each_split_epoch_done = []; counter = 0
    while epoch < opt.max_epoch:
        start = time.time()
        if epoch_done:
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = max(0, int((epoch - opt.learning_rate_decay_start)//2.))
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
            epoch_done = False

        if (opt.use_warmup == 1) and (iteration < opt.noamopt_warmup):
            opt.current_lr = opt.learning_rate * (iteration + 1) / opt.noamopt_warmup
            utils.set_lr(optimizer, opt.current_lr)

        # Load data from train split
        optimizer.zero_grad()
        data = split_loader.get_batch_all_tag('train')
        counter += len(data['AF_feat'][0])

        AF_feat = data['AF_feat']
        fr_path_feat = data['fr_path_feat']
        bk_path_feat = data['bk_path_feat']
        fr_intersect_edge_batch = data['fr_intersect_edge_batch']
        bk_intersect_edge_batch = data['bk_intersect_edge_batch']
        fr_intersect_feat_batch = data['fr_intersect_feat_batch']
        bk_intersect_feat_batch = data['bk_intersect_feat_batch']

        labels = torch.reshape(torch.tensor(data['labels'], dtype=torch.float32),[-1, 1]).cuda()
        loss_amplify = torch.reshape(torch.tensor(data['loss_amplify'], dtype=torch.float32),[-1, 1]).cuda()
        model_out = dp_lw_model([AF_feat, fr_path_feat, bk_path_feat, fr_intersect_edge_batch, bk_intersect_edge_batch,
                                 fr_intersect_feat_batch, bk_intersect_feat_batch], labels, loss_amplify,
                                 iter=iteration)

        loss = model_out['total_loss'].mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        train_loss = loss.item()
        end = time.time()
        if iteration%30==0:
            print(f"iter {iteration} (epoch {epoch}), loss amp: {round(split_loader.loss_amplify, 2)}, sample_num {len(AF_feat[0])}, loss_total = {train_loss}, time/batch = {end - start}")
            avg_acc = 0
            avg_prec = 0
            avg_rec = 0
            for split_num in range(opt.split_number):
                print(f'Hour {split_num} --- ', end=' ')
                for metric_name, metric_score in model_out['ED_split_number_{}_metrics'.format(split_num)].items():
                    print('{}: {} ;'.format(metric_name, round(metric_score, 4)), end=' ')
                    avg_acc  += metric_score if metric_name == 'Accuracy' else 0
                    avg_prec += metric_score if metric_name == 'Precision' else 0
                    avg_rec += metric_score if metric_name == 'Recall' else 0
                print('Loss: {} '.format(round(float(model_out['ED_split_number_{}_loss'.format(split_num)]), 4)))

            print('Avg Precision is {}'.format(round(avg_prec /  opt.split_number, 4)))
            print('Avg Recall is {}'.format(round(avg_rec /  opt.split_number, 4)))
            print('Consistency Score is {}'.format(model_out['Consistent_F1']))
            print('Early weighted F1 is {}'.format(model_out['Early_weighted_F1']))


        # Update the iteration and epoch
        iteration += 1
        for split_name, split_warpped in data['bounds']['wrapped'].items():
            if True == split_warpped and [split_name, split_warpped] not in each_split_epoch_done:
                each_split_epoch_done.append([split_name, split_warpped])
                if len(each_split_epoch_done) == 2:
                    epoch += 1
                    epoch_done = True
                    each_split_epoch_done = []

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if opt.noamopt: opt.current_lr = optimizer.rate()
            elif opt.reduce_on_plateau: opt.current_lr = optimizer.current_lr
            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr

        # update infos
        infos['iter'] = iteration
        infos['epoch'] = epoch
        infos['iterators'] = split_loader.iterators
        infos['split_ix'] = split_loader.split_ix


        # make evaluation on validation set, and save model
        if epoch_done:
            # eval model
            val_loss, val_metrics_score = eval_utils.eval_split(dp_lw_model, split_loader, split ='val')
            test_loss, test_metrics_score = eval_utils.eval_split(dp_lw_model, split_loader, split ='test')
            if opt.reduce_on_plateau: optimizer.scheduler_step(val_loss)

            # Save model if is improving on validation result
            current_score = test_metrics_score['Consistent_F1'] * test_metrics_score['Early_weighted_F1']
            best_flag = False
            if best_val_test_score is None or current_score > best_val_test_score:
                best_perfor_each_model[opt.model_index] = [val_metrics_score, test_metrics_score]
                best_val_test_score = current_score
                best_flag = True

            # Dump misc information
            infos['best_val_test_score'] = best_val_test_score
            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history

            if opt.save_history_ckpt: save_checkpoint(dp_lw_model, infos, optimizer, model_name=f'{opt.id}')
            if best_flag: save_checkpoint(dp_lw_model, infos, optimizer, model_name=f'{opt.id}_best')


if __name__ == '__main__':
    opt = opts.parse_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt.id = opt.illicit_name
    best_perfor_each_model = {}
    train(opt)


