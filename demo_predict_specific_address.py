#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import model

from opts import parse_opt
from misc.loss_wrapper import LossWrapper
from path_related_tool_box.feat_processing_for_prediction import *
opt = parse_opt()



if __name__ == '__main__':
    # 1. Load specified address info
    tmp_save_path = './tmp_Path_Graph_data/'
    picked_node_id = 5841704807

    AF_feat_batch, fr_path_feat, bk_path_feat, \
    fr_intersect_edge, bk_intersect_edge, \
    fr_intersect_feat, bk_intersect_feat = feat_process_for_predict(tmp_save_path, picked_node_id)


    # 2. Load all models
    for illicit_name in ['Hack', 'Ransomware', 'Darknet']:
        print(f'\nUsing {illicit_name} Model')

        # Load Data
        opt.illicit_name = illicit_name
        opt.id = opt.illicit_name
        opt.batch_size = 1

        # Load Model
        model_core = model.setup(opt)
        dp_lw_model = LossWrapper(model_core, opt)
        dp_lw_model.load_state_dict(torch.load(os.path.join(opt.start_from, f'{opt.illicit_name}.pth')))
        dp_lw_model = dp_lw_model.cuda()
        dp_lw_model.eval()

        # Testing
        labels_holder = torch.ones([1,1], dtype=torch.float32).cuda()
        loss_amp_holder = torch.ones([1,1], dtype=torch.float32).cuda()
        with torch.no_grad():
            out, each_split_prediction, ground_labels = \
                dp_lw_model([AF_feat_batch, fr_path_feat, bk_path_feat,
                             fr_intersect_edge, bk_intersect_edge, fr_intersect_feat, bk_intersect_feat],
                             labels_holder, loss_amp_holder, return_pred_switch=True, eval_switch=False)
        prediction = each_split_prediction.detach().cpu().tolist()[0]
        for hour_idx in range(23):
            print(f'Hour {hour_idx+1} / Type {illicit_name} : Probability : {round(float(prediction[hour_idx]), 2)}')

        # Clean up
        del dp_lw_model, model_core