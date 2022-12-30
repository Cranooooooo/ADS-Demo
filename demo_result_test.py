#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import model
import os

from dataloader import DataLoader
from misc import eval_utils
from opts import parse_opt
from misc.loss_wrapper import LossWrapper
opt = parse_opt()


if __name__ == '__main__':

    for illicit_name in ['Darknet', 'Hack', 'Ransomware']:
        # Load Data
        opt.illicit_name = illicit_name
        opt.id = opt.illicit_name
        split_loader = DataLoader(opt)

        # Load Model
        model_core = model.setup(opt)
        dp_lw_model = LossWrapper(model_core, opt)
        dp_lw_model.load_state_dict(torch.load(os.path.join(opt.start_from, f'{opt.illicit_name}.pth')))
        dp_lw_model = dp_lw_model.cuda()

        # Testing
        test_loss, test_metrics_score = eval_utils.eval_split(dp_lw_model, split_loader, split='test')

        # Clean up
        del dp_lw_model, model_core