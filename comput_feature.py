#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/2/18 5:04 PM
# @Author  : HOU NANA
# @Site    : http://github.com/nanahou
# @File    : comput_feature.py


from global_tools.audioread import audioread
from global_tools.normhamming import normhamming
from global_tools.plot_spectrum import *
from global_tools.sigproc import *
import numpy as np


def get_power_spec(filename, fft_len, frame_shift):
    # process clean dataset
    rate, sig = audioread(filename)
    frames = framesig(sig, fft_len, frame_shift, lambda x: normhamming(x), True)
    # 2nd get power
    power_spec = powspec(frames, fft_len)
    power_spec =  np.absolute(power_spec)
    return power_spec


def comp_delta4tensor(feature, DELTAWINDOW, n_frame):
    temp1 = np.zeros((1, DELTAWINDOW), dtype=np.int)
    temp2 = np.arange(0, n_frame, 1)
    temp3 = np.ones((1, DELTAWINDOW), dtype=np.int)*(n_frame-1)
    idx = np.append(temp1, temp2)
    idx = np.append(idx, temp3)
    # for test, for train not sure
    static_coef = feature[idx.tolist(), :].clone()
    delta_coef = 0
    i = DELTAWINDOW + 1
    denom = np.sum(np.square(np.arange(1, DELTAWINDOW + 1, 1))) * 2.0
    for j in range(1, DELTAWINDOW+1):
        delta_coef = delta_coef + torch.mul((torch.add(static_coef[i+j-1:i+j+n_frame-1, :].clone(), (-1.0)*static_coef[i-j-1:i-j+n_frame-1, :].clone())), (j / denom))
    return delta_coef


def comp_dynamic_features(log_spec):
    DELTAWINDOW = 2
    n_frame, d_frame = log_spec.shape
    feature = log_spec[:, :]
    delta = comp_delta4tensor(feature, DELTAWINDOW, n_frame)
    acc = comp_delta4tensor(delta, DELTAWINDOW, n_frame)
    delta_feature = torch.cat((log_spec, delta), dim=1)
    acc_feature = torch.cat((log_spec, delta, acc), dim=1)
    return delta_feature, acc_feature


def comp_dynamic_features_4batch(log_spec):
    DELTAWINDOW = 2
    batch_size, n_frame, d_frame = log_spec.shape
    dynamic_list = []
    for idx in range(0, batch_size):
        feature = log_spec[idx, :, :]
        delta = comp_delta4tensor(feature, DELTAWINDOW, n_frame)
        acc = comp_delta4tensor(delta, DELTAWINDOW, n_frame)

        dynamic_feature = torch.cat((feature, delta, acc), dim=1)

        dynamic_list.append(dynamic_feature)

    dynamic_list = torch.stack(dynamic_list)

    return dynamic_list


def fill_const(data, batch_lens):
    data_tmp = data.clone()

    i = 0
    for idx in batch_lens:
        if idx == batch_lens[0]:
            i += 1
        else:
            data_tmp[i, idx:, :] = 0.01
            i += 1

    return data_tmp


