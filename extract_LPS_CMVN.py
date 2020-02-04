#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/17/2019 3:33 PM
# @Author  : HOU NANA
# @Site    : http://github.com/nanahou
# @File    : extract_LPS_CMVN.py


'''
    This file is for extracting power_d, dynamic_c, and dynamic_d and then computing cmvn using own data set
    not all because there is no big difference.
'''
import numpy as np
import torch
import pickle
import time
import os
import sys
sys.path.append('/home/hounana/pytorch/enhancement/')
from scripts.comput_feature import *
from scripts.cmvn import comp_cmvn4tensor_varylen
from scripts.audioread import audioread
from scripts.normhamming import normhamming
from scripts.plot_spectrum import *
from scripts.sigproc import *


def get_power_spec(filename, fft_len, frame_shift):
    # 1st process clean dataset
    rate, sig, nbits = audioread(filename)
    frames = framesig(sig, fft_len, frame_shift, lambda x: normhamming(x), True)
    # 2nd get power
    power_spec = powspec(frames, fft_len)
    power_spec =  np.absolute(power_spec)
    return power_spec

# ---------------------extract training 8k, 16k pairs files -------------------------------
def main():
    t_start = time.time()

    thred = -4
    fft_len_16k, frame_shift_16k = 512, 256
    fft_len_8k, frame_shift_8k = 256, 128
    data_path_16k = '/data/disk3/hounana/Valentini-Botinhao_1s/16k/clean_testset_wav_1s/'
    data_path_8k = '/data/disk3/hounana/Valentini-Botinhao_1s/8k/clean_testset_wav_1s/'

    LPS_path_16k = '/data/disk3/hounana/Valentini-Botinhao_1s/LPS/clean_testset_wav_1s_LPS_16k/'
    LPS_path_8k = '/data/disk3/hounana/Valentini-Botinhao_1s/LPS/clean_testset_wav_1s_LPS_8k/'

    cmvn_path_16k = '/data/disk3/hounana/Valentini-Botinhao_1s/CMVN/clean_testset_wav_1s_16k/'
    cmvn_path_8k = '/data/disk3/hounana/Valentini-Botinhao_1s/CMVN/clean_testset_wav_1s_8k/'

    save_path_mean_16k = '/data/disk3/hounana/Valentini-Botinhao_1s/tt_mean_16k.pkl'
    save_path_std_16k = '/data/disk3/hounana/Valentini-Botinhao_1s/tt_std_16k.pkl'
    save_path_mean_8k = '/data/disk3/hounana/Valentini-Botinhao_1s/tt_mean_8k.pkl'
    save_path_std_8k = '/data/disk3/hounana/Valentini-Botinhao_1s/tt_std_8k.pkl'

    for dir in [LPS_path_16k, LPS_path_8k, cmvn_path_16k, cmvn_path_8k]:
        if not os.path.exists(dir):
            os.makedirs(dir)


    data_list = [x for x in os.listdir(data_path_16k) if x.endswith(".wav")]
    # data_list = data_list[0:5]

    count = 1.0
    for item in data_list:
        item_8k = data_path_8k + item
        item_16k = data_path_16k + item

        file_lps_16k = LPS_path_16k + item[:thred] + '.pkl'
        file_lps_8k = LPS_path_8k + item[:thred] + '.pkl'

        # extract magnitude and power
        power_16k = get_power_spec(item_16k, fft_len_16k, frame_shift_16k)
        power_16k = torch.from_numpy(power_16k.astype(np.float)).float()

        power_8k = get_power_spec(item_8k, fft_len_8k, frame_shift_8k)
        power_8k = torch.from_numpy(power_8k.astype(np.float)).float()

        log_16k = torch.log(power_16k)
        log_8k = torch.log(power_8k)
        # print(log_16k.size())
        # print(log_8k.size())

        with open(file_lps_16k, 'wb') as out_dynamic_c:
            pickle.dump(log_16k, out_dynamic_c, True)
        with open(file_lps_8k, 'wb') as out_dynamic_d:
            pickle.dump(log_8k, out_dynamic_d, True)

        if count % 1000 == 0:
            print('get features: [{}/{} ({:.0f}%)]'.format(count, len(data_list), 100. * count / len(data_list)))
        count = count + 1

    # print(noisy_list)
    comp_cmvn4tensor_varylen(data_list, LPS_path_16k, 257, cmvn_path_16k, save_path_mean_16k, save_path_std_16k)
    comp_cmvn4tensor_varylen(data_list, LPS_path_8k, 129, cmvn_path_8k, save_path_mean_8k, save_path_std_8k)

    print('consuming: %f hours' % ((time.time() - t_start) / 3600))


# ---------------------extract test files -------------------------------
# def main():
#     t_start = time.time()
#
#     thred = -4
#     fft_len_8k, frame_shift_8k = 256, 128
#     data_path_8k = '/data/disk3/hounana/Valentini-Botinhao_1s/8k/clean_testset_wav/'
#
#     LPS_path_8k = '/data/disk3/hounana/Valentini-Botinhao_1s/LPS/clean_testset_wav_8k/'
#     cmvn_path_8k = '/data/disk3/hounana/Valentini-Botinhao_1s/CMVN/clean_testset_wav_8k/'
#
#     for dir in [LPS_path_8k, cmvn_path_8k]:
#         if not os.path.exists(dir):
#             os.makedirs(dir)
#
#     data_list = [x for x in os.listdir(data_path_8k) if x.endswith(".wav")]
#     # data_list = data_list[0:5]
#
#     count = 1.0
#     for item in data_list:
#         item_8k = data_path_8k + item
#
#         file_lps_8k = LPS_path_8k + item[:thred] + '.pkl'
#
#         # extract magnitude and power
#         power_8k = get_power_spec(item_8k, fft_len_8k, frame_shift_8k)
#         power_8k = torch.from_numpy(power_8k.astype(np.float)).float()
#
#         log_8k = torch.log(power_8k)
#         # print(log_16k.size())
#         # print(log_8k.size())
#
#         with open(file_lps_8k, 'wb') as out_dynamic_d:
#             pickle.dump(log_8k, out_dynamic_d, True)
#
#         if count % 1000 == 0:
#             print('get features: [{}/{} ({:.0f}%)]'.format(count, len(data_list), 100. * count / len(data_list)))
#         count = count + 1
#
#     # print(noisy_list)
#     comp_cmvn4tensor_varylen(data_list, LPS_path_8k, 129, cmvn_path_8k)
#
#     print('consuming: %f hours' % ((time.time() - t_start) / 3600))


if __name__ == '__main__':
    main()
