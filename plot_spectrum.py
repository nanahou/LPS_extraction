#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/1/18 11:50 AM
# @Author  : HOU NANA
# @Site    : http://github.com/nanahou
# @File    : plot_spectrum.py

import torch
import matplotlib.pyplot as plt


def plot_spectrum(d_spectrum, name):
    plt.interactive(False)
    color_map = plt.get_cmap('jet')
    plt.figure()
    d_spectrum = torch.squeeze(d_spectrum)
    print(d_spectrum.size())
    d_spectrum = d_spectrum.data.cpu().numpy()
    plt.imshow(d_spectrum, cmap=color_map)
    # plt.imshow(d_spectrum)
    plt.title(str(name))
    plt.show(block=True)
    input("Press Enter to exit..")
    plt.close('all')

