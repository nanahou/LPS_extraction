#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017  Chenglin Xu

"""
normalized square root hamming periodic window
"""

import numpy
from scipy.signal import hamming


def normhamming(fft_len):
    if fft_len == 512:
        frame_shift = 160
    elif fft_len == 256:
        frame_shift = 128
    else:
        print("Wrong fft_len, current only support 16k/8k sampling rate wav")
        exit(1)
    win = numpy.sqrt(hamming(fft_len, False))
    win = win/numpy.sqrt(numpy.sum(numpy.power(win[0:fft_len:frame_shift],2)))
    return win
