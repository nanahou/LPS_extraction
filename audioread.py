#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16/11/17
# @Author  : NANA HOU
# @Site    : https://github.com/nanahou
# @File    : audioread.py


import scipy.io.wavfile as wav
import numpy as np


def audioread(filename):
    (rate, sig) = wav.read(filename)
    # print('just read:', sig)
    if sig.dtype == 'int16':
        nb_bits = 16
    elif sig.dtype == 'int32':
        nb_bits = 32
    else:
        print('no type match!', sig.dtype)
    
    max_nb_bit = float(2 ** (nb_bits - 1))
    sig = sig / (max_nb_bit + 1.0)
    return rate, sig, nb_bits
