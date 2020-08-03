#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Takenori Yoshimura
# Licensed under the MIT license

import librosa
import numpy as np

import horoscopy
from horoscopy.utils import read_binary

from .utils import get_data


def test_stft_to_mcep(wav_file=get_data('example.wav'),
                      mcep_file=get_data('example.mcep.from.sptk'),
                      n_fft=512, hop_length=80, win_length=400,
                      win_func='blackman', order=24):
    y, sr = librosa.load(wav_file, sr=None)
    y *= 32768
    S = np.abs(librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=horoscopy.window.get_sptk_window(win_func, win_length),
        pad_mode='constant'))
    actual = horoscopy.stft_to_mcep(S, M=order)

    target = read_binary(mcep_file)
    target = np.transpose(np.reshape(target, (-1, order + 1)))
    np.testing.assert_array_almost_equal(actual, target)
