#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Takenori Yoshimura
# Licensed under the MIT license

import numpy as np

import horoscopy


np.random.seed(12345)


def test_identity(m=4):
    c = np.random.rand(m + 1)
    g = horoscopy.freqt(c, M=m, alpha=0)
    np.testing.assert_array_almost_equal(c, g)


def test_reversible(m=4, M=20, a=0.42):
    c = np.random.rand(m + 1)
    g = horoscopy.freqt(c, M=M, alpha=a)
    c2 = horoscopy.freqt(g, M=m, alpha=-a)
    np.testing.assert_array_almost_equal(c, c2)


def test_zeroth_order(m=0, M=0, a=0.1):
    c = np.random.rand(m + 1)
    g = horoscopy.freqt(c, M=m, alpha=a, recursive=False)
    np.testing.assert_array_almost_equal(c, g)


def test_matrix_input(m=4, M=4, a=0.42):
    c = np.random.rand(m + 1)
    g = horoscopy.freqt(c, M=M, alpha=a)
    c2 = np.expand_dims(c, axis=-1)
    g2 = horoscopy.freqt(c2, M=M, alpha=a)[:, 0]
    np.testing.assert_array_almost_equal(g, g2)
