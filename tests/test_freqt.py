#!/usr/bin/env python
# -*- coding: utf-8 -*-

import horoscopy
import numpy as np


def test_identity(m=4):
    c = np.random.rand(m + 1)
    g = horoscopy.freqt(c, M=m, alpha=0)
    np.testing.assert_array_almost_equal(c, g)


def test_reversible(m=4, M=20, a=0.42):
    c = np.random.rand(m + 1)
    g = horoscopy.freqt(c, M=M, alpha=a)
    c2 = horoscopy.freqt(g, M=m, alpha=-a)
    np.testing.assert_array_almost_equal(c, c2)
