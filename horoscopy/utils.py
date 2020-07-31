#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def asarray(a):
    """Convert array-like input to numpy array.

    This function is implemented referring to scipy.

    Parameters
    ----------
    a : array_like
        Array-like input.


    Returns
    -------
    a : np.ndarray
        Converted validated array.

    """

    if type(a) is np.ndarray:
        return a

    a = np.asarray(a)
    if a.dtype is np.dtype('O'):
        raise ValueError('object arrays are not supported')

    return a


def check_alpha(alpha):
    """Check whether given alpha is valid or not.

    Parameters
    ----------
    alpha : float [scalar]
        Frequency warping factor.

    """

    if np.abs(alpha) >= 1.0:
        raise ValueError('|alpha| must be less than 1.0')
