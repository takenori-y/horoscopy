#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def _asarray(a, as_matrix=False):
    """Convert array-like input to numpy array.

    Parameters
    ----------
    a : array_like
        Array-like input.

    as_matrix : bool
        If true and input is vector, append an additional axis to the input.

    Returns
    -------
    a : np.ndarray
        Converted validated array.

    Notes
    -----
    This function is implemented referring to scipy.

    """

    a = np.asarray(a)
    if a.dtype is np.dtype('O'):
        raise ValueError('object arrays are not supported')
    if as_matrix and a.ndim == 1:
        a = np.expand_dims(a, axis=-1)

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
