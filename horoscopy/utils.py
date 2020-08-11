#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Takenori Yoshimura
# Licensed under the MIT license

import os
import struct

import numpy as np


def _asarray(a):
    """Convert array-like input to numpy array.

    Parameters
    ----------
    a : array_like
        Array-like input.

    Returns
    -------
    a : np.ndarray
        Converted validated array.

    Notes
    -----
    This function is implemented referring to scipy.

    """

    if type(a) is np.ndarray:
        return a

    a = np.asarray(a)
    if a.dtype is np.dtype('O'):
        raise ValueError('object arrays are not supported')
    return a


def _dtype_to_pack_info(dtype):
    """Get information from data type string for pack and unpack.

    Parameters
    ----------
    dtype : str
        One of the following string values: 'char', 'short', 'int', 'float',
        'double'.

    Returns
    -------
    c : str
        Format character.

    size : int
        Size required by the format.

    """

    dic = {
        'char' : ('c', 1),
        'short' : ('h', 2),
        'int' : ('i', 4),
        'float' : ('f', 4),
        'double' : ('d', 8),
    }

    if dtype not in dic.keys():
        raise NotImplementedError('Unexpected data type: ' + dtype)
    return dic[dtype]


def check_alpha(alpha):
    """Check whether given alpha is valid or not.

    Parameters
    ----------
    alpha : float [scalar]
        Frequency warping factor.

    """

    if np.abs(alpha) >= 1.0:
        raise ValueError('|alpha| must be less than 1.0')


def sr_to_alpha(sr, N=10, step=0.01):
    """Compute frequency warping factor under given sampling rate.

    Parameters
    ----------
    sr : float > 0 [scalar]
        Sampling rate in Hz.

    N : int >= 2 [scalar]
        Number of sample points in the frequency domain.

    step : float > 0 [scalar]
        Step size used in grid search.

    Returns
    -------
    alpha: float [scalar]
        Frequency warping factor.

    """

    def make_warped_freq(alpha, N):
        """Compute phase characteristic of the 1st order all-pass filter.
        """
        omega = np.arange(N) * (np.pi / (N - 1))
        alpha2 = alpha * alpha
        numer = (1 - alpha2) * np.sin(omega)
        denom = (1 + alpha2) * np.cos(omega) - 2 * alpha
        warped_omega = np.arctan(numer / denom)
        warped_omega[warped_omega < 0] += np.pi  # Phase unwrapping.
        return warped_omega

    def make_mel_freq(sr, N):
        """Compute mel-frequencies based on G. Fant and normalize them.
        """
        freq = np.arange(N) * (0.5 * sr / (N - 1))
        mel_freq = np.log(1 + freq / 1000)
        mel_freq = mel_freq * (np.pi / mel_freq[-1])
        return mel_freq

    if sr <= 0:
        raise ValueError('Sample rate must be a positive number')

    if N <= 1:
        raise ValueError('Number of sample points must be greater than 1')

    if step <= 0:
        raise ValueError('Step size must be a positive number')

    # Search appropriate alpha in terms of L2 distance.
    grid_alpha = np.arange(0, 1, step)
    target = make_mel_freq(sr, N)
    dist = [np.sum(np.square(target - make_warped_freq(a, N)))
            for a in grid_alpha]
    alpha = grid_alpha[np.argmin(dist)]

    return alpha


def read_binary(filename, dtype='double'):
    """Read a binary file.

    Parameters
    ----------
    filename : str
       Filename to read.

    dtype : str
       Input data type.

    Returns
    -------
    data : np.ndarray
       Loaded data.

    """

    if not os.path.exists(filename):
        raise OSError('No such file (%s).' % filename)

    format_char, byte_size = _dtype_to_pack_info(dtype)
    with open(filename, 'rb') as f:
        file_size = os.path.getsize(filename)
        data = struct.unpack(format_char * (file_size // byte_size),
                             f.read(file_size))

    return np.asarray(data, dtype=np.float64)
