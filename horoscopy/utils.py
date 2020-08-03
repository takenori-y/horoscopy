#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Takenori Yoshimura
# Licensed under the MIT license

import os
import struct

import numpy as np


def _asarray(a, as_matrix=False):
    """Convert array-like input to numpy array.

    Parameters
    ----------
    a : array_like
        Array-like input.

    as_matrix : bool
        If True and input is vector, append an additional axis to the input.

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


def _safe_squeeze(x, axis=1):
    """Remove single-dimensional entry from the shape of an array safely.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    axis : int >= 0 [scalar]
        A single-dimensional entry.

    Returns
    -------
    y : np.ndarray
        The input array.

    """

    if x.shape[axis] == 1:
        x = np.squeeze(x, axis=axis)
    return x


def check_alpha(alpha):
    """Check whether given alpha is valid or not.

    Parameters
    ----------
    alpha : float [scalar]
        Frequency warping factor.

    """

    if np.abs(alpha) >= 1.0:
        raise ValueError('|alpha| must be less than 1.0')


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
