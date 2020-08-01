#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Takenori Yoshimura
# Licensed under the MIT license

import numpy as np

from .utils import _asarray, _safe_squeeze, check_alpha


def freqt(C, M=24, alpha=0.42):
    """Perform frequency transform.

    Parameters
    ----------
    C : array_like [shape=(m + 1,) or (m + 1, T)]
        Minimum phase sequence.

    M : int >= 0 [scalar]
        Order of warped sequence.

    alpha : float in (-1, 1) [scalar]
        Frequency warping factor.

    Returns
    -------
    G : np.ndarray [shape=(M + 1), or (M + 1, T)]
        Frequency warped sequence.

    References
    ----------
    A. V. Oppenheim and D. H. Johnson, ``Discrete representation of signals,''
    in Proceedings of the IEEE, vol. 60, no. 6, pp. 681--691, 1972.

    """

    C = _asarray(C, as_matrix=True)
    if C.ndim != 2:
        raise ValueError('C must be 2-D matrix or 1-D vector')

    m = C.shape[0] - 1
    T = C.shape[1]

    if m < 0:
        raise ValueError('Order m must be a non-negative integer')

    if M < 0:
        raise ValueError('Order M must be a non-negative integer')

    check_alpha(alpha)
    beta = 1 - alpha * alpha

    L = M + 1
    D = np.zeros((L, T))
    G = np.zeros((L, T))
    for i in range(m, -1, -1):
        D[0] = G[0]
        G[0] = C[i] + alpha * D[0]
        if 1 < L:
            D[1] = G[1]
            G[1] = beta * D[0] + alpha * D[1]
        for j in range(2, L):
            D[j] = G[j]
            G[j] = D[j - 1] + alpha * (D[j] - G[j - 1])

    return _safe_squeeze(G)
