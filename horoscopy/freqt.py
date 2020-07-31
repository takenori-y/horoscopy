#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .utils import asarray, check_alpha


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

    L = M + 1

    C = asarray(C)
    dim = C.ndim
    if dim == 1:
        out_size = (L,)
    elif dim == 2:
        T = C.shape[1]
        out_size = (L, T)
    else:
        raise ValueError('Unexpected input dim: ' + str(dim))

    m = C.shape[0] - 1
    if m < 0:
        raise ValueError('Order m must be a non-negative integer')

    if M < 0:
        raise ValueError('Order M must be a non-negative integer')

    check_alpha(alpha)
    beta = 1 - alpha * alpha

    D = np.zeros(out_size)
    G = np.zeros(out_size)
    for i in range(m, -1, -1):
        D[0] = G[0]
        G[0] = C[i] + alpha * D[0]
        if 1 < L:
            D[1] = G[1]
            G[1] = beta * D[0] + alpha * D[1]
        for j in range(2, L):
            D[j] = G[j]
            G[j] = D[j - 1] + alpha * (D[j] - G[j - 1])

    return G
