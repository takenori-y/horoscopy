#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Takenori Yoshimura
# Licensed under the MIT license

import numpy as np

from .utils import _asarray, _safe_squeeze


def solve_toeplitz_plus_hankel(t, h, b):
    """Solve a Toeplitz plus Hankel system.

    Parameters
    ----------
    t : (array_like, array_like)
        First column(s) and first row(s) of the Toeplitz matrix.

    h : (array-like, array_like)
        First column(s) and last row(s) of the Hankel matrix.

    b : array-like [shape=(M,) or (M, K)]
        Constant vector(s). Right-hand side in ``(T + H) x = b``.

    Returns
    -------
    a : np.ndarray [shape=(M,) or (M, K)]
        Solution of the Toeplitz plus Hankel system.

    References
    ----------
    G. Merchant and T. Parks, ``Efficient solution of a Toeplitz-plus-Hankel
    coefficient matrix system of equations,'' IEEE Transactions on Acoustics,
    Speech, and Signal Processing, vol. 30, no. 1, pp. 40--44, 1982.

    """

    def mv(m, v):
        return np.squeeze(np.matmul(m, np.expand_dims(v, axis=-1)), axis=-1)

    def cross_transpose(m):
        n = np.zeros(m.shape)
        n[:, 0, 0] = m[:, 1, 1]
        n[:, 0, 1] = m[:, 1, 0]
        n[:, 1, 0] = m[:, 0, 1]
        n[:, 1, 1] = m[:, 0, 0]
        return n

    if not isinstance(t, tuple) and len(t) != 2:
        raise ValueError('t must be 2-dim tuple')

    if not isinstance(h, tuple) and len(h) != 2:
        raise ValueError('h must be 2-dim tuple')

    t_c, t_r = t
    t_c = _asarray(t_c, as_matrix=True)
    t_r = _asarray(t_r, as_matrix=True)
    if t_c.ndim != t_r.ndim or t_c.ndim != 2:
        raise ValueError('t must be 2-D matrices or 1-D vectors')

    h_c, h_r = h
    h_c = _asarray(h_c, as_matrix=True)
    h_r = _asarray(h_r, as_matrix=True)
    if h_c.ndim != h_r.ndim or h_c.ndim != 2:
        raise ValueError('h must be 2-D matrices or 1-D vectors')

    b = _asarray(b, as_matrix=True)
    if b.ndim != 2:
        raise ValueError('b must be 2-D matrix or 1-D vector')
    M, K = b.shape

    # Step 1:
    if 1:
        # Set R.
        R = np.zeros((M, K, 2, 2))
        R[:, :, 0, 0] = t_c
        R[:, :, 1, 1] = t_r
        R[:, :, 0, 1] = h_c
        R[:, :, 1, 0] = h_r[::-1]

        # Apply coefficients modification.
        s = 1 if M % 2 == 0 else 0
        d0 = t_c[0]
        R[::2, :, 0, 0] += d0
        R[::2, :, 1, 1] += d0
        R[s::2, :, 0, 1] -= d0
        R[s::2, :, 1, 0] -= d0

        # Set X_0.
        X = np.zeros((M, K, 2, 2))
        X[0, :, 0, 0] = 1
        X[0, :, 1, 1] = 1
        X_prev = np.zeros((M, K, 2, 2))

        # Set p_0.
        b_bar = np.zeros((K, 2))
        b_bar[:, 0] = b[0]
        b_bar[:, 1] = b[M - 1]
        p = np.zeros((M, K, 2))
        p[0] = mv(np.linalg.inv(R[0]), b_bar)

        # Set V_x.
        V_x = np.copy(R[0])

    # Step 2:
    for i in range(1, M):
        # a: Calculate E_x.
        E_x = np.zeros((K, 2, 2))
        for j in range(i):
            E_x += np.matmul(R[i - j], X[j])

        # b: Calculate e_p.
        e_p = np.zeros((K, 2))
        for j in range(i):
            e_p += mv(R[i - j], p[j])

        # c: Calculate B_x.
        B_x = np.matmul(np.linalg.inv(cross_transpose(V_x)), E_x)

        # d: Update X and V_x.
        for j in range(1, i):
            X[j] -= np.matmul(cross_transpose(X_prev[i - j]), B_x)
        X[i] = -B_x
        X_prev[1:i + 1] = X[1:i + 1]
        V_x -= np.matmul(cross_transpose(E_x), B_x)

        # e: Calculate g.
        b_bar[:, 0] = b[i]
        b_bar[:, 1] = b[M - 1 - i]
        g = mv(np.linalg.inv(cross_transpose(V_x)), b_bar - e_p)

        # f: Update p.
        for j in range(i):
            p[j] += mv(cross_transpose(X[i - j]), g)
        p[i] = g

    # Step 3:
    if 1:
        # Extract solution vector.
        a = p[:, :, 0]

    return _safe_squeeze(a)
