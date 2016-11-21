#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot


def randomized_range_finder_rsymevd(A, size, n_iter,
                                    power_iteration_normalizer='qr',
                                    random_state=None):
    '''Computes an orthonormal matrix whose range approximates 
    the range of symmetric matrix A.

    Parameters
    -----------
    A : sparse matrix
        Matrix to decompose
    size : integer
        Number of columns of the return array
    n_iter : integer
        Number of power iterations used to stabilize the result
    power_iteration_normalizer : 'qr' (default), 'lu'
        Whether the power iterations are normalized with step-by-step
        QR decomposition ('qr') or LU decomposition ('lu').
    random_state : RandomState or an int seed, default None
        A random number generator instance
    '''
    random_state = check_random_state(random_state)

    Q = random_state.normal(size=(A.shape[1], size))

    for i in range(n_iter):
        if power_iteration_normalizer == 'qr':
            Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode='economic')
        elif power_iteration_normalizer == 'lu':
            Q, _ = linalg.lu(safe_sparse_dot(A, Q), permute_l=True)

    Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode='economic')

    return Q


def randomized_symevd(M, n_components, n_oversamples=10, n_iter=3,
                      power_iteration_normalizer='qr', random_state=0):
    '''Computes a trucated randomized eigenvalue decomposition 
    for a symmetric matrix M.
    Algorithm 5.3 of [Halko+2009]_.
    See also original implementation [sklearn]_.

    Parameters
    -----------
    M : sparse matrix
        Matrix to decompose
    n_components : int
        Number of eigenvalues and vectors to extract.
    n_oversamples : int, default 10
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning.
    n_iter : int, default 3
        Number of power iterations.
    power_iteration_normalizer : 'qr' (default), 'lu'
        Whether the power iterations are normalized with step-by-step
        QR decomposition ('qr') or LU decomposition ('lu').
    random_state : int, default 0
        A random number generator instance to make behavior.

    Notes
    ------
    .. [Halko+2009]
        Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). 
        Finding structure with randomness: Probabilistic algorithms 
        for constructing approximate matrix decompositions. 
        SIAM Review, 53(2), 217–288.
    .. [sklearn] `sklearn.utils.extmath.randomized_svd`
    '''

    random_state = check_random_state(random_state)
    n_random = 2 * n_components + n_oversamples
    n1, n2 = M.shape

    assert n1 == n2, 'M is not square matrix.'
    # assert (M != M.T).nnz == 0, 'M is not symmetric matrix.'  ## TODO: this does not work well

    Q = randomized_range_finder_rsymevd(M, n_random, n_iter,
                                        power_iteration_normalizer,
                                        random_state)
    B = safe_sparse_dot(Q.T, M)
    B = safe_sparse_dot(B, Q)

    eigenvalues, Uhat = linalg.eigh(B)
    del B
    U = np.dot(Q, Uhat)

    return U[:, -n_components:], eigenvalues[-n_components:]
