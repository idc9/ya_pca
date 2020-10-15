"""
Code borrowed from
http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/
https://gist.github.com/ahwillia/65d8f87fcd4bded3676d67b55c1a3954
"""
import numpy as np
from sklearn.utils import check_random_state
from copy import deepcopy
from sklearn.impute import SimpleImputer

from ya_pca.linalg_utils import svd_wrapper, rand_orthog


def svd_missing(X, M, rank, U_init='impute_mean', max_n_steps=100,
                atol=1e-6, random_state=None):
    """
    Computes SVD with missing data using the alternating algorithm described in http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/.

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The data matrix.

    M: array-like, (n_samples, n_features)
        The binary matrix indicating missing values (0 means missing).

    rank: int
        SVD rank to compute.

    U_init: str
        How to initialize the left singular vectors.
        Must be one of

        'random'
            Sample a random orthonormal matrix.

        'impute_mean'
            Impute the missing entries with the column means then compute the SVD.

    max_n_steps: int
        Maximum number of steps.

    atol: float
        Absolute tolerance for stopping criteria.

    random_state: None, int
        Random seed for random initalization.

    Output
    ------
    U, V, opt_history

    U: array-like, (n_samples, rank)
        The left singular values.

    V: array-like, (n_samples, rank)
        The right singular values.

    opt_history: list
        The loss values at each step.

    Note there is no normalization for the left/right singular vectors.

    """
    rng = check_random_state(random_state)

    assert M.dtype == bool
    assert all(M.mean(axis=0) != 0)
    assert all(M.mean(axis=1) != 0)

    # initialize U randomly
    if type(U_init) == str:
        if U_init == 'random':
            U = rand_orthog(X.shape[0], rank, random_state=rng)

        elif U_init == 'impute_mean':

            X_filled = SimpleImputer(strategy='mean').fit_transform(X)

            # X_filled = X.copy()
            # X_filled[~M] = np.nan
            # m = np.nanmean(X_filled, axis=0)
            # for j in range(X.shape[1]):
            #     nan_idxs = np.where(~M[:, j])[0]
            #     X_filled[nan_idxs, j] = m[j]
            U = svd_wrapper(X_filled, rank=rank)[0]

    loss_history = []
    prev_loss = np.nan
    for step in range(max_n_steps):

        Vt = censored_lstsq(U, X, M)
        U = censored_lstsq(Vt.T, X.T, M.T).T

        resid = np.dot(U, Vt) - X
        loss_val = np.mean(resid[M]**2)
        loss_history.append(loss_val)

        if step >= 1:
            diff = prev_loss - loss_val
            if diff < atol:
                break
            else:
                prev_loss = deepcopy(loss_val)

    return U, Vt.T, loss_history


def censored_lstsq(A, B, M):
    """Solves least squares problem with missing data in B
    Note: uses a broadcasted solve for speed.
    Args
    ----
    A: (ndarray) : n x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    if A.ndim == 1:
        A = A[:, None]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:, :, None]  # n x r x 1 tensor
    T = np.matmul(A.T[None, :, :], M.T[:, :, None] * A[None, :, :])  # n x r x r tensor
    try:
        # transpose to get r x n
        return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T

    except:
        r = T.shape[1]
        T[:, np.arange(r), np.arange(r)] += 1e-6
        return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T
