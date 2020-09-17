"""
Code based on https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/decomposition/_pca.py
"""
import numpy as np
from math import log
from scipy.special import gammaln
import pandas as pd


def select_rank_minka(shape, svals):
    """
    Selects the PCA rank using the method from (Minka, 2000)

    Parameters
    ----------
    shape: tuple (n_samples, n_features)

    svals: array-like, (n_features, )
        All singular values of the data matrix X.


    Output
    ------
    rank_est, out

    rank_est: int
        Estimated rank

    out: dict
        All output from rank selection algorithm.
    """
    # assert shape[0] > shape[1]

    n_samples, n_features = shape
    # assert n_samples > n_features
    assert len(svals) == n_features

    cov_evals = svals ** 2 / n_samples

    ranks = np.arange(1, n_features)  # 1 to n_features - 1

    log_liks = np.empty_like(ranks)

    for idx, rank in enumerate(ranks):
        log_liks[idx] = get_log_lik(cov_evals=cov_evals,
                                    rank=rank,
                                    shape=shape)

    log_liks = pd.Series(log_liks, index=ranks)
    log_liks.name = 'log_lik'
    log_liks.index.name = 'rank'

    rank_est = log_liks.idxmax()

    return rank_est, {'log_liks': log_liks,
                      'cov_evals': cov_evals}


def get_log_lik(cov_evals, rank, shape):
    """Compute the log-likelihood of a rank ``rank`` dataset.

    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``.

    Parameters
    ----------
    cov_evals : array of shape (n_features, )
        Eigen values of the sample covarinace matrix.

    rank : int
        Tested rank value. It should be strictly lower than n_features,
        otherwise the method isn't specified (division by zero in equation
        (31) from the paper).
    shape : tuple (n_samples, n_features)
        Shape of the data set

    Returns
    -------
    ll : float,
        The log-likelihood

    Notes
    -----
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
    """

    n_samples, n_features = shape
    if not 1 <= rank <= n_features - 1:
        raise ValueError("the tested rank should be in [1, n_features - 1]")

    eps = 1e-15

    if cov_evals[rank - 1] < eps:
        # When the tested rank is associated with a small eigenvalue, there's
        # no point in computing the log-likelihood: it's going to be very
        # small and won't be the max anyway. Also, it can lead to numerical
        # issues below when computing pa, in particular in log((spectrum[i] -
        # spectrum[j]) because this will take the log of something very small.
        return -np.inf

    pu = -rank * log(2.)
    for i in range(1, rank + 1):
        pu += (gammaln((n_features - i + 1) / 2.) -
               log(np.pi) * (n_features - i + 1) / 2.)

    pl = np.sum(np.log(cov_evals[:rank]))
    pl = -pl * n_samples / 2.

    v = max(eps, np.sum(cov_evals[rank:]) / (n_features - rank))
    pv = -np.log(v) * n_samples * (n_features - rank) / 2.

    m = n_features * rank - rank * (rank + 1.) / 2.
    pp = log(2. * np.pi) * (m + rank) / 2.

    pa = 0.
    spectrum_ = cov_evals.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(cov_evals)):
            pa += log((cov_evals[i] - cov_evals[j]) *
                      (1. / spectrum_[j] - 1. / spectrum_[i])) + log(n_samples)

    ll = pu + pl + pv + pp - pa / 2. - rank * log(n_samples) / 2.

    return ll
