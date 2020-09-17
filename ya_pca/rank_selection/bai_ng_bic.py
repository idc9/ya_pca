import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ya_pca.linalg_utils import svd_wrapper

# TODO: these formulas give weird results for the larger components


def select_rank_bai_ng_bic(X, who=1, UDV=None, max_rank=None):
    """
    Determining the number of factors in approximate factor models, (Bai and Ng, 2002)

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The data matrix.

    who: int
        Which BIC formula to use. Must be one of [0, 1, 2, 3]

    UDV: None, output of pca.linalg_utils.svd_wrapper
        (Optional) Precomputed SVD of X

    max_rank: None, int
        The maximum rank to look at.

    Output
    ------
    rank_est, out

    rank_est: int
        Estimated rank

    out: dict
        All output from rank selection algorithm.
    """
    if UDV is None:
        UDV = svd_wrapper(X, rank=max_rank)

    n, d = X.shape
    if max_rank is None:
        max_rank = UDV[0].shape[1]

    ranks = np.arange(1, max_rank)

    def get_error(K):
        if K == 0:
            return 0

        U, D, V = UDV
        X_hat = (U[:, 0:K] * D[0:K]) @ V[:, 0:K].T
        return np.linalg.norm(X - X_hat) ** 2 / np.product(X.shape)

    sq_errors = [get_error(K=r) for r in ranks]

    bic = {0: _bic_0, 1: _bic_1, 2: _bic_2, 3: _bic_3}[int(who)]

    bic_vals = [bic(sq_error=sq_errors[k], k=ranks[k], n=n, d=d)
                for k in range(len(ranks))]

    bic_vals = pd.Series(bic_vals, index=ranks)
    bic_vals.index.name = 'rank'
    bic_vals.name = 'BIC_' + str(who)

    # rank_est = ranks[np.argmin(bic_vals)]
    rank_est = bic_vals.idxmin()

    return rank_est, bic_vals


def _bic_0(sq_error, k, n, d):
    return np.log(sq_error) + k * g_0(n=n, d=d)


def _bic_1(sq_error, k, n, d):
    return np.log(sq_error) + k * g_1(n=n, d=d)


def _bic_2(sq_error, k, n, d):
    return np.log(sq_error) + k * g_2(n=n, d=d)


def _bic_3(sq_error, k, n, d):
    return np.log(sq_error) + k * g_3(n=n, d=d)


def g_0(n, d):
    return np.log(d) / d


def g_1(n, d):
    return ((n + d) / (n * d)) * np.log((n * d) / (n + d))


def g_2(n, d):
    return ((n + d) / (n * d)) * np.log(min(n, d))


def g_3(n, d):
    return np.log(min(n, d)) / min(n, d)


def plot_bic(bic_vals):
    """
    Plots the BIC values.
    """
    assert bic_vals.index.name == 'rank'

    rank_est = bic_vals.idxmin()
    plt.plot(bic_vals.index, bic_vals, marker='.', color='black')
    plt.axvline(rank_est, label='Estimated rank: {}'.format(rank_est),
                color='red')
    plt.xlabel("Rank")
    plt.ylabel(bic_vals.name)
