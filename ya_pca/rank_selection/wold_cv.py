from sklearn.utils import check_random_state
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from ya_pca.svd_missing import svd_missing
from ya_pca.linalg_utils import rand_orthog


def select_rank_wold_cv(X, max_rank, p_holdout=0.3, opt_kws={}, n_folds=5,
                        random_state=None, rotate=False):
    """
    Estimates the PCA rank using the Wold style cross- validation method discussed in (Owen and Perry, 2009)

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The data matrix.

    max_rank: None, int
        (optional) Maximum rank to compute.

    p_holdout: float
        Proportion for hold outs.

    opt_kws: dict
        Optimization key word arguments for pca.svd_missing.svd_missing

    n_folds: int
        Number of cross-validation folds.


    random_state: None, int
        Random seed.

    rotate: bool
        Whether or not to apply a random rotation to the data matrix.
        This can help if the true PCs are sparse.

    Output
    ------
    rank_est, out

    rank_est: int
        Estimated rank

    out: dict
        All output from rank selection algorithm.
    """

    rng = check_random_state(random_state)

    if rotate:
        X = rand_orthog(n=X.shape[0], K=X.shape[0], random_state=rng) @ X
        X = X @ rand_orthog(n=X.shape[1], K=X.shape[1], random_state=rng)

    ranks = np.arange(1, max_rank + 1)
    fold_errors = np.zeros((n_folds, len(ranks)))
    for fold in range(n_folds):
        M = rng.rand(*X.shape) > p_holdout

        for rank_idx, rank in enumerate(ranks):
            out = svd_missing(X, M=M, rank=rank, random_state=rng,
                              **opt_kws)

            U, V = out[0], out[1]

            resid = np.dot(U, V.T) - X
            fold_errors[fold, rank_idx] = np.mean(resid[~M]**2)

    errors = fold_errors.mean(axis=0)
    errors = pd.Series(errors, index=ranks)
    errors.index.name = 'rank'
    errors.name = 'wold_cv_error'

    rank_est = errors.idxmin()

    return rank_est, {'fold_errors': fold_errors,
                      'errors': errors}


def plot_wold_cv(errors):
    """
    Plots cross validation errors for Wold CV rank selection.
    """

    plt.plot(errors.index, errors, marker='.', color='black')

    if errors.index.name == 'rank':
        rank_est = errors.idxmin()
        plt.axvline(rank_est, label='Rank est: {}'.format(rank_est),
                    color='red')
