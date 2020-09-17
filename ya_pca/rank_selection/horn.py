from sklearn.utils import check_random_state
import numpy as np
import matplotlib.pyplot as plt

from ya_pca.utils import sample_parallel
from ya_pca.linalg_utils import svd_wrapper


def select_rank_horn(X, n_perm=1000, q=0.95, max_rank=None, svals=None,
                     random_state=None, n_jobs=None, backend=None):
    """
    Selects the PCA rank using Horn's Parallel Analysis.
    See https://arxiv.org/pdf/1710.00479.pdf.

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The data matrix.

    n_perm: int
        Number of permutations.

    q: float
        The cutoff quantile between 0 and 1.
        Observed singular values above this null quantile are deemed to be signal.

    max_rank: None, int
        Maximum PCA rank to check.

    svals: None, array-like
        (optional) Precomputed singular values.

    random_state: None, int
        Random seed for sampling.

    n_jobs: None, int
        Parallelize the permutation samples.

    Output
    ------
    rank_est, out

    rank_est: int
        Estimated rank

    out: dict
        All output from rank selection algorithm.
    """
    # TODO: better name for n_samples
    if svals is not None:
        obs_svals = svals
    else:
        obs_svals = svd_wrapper(X, rank=max_rank)[1]

    perm_samples = sample_parallel(fun=sample_horn,
                                   kws={'X': X},
                                   # random_state=random_state,
                                   n_jobs=n_jobs)
    perm_samples = np.array(perm_samples)

    sval_cutoff = np.quantile(perm_samples, q=q)
    rank_est = sum(obs_svals > sval_cutoff)

    return rank_est, {'perm_samples': perm_samples,
                      'obs_svals': obs_svals,
                      'sval_cutoff': sval_cutoff}


def sample_horn(X, random_state=None):
    """
    Permutes each colum the computes the largest singular value.
    """
    rng = check_random_state(random_state)

    idxs = np.arange(X.shape[0])
    X_perm = np.zeros_like(X)

    for j in range(X.shape[1]):
        perm_idxs = rng.permutation(idxs)
        X_perm[:, j] = X[:, j][perm_idxs]

    return np.linalg.norm(X_perm, ord=2)


def plot_horn(obs_svals, perm_samples, sval_cutoff):
    """
    Visualization for Horn's Parallel Analysis.
    Histogram of null distribution.
    """

    plt.hist(perm_samples, color='black')
    for sval in obs_svals:
        if sval > sval_cutoff:
            ls = '-'
            # alpha = 0.5
            lw = 2
        else:
            ls = '--'
            lw = 1
            # alpha = 1
        plt.axvline(sval, color='blue', ls=ls, lw=lw)
    plt.axvline(sval_cutoff, label='Threshold: {:1.3f}'.format(sval_cutoff),
                color='red', lw=2)
