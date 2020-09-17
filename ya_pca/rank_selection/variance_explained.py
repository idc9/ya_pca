import numpy as np
from scipy.sparse import issparse

from ya_pca.linalg_utils import svd_wrapper


def select_rank_variance_explained(X, q=0.95, svals=None):
    """
    Selects the PCA rank by finding the components which explains the desired amount of variance.

    Parameters
    ----------
    X: array-like, (n_samples, n_features)

    q: float
        The desired percent of variance to be explained

    svals: array-like, None
        (Optional) Precomputed singular values.

    Output
    ------
    rank_est, out

    rank_est: int
        Estimated rank

    out: dict
        All output from rank selection algorithm.
    """

    if svals is None:
        svals = svd_wrapper(X)[1]

    tot_variance = safe_frob_norm(X) ** 2
    var_expl_prop = svals ** 2 / tot_variance

    assert np.allclose(var_expl_prop.sum(), 1)

    var_expl_cum = np.cumsum(var_expl_prop)
    rank_est = np.where(var_expl_cum > q)[0].min()

    return rank_est, {'tot_variance': tot_variance,
                      'var_expl_prop': var_expl_prop,
                      'var_expl_cum': var_expl_cum}


def safe_frob_norm(X):
    """
    Calculates the Frobenius norm of X whether X is dense or sparse.

    Currently, neither scipy.linalg.norm nor numpy.linalg.norm work for
    sparse matrices.
    """
    if issparse(X):
        return np.sqrt(sum(X.data ** 2))
    else:
        return np.linalg.norm(np.array(X), ord='fro')
