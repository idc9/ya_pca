from ya_pca.linalg_utils import svd_wrapper

from ya_pca.rank_selection.rmt_threshold import select_rank_rmt_threshold
from ya_pca.rank_selection.bai_ng_bic import select_rank_bai_ng_bic
from ya_pca.rank_selection.bi_cv import select_rank_bi_cv
from ya_pca.rank_selection.wold_cv import select_rank_wold_cv
from ya_pca.rank_selection.profile_likelihood import select_rank_prof_lik
from ya_pca.rank_selection.horn import select_rank_horn
from ya_pca.rank_selection.variance_explained import \
    select_rank_variance_explained
from ya_pca.rank_selection.minka import select_rank_minka

_valid_methods = ['rmt_threshold', 'bi_cv', 'wold_cv', 'bai_ng_bic',
                  'horn', 'prof_lik', 'minka', 'var_expl']


def select_rank(X, method='rmt_threshold', UDV=None,
                max_rank=None, random_state=None, **kwargs):

    """
    Selects the PCA rank.

    Parameters
    -----------
    X: array-like, (n_samples, n_features)
        The data matrix.

    method: str
        Which rank selection method to use.

    UDV: None, tuple
        (Optional) Precomputed SVD.

    max_rank: None, int
        Maximum rank to look at.

    random_state: None, int
        Random seed.

    **kwargs:
        Key word arguments for rank selection method.

    Output
    ------
    UDV, rank_est, out

    UDV: tuple
        Truncated left svecs, svals, and right svecs
    rank_est: int
        Estimated rank

    out: dict
        All output from rank selection algorithm.
    """

    shape = X.shape

    if UDV is None:
        UDV = svd_wrapper(X, max_rank=max_rank)
    else:
        max_rank = UDV[0].shape[1]

    svals = UDV[1]

    if method == 'rmt_threshold':
        # rank_est, out = select_rank_rmt_threshold(shape=shape, svals=svals,
        #                                           **kwargs)
        rank_est, out = select_rank_rmt_threshold(X=X, UDV=UDV, **kwargs)

    elif method == 'bi_cv':
        rank_est, out = select_rank_bi_cv(X=X, max_rank=max_rank,
                                          random_state=random_state,
                                          **kwargs)

    elif method == 'wold_cv':
        rank_est, out = select_rank_wold_cv(X=X, max_rank=max_rank,
                                            random_state=random_state,
                                            **kwargs)

    elif method == 'bai_ng_bic':
        rank_est, out = select_rank_bai_ng_bic(X=X, UDV=UDV,
                                               max_rank=max_rank,
                                               **kwargs)

    elif method == 'horn':
        rank_est, out = select_rank_horn(X=X, svals=svals, max_rank=max_rank,
                                         random_state=random_state)

    elif method == 'prof_lik':
        rank_est, out = select_rank_prof_lik(svals=svals, **kwargs)

    elif method == 'minka':
        rank_est, out = select_rank_minka(shape=shape, svals=svals)

    elif method == 'var_expl':
        rank_est, out = select_rank_variance_explained(X=X,
                                                       svals=svals, **kwargs)

    else:
        raise ValueError("Invalid input for method: {}. Must be "
                         "one of {}.".format(method, _valid_methods))

    # TODO: what to do if rank == 0
    UDV = truncate_svd(UDV, rank_est)

    return UDV, rank_est, out


def truncate_svd(UDV, rank):
    return UDV[0][:, 0:rank], UDV[1][0:rank], UDV[2][:, 0:rank]
