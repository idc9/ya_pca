import numpy as np

from ya_pca.rank_selection.marcenko_pastur import compute_mp_quantile
from ya_pca.linalg_utils import svd_wrapper
from ya_pca.rank_selection.soft_impute_cv import soft_impute_cv
from ya_pca.rank_selection.variance_explained import safe_frob_norm

from warnings import warn


def estimate_noise(X, method='mp', UDV=None, **kwargs):
    """
    Estimates the noise level for a low rank PCA model.

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The data matrix.

    method: str
        Which method to use. Must be one of TODO: document
        'mp':

        'cv_soft_impute':

        'naive':

        'sn':

    UDV: None, tuple
        Precomputed SVD.

    **kwargs:
        Key word arguments for the noise estimation method.
    """

    assert method.lower() in ['mp', 'cv_soft_impute', 'naive', 'sn']

    if method.lower() == 'mp':
        if UDV is None:
            UDV = svd_wrapper(X)
        svals = UDV[1]
        return estimate_noise_mp_quantile(svals=svals, shape=X.shape,
                                          **kwargs)

    elif method.lower() == 'cv_soft_impute':
        return estimate_noise_soft_impute(X, UDV=UDV, **kwargs)[0]

    elif method.lower() == 'naive':
        return estimate_noise_naive_rank_based(X, UDV=UDV, **kwargs)[0]

    elif method.lower() == 'sn':
        return estimate_noise_shabalin_nobel(X, UDV=UDV, **kwargs)[0]


# def estimate_noise_mp_median(svals, shape):
#     """

#     Parameters
#     ----------
#     svals: (min(shape), )
#         Singular values of the data matrix.

#     shape: tuple, (n_samples, n_features)
#         Shape of the data matrix.

#     """
#     # TODO: perhaps allow lower quantiles
#     if shape[0] == shape[1]:
#         raise NotImplementedError
#         # TODO :figure out what to do in this case

#     n_samples, n_features = shape
#     beta = n_features / n_samples

#     mp_median = compute_mp_quantile(beta, q=0.5)

#     median_sval = np.median(svals)

#     return float(median_sval / (np.sqrt(n_samples) * mp_median))


def estimate_noise_mp_quantile(svals, shape, q=0.5):
    """
    Estimates the noise level using the median of the Marcenko Pastur distribution as discussed in (Gavish and Donoho, 2014).

    Parameters
    ----------
    svals: (min(shape), )
        Singular values of the data matrix.

    shape: tuple, (n_samples, n_features)
        Shape of the data matrix.

    Output
    ------
    noise_est: float
        The estimated noise standard deviation.
    """

    # TODO: perhaps allow lower quantiles
    if shape[0] == shape[1]:
        raise NotImplementedError

    n_samples, n_features = shape
    beta = n_features / n_samples
    mult = n_samples

    if beta > 1:
        beta = 1 / beta
        mult = n_features

    # if the user has only provided a subset of the svals
    # then override the q input and use the smallest sval
    if len(svals) < min(shape):
        new_q = 1 - (len(svals) / min(shape))

        if new_q > q:
            warn("Only {}/{} singular values provided, but asked for the "
                 "{}th quantile. Falling back on the {}th quantile/".
                 format(len(svals), min(shape), q, new_q))

            q = new_q

        # pad with zeros to the quantile function below words
        n_to_add = min(shape) - len(svals)
        svals = np.concatenate([svals,
                                np.repeat(min(svals), repeats=n_to_add)])
        # TODO: test this!

    mp_quant = compute_mp_quantile(beta, q=q)
    sval_quan = np.quantile(svals, q=q)

    return float(sval_quan / (np.sqrt(mult) * mp_quant))


def estimate_noise_soft_impute(X, UDV=None, c=2 / 3, K=5, n_lambd_vals=100,
                               max_rank=None, max_rank_si=None, **kws):
    """

    Estimates the noise level using SoftImpute according to the scheme
    developed in (Choi et al, 2017)
    """

    if max_rank_si is None and max_rank is not None:
        max_rank_si = max_rank

    lambd_cv, out = soft_impute_cv(X, K=K, n_lambd_vals=n_lambd_vals,
                                   max_rank=max_rank_si, **kws)

    if UDV is None:
        UDV = svd_wrapper(X, rank=max_rank)
    U, svals, V = UDV

    n_svals_computed = len(svals)
    m = min(X.shape)
    _svals = np.concatenate([svals,
                             np.zeros(m - n_svals_computed)])

    ndof = (_svals > lambd_cv).sum()

    if ndof >= n_svals_computed:  # ge in case lambd_cv == 0
        warn("max_rank may be too small and have affected soft_imputs's"
             " noise estimate.")

    soft_thresh_svals = soft_thresh(svals, lambd_cv)
    B_hat = (U * soft_thresh_svals) @ V.T
    RSS = ((X - B_hat) ** 2).sum()
    n, d = X.shape

    # denom = (n * (d - c * ndof))
    denom = max(X.shape) * (min(X.shape) - c * ndof)
    sigma_sq_est = RSS / denom

    sigma_est = np.sqrt(sigma_sq_est)
    return sigma_est, {'svals': svals,
                       'ndof': ndof,
                       'RSS': RSS,
                       'soft_thresh_svals': soft_thresh_svals,
                       'denom': denom,
                       'c': c,
                       'shape': X.shape,
                       'lambd_cv': lambd_cv,
                       'cv_out': out}


def soft_thresh(x, lambd):
    return np.maximum(x - lambd, 0)


def estimate_noise_shabalin_nobel(X, UDV=None):
    """
    Estimates the noise level using the method presented in Section 4.1
    of (Shablin and Nobel, 2010)
    """
    # TODO: implement this!
    raise NotImplementedError


def estimate_noise_naive_rank_based(X, rank, UDV=None):
    """
    Estimates the noise level using the mean of the smallest squared
    singular vaules.

    Let s_1, ...., s_m be the singular values of X (n x d) where m = min(n, d).
    Suppose we know the true rank is R. Then let

    (1/sqrt(max(n, d))) * (1/(m - R)) * sum_{j=R+1}^m s_j^2
    is an estimate of the noise level.

    Parameters
    ----------
    X: array-like

    rank: int, None

    UDV:

    Output
    ------
    sigma_est, out

    sigma_est: float
        Estimate of the noise standard deviation.

    """

    if UDV is None:
        svals = svd_wrapper(X, rank=rank)[1]
    else:
        svals = UDV[1]
        assert len(svals) >= rank
        svals = svals[0:min(len(svals), rank)]

    tot_svsq_sum = safe_frob_norm(X) ** 2
    sum_smallest_svals_sq = tot_svsq_sum - sum(svals ** 2)

    m = min(X.shape)

    var_est = (1 / max(X.shape)) * (1 / (m - rank)) * sum_smallest_svals_sq

    return np.sqrt(var_est), {'tot_svsq_sum': tot_svsq_sum,
                              'sum_smallest_svals_sq': sum_smallest_svals_sq,
                              'svals': svals,
                              'rank': rank}
