import numpy as np
from numbers import Number

from ya_pca.rank_selection.noise_estimates import estimate_noise
from ya_pca.linalg_utils import svd_wrapper


def select_rank_rmt_threshold(X, thresh_method='dg',
                              noise_est='mp',
                              noise_est_kwargs={},
                              UDV=None):
    """
    Selects the PCA rank using one of the random matrix theory based thresholds.

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The data matrix.

    thresh_method: str
        Which thesholding method to use. Must be one of
        'dg': Donoho-Gavish method

        'mpe': Marcenko Pastur edge method

    noise_est: float, str
        Either the noise estimate or a method to estimate the noise.

    noise_est_kwargs: dict
        Key word arguments for noise estimation method.

    UDV: None, tuple
        Precomputed SVD.

    Output
    ------
    rank_est, out

    rank_est: int
        Estimated rank

    out: dict
        All output from rank selection algorithm.
    """

    if UDV is None:
        UDV = svd_wrapper(X)

    if not isinstance(noise_est, Number):
        noise_est = estimate_noise(X, method=noise_est,
                                   UDV=UDV, **noise_est_kwargs)

    assert thresh_method in ['mpe', 'dg']

    if thresh_method == 'dg':
        thresh = donoho_gavish_threshold(shape=X.shape,
                                         sigma=noise_est)

    elif thresh_method == 'mpe':
        thresh = marcenko_pastur_edge_threshold(shape=X.shape,
                                                sigma=noise_est)

    svals = UDV[1]
    rank_est = sum(np.array(svals) > thresh)

    return rank_est, {'svals': svals,
                      'shape': X.shape,
                      'thresh': thresh,
                      'noise_estimate': noise_est}


def marcenko_pastur_edge_threshold(shape, sigma):
    """
    See (Gavish and Donoho, 2014).

    Parameters
    ----------
    shape: tuple (n_samples, n_features)
        Shape of the data matrix.

    sigma: float
        Estiamte of the noise standard deviation.

    Output
    ------
    sigular_value_threshold: float

    """
    n_samples, n_features = shape
    beta = n_features / n_samples
    mult = n_samples

    if beta > 1:
        beta = 1 / beta
        mult = n_features

    return (1 + np.sqrt(beta)) * np.sqrt(mult) * sigma


def dg_threshold(beta):
    return np.sqrt(2 * (beta + 1) + 8 * beta / (beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1)))


def donoho_gavish_threshold(shape, sigma):
    """
    (Gavish and Donoho, 2014)

    Parameters
    ----------
    shape: tuple (n_samples, n_features)
        Shape of the data matrix.

    sigma: float
        Estiamte of the noise standard deviation.


    Output
    ------
    sigular_value_threshold: float

    """
    n_samples, n_features = shape
    beta = n_features / n_samples
    mult = n_samples

    # TODO: is this what we want to do?
    if beta > 1:
        beta = 1 / beta
        mult = n_features

    if n_samples == n_features:
        lambd = 4 / np.sqrt(3)
    else:
        lambd = dg_threshold(beta)

    return lambd * np.sqrt(mult) * sigma
