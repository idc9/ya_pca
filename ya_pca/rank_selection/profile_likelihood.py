"""
Most of the code is borrowed form https://github.com/neurodata/graspy/blob/master/graspy/embed/svd.py
"""
import numpy as np
from scipy.stats import norm


def select_rank_prof_lik(svals, n_elbows=2):
    """
    Parameters
    ----------
    svals:
        The singular values.

    n_elbows: int
        TODO: documents

    Output
    ------
    rank_est, out

    rank_est: int
        Estimated rank

    out: dict
        All output from rank selection algorithm.
    """

    idx = 0
    elbows = []
    values = []
    likelihoods = []

    for _ in range(n_elbows):
        arr = svals[idx:]

        if arr.size <= 1:  # Cant compute likelihoods with 1 numbers
            break

        lq = _compute_likelihood(arr)
        idx += np.argmax(lq) + 1
        elbows.append(idx)
        values.append(svals[idx - 1])
        likelihoods.append(lq)

    rank_est = elbows[-1]

    return rank_est, {'elbows': elbows,
                      'values': values,
                      'likelihoods': likelihoods}


def _compute_likelihood(arr):
    """
    Computes the log likelihoods based on normal distribution given
    a 1d-array of sorted values. If the input has no variance,
    the likelihood will be nan.
    """
    n_elements = len(arr)
    likelihoods = np.zeros(n_elements)

    for idx in range(1, n_elements + 1):
        # split into two samples
        s1 = arr[:idx]
        s2 = arr[idx:]

        # deal with when input only has 2 elements
        if (s1.size == 1) & (s2.size == 1):
            likelihoods[idx - 1] = -np.inf
            continue

        # compute means
        mu1 = np.mean(s1)
        if s2.size != 0:
            mu2 = np.mean(s2)
        else:
            # Prevent numpy warning for taking mean of empty array
            mu2 = -np.inf

        # compute pooled variance
        variance = ((np.sum((s1 - mu1) ** 2) + np.sum((s2 - mu2) ** 2))) / (
            n_elements - 1 - (idx < n_elements)
        )
        std = np.sqrt(variance)

        # compute log likelihoods
        likelihoods[idx - 1] = np.sum(norm.logpdf(s1, loc=mu1, scale=std)) + np.sum(
            norm.logpdf(s2, loc=mu2, scale=std)
        )

    return likelihoods
