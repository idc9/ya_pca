import numpy as np


def select_rank_psueodrank(shape, svals, alpha=0.05):
    """

    (Choi et al. 2017)
    """

    n, d = shape

    def mu(k):
        """
        mu_{n, d - k}
        """
        return (np.sqrt(n - .5) + np.sqrt(d - k - .5)) ** 2

    def sigma(k):
        """
        sigma_{n, d - k}
        """
        return (np.sqrt(n - .5) + np.sqrt(d - k - .5)) * \
            ((1 / np.sqrt(n - .5)) + (1 / np.sqrt(d - k - .5))) ** (1 / 3)

    ranks = np.arange(1, min(shape))  # 1 to min(n, d) - 1

    stats = []
    rejects = []
    cutoff = qtw(1 - alpha)
    rank_est = 0
    for k in ranks:
        # H0: rank <= k - 1
        stat = (svals[k - 1] ** 2 - mu(k)) / sigma(k)
        reject = stat > cutoff

        stats.append(stat)
        rejects.append(reject)

        if reject:
            rank_est = k
        else:
            break

    return rank_est, {'rejects': np.array(rejects),
                      'stats': np.array(stats),
                      'cutoff': np.array(cutoff)}


def qtw(q):
    """
    Quantile function of Tracy-Widom distribution.
    """
    raise NotImplementedError
