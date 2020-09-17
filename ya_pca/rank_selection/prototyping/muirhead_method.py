import numpy as np
from scipy.stats import chi2


def select_rank_muirhead(svals, shape, alpha=0.05):
    """
    Muirhead 1982 or section 2.2.3 of Choi et al. 2017
    """
    n, d = shape

    def get_q(k):
        return d - k + 1

    def get_ell_bar(k):
        q = get_q(k)
        return sum(svals[k - 1:] ** 2) / q

    # def V(k):
    #     q = get_q(k)

    #     return (((n - 1) ** (q - 1)) * np.product(svals[k - 1:] ** 2)) / \
    #         (((1 / q) * sum(svals[k - 1:] ** 2)) ** q)

    def logV(k):
        q = get_q(k)

        return (q - 1) * np.log(n - 1) + 2 * sum(np.log(svals[k - 1:])) \
            - q * np.log((1 / q) * sum(svals[k - 1:] ** 2))

    ranks = np.arange(1, min(shape))  # 1 to min(n, d) - 1
    cutoffs = []
    stats = []
    rejects = []
    dofs = []
    rank_est = 0
    for k in ranks:
        q = get_q(k)
        ell_bar = get_ell_bar(k)

        first = n - k - ((2 * q ** 2 + q + 2) / (6 * q))

        second = 0
        for j in range(k):
            second += ell_bar ** 2 / (svals[j] ** 2 - ell_bar) ** 2

        stat = - (first + second) * logV(k)  # np.log(V(k))

        dof = ((q + 2) * (q - 1)) / 2

        cutoff = chi2.ppf(q=1 - alpha, df=dof)

        reject = stat > cutoff

        stats.append(stat)
        cutoffs.append(cutoff)
        rejects.append(reject)
        dofs.append(dof)

        if reject:
            rank_est = k
        else:
            break

    return rank_est, {'stats': np.array(stats),
                      'cutoffs': np.array(cutoffs),
                      'rejects': np.array(rejects),
                      'dofs': np.array(dofs)}
