"""
Some toy data distributions
"""
import numpy as np
from sklearn.utils import check_random_state
from itertools import product

from ya_pca.linalg_utils import rand_orthog


def rand_factor_model(n_samples=100, n_features=20,
                      rank=10,
                      m=1.5,
                      sigma=1.0, random_state=None):
    """

    X = sqrt(n) UV^T + E
    U = scores (n_samples x rank) has iid normal entries
    V = loadings (n_features x rank) is a column orthonormal matrix
    E (n_samples x n_features) has iid N(0, sigma^2) entries

    Follows Choi et al. 2017 section 2.2.3.
    Parameters
    ----------
    n_samples: int
        Number of samples.

    n_features: int
        Number of features.

    rank: int
        Rank of the true signal.

    m: float
        Signal strength.

    sigma: float
        Standard deviation of the noise.

    random_state: None, int
        Seed for generating the data

    Output
    ------
    X, out

    X: array-like (n_samples, n_features)
        The sampled data matrix

    out: dict
        The true PCA scores, svals and loadings
    """

    rng = check_random_state(random_state)

    # scores = rng.normal(size=(n_samples, rank))
    U = rand_orthog(n_samples, rank, random_state=rng)
    V = rand_orthog(n_features, rank, random_state=rng)

    svals = m * np.arange(1, rank + 1) * sigma * (n_samples * n_features) ** (.25)
    svals = np.sort(svals)[::-1]

    E = rng.normal(size=(n_samples, n_features), scale=sigma)

    X = (U * svals).dot(V.T) + E

    # return X, scores, loadings, E
    return X, {'U': U,
               'svals': svals,
               'V': V,
               'E': E}


def perry_sim_dist(strong=True, sparse=False, noise='white',
                   random_state=None):
    """
    Toy PCA data example from Section 5.4 of (Perry, 2009)

    Parameters
    ----------
    strong: bool
        Strong or weak signal.

    sparse: bool
        Sparse or dense loadings.

    noise: str
        Which kind of noise. Must be one of ['white', 'heavy', 'colored']

    random_state: None, int
        Random seed for the data.

    Output
    ------
    X, out

    X: array-like (n_samples, n_features)
        The sampled data matrix

    out: dict
        The true PCA scores, svals and loadings
    """
    assert noise in ['white', 'heavy', 'colored']
    rng = check_random_state(random_state)

    n = 100
    d = 50
    K = 6

    svals = np.arange(5, 10 + 1)
    if strong:
        svals = svals * np.sqrt(n)

    if sparse:
        s = .1

        u = rng.uniform(size=(n, K))
        zero_mask = u > 1 - s
        neg_mask = u < s / 2

        U = np.ones((n, K))
        U[zero_mask] = 0.0
        U[neg_mask] = -1
        U = U * (1 / np.sqrt(s * n))

        v = rng.uniform(size=(d, K))
        zero_mask = v > 1 - s
        neg_mask = v < s / 2

        V = np.ones((d, K))
        V[zero_mask] = 0.0
        V[neg_mask] = -1
        V = V * (1 / np.sqrt(s * d))

    else:
        U = rng.normal(size=(n, K), scale=1 / np.sqrt(n))
        V = rng.normal(size=(d, K), scale=1 / np.sqrt(d))

    assert noise in ['white', 'heavy', 'colored']

    if noise == 'white':
        E = rng.normal(size=(n, d))

    elif noise == 'heavy':
        df = 3
        sig = np.sqrt(df / (df - 2))
        E = (1.0 / sig) * rng.standard_t(df=df, size=(n, d))

    elif noise == 'colored':
        v1, v2 = 3, 3
        sigma_sq = 1.0 / rng.chisquare(df=v1, size=n)
        tau_sq = 1.0 / rng.chisquare(df=v2, size=d)

        c = np.sqrt((1 / (v1 - 2)) + (1 / (v2 - 2)))

        E = rng.normal(size=(n, d))

        E = np.zeros((n, d))
        for i, j in product(range(n), range(d)):
            E[i, j] = rng.normal(scale=np.sqrt(sigma_sq[i] + tau_sq[j]))
        E = (1 / c) * E

    X = np.sqrt(n) * (U * svals) @ V.T + E

    return X, {'rank': K, 'U': U, 'svals': svals, 'V': V, 'E': E}
