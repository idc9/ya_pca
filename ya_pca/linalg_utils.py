from scipy.linalg import eigh
from scipy.linalg import svd as full_svd
from sklearn.utils.extmath import svd_flip
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.utils import check_random_state


def svd_wrapper(X, rank=None):
    """
    Computes the (possibly partial) SVD of a matrix. Handles the case where
    X is either dense or sparse.

    Parameters
    ----------
    X: array-like,  shape (N, D)

    rank: rank of the desired SVD (required for sparse matrices)

    Output
    ------
    U, D, V

    U: array-like, shape (N, rank)
        Orthonormal matrix of left singular vectors.

    D: list, shape (rank, )
        Singular values in non-increasing order (e.g. D[0] is the largest).

    V: array-like, shape (D, rank)
        Orthonormal matrix of right singular vectors

    """
    # TODO: give user option to compute randomized SVD

    full = False  # compute the Full SVD
    if rank is None or rank == min(X.shape):
        full = True

    if issparse(X) or not full:
        assert rank <= min(X.shape) - 1  # svds cannot compute the full svd
        scipy_svds = svds(X, rank)
        U, D, V = fix_scipy_svds(scipy_svds)

    else:
        U, D, V = full_svd(X, full_matrices=False)
        V = V.T

        if rank:
            U = U[:, :rank]
            D = D[:rank]
            V = V[:, :rank]

    # enfoce deterministic output
    U, V = svd_flip(U, V.T)
    V = V.T

    return U, D, V


def pca(X, rank=None):
    X = np.array(X)
    m = X.mean(axis=0)
    return (*svd_wrapper(X - m, rank=rank), m)


def fix_scipy_svds(scipy_svds):
    """
    scipy.sparse.linalg.svds orders the singular values backwards,
    this function fixes this insanity and returns the singular values
    in decreasing order

    Parameters
    ----------
    scipy_svds: the out put from scipy.sparse.linalg.svds

    Output
    ------
    U, D, V
    ordered in decreasing singular values
    """
    U, D, V = scipy_svds

    sv_reordering = np.argsort(-D)

    U = U[:, sv_reordering]
    D = D[sv_reordering]
    V = V.T[:, sv_reordering]

    return U, D, V


def eigh_wrapper(A, B=None, rank=None, eval_descending=True):
    """
    Symmetrics eigenvector or genealized eigenvector problem.

    A v = lambda v

    or

    A v = labmda B v

    where A (and B) are symmetric (hermetian).

    Parameters
    ----------
    A: array-like, shape (n x n)

    B: None, array-like, shape (n x n)

    rank: None, int
        Number of

    eval_descending: bool
        Whether or not to compute largest or smallest eigenvalues.
        If True, will compute largest rank eigenvalues and
        eigenvalues are returned in descending order. Otherwise,
        computes smallest eigenvalues and returns them in ascending order.

    Output
    ------
    evals, evecs

    """

    if rank is not None:
        n_max_evals = A.shape[0]

        if eval_descending:
            eigvals_idxs = (n_max_evals - rank, n_max_evals - 1)
        else:
            eigvals_idxs = (0, rank - 1)
    else:
        eigvals_idxs = None

    evals, evecs = eigh(a=A, b=B, eigvals=eigvals_idxs)

    if eval_descending:
        ev_reordering = np.argsort(-evals)
        evals = evals[ev_reordering]
        evecs = evecs[:, ev_reordering]

    evecs = svd_flip(evecs, evecs.T)[0]

    return evals, evecs


def rand_orthog(n, K, random_state=None):
    """
    Samples a random orthonormal matrix. See Section A.1.1 of https://arxiv.org/pdf/0909.3052.pdf

    Output
    ------
    A: array-like, (n, K)
        A random, column orthonormal matrix.
    """
    rng = check_random_state(random_state)

    Z = rng.normal(size=(n, K))
    Q, R = np.linalg.qr(Z)

    s = np.ones(K)
    neg_mask = rng.uniform(size=K) > .5
    s[neg_mask] = -1

    return Q * s


def rand_orthog_qr(n, K, random_state=None):
    """
    Samples a random, column orthonormal matrix of size n x K
    using a QR decomposition.

    Output
    ------
    A: array-like, (n, K)
        A random, column orthonormal matrix.
    """
    rng = check_random_state(random_state)

    A = rng.normal(size=(n, K))
    A, _ = np.linalg.qr(A)

    return A
