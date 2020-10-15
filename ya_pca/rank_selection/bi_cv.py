import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
import pandas as pd

from warnings import warn

from ya_pca.linalg_utils import svd_wrapper, rand_orthog


def select_rank_bi_cv(X, max_rank=None, krow=2, kcol=2,
                      random_state=None, rotate=False):
    """
    Estimates the PCA rank using the bi-cross validation method discussed in
    (Owen and Perry, 2009).

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The data matrix.

    max_rank: None, int
        (optional) Maximum rank to compute.

    krow, kcol: int
        Number of row/column folds.

    random_state: None, int
        Random seed.

    rotate: bool
        Whether or not to apply a random rotation to the data matrix.
        This can help if the true PCs are sparse.

    Output
    ------
    rank_est, out

    rank_est: int
        Estimated rank

    out: dict
        All output from rank selection algorithm.
    """
    rng = check_random_state(random_state)
    bicv_splitter = BiCvFolds(shape=X.shape, krow=krow, kcol=kcol,
                              random_state=random_state)

    # setup rank sequence
    if max_rank is None:
        max_rank = bicv_splitter.max_rank_
    max_rank = min(max_rank, bicv_splitter.max_rank_)
    ranks = np.arange(0, max_rank + 1)  # 0, 1, ..., max_rank

    # possibly rotate columns and rows of X
    if rotate:
        X = rand_orthog(n=X.shape[0], K=X.shape[0], random_state=rng) @ X
        X = X @ rand_orthog(n=X.shape[1], K=X.shape[1], random_state=rng)

    test_row_folds = []
    test_col_folds = []
    grid_errors = np.zeros((krow, kcol, len(ranks)))

    for X_tr, Y_tr, X_tst, Y_tst, row_idx, col_idx, test_rows, test_cols in \
            bicv_splitter.iter_matrix(X):

        # compute SVD of training block
        UDV_tr = svd_wrapper(X_tr, rank=max_rank)

        # compute error or each test block
        for rank_idx, rank in enumerate(ranks):

            grid_errors[row_idx, col_idx, rank_idx] = \
                get_error(UDV_tr=UDV_tr, rank=rank,
                          Y_tr=Y_tr, X_tst=X_tst, Y_tst=Y_tst)

            test_col_folds.append(test_rows)
            test_row_folds.append(test_cols)

    # sum over folds
    errors = grid_errors.sum(axis=(0, 1))
    errors = errors / (np.product(X.shape))  # divide by number of elements

    # format
    errors = pd.Series(errors, index=ranks)
    errors.index.name = 'rank'
    errors.name = 'cv_mse'

    # pick the best rank!
    rank_est = errors.idxmin()

    if rank_est == max_rank:
        warn("Bi-CV picked the largest possible rank of {}, which suggests "
             "we did not examine a enough rank. Try increasing krow/kcol.".
             format(rank_est))

    return rank_est, {'errors': errors,
                      'grid_errors': grid_errors,
                      'test_row_folds': test_row_folds,
                      'test_col_folds': test_col_folds}


def get_folds(n, n_splits=2, random_state=None):
    rng = check_random_state(random_state)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=rng)

    return [tst_idxs for (tr_idxs, tst_idxs) in splitter.split(np.arange(n))]


class BiCvFolds(object):
    """
    Iterates folds for bi-cross validation.
    """

    def __init__(self, shape, krow=2, kcol=2, random_state=None):

        rng = check_random_state(random_state)

        self.shape = shape
        self.row_splitter = KFold(n_splits=krow, shuffle=True,
                                  random_state=rng)
        self.col_splitter = KFold(n_splits=kcol, shuffle=True,
                                  random_state=rng)

        self.krow = krow
        self.kcol = kcol
        self.random_state = rng

    @property
    def max_rank_(self):
        n_rows, n_cols = self.shape
        return int(np.floor(min(n_rows - (n_rows / self.krow),
                                n_cols - (n_cols / self.kcol))))

    def split(self):
        row_folds = self.row_splitter.split(np.arange(self.shape[0]))
        col_folds = self.col_splitter.split(np.arange(self.shape[1]))
        for row_idx, (train_rows, test_rows) in enumerate(row_folds):
            for col_idx, (train_cols, test_cols) in enumerate(col_folds):
                yield row_idx, train_rows, test_rows, \
                    col_idx, train_cols, test_cols

    def iter_matrix(self, X):
        for row_idx, train_rows, test_rows, \
                col_idx, train_cols, test_cols in self.split():

            X_tr = X[train_rows, :][:, train_cols]
            Y_tr = X[train_rows, :][:, test_cols]

            X_tst = X[test_rows, :][:, train_cols]
            Y_tst = X[test_rows, :][:, test_cols]

            yield X_tr, Y_tr, X_tst, Y_tst, \
                row_idx, col_idx, test_rows, test_cols


def get_error(UDV_tr, rank, Y_tr, X_tst, Y_tst):
    """
    Computes the type (I) residual from (Owen and Perry, 2009) i.e. Equation (3.3)
    """
    if rank == 0:
        return np.linalg.norm(Y_tst) ** 2

    # training PCA scores
    U_tr, D_tr, V_tr = UDV_tr[0][:, 0:rank], UDV_tr[1][0:rank], \
        UDV_tr[2][:, 0:rank]

    # TODO: safe invert D_tr
    # regress Y train  on train projections
    beta_tr = (U_tr * safe_invert(D_tr)).T @ Y_tr

    # project test data onto PCA subspace
    # then compute predictions
    Y_tst_pred = X_tst @ V_tr[:, 0:rank] @ beta_tr

    return np.linalg.norm(Y_tst - Y_tst_pred) ** 2


def safe_invert(x, eps=1e-10):
    """
    Safely inverts the elements of an array.
    Zero elements are inverted to 0.
    """
    non_zero_mask = np.abs(x) > eps

    x_inv = np.zeros(len(x))
    x_inv[non_zero_mask] = 1 / np.array(x)[non_zero_mask]
    return x_inv
