import numpy as np
from itertools import product
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

# from fancyimpute import SoftImpute


def soft_impute_cv(X, K=5, n_lambd_vals=100,
                   max_iters=100, convergence_threshold=0.001,
                   max_rank=None, verbose=False,
                   n_jobs=None, backend=None):
    """
    Performs Wold style cross-validation to select the lambda value for the nuclear norm penalized version of PCA. See (Choi et al, 2017).

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The data matrix.

    K: int

    n_lambd_vals: int

    UDV: None or output of svd_wrapper

    max_rank: None, int

    n_jobs: None, int

    max_iters, convergence_threshold, max_rank, verbose are all arguments
    to fancyimpute.SoftImpute

    Output
    ------
    best_lambd, out

    best_lambd: float
        The selected lambda value.

    out: dict
        Detailed cross-validation output.
    """

    # only import from fancyimpute if we actually need it
    # TODO-HACK: figure out better way to deal with this
    from fancyimpute import SoftImpute

    assert X.ndim == 2
    lambd_max = np.linalg.norm(X, ord=2)  # largest sval
    lambd_vals = np.linspace(start=0, stop=lambd_max, num=n_lambd_vals)

    array_indices = np.array([(i, j) for (i, j) in
                              product(range(X.shape[0]), range(X.shape[1]))])
    index_indices = np.arange(array_indices.shape[0])
    splitter = KFold(n_splits=K, shuffle=True, random_state=234)

    cv_mse = np.empty((K, n_lambd_vals))
    for fold_idx, (tr_idxs, tst_idxs) in \
            enumerate(splitter.split(index_indices)):

        # setup missing mask
        missing_mask = np.zeros_like(X)
        missing_mask = missing_mask.reshape(-1)
        missing_mask[tst_idxs] = 1
        missing_mask = missing_mask.reshape(X.shape)
        missing_mask = missing_mask.astype(bool)

        X_incomplete = X.copy()
        X_incomplete[missing_mask] = np.nan

        def fun(lambd_val):
            # fit soft impute for each lambda value
            si = SoftImpute(shrinkage_value=lambd_val,
                            init_fill_method='mean',
                            max_rank=max_rank,
                            verbose=verbose,
                            max_iters=max_iters,
                            convergence_threshold=convergence_threshold)
            X_filled = si.fit_transform(X_incomplete.copy())
            return ((X_filled[missing_mask] - X[missing_mask]) ** 2).mean()

        if n_jobs is not None:
            MSEs = Parallel(n_jobs=n_jobs, backend=backend)(delayed(fun)(
                lambd_val) for lambd_val in lambd_vals)

        else:
            MSEs = [fun(lambd_vals) for lambd_vals in lambd_vals]

        cv_mse[fold_idx, :] = MSEs

        # # TODO: parallelize this!
        # for lambd_idx, lambd_val in enumerate(lambd_vals):

        #     # fit soft impute for each lambda value
        #     si = SoftImpute(shrinkage_value=lambd_val,
        #                     init_fill_method='mean',
        #                     max_rank=max_rank,
        #                     verbose=verbose,
        #                     max_iters=max_iters,
        #                     convergence_threshold=convergence_threshold)
        #     X_filled = si.fit_transform(X_incomplete.copy())

        #     # get test error
        #     cv_mse[fold_idx, lambd_idx] = \
        #         ((X_filled[missing_mask] - X[missing_mask]) ** 2).mean()

    tune_mse = cv_mse.mean(axis=0)
    best_lambd = lambd_vals[tune_mse.argmin()]

    return best_lambd, {'tune_mse': tune_mse,
                        'cv_mse': cv_mse,
                        'lambd_vals': lambd_vals}
