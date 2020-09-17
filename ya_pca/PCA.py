import numpy as np
from numbers import Number
from time import time
from scipy.sparse import issparse

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from warnings import warn

from ya_pca.linalg_utils import svd_wrapper
from ya_pca.rank_selection.rank_selection import select_rank
from ya_pca.rank_selection.variance_explained import safe_frob_norm

# TODO: create a rank selection object. Also re-write fit to possibly use
# the rank selection's max_rank/SVD


class PCA(BaseEstimator, TransformerMixin):
    """
    Principal components analysis.

    Parameters
    ----------
    n_components: None, str, int
        How to choose the number of components.
        If None, will compute all components, if int will compute
        this user specified number of componets. If str, must be one
        of the rank selection methods.

    center: bool
        Whether or not to center the data.

    max_rank: None, int
        Maximum SVD rank to compute for rank selection. Only used
        if also employing a rank selection method.

    rank_sel_kws: dict
        Key work arguments for the given rank selcetion method.

    score_mode: str
        How to score samples. Must be one of ["euclid", "log_lik"].
        If "euclid", computes Euclidean distance from the sample to its
        predcition. If "log_lik", uses the factor model log likelihood.

    Attributes
    ----------
    n_components_: int
        Estimated or specified number of components.

    scores_: array-like, (n_samples, n_components)
        The normalized PCA scores i.e. left singular vectors.

    unnorm_scores_: array-like, (n_samples, n_components)
        The unnormalized PCA scores i.e. left singular vectors scaled by the singular values.

    svals_: array-like, (n_components, )
        The singular values.

    loadings_:
        The normalized PCA loadings i.e. right singular vectors.

    all_svals_: array-like, (n_components_computed, )
        All singular values that were computed e.g. if we compute a full SVD then perform rank selection this stores all the svals.

    cov_evals_: array-like, (n_components, )
        Eigenvalues of the sample covariance matrix.

    var_expl_prop_: array-like, (n_components)
        Variance explained proportion for each component.

    var_expl_cum_
        Cumulative variance explained.

    tot_variance_: float
        The total variance i.e. the square Frobenius norm of the data matrix.

    var_expl_prop_all_: array-like, (n_components_computed, )
        Variance explained proportion for all the computed singular values.

    rank_sel_out_: dict
        Output from the rank selection method

    rank_sel_rank_est_: int
        The estimated rank.

    UDV_:
        Returns U, D, V
        where U = left singular vectors, D = svals, V = right svecs

    metadata_: dict
        Some metadata e.g. fit time.
    """
    def __init__(self, n_components=None, center=True, max_rank=None,
                 rank_sel_kws={}, score_mode='euclid'):

        self.n_components = n_components
        self.center = center
        self.max_rank = max_rank
        self.rank_sel_kws = rank_sel_kws

        self.score_mode = score_mode

    def fit(self, X):

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if issparse(X):
            raise TypeError('PCA does not support sparse input. See '
                            'TruncatedSVD for a possible alternative.')

        # X = self._validate_data(X, dtype=[np.float64, np.float32],
        #                         ensure_2d=True, copy=self.copy)

        self.metadata_ = {'shape': X.shape}
        # if isinstance(X, pd.DataFrame):
        #     sample_names = X.index.values
        #     var_names = X.columns.values

        # possibly center X
        if self.center:
            # assert X is not sparse
            X = np.array(X)
            self.center_ = X.mean(axis=0)
            X = X - self.center_

        # smallest rank the SVD can be
        m = min(X.shape)

        # determine SVD rank to compute
        if self.n_components is None:
            rank2compute = m
            assert self.max_rank is None

        elif isinstance(self.n_components, Number):
            assert self.max_rank is None
            rank2compute = min(self.n_components, m)

        elif isinstance(self.n_components, str):
            if self.max_rank is None:
                rank2compute = m
            else:
                rank2compute = min(self.max_rank, m)

        # compute SVD of data matrix
        start_time = time()
        UDV = svd_wrapper(X, rank=rank2compute)
        self.metadata_['svd_runtime'] = time() - start_time

        self.scores_ = UDV[0]
        self.svals_ = UDV[1]
        self.loadings_ = UDV[2]

        self.all_svals_ = UDV[1]

        # Should we add nans to make really clear we dont have these
        # if len(self.all_svals_) < m:
        #     self.all_svals_ = \
        #         np.concatenate([self.all_svals_,
        #                         [np.nan] * (m - len(self.all_svals_))])

        # compute variance explained, before doing rank selection
        self.tot_variance_ = safe_frob_norm(X) ** 2
        self.var_expl_prop_all_ = self.all_svals_ ** 2 / self.tot_variance_

        # rank selection
        if isinstance(self.n_components, str):

            start_time = time()
            UDV, rank_est, out = select_rank(X=X, method=self.n_components,
                                              UDV=UDV, **self.rank_sel_kws)

            self.metadata_['rank_sel_runtime'] = time() - start_time

            self.rank_sel_out_ = out
            self.rank_sel_rank_est_ = rank_est

            self.set_n_components(rank=rank_est)

            if rank_est == rank2compute:
                warn("The rank selection picked the largest possible"
                     " rank of {}, which may indicate an issue with"
                     " the rank selection method. It may be a good idea"
                     " to look at diagnostics for the rank selection method."
                     .format(rank_est))

        # TODO: possibly get noise estimate

        return self

    def set_n_components(self, rank):
        assert rank <= self.n_components_

        self.scores_ = self.scores_[:, 0:rank]
        self.svals_ = self.svals_[0:rank]
        self.loadings_ = self.loadings_[:, 0:rank]
        return self

    def check_fitted(self):
        if hasattr(self, 'scores_'):
            return True
        else:
            return False

    @property
    def n_components_(self):
        """
        Estimated (or specified) PCA rank
        """
        if self.check_fitted():
            return self.loadings_.shape[1]

    @property
    def cov_evals_(self):
        """
        Eigenvalues of the sample covariance matrix
        """
        n_samples_tr = self.metadata_['shape'][0]
        # TODO: double check
        return (1 / (n_samples_tr - 1)) * self.svals_ ** 2

    @property
    def unnorm_scores_(self):
        """
        Unnormalized scores i.e. if X = UDV.T then this is UD
        """

        # TODO: maybe better name
        if self.check_fitted():
            return self.scores_ * self.svals_

    @property
    def var_expl_prop_(self):
        """
        Variance explaiend proportion for each component.
        """
        if self.check_fitted():
            return self.svals_ ** 2 / self.tot_variance_

    @property
    def var_expl_cum_(self):
        if self.check_fitted():
            return np.cumsum(self.var_expl_prop_)

    @property
    def UDV_(self):
        if self.check_fitted():
            return self.scores_, self.svals_, self.loadings_

    def fit_transform(self, X):
        self.fit(X)
        return self.unnorm_scores_

    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        Examples
        --------
        TODO
        """
        check_is_fitted(self, attributes='loadings_')

        X = check_array(X)
        if hasattr(self, 'center_'):
            X = X - self.center_

        projections = np.dot(X, self.loadings_)
        # if self.whiten:
        #     X_transformed /= np.sqrt(self.explained_variance_)
        return projections

    def inverse_transform(self, X):
        """Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.
        Returns
        -------
        Y: array-like, shape (n_samples, n_features)


        """
        # if self.whiten:
        #     return np.dot(X, np.sqrt(self.explained_variance_[:, np.newaxis]) *
        #                     self.components_) + self.mean_
        # else:
        Y = np.dot(X, self.loadings_)
        if hasattr(self, 'center_'):
            Y = Y + self.center_
        return Y

    def score_samples(self, X):
        if self.score_mode == 'euclid':
            return self.score_samples_euclid(X)

        elif self.score_mode == 'log_lik':
            return self.score_samples_log_lik(X)

        else:
            valid = ['euclid', 'log_lik']
            raise ValueError("Invalid inpute for score_mode: {}."
                             "Must be one of {}.".format(valid))

    def score_samples_euclid(self, X):
        """
        Compute the reconstruction error i.e. the Frobenius norm of the residuals matrix.
        """
        proj = self.transform(X)
        pred = self.inverse_transform(proj)
        resid = X - pred
        return np.linalg.norm(resid, ord=2, axis=1)

    def score_samples_log_lik(self, X):
        """Return the log-likelihood of each sample.
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X : array, shape(n_samples, n_features)
            The data.
        Returns
        -------
        ll : array, shape (n_samples,)
            Log-likelihood of each sample under the current model.
        """
        raise NotImplementedError
        check_is_fitted(self, 'loadings_')

        X = check_array(X)
        if hasattr(self, 'center_'):
            X = X - self.center_

        # n_features = X.shape[1]
        # precision = self.get_precision()
        # log_like = -.5 * (X * (np.dot(X, precision))).sum(axis=1)
        # log_like -= .5 * (n_features * log(2. * np.pi) -
        #                   fast_logdet(precision))
        # return log_like

    def score(self, X, y=None):
        return np.mean(self.score_samples(X))
