from scipy.integrate import quad
from scipy.optimize import root_scalar
import numpy as np

# TODO: document beta


def get_mp_pdf(beta):
    """
    Gets the MP PDF.

    Parameters
    ----------
    beta: float
        TODO

    Output
    ------
    pdf, beta_minus, beta_plus

    pdf: callable
        The pdf function.

    beta_minus, beta_plus: float
        The lower and upper bound of the support.
    """

    beta_minus = (1 - np.sqrt(beta)) ** 2
    beta_plus = (1 + np.sqrt(beta)) ** 2

    def pdf(x):
        # assert (beta_minus <= t) and (t <= beta_plus)
        return np.sqrt((beta_plus - x) * (x - beta_minus)) / (2 * np.pi * x * beta)

    return pdf, beta_minus, beta_plus


def get_mp_cdf(beta):
    """
    Gets the MP CDF

    Parameters
    ----------
    beta: float
        TODO

    Output
    ------
    cdf, beta_minus, beta_plus

    cdf: callable
        The cdf function.

    beta_minus, beta_plus: float
        The lower and upper bound of the support.
    """
    pdf, beta_minus, beta_plus = get_mp_pdf(beta)

    def cdf(x):
        return quad(func=pdf, a=beta_minus, b=x)[0]

    return cdf, beta_minus, beta_plus


def compute_mp_quantile(beta, q=0.5):
    """
    Computes the qualtile for the MP distribution.

    Parameters
    ----------
    beta: float
        TODO

    q: float
        The desired quantile.

    """
    assert 0 < q and q < 1
    cdf, beta_minus, beta_plus = get_mp_cdf(beta)

    def root_func(x):
        return cdf(x) - q

    return root_scalar(f=root_func, method=None,
                       bracket=(beta_minus, beta_plus)).root
