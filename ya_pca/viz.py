import numpy as np
import matplotlib.pyplot as plt


def scree_plot(svals, **kws):
    """
    Scree plot of singular values with nice formatting (including 1 indexing).

    Parameters
    ----------
    svals: array-like, (n, )

    **kws: key word arguments for plt.plot
    """
    index = np.arange(1, len(svals) + 1)
    plt.plot(index, svals, marker='.', **kws)
    plt.xlabel("Component")
    # plt.ylabel("Singular_value")
    plt.xlim(0)
    # plt.ylim(0)
