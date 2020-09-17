from joblib import Parallel, delayed
from sklearn.utils import check_random_state
import numpy as np

from inspect import getargspec


def sample_random_seeds(n_seeds, random_state=None):
    """
    Samples a bunch of random seeds.

    Parameters
    ----------
    n_seeds: int
        Number of seeds to samaple.

    random_state: None, int
        Metaseed that determines the random seeds.
    """
    # get seeds for each sample
    rng = check_random_state(random_state)
    return rng.randint(np.iinfo(np.int32).max, size=n_seeds)


def sample_parallel(fun, kws={}, n_samples=100,
                    random_state=None, n_jobs=None, backend=None):
    """
    Computes samples possibly in parralel using from sklearn.externals.joblib

    Parameters
    ---------
    fun: callable
        The sampling function. It should take random_state as a key word argument.

    n_samples: int
        Number of samples to draw.

    n_jobs: None, -1, int
        Number of cores to use. If None, will not sample in parralel.
         If -1 will use all available cores.

     **kwargs: args and key word args for fun

    Output
    ------
    samples: list
        Each entry of samples is the output of one call to
        fun(*args, **kwargs)
    """
    if 'random_state' not in getargspec(fun).args:
        raise ValueError("func must take 'random_state' as an argument")

    seeds = sample_random_seeds(n_samples, random_state)

    if n_jobs is not None:
        return Parallel(n_jobs=n_jobs, backend=backend) \
            (delayed(fun)(random_state=seeds[s],
                          **kws) for s in range(n_samples))

    else:
        return [fun(random_state=seeds[s], **kws)
                for s in range(n_samples)]
