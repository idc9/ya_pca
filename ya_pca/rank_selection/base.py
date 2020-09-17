
# TODO: finish this!
class PCARankSelection(object):

    def __init__(self, max_rank=None):
        self.max_rank = max_rank

    @property
    def max_rank_(self):
        if hasattr(self, '_max_rank_'):
            return self._max_rank_
        else:
            return max_rank

    def fit_transform(self, X, UDV=None):

        UDV = svd_wrapper(X, rank=self.max_rank_)



        return UDV
