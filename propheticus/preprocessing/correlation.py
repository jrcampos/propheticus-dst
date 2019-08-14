import numpy
from sklearn.base import BaseEstimator, TransformerMixin


class Correlation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        if hasattr(self, 'indexes_'):
            del self.indexes_

    def fit(self, X, Y=None):
        self._reset()

        self.indexes_ = []

        # TODO: this should validate that the existing dataset does not contain features with 0 variance, otherwise will throw error
        # is_tnan = numpy.transpose(X)
        corr = numpy.corrcoef(numpy.transpose(X))
        for i in range(len(corr) - 1):
            if len(numpy.argwhere(corr[i][i + 1:] > self.threshold)) > 0:
                self.indexes_.append(i)

        return self

    def transform(self, X):
        return numpy.delete(X, self.indexes_, 1)
