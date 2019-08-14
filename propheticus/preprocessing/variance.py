import numpy
import sklearn.feature_selection
from sklearn.base import BaseEstimator, TransformerMixin


class Variance(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0):
        self.threshold = threshold

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        if hasattr(self, 'indexes_'):
            del self.indexes_

    def fit(self, X, Y=None):
        self._reset()

        oVT = sklearn.feature_selection.VarianceThreshold()
        oVT.fit_transform(X)

        self.indexes_ = numpy.where(oVT.variances_ <= self.threshold)[0]

        return self

    def transform(self, X):
        return numpy.delete(X, self.indexes_, 1)
