import numpy
from sklearn.base import BaseEstimator, TransformerMixin


class EigenValues(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        if hasattr(self, 'indexes_'):
            del self.indexes_

    def fit(self, X, y=None):
        self._reset()

        self.indexes_ = []

        corr = numpy.corrcoef(X, rowvar=0)
        w, v = numpy.linalg.eig(corr)
        for i, eigenvalue in enumerate(w):
            if i not in self.indexes_ and abs(eigenvalue) < self.threshold:
                self.indexes_.append(i)

        return self

    def transform(self, X):
        return numpy.delete(X, self.indexes_, 1)
