import numpy
import sklearn.feature_selection
from sklearn.base import BaseEstimator, TransformerMixin


class MIC(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8, random_state=None):
        self.threshold = threshold
        self.random_state = random_state

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        if hasattr(self, 'indexes_'):
            del self.indexes_

    def fit(self, X, Y=None):
        self._reset()

        mi = sklearn.feature_selection.mutual_info_classif(X, Y, random_state=self.random_state)
        self.indexes_ = sorted(range(len(mi)), key=lambda i: mi[i])[int(self.threshold * len(mi)):]  # Removes 1 - 0.X

        return self

    def transform(self, X):
        return numpy.delete(X, self.indexes_, 1)
