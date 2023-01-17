import numpy
import sklearn.feature_selection
from sklearn.base import BaseEstimator, TransformerMixin

import propheticus.shared

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

        oVT = sklearn.feature_selection.VarianceThreshold()
        oVT.fit(X)
        NullIndexes = numpy.where(oVT.variances_ == 0)[0]
        if len(NullIndexes) > 0:
           propheticus.shared.Utils.printFatalMessage(f'Data contained variables with null variance ({NullIndexes}), which is not possible to calculate the correlation matrix. Combine with variance feature selection')

        corr = numpy.corrcoef(numpy.transpose(X))
        for i in range(len(corr) - 1):
            if len(numpy.argwhere(corr[i][i + 1:] > self.threshold)) > 0:
                self.indexes_.append(i)

        return self

    def transform(self, X):
        return numpy.delete(X, self.indexes_, 1)
