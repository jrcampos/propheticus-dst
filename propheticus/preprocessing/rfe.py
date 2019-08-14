import numpy
import sklearn.feature_selection
from sklearn.base import BaseEstimator, TransformerMixin
import propheticus.shared.Utils
import importlib
import copy

class RFE(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_package=None, estimator_callable=None, estimator_arguments=None, random_state=None, **kwargs):
        self.estimator_package = estimator_package
        self.estimator_callable = estimator_callable
        self.estimator_arguments = copy.deepcopy(estimator_arguments)
        self.random_state = random_state
        self.kwargs = kwargs

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        if hasattr(self, 'indexes_'):
            del self.indexes_

    def fit(self, X, Y=None):
        self._reset()

        if 'random_state' in self.estimator_arguments:
            self.estimator_arguments['random_state'] = self.random_state

        if self.estimator_callable is not None and 'estimator' not in self.kwargs:
            loaded_module = importlib.import_module(self.estimator_package)
            model = getattr(loaded_module, self.estimator_callable)(**self.estimator_arguments)
            self.kwargs['estimator'] = model

        # model = sklearn.tree.DecisionTreeClassifier(random_state=self.random_state)

        def RFECVWrapper():
            def custom_scorer(estimator, X, Y):
                Predictions = estimator.predict(X)
                score = sklearn.metrics.recall_score(Y, Predictions, average='weighted')
                return score

            return custom_scorer

        if 'scoring' not in self.kwargs:
            self.kwargs['scoring'] = RFECVWrapper()

        oRFE = sklearn.feature_selection.RFECV(
            cv=sklearn.model_selection.StratifiedKFold(3, shuffle=True, random_state=self.random_state),
            **self.kwargs
        )
        oRFE.fit(X, Y)
        self.indexes_ = [index for index, header in enumerate(X[0]) if index not in oRFE.get_support(indices=True)]

        return self

    def transform(self, X):
        return numpy.delete(X, self.indexes_, 1)
