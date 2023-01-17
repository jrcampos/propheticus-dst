import numpy
import operator
from sklearn.base import BaseEstimator, ClassifierMixin
import os
import sklearn.preprocessing

import propheticus
import propheticus.shared

import sys
sys.path.append(propheticus.Config.framework_selected_instance_path)
if os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, 'InstanceConfig.py')):
    from InstanceConfig import InstanceConfig as Config
else:
    import propheticus.Config as Config

class Stacking(BaseEstimator, ClassifierMixin):
    def __init__(self, algorithm, random_state, voting, algorithm_parameters=None):
        self.algorithm = algorithm
        self.algorithm_parameters = algorithm_parameters
        self.random_state = random_state
        self.voting = voting

        self.classes_ = None
    
    def fit(self, predictions, probabilities, Y):
        AlgorithmCallDetails = Config.ClassificationAlgorithmsCallDetails[self.algorithm]
        AlgorithmCallArguments = self.algorithm_parameters if self.algorithm_parameters else {}

        if self.voting == 'hard':
            DirectPredictions = numpy.transpose(list(predictions.values()))
            self.encoder = sklearn.preprocessing.OneHotEncoder()
            DirectPredictions = self.encoder.fit_transform(DirectPredictions).toarray()
            TrainData = DirectPredictions

        elif self.voting == 'soft':
            DirectProbabilities = _TEMP = numpy.swapaxes(list(probabilities.values()), 0, 1)
            DirectProbabilities = [list(numpy.array(row).flat) for row in DirectProbabilities]
            TrainData = DirectProbabilities

        self.ensemble_model = propheticus.shared.Utils.dynamicAPICall(AlgorithmCallDetails, AlgorithmCallArguments, self.random_state)
        self.ensemble_model.fit(TrainData, Y)

        self.classes_ = sorted(set(Y))
        return self

    def predict(self, predictions, probabilities):

        if self.voting == 'hard':
            DirectPredictions = numpy.transpose(list(predictions.values()))
            DirectPredictions = self.encoder.transform(DirectPredictions).toarray()
            TestData = DirectPredictions

        elif self.voting == 'soft':
            DirectProbabilities = numpy.swapaxes(list(probabilities.values()), 0, 1)
            DirectProbabilities = [list(numpy.array(row).flat) for row in DirectProbabilities]
            TestData = DirectProbabilities

        FinalPredictions = self.ensemble_model.predict(TestData)

        return FinalPredictions

    def predict_proba(self, predictions, probabilities):
        if self.voting == 'hard':
            DirectPredictions = numpy.transpose(list(predictions.values()))
            DirectPredictions = self.encoder.transform(DirectPredictions)
            TestData = DirectPredictions

        elif self.voting == 'soft':
            DirectProbabilities = numpy.swapaxes(list(probabilities.values()), 0, 1)
            DirectProbabilities = [list(numpy.array(row).flat) for row in DirectProbabilities]
            TestData = DirectProbabilities

        FinalProbabilities = self.ensemble_model.predict_proba(TestData)

        return FinalProbabilities
