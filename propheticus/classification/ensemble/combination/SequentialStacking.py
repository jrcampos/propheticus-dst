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

class SequentialStacking(BaseEstimator, ClassifierMixin):
    def __init__(self, algorithm, threshold, random_state, algorithm_parameters=None):
        self.algorithm = algorithm
        self.algorithm_parameters = algorithm_parameters
        self.random_state = random_state
        self.threshold = threshold

        self.classes_ = None
    
    def fit(self, predictions, probabilities, Y):
        AlgorithmCallDetails = Config.ClassificationAlgorithmsCallDetails[self.algorithm]
        AlgorithmCallArguments = self.algorithm_parameters if self.algorithm_parameters else {}

        DirectProbabilities = numpy.swapaxes(list(probabilities.values()), 0, 1)
        DirectProbabilities = [list(numpy.array(row).flat) for row in DirectProbabilities]
        TrainData = DirectProbabilities

        self.ensemble_model = propheticus.shared.Utils.dynamicAPICall(AlgorithmCallDetails, AlgorithmCallArguments, self.random_state)
        self.ensemble_model.fit(TrainData, Y)

        self.classes_ = sorted(set(Y))
        return self

    def predict(self, predictions, probabilities, TargetTest):
        DirectPredictions = numpy.transpose(list(predictions.values()))
        DirectPredictions = [item for sublist in DirectPredictions for item in sublist]

        DirectProbabilities = numpy.swapaxes(list(probabilities.values()), 0, 1)
        DirectProbabilities = numpy.array([list(numpy.array(row).flat) for row in DirectProbabilities])

        RecheckIndexes = []
        for index, _probabilities in enumerate(DirectProbabilities):
            max_prob = max(_probabilities)
            if float(max_prob) < self.threshold:
                RecheckIndexes.append(index)

        if len(RecheckIndexes) > 0:
            corrections = 0
            introduced_errors = 0
            RecheckPredictions = self.ensemble_model.predict(DirectProbabilities[RecheckIndexes])
            for index, new_prediction in enumerate(RecheckPredictions):
                ref_index = RecheckIndexes[index]
                previous_prediction = DirectPredictions[ref_index]

                DirectPredictions[ref_index] = new_prediction

                target_value = TargetTest[ref_index]
                if previous_prediction != target_value:
                    a = 0

                if previous_prediction != new_prediction:
                    if new_prediction == target_value:
                        corrections += 1
                    elif new_prediction != target_value:
                        introduced_errors += 1

            # print(f'Corrections: {corrections};Errors: {introduced_errors}')

        return numpy.array(DirectPredictions)

    def predict_proba(self, predictions, probabilities):
        DirectProbabilities = numpy.swapaxes(list(probabilities.values()), 0, 1)
        DirectProbabilities = [list(numpy.array(row).flat) for row in DirectProbabilities]

        RecheckIndexes = []
        for index, _probabilities in enumerate(DirectProbabilities):
            max_prob = max(_probabilities)
            if float(max_prob) < self.threshold:
                RecheckIndexes.append(index)

        if len(RecheckIndexes) > 0:
            RecheckPredictions = self.ensemble_model.predict_proba(numpy.array(DirectProbabilities)[RecheckIndexes])
            for index, new_prediction in enumerate(RecheckPredictions):
                DirectProbabilities[RecheckIndexes[index]] = new_prediction

        return numpy.array(DirectProbabilities)
