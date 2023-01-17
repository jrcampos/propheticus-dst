import numpy
import operator
import propheticus.shared
from sklearn.base import BaseEstimator, ClassifierMixin

class Voting(BaseEstimator, ClassifierMixin):
    def __init__(self, voting):
        self.voting = voting
        self.classes_ = None
    
    def fit(self, predictions, probabilities, Y):
        self.classes_ = sorted(set(Y))
        return self

    def predict(self, predictions, probabilities):
        FinalPredictions = []

        DirectPredictions = numpy.transpose(list(predictions.values()))
        DirectProbabilities = numpy.swapaxes(list(probabilities.values()), 0, 1)

        for index, Predictions in enumerate(DirectPredictions):
            if self.voting == 'hard':
                _Predictions = {}
                for prediction in Predictions:
                    _Predictions[prediction] = _Predictions.get(prediction, 0) + 1
    
                maj_voted = max(_Predictions.items(), key=operator.itemgetter(1))
                FinalPredictions.append(maj_voted[0])
    
            elif self.voting == 'soft':
                probabilities_average = numpy.mean(DirectProbabilities[index], axis=0)
                selected_class_index = probabilities_average.argmax()
                FinalPredictions.append(self.classes_[selected_class_index])
            else:
                propheticus.shared.Utils.printFatalMessage(f'Invalid self.voting parameter passed: {self.voting}')
    
        return numpy.array(FinalPredictions)

    def predict_proba(self, predictions, probabilities):
        InvX = numpy.swapaxes(list(probabilities.values()), 0, 1)
        _probabilities = [numpy.mean(Probabilities, axis=0) for Probabilities in InvX]
        return numpy.array(_probabilities)
