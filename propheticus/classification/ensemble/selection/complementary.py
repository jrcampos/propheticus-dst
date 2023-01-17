import numpy
import operator
import collections
import itertools
import multiprocessing

import propheticus.shared


def _complementary(TargetCount, Experiments, Comparisons):
    Gains = []
    for compare_algorithms, AlgorithmsComparisons in Comparisons.items():
        exp_a, exp_b = compare_algorithms.split(' > ')
        if exp_a not in Experiments or exp_b not in Experiments:
            continue

        for index, Indexes in enumerate(AlgorithmsComparisons.values()):
            for target, diff in collections.Counter(Indexes).items():
                Gains.append(diff / TargetCount[target])

    return '>'.join(Experiments), sum(Gains) / len(Gains)


def complementary(Target, min_models, max_models, **kwargs):
    PredictionsByAlgorithms = kwargs
    TargetCount = collections.Counter(Target)

    Comparisons = {}
    for _algorithm in sorted(list(PredictionsByAlgorithms.keys())):
        AlgorithmPredictions = PredictionsByAlgorithms[_algorithm]
        for _algorithm2 in sorted(list(PredictionsByAlgorithms.keys())):
            AlgorithmPredictions2 = PredictionsByAlgorithms[_algorithm2]

            compare_algorithms = ' > '.join(sorted([_algorithm, _algorithm2]))
            if _algorithm == _algorithm2 or compare_algorithms in Comparisons:
                continue

            Comparisons[f'{compare_algorithms}'] = {f'{_algorithm}': [], f'{_algorithm2}': []}

            AlgorithmPredictions = numpy.array(AlgorithmPredictions)
            AlgorithmPredictions2 = numpy.array(AlgorithmPredictions2)

            EqualPredictions = (AlgorithmPredictions == AlgorithmPredictions2)
            if EqualPredictions.all():
                propheticus.shared.Utils.printWarningMessage(f'Algorithms passed to the ensemble have exactly the same results! {_algorithm} - {_algorithm2}')

            DifferentPredictions = numpy.where(EqualPredictions == numpy.bool_(False))[0]
            for index in DifferentPredictions:
                if AlgorithmPredictions[index] == Target[index]:
                    Comparisons[f'{compare_algorithms}'][f'{_algorithm}'].append(AlgorithmPredictions[index])
                elif AlgorithmPredictions2[index] == Target[index]:
                    Comparisons[f'{compare_algorithms}'][f'{_algorithm2}'].append(AlgorithmPredictions2[index])

    ExpGain = {}
    _max_models = min(len(PredictionsByAlgorithms.keys()), max_models) + 1
    for i in range(min_models, _max_models):
        for Experiments in itertools.combinations(PredictionsByAlgorithms.keys(), i):
            Gains = []
            for compare_algorithms, AlgorithmsComparisons in Comparisons.items():
                exp_a, exp_b = compare_algorithms.split(' > ')
                if exp_a not in Experiments or exp_b not in Experiments:
                    continue

                for index, Indexes in enumerate(AlgorithmsComparisons.values()):
                    for target, diff in collections.Counter(Indexes).items():
                        Gains.append(diff / TargetCount[target])

            ExpGain['>'.join(Experiments)] = sum(Gains) / len(Experiments)

    SelectedExperiments = max(ExpGain.items(), key=operator.itemgetter(1))[0].split('>')
    return SelectedExperiments
