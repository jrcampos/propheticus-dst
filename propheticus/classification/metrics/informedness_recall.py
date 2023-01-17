import propheticus.shared.Utils
import sklearn.metrics
import sklearn.preprocessing
import collections

def informedness_recall(y_true, y_pred, average=None, **kwargs):
    metric = propheticus.shared.Utils.computeMetricsByClass(y_true, y_pred, 'informedness', average, **kwargs)
    if metric is None:
        raise ValueError(f'Informedness_recall metric returned None.' + ' Only one class present in y_true' if len(set(y_true)) == 1 else '')

    sklearn_informedness = sklearn.metrics.balanced_accuracy_score(y_true, y_pred, adjusted=True)
    rounded_sklearn_informedness = round(sklearn_informedness, 4)
    rounded_metric = round(metric, 4)
    if len(set(y_true)) == 2 and len(set(y_pred)) == 2 and rounded_sklearn_informedness != rounded_metric:
        propheticus.shared.Utils.printFatalMessage(f'Computed informedness is different than sklearn values! Computed: {rounded_metric}, sklearn: {rounded_sklearn_informedness}. Maybe the positive class is not defined? {collections.Counter(y_pred)} vs {collections.Counter(y_true)}(2)')

    max_informedness = 1
    min_informedness = -1
    normalized_sklearn_informedness = (metric - min_informedness) / (max_informedness - min_informedness)

    sklearn_recall = sklearn.metrics.recall_score(y_true, y_pred, labels=kwargs['labels'], average=average)

    normalized_informedness_recall = normalized_sklearn_informedness * sklearn_recall
    return normalized_informedness_recall
