import propheticus.shared.Utils
import sklearn.metrics

def informedness(y_true, y_pred, average=None, **kwargs):
    metric = propheticus.shared.Utils.computeMetricsByClass(y_true, y_pred, 'informedness', average, **kwargs)

    if average is not None:
        sklearn_informedness = sklearn.metrics.balanced_accuracy_score(y_true, y_pred, adjusted=True)
        rounded_sklearn_informedness = round(sklearn_informedness, 4)
        rounded_metric= round(metric, 4)
        if len(set(y_true)) == 2 and rounded_sklearn_informedness != rounded_metric:
            propheticus.shared.Utils.printFatalMessage(f'Computed informedness is different than sklearn values! Computed: {rounded_metric}, sklearn: {rounded_sklearn_informedness}')

    if average is None:
        return '\n'.join([str(i) + ': ' + str(value) for i, value in enumerate(metric.values())])
    else:
        return metric
