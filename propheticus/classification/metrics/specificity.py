import propheticus.shared.Utils

def specificity(y_true, y_pred, average=None, **kwargs):
    metric = propheticus.shared.Utils.computeMetricsByClass(y_true, y_pred, 'specificity', average, **kwargs)
    if average is None:
        return '\n'.join([str(i) + ': ' + str(value) for i, value in enumerate(metric.values())])
    else:
        return metric
