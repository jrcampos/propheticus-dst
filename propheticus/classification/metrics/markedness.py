import propheticus.shared.Utils

def markedness(y_true, y_pred, average=None, **kwargs):
    metric = propheticus.shared.Utils.computeMetricsByClass(y_true, y_pred, 'markedness', average, **kwargs)
    if metric is None:
        raise ValueError(f'Markedness metric returned None.' + ' Only one class present in y_true' if len(set(y_true)) == 1 else '')

    if average is None:
        return '\n'.join([str(i) + ': ' + str(value) for i, value in enumerate(metric.values())])
    else:
        return metric
