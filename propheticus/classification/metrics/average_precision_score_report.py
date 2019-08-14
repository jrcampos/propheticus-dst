import sklearn.metrics

def average_precision_score_report(y_true, y_score, **kwargs):
    Reports = []
    for i, score in enumerate(sklearn.metrics.average_precision_score(y_true, y_score, **kwargs)):
        Reports.append(str(i) + ': ' + str(score))

    return '\n'.join(Reports)
