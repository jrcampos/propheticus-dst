import sklearn.metrics

def roc_auc_score_report(y_true, y_score, **kwargs):
    Reports = []
    for i, score in enumerate(sklearn.metrics.roc_auc_score(y_true, y_score, **kwargs)):
        Reports.append(str(i) + ': ' + str(score))

    return '\n'.join(Reports)
