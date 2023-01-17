"""
Wrapper class to extend the PyClustering class to the scikit interface
"""
import pyclustering.cluster
import pyclustering.cluster.cure
import numpy
from sklearn.base import BaseEstimator, ClusterMixin

class py_clustering(BaseEstimator, ClusterMixin):
    def __init__(self, algorithm, **kwargs):
        self.algorithm = algorithm
        self.arguments = kwargs
        self.labels_ = None

    def fit(self, X):
        if self.algorithm == 'cure':
            _X = [Item.tolist() for Item in X]
            model = pyclustering.cluster.cure.cure(data=_X, **self.arguments)

        model.process()
        DataByClusters = model.get_clusters()

        Data = [None] * len(X)
        for index, Cluster in enumerate(DataByClusters):
            for item_index in Cluster:
                Data[item_index] = index

        self.labels_ = numpy.array(Data)

class CURE(py_clustering):
    def __init__(self, **kwargs):
        super(CURE, self).__init__('cure', **kwargs)
        
