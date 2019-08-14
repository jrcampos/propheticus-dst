import numpy

class DimensionalityReduction(object):
    DimensionalityReductionCallDetails = {
        'variance': {
            'package': 'propheticus.preprocessing',
            'callable': 'Variance',
            'parameters': {
                'threshold': {'type': 'float'},
            }
        },
        'correlation': {
            'package': 'propheticus.preprocessing',
            'callable': 'Correlation',
            'parameters': {
                'threshold': {'type': 'float'},
                'propheticus_order': {'type': 'int'}
            }
        },
        'eigenvalues': {
            'package': 'propheticus.preprocessing',
            'callable': 'EigenValues',
            'parameters': {
                'threshold': {'type': 'float'},
                'propheticus_order': {'type': 'int'}
            }
        },
        'mic': {
            'package': 'propheticus.preprocessing',
            'callable': 'MIC',
            'parameters': {
                'threshold': {'type': 'float'},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_order': {'type': 'int'}
            }
        },
        'rfe': {
            'package': 'propheticus.preprocessing',
            'callable': 'RFE',
            'parameters': {
                'estimator_package': {'type': 'str', 'default': 'sklearn.tree'},
                'estimator_callable': {'type': 'str', 'default': 'DecisionTreeClassifier'},
                'estimator_arguments': {'type': 'dict', 'default': {'random_state': None}},
                'step': {'type': 'int', 'default': 2},
                'scoring': {'type': 'str'},
                'n_jobs': {'type': 'int'},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_order': {'type': 'int'}
            }
        },
        'pca': {
            'package': 'sklearn.decomposition',
            'callable': 'PCA',
            'parameters': {
                'n_components': {'type': ['int', 'float', 'None', 'str'], 'default': 0.99},
                'whiten': {'type': 'bool', 'values': [True, False]},
                'svd_solver': {'type': 'str', 'values': ['auto', 'full', 'arpack', 'randomized']},
                'tol': {'type': 'float'},
                'iterated_power': {'type': ['int', 'str'], 'values': {'str': 'auto'}},
                'random_state': {'hide': True, 'type': ['int', 'None']},
                'propheticus_order': {'type': 'int'}
            }
        },
    }
