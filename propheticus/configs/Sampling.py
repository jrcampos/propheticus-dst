import numpy

class Sampling(object):
    oversampling_ratio = 3
    undersampling_maj_to_min_ratio = 1
    SamplingCallDetails = {
        # OVERSAMPLING
        'SMOTE': {
            'package': 'imblearn.over_sampling',
            'callable': 'SMOTE',
            'type': 'oversampling',
            'parameters': {
                'sampling_strategy': {'type': ['float', 'dict', 'str', 'callable'], 'values': {'str': ['majority', 'not majority', 'not minority', 'all', 'auto']}},
                'k_neighbors': {'type': ['int', 'obj']},
                'n_jobs': {'type': ['int', 'obj']},
                'random_state': {'hide': True, 'type': ['int', 'None']},
                'propheticus_ratio': {'type': 'int', 'default': oversampling_ratio},
                'propheticus_order': {'type': 'int'}
            }
        },
        # TODO: REVIEW FOLLOWING TECHNIQUE; WAS NOT OVERSAMPLING AS CONFIGURED
        # 'ADASYN': {
        #     'package': 'imblearn.over_sampling',
        #     'callable': 'ADASYN',
        #     'type': 'oversampling',
        #     'parameters': {
        #         'sampling_strategy': {'type': ''},
        #         'n_neighbors': {'type': ''},
        #         'n_jobs': {'type': ''},
        #         'random_state': {'hide': True, 'type': ''}
        #     ]
        # },
        'SMOTEENN': {
            'package': 'imblearn.combine',
            'callable': 'SMOTEENN',
            'type': 'oversampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': oversampling_ratio},
                'propheticus_order': {'type': ''}
            }
        },
        'SMOTETomek': {
            'package': 'imblearn.combine',
            'callable': 'SMOTETomek',
            'type': 'oversampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': oversampling_ratio},
                'propheticus_order': {'type': ''}
            }
        },
        'RandomOverSampler': {
            'package': 'imblearn.over_sampling',
            'callable': 'RandomOverSampler',
            'type': 'oversampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': oversampling_ratio},
                'propheticus_order': {'type': ''}
            }
        },


        # UNDERSAMPLING
        'RandomUnderSampler': {
            'package': 'imblearn.under_sampling',
            'callable': 'RandomUnderSampler',
            'type': 'undersampling',
            'parameters': {
                'sampling_strategy': {'type': ['float', 'dict', 'str', 'callable'], 'values': {'str': ['majority', 'not majority', 'not minority', 'all', 'auto']}},
                'replacement': {'type': 'bool', 'values': [True, False]},
                'random_state': {'hide': True, 'type': ['int', 'None']},
                'propheticus_ratio': {'type': 'int', 'default': undersampling_maj_to_min_ratio},
                'propheticus_order': {'type': 'int'}
            }
        },
        'CondensedNearestNeighbour': {
            'package': 'imblearn.under_sampling',
            'callable': 'CondensedNearestNeighbour',
            'type': 'undersampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'return_indices': {'type': ''},
                'n_neighbors': {'type': ''},
                'n_seeds': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': undersampling_maj_to_min_ratio},
                'propheticus_order': {'type': ''}
            }
        },
        'EditedNearestNeighbours': {
            'package': 'imblearn.under_sampling',
            'callable': 'EditedNearestNeighbours',
            'type': 'undersampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'return_indices': {'type': ''},
                'n_neighbors': {'type': ''},
                'kind_sel': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': undersampling_maj_to_min_ratio},
                'propheticus_order': {'type': ''}
            }
        },
        'RepeatedEditedNearestNeighbours': {
            'package': 'imblearn.under_sampling',
            'callable': 'RepeatedEditedNearestNeighbours',
            'type': 'undersampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'return_indices': {'type': ''},
                'n_neighbors': {'type': ''},
                'max_iter': {'type': ''},
                'kind_sel': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': undersampling_maj_to_min_ratio},
                'propheticus_order': {'type': ''}
            }
        },
        'AllKNN': {
            'package': 'imblearn.under_sampling',
            'callable': 'AllKNN',
            'type': 'undersampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'return_indices': {'type': ''},
                'n_neighbors': {'type': ''},
                'allow_minority': {'type': ''},
                'kind_sel': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': undersampling_maj_to_min_ratio},
                'propheticus_order': {'type': ''}
            }
        },
        'InstanceHardnessThreshold': {
            'package': 'imblearn.under_sampling',
            'callable': 'InstanceHardnessThreshold',
            'type': 'undersampling',
            'parameters': {
                'estimator': {'type': ''},
                'sampling_strategy': {'type': ''},
                'return_indices': {'type': ''},
                'cv': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': undersampling_maj_to_min_ratio},
                'propheticus_order': {'type': ''}
            }
        },
        'NearMiss': {
            'package': 'imblearn.under_sampling',
            'callable': 'NearMiss',
            'type': 'undersampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'return_indices': {'type': ''},
                'version': {'type': ''},
                'n_neighbors': {'type': ''},
                'n_neighbors_ver3': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': undersampling_maj_to_min_ratio},
                'propheticus_order': {'type': ''}
            }
        },
        'NeighbourhoodCleaningRule': {
            'package': 'imblearn.under_sampling',
            'callable': 'NeighbourhoodCleaningRule',
            'type': 'undersampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'return_indices': {'type': ''},
                'n_neighbors': {'type': ''},
                'threshold_cleaning': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': undersampling_maj_to_min_ratio},
                'propheticus_order': {'type': ''}
            }
        },
        'OneSidedSelection': {
            'package': 'imblearn.under_sampling',
            'callable': 'OneSidedSelection',
            'type': 'undersampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'return_indices': {'type': ''},
                'n_neighbors': {'type': ''},
                'n_seeds': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': undersampling_maj_to_min_ratio},
                'propheticus_order': {'type': ''}
            }
        },
        'TomekLinks': {
            'package': 'imblearn.under_sampling',
            'callable': 'TomekLinks',
            'type': 'undersampling',
            'parameters': {
                'sampling_strategy': {'type': ''},
                'return_indices': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'propheticus_ratio': {'type': '', 'default': undersampling_maj_to_min_ratio},
                'propheticus_order': {'type': ''}
            }
        },
    }
