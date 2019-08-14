import numpy

class Classification(object):
    ClassificationPerformanceMetrics = {
        'acc_score': {
            'package': 'sklearn.metrics',
            'callable': 'accuracy_score',
            'use_binarized_results': True,
            'parameters': {
                'normalize': {'type': ''},
                'sample_weight': {'type': ''}
            }
        },
        'roc_auc_score': {
            'package': 'sklearn.metrics',
            'callable': 'roc_auc_score',
            'use_binarized_results': True,
            'use_prediction_probabilities': True,
            'parameters': {
                'average': {'type': '', 'default': 'weighted'},
                'sample_weight': {'type': ''},
                'max_fpr': {'type': ''}
            }
        },
        'roc_auc_score_report': {
            'package': 'propheticus.classification.metrics',
            'callable': 'roc_auc_score_report',
            'use_binarized_results': True,
            'use_prediction_probabilities': True,
            'parameters': {
                'average': {'type': '', 'default': None},
                'sample_weight': {'type': ''},
                'max_fpr': {'type': ''}
            }
        },
        'precision_score': {
            'package': 'sklearn.metrics',
            'callable': 'precision_score',
            'parameters': {
                'labels': {'type': '', 'default': 'propheticus_positive_labels'},
                'pos_label': {'type': ''},
                'average': {'type': '', 'default': 'weighted'},
                'sample_weight': {'type': ''}
            }
        },
        'f05_score': {
            'package': 'sklearn.metrics',
            'callable': 'fbeta_score',
            'parameters': {
                'beta': {'type': '', 'default': 0.5},
                'labels': {'type': '', 'default': 'propheticus_positive_labels'},
                'pos_label': {'type': ''},
                'average': {'type': '', 'default': 'weighted'},
                'sample_weight': {'type': ''}
            }
        },
        'f1_score': {
            'package': 'sklearn.metrics',
            'callable': 'f1_score',
            'parameters': {
                'labels': {'type': '', 'default': 'propheticus_positive_labels'},
                'pos_label': {'type': ''},
                'average': {'type': '', 'default': 'weighted'},
                'sample_weight': {'type': ''}
            }
        },
        'f2_score': {
            'package': 'sklearn.metrics',
            'callable': 'fbeta_score',
            'parameters': {
                'beta': {'type': '', 'default': 2},
                'labels': {'type': '', 'default': 'propheticus_positive_labels'},
                'pos_label': {'type': ''},
                'average': {'type': '', 'default': 'weighted'},
                'sample_weight': {'type': ''}
            }
        },
        'recall_score': {
            'package': 'sklearn.metrics',
            'callable': 'recall_score',
            'parameters': {
                'labels': {'type': '', 'default': 'propheticus_positive_labels'},
                'pos_label': {'type': ''},
                'average': {'type': '', 'default': 'weighted'},
                'sample_weight': {'type': ''}
            }
        },
        'log_loss': {
            'package': 'sklearn.metrics',
            'callable': 'log_loss',
            'use_binarized_results': True,
            'parameters': {
                'eps': {'type': ''},
                'normalize': {'type': ''},
                'sample_weight': {'type': ''},
                # 'labels': {'type': '', 'default': 'propheticus_positive_labels'}
            }
        },
        'hamming_loss': {
            'package': 'sklearn.metrics',
            'callable': 'hamming_loss',
            'use_binarized_results': True,
            'parameters': {
                # 'labels': {'type': '', 'default': 'propheticus_positive_labels'},
                'sample_weight': {'type': ''}
            }
        },
        'average_precision_score': {
            'package': 'sklearn.metrics',
            'callable': 'average_precision_score',
            'use_binarized_results': True,
            'use_prediction_probabilities': True,
            'parameters': {
                'average': {'type': '', 'default': 'weighted'},
                'pos_label': {'type': ''},
                'sample_weight': {'type': ''}
            }
        },
        'average_precision_score_report': {
            'package': 'propheticus.classification.metrics',
            'callable': 'average_precision_score_report',
            'use_binarized_results': True,
            'use_prediction_probabilities': True,
            'parameters': {
                'average': {'type': '', 'default': None},
                'pos_label': {'type': ''},
                'sample_weight': {'type': ''}
            }
        },
        'classification_report': {
            'package': 'sklearn.metrics',
            'callable': 'classification_report',
            'use_binarized_results': True,
            'parameters': {
                'labels': {'type': ''},
                'target_names': {'type': ''},
                'sample_weight': {'type': ''},
                'digits': {'type': '', 'default': 4},
                'output_dict': {'type': ''}
            }
        },
        'confusion_matrix': {
            'package': 'sklearn.metrics',
            'callable': 'confusion_matrix',
            'parameters': {
                'labels': {'type': ''},
                'sample_weight': {'type': ''}
            }
        },
        'specificity': {
            'package': 'propheticus.classification.metrics',
            'callable': 'specificity',
            'parameters': {
                'average': {'type': '', 'default': 'weighted'},
                'labels': {'type': '', 'default': 'propheticus_positive_labels'}
            }
        },
        'specificity_report': {
            'package': 'propheticus.classification.metrics',
            'callable': 'specificity',
            'parameters': {
                'average': {'type': '', 'default': None}
            }
        },
        'informedness': {
            'package': 'propheticus.classification.metrics',
            'callable': 'informedness',
            'parameters': {
                'average': {'type': '', 'default': 'weighted'},
                'labels': {'type': '', 'default': 'propheticus_positive_labels'}
            }
        },
        'informedness_report': {
            'package': 'propheticus.classification.metrics',
            'callable': 'informedness',
            'parameters': {
                'average': {'type': '', 'default': None}
            }
        },
        'markedness': {
            'package': 'propheticus.classification.metrics',
            'callable': 'markedness',
            'parameters': {
                'average': {'type': '', 'default': 'weighted'},
                'labels': {'type': '', 'default': 'propheticus_positive_labels'}
            }
        },
        'markedness_report': {
            'package': 'propheticus.classification.metrics',
            'callable': 'markedness',
            'parameters': {
                'average': {'type': '', 'default': None}
            }
        },

    }

    ClassificationReportHeaders = {
        'classification_report': 'Classification Report',
        'f05_score': 'F0.5-Score',
        'f1_score': 'F1-Score',
        'f2_score': 'F2-Score',
        'roc_auc_score': 'ROC-AUC',
        'roc_auc_score_report': 'ROC-AUC Report',
        'average_precision_score': 'Aver.Prec.',
        'average_precision_score_report': 'Aver.Prec.Report',
        'acc_score': 'Accuracy',
        'precision_score': 'Precision',
        'recall_score': 'Recall',
        'specificity': 'Specificity',
        'specificity_report': 'Specificity Report',
        'informedness': 'Informedness',
        'informedness_report': 'Informedness Report',
        'markedness': 'Markedness',
        'markedness_report': 'Markedness Report',
        'confusion_matrix': 'Confusion Matrix',
        'log_loss': 'LogLoss',
        'hamming_loss': 'Hamming Loss',
        'duration_run': 'Time: Run',
        'duration_train': 'Time: Train',
        'duration_test': 'Time: Test',
        'duration_dimensionality_reduction': 'Time: Dim.Red.',
        'duration_data_balancing': 'Time: DataBalanc.',
        'logs': 'Logs'
    }

    ClassificationAlgorithmsCallDetails = {
        'keras_deep_learning': {
            'package': 'propheticus.classification.algorithms',
            'callable': 'CustomKerasClassifier',
            'dataset_metadata_argument': True,
            'parameters': {
                'build_fn': {'type': ''},  # NOTE: now uses a custom method to create the model, this will override it
                'epochs': {'type': '', 'default': 150},
                'batch_size': {'type': '', 'default': 10},
                'verbose': {'type': '', 'default': 0}
            }
        },
        'adaboost': {
            'package': 'sklearn.ensemble',
            'callable': 'AdaBoostClassifier',
            'parameters': {
                'base_estimator': {'type': ''},
                'n_estimators': {'type': 'int'},
                'learning_rate': {'type': ''},
                'algorithm': {'type': 'str', 'values': ['SAMME', 'SAMME.R']},
                'random_state': {'hide': True, 'type': ''}
            }
        },
        'bagging': {
            'package': 'sklearn.ensemble',
            'callable': 'BaggingClassifier',
            'parameters': {
                'base_estimator': {'type': ''},
                'n_estimators': {'type': '', 'default': 100},
                'max_samples': {'type': ''},
                'max_features': {'type': ''},
                'bootstrap': {'type': ''},
                'bootstrap_features': {'type': ''},
                'oob_score': {'type': ''},
                'warm_start': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'verbose': {'type': ''}
            },
            'grid': [
                {
                    'n_estimators': [100],
                    'max_features': list(numpy.linspace(0.1, 1.0, 2, endpoint=True)),
                    'bootstrap': [False, True],  # , True
                    'bootstrap_features': [False, True],  # , True
                }
            ]
        },
        'decision_tree': {
            'package': 'sklearn.tree',
            'callable': 'DecisionTreeClassifier',
            'parameters': {
                'criterion': {'type': 'str', 'values': ['gini', 'entropy']},
                'splitter': {'type': 'str', 'values': ['best', 'random']},
                'max_depth': {'type': ['int', 'None']},
                'min_samples_split': {'type': ['int', 'float']},
                'min_samples_leaf': {'type': ['int', 'float']},
                'min_weight_fraction_leaf': {'type': 'float'},
                'max_features': {'type': ['int', 'float', 'str', 'None']},
                'random_state': {'hide': True, 'type': ['int', 'None']},
                'max_leaf_nodes': {'type': ['int', 'None']},
                'min_impurity_decrease': {'type': 'float'},
                'class_weight': {'type': ['dict', 'list-dicts', 'balanced', 'None']},
                'presort': {'type': 'bool', 'values': [True, False]}
            },
            'grid': [
                {
                    'criterion': ['gini', 'entropy'],
                    # 'splitter': ['best', 'random'],
                    # 'max_depth': list(range(1, 30, 3)) + [None],
                    'min_samples_split': numpy.linspace(0.1, 1.0, 5, endpoint=True),
                    # 'min_samples_leaf': numpy.linspace(0.1, 0.5, 5, endpoint=True),
                    # 'max_features': numpy.linspace(0.1, 1.0, 5, endpoint=True)
                }
            ]
        },
        'extra_trees': {
            'package': 'sklearn.ensemble',
            'callable': 'ExtraTreesClassifier',
            'parameters': {
                'n_estimators': {'type': ''},
                'criterion': {'type': ''},
                'max_depth': {'type': ''},
                'min_samples_split': {'type': ''},
                'min_samples_leaf': {'type': ''},
                'min_weight_fraction_leaf': {'type': ''},
                'max_features': {'type': ''},
                'max_leaf_nodes': {'type': ''},
                'min_impurity_decrease': {'type': ''},
                'min_impurity_split': {'type': ''},
                'bootstrap': {'type': ''},
                'oob_score': {'type': ''},
                'n_jobs': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'verbose': {'type': ''},
                'warm_start': {'type': ''},
                'class_weight': {'type': ''}
            },
            'grid': [
                {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    # 'max_depth': list(range(1, 40, 10)) + [None],
                    'min_samples_split': [0.1, 2],
                    'min_samples_leaf': [0.1, 1],
                    'max_features': list(numpy.linspace(0.1, 1.0, 2, endpoint=True)) + ['auto'],
                    'bootstrap': [False, True],  # , True
                }
            ]
        },
        'gaussian_process': {
            'package': 'sklearn.gaussian_process',
            'callable': 'GaussianProcessClassifier',
            'parameters': {
                'kernel': {'type': ''},
                'optimizer': {'type': ''},
                'n_restarts_optimizer': {'type': ''},
                'max_iter_predict': {'type': ''},
                'warm_start': {'type': ''},
                'copy_X_train': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'multi_class': {'type': ''},
                'n_jobs': {'type': ''}
            }
        },
        'gradient_boosting': {
            'package': 'sklearn.ensemble',
            'callable': 'GradientBoostingClassifier',
            'parameters': {
                'loss': {'type': ''},
                'learning_rate': {'type': ''},
                'n_estimators': {'type': ''},
                'subsample': {'type': ''},
                'criterion': {'type': ''},
                'min_samples_split': {'type': ''},
                'min_samples_leaf': {'type': ''},
                'min_weight_fraction_leaf': {'type': ''},
                'max_depth': {'type': ''},
                'min_impurity_decrease': {'type': ''},
                'min_impurity_split': {'type': ''},
                'init': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'max_features': {'type': ''},
                'verbose': {'type': ''},
                'max_leaf_nodes': {'type': ''},
                'warm_start': {'type': ''},
                'presort': {'type': ''},
                'validation_fraction': {'type': ''},
                'n_iter_no_change': {'type': ''},
                'tol': {'type': ''}
            },
            'grid': [
                {
                    'loss': ['deviance', 'exponential'],
                    'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                    'n_estimators': [4, 16, 64, 100, 200],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': list(range(1, 30, 3)) + [None],
                    'min_samples_split': list(numpy.linspace(0.1, 1.0, 5, endpoint=True)) + [2],
                    'min_samples_leaf': list(numpy.linspace(0.1, 0.5, 5, endpoint=True)) + [1],
                    'max_features': numpy.linspace(0.1, 1.0, 5, endpoint=True)
                }
            ]
        },
        'knn': {
            'package': 'sklearn.neighbors',
            'callable': 'KNeighborsClassifier',
            'parameters': {
                'n_neighbors': {'type': ''},
                'weights': {'type': ''},
                'algorithm': {'type': ''},
                'leaf_size': {'type': ''},
                'p': {'type': ''},
                'metric': {'type': ''},
                'metric_params': {'type': ''},
                'n_jobs': {'type': ''}
            },
            'grid': [
                {
                    'n_neighbors': range(2, 16, 4),
                    'p': range(1, 6, 2)
                }
            ]
        },
        'logistic_regression': {
            'package': 'sklearn.linear_model',
            'callable': 'LogisticRegression',
            'parameters': {
                'penalty': {'type': ''},
                'dual': {'type': ''},
                'tol': {'type': ''},
                'C': {'type': ''},
                'fit_intercept': {'type': ''},
                'intercept_scaling': {'type': ''},
                'class_weight': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'solver': {'type': ''},
                'max_iter': {'type': ''},
                'multi_class': {'type': ''},
                'verbose': {'type': ''},
                'warm_start': {'type': ''},
                'n_jobs': {'type': ''}
            },
            'grid': [
                {
                    'C': numpy.linspace(0.1, 1.0, 5, endpoint=True),
                    'solver': ['liblinear', 'lbfgs', 'newton-cg']
                }
            ]
        },
        'naive_bayes': {
            'package': 'sklearn.naive_bayes',
            'callable': 'GaussianNB',
            'parameters': {
                'priors': {'type': ''},
                'var_smoothing': {'type': ''}
            }
        },
        'neural_network': {
            'package': 'sklearn.neural_network',
            'callable': 'MLPClassifier',
            'parameters': {
                'hidden_layer_sizes': {'type': ''},
                'activation': {'type': '', 'values': ['identity', 'logistic', 'tanh', 'relu']},
                'solver': {'type': '', 'values': ['lbfgs', 'sgd', 'adam']},
                'alpha': {'type': ''},
                'batch_size': {'type': ''},
                'learning_rate': {'type': '', 'values': ['constant', 'invscaling', 'adaptive']},
                'learning_rate_init': {'type': ''},
                'power_t': {'type': ''},
                'max_iter': {'type': ''},
                'shuffle': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'tol': {'type': ''},
                'verbose': {'type': ''},
                'warm_start': {'type': ''},
                'momentum': {'type': ''},
                'nesterovs_momentum': {'type': ''},
                'early_stopping': {'type': ''},
                'validation_fraction': {'type': ''},
                'beta_1': {'type': ''},
                'beta_2': {'type': ''},
                'epsilon': {'type': ''},
                'n_iter_no_change': {'type': ''}
            },
            'grid': [
                {
                    'hidden_layer_sizes': [(100,), (100, 20)],
                    'activation': ['logistic', 'relu'],  # 'identity', 'tanh',
                    'solver': ['lbfgs', 'adam'],  #  'sgd',
                    'alpha': [0.001, 0.1],
                    'learning_rate': ['constant', 'adaptive']  # 'invscaling',
                }
            ]
        },
        'random_forests': {
            'package': 'sklearn.ensemble',
            'callable': 'RandomForestClassifier',
            'parameters': {
                'n_estimators': {'type': 'int', 'default': 100},
                'criterion': {'type': 'str', 'values': ['gini', 'entropy']},
                'max_depth': {'type': ['int', 'None']},
                'min_samples_split': {'type': ['int', 'float']},
                'min_samples_leaf': {'type': ['int', 'float']},
                'min_weight_fraction_leaf': {'type': 'float'},
                'max_features': {'type': ['int', 'float', 'str', 'None']},
                'max_leaf_nodes': {'type': ['int', 'None']},
                'min_impurity_decrease': {'type': 'float'},
                'min_impurity_split': {'type': 'float'},
                'bootstrap': {'type': 'bool', 'values': [True, False]},
                'oob_score': {'type': 'bool', 'values': [True, False]},
                'n_jobs': {'type': ['int', 'None']},
                'random_state': {'hide': True, 'type': ['int', 'None']},
                'verbose': {'type': 'int'},
                'warm_start': {'type': 'bool', 'values': [True, False]},
                'class_weight': {'type': ['dict', 'list-dicts', 'balanced', 'None']}
            },
            'grid': [
                {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    # 'max_depth': list(range(1, 40, 10)) + [None],
                    'min_samples_split': [0.1, 2],
                    # 'min_samples_leaf': [0.1, 1],
                    # 'max_features': list(numpy.linspace(0.1, 1.0, 2, endpoint=True)) + ['auto'],
                    'bootstrap': [False],  # , True
                    # 'class_weight': ['balanced', {'Non-Vulnerable': 1, 'Vulnerable': 10}, None]
                    'class_weight': ['balanced', None]
                }
            ]
            # [
            #     {
            #         'n_estimators': [100],
            #         'criterion': ['gini', 'entropy'],
            #         # 'max_depth': list(range(1, 40, 10)) + [None],
            #         'min_samples_split': list(numpy.linspace(0.1, 1.0, 3, endpoint=True)),
            #         'min_samples_leaf': list(numpy.linspace(0.1, 0.5, 3, endpoint=True)),
            #         'max_features': numpy.linspace(0.1, 1.0, 4, endpoint=True),
            #         'bootstrap': [False],  # , True
            #         'class_weight': ['balanced', {'Non-Vulnerable': 1, 'Vulnerable': 10}, None]
            #     }
            # ]
        },
        'svm': {
            'package': 'sklearn.svm',
            'callable': 'SVC',
            'parameters': {
                'C': {'type': ''},
                'kernel': {'type': ''},
                'degree': {'type': ''},
                'gamma': {'type': ''},
                'coef0': {'type': ''},
                'shrinking': {'type': ''},
                'probability': {'type': '', 'default': True},
                'tol': {'type': ''},
                'cache_size': {'type': ''},
                'class_weight': {'type': ''},
                'verbose': {'type': ''},
                'max_iter': {'type': ''},
                'decision_function_shape': {'type': ''},
                'random_state': {'hide': True, 'type': ''}
            },
            'grid': [
                {
                    'kernel': ['linear'],
                    'C': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                },
                {
                    'kernel': ['rbf'],
                    'C': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                    'gamma': [0.1, 1, 10]
                },
                {
                    'kernel': ['poly'],
                    'C': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                    'gamma': [0.1, 1, 10],
                    'degree': range(2, 8, 2)
                },
            ]
        }
    }
