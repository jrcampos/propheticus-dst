import numpy

class Clustering(object):
    ClusteringPerformanceMetrics = {
        'homogeneity': {
            'package': 'sklearn.metrics',
            'callable': 'homogeneity_score',
            'parameters': {
            }
        },
        'completeness': {
            'package': 'sklearn.metrics',
            'callable': 'completeness_score',
            'parameters': {
            }
        },
        'v_measure': {
            'package': 'sklearn.metrics',
            'callable': 'v_measure_score',
            'parameters': {
            }
        },
        'adjustment_rand': {
            'package': 'sklearn.metrics',
            'callable': 'adjusted_rand_score',
            'parameters': {
            }
        },
        'silhouette': {
            'package': 'sklearn.metrics',
            'callable': 'silhouette_score',
            'use_data_as_truth': True,
            'parameters': {
                'metric': {'type': '', 'default': 'euclidean'},
                'sample_size': {'type': ''},
                'random_state': {'hide': True, 'type': ''}
            }
        },
    }

    ClusteringReportHeaders = {
        'clusters': 'Clusters',
        'homogeneity': 'Homogeneity',
        'completeness': 'Completeness',
        'v_measure': 'V-Measure',
        'adjustment_rand': 'Adjustment Rand',
        'silhouette': 'Silhouette',
    }

    ClusteringAlgorithmsCallDetails = {
        'cure': {
            'package': 'propheticus.clustering.algorithms',
            'callable': 'CURE',
            'parameters': {
                'number_cluster': {'type': ''},
                'number_represent_points': {'type': ''},
                'compression': {'type': ''},
                'ccore': {'type': ''}
            },
            'grid': [
                {
                    'number_cluster': [2, 4, 6]
                }
            ]
        },
        'birch': {
            'package': 'sklearn.cluster',
            'callable': 'Birch',
            'parameters': {
                'threshold': {'type': ''},
                'branching_factor': {'type': ''},
                'n_clusters': {'type': ''},
                'compute_labels': {'type': ''},
                'copy': {'type': ''}
            },
            'grid': [
                {
                    'n_clusters': [2, 4, 6]
                }
            ]
        },
        # 'gaussian_mixture': {
        #     'package': 'sklearn.mixture',
        #     'callable': 'GaussianMixture',
        #     'parameters': {
        #         'n_components': {'type': ''},
        #         'covariance_type': {'type': ''},
        #         'tol': {'type': ''},
        #         'reg_covar': {'type': ''},
        #         'max_iter': {'type': ''},
        #         'n_init': {'type': ''},
        #         'init_params': {'type': ''},
        #         'weights_init': {'type': ''},
        #         'means_init': {'type': ''},
        #         'precisions_init': {'type': ''},
        #         'random_state': {'hide': True, 'type': ''},
        #         'warm_start': {'type': ''},
        #         'verbose': {'type': ''},
        #         'verbose_interval': {'type': ''}
        #     },
        #     'grid': [
        #         {
        #             'n_components': [1, 4, 8],
        #             'covariance_type': ['spherical', 'tied', 'diag', 'full']
        #         }
        #     ]
        # },
        'kmeans': {
            'package': 'sklearn.cluster',
            'callable': 'KMeans',
            'parameters': {
                'n_clusters': {'type': ''},
                'init': {'type': ''},
                'n_init': {'type': ''},
                'max_iter': {'type': ''},
                'tol': {'type': ''},
                'precompute_distances': {'type': ''},
                'verbose': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'copy_x': {'type': ''},
                'n_jobs': {'type': '', 'default': -1},
                'algorithm': {'type': ''}
            },
            'grid': [
                {
                    'n_clusters': [2, 4, 6]
                }
            ]
        },
        'spectral': {
            'package': 'sklearn.cluster',
            'callable': 'SpectralClustering',
            'parameters': {
                'n_clusters': {'type': ''},
                'eigen_solver': {'type': ''},
                'n_components': {'type': ''},
                'random_state': {'hide': True, 'type': ''},
                'n_init': {'type': ''},
                'gamma': {'type': ''},
                'affinity': {'type': ''},
                'n_neighbors': {'type': ''},
                'eigen_tol': {'type': ''},
                'assign_labels': {'type': ''},
                'degree': {'type': ''},
                'coef0': {'type': ''},
                'kernel_params': {'type': ''},
                'n_jobs': {'type': '', 'default': -1},
                'verbose': {'type': ''},
            }
        },
        'dbscan': {
            'package': 'sklearn.cluster',
            'callable': 'DBSCAN',
            'callback': {'package': 'propheticus.core.Clustering', 'callable': 'DBSCAN'},
            'parameters': {
                'eps': {'type': 'float'},
                'min_samples': {'type': 'int'},
                'metric': {'type': 'str'},
                'metric_params': {'type': 'dict'},
                'algorithm': {'type': 'str', 'values': ['auto’, ‘ball_tree’, ‘kd_tree’, ‘brute']},
                'leaf_size': {'type': 'int'},
                'p': {'type': 'float'},
                'n_jobs': {'type': '', 'default': -1}
            },
            'grid': [
                {
                    'eps': numpy.arange(0.1, 5, 0.4),
                    'min_samples': [2, 3, 4]
                }
            ]
        },
        'hierarchical_clustering': {
            'package': 'sklearn.cluster',
            'callable': 'AgglomerativeClustering',
            'callback': {'package': 'propheticus.core.Clustering', 'callable': 'hierarchical_clustering'},
            'parameters': {
                'n_clusters': {'type': ''},
                'affinity': {'type': ''},
                'memory': {'type': ''},
                'connectivity': {'type': ''},
                'compute_full_tree': {'type': ''},
                'linkage': {'type': ''},
                'pooling_func': {'type': ''}
            },
            'grid': [
                {
                    'n_clusters': [2, 4, 6],
                    'linkage': ['average', 'complete', 'single'],
                    'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
                },
                {
                    'n_clusters': [2, 4, 6],
                    'linkage': ['ward']
                }
            ]
        },
        'optics': {
            'package': 'sklearn.cluster',
            'callable': 'OPTICS',
            'parameters': {
                'min_samples': {'type': 'int'},
                'max_eps': {'type': 'float'},
                'metric': {'type': 'str'},
                'p': {'type': 'int'},
                'cluster_method': {'type': 'str'},
                'eps': {'type': 'float'},
                'xi': {'type': 'float'},
                'predecessor_correction': {'type': 'boolean'},
                'min_cluster_size': {'type': 'int'},
                'algorithm': {'type': 'str', 'values': ['auto’, ‘ball_tree’, ‘kd_tree’, ‘brute']},
                'leaf_size': {'type': 'int'},
                'p': {'type': 'float'},
                'n_jobs': {'type': '', 'default': -1}
            },
        },
    }
