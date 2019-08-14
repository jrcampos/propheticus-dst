"""
Contains all the available static configurations
"""
import os
import sys
import multiprocessing
import json

import propheticus.configs

class Config(
    propheticus.configs.Classification,
    propheticus.configs.Clustering,
    propheticus.configs.Sampling,
    propheticus.configs.DimensionalityReduction,
    propheticus.configs.Preprocessing
):
    """
    Contains all the available static configurations

    ...

    Attributes
    ----------
    log_times : bool
        Used to control whether or not to log (show messages) regarding the execution times
    log_status : bool
        Used to control whether or not to log (show messages) regarding the status messages (non-error)
    max_alert_target_correlation : float
        Used to control whether or not to log (show messages) regarding the execution times
    max_alert_dt_feature_importance : float
        Threshold (max) to control whether or not a alert log message is displayed concerning a feature importance in DT algorithms
    use_transparency : bool
        Controls if generated images have transparent background
    publication_format : bool
        Controls if generated images contain detailed information (numbers instead of %, descriptions, ...)
    hash_config_basename : bool
        Controls if generated files' names are hashed or not
    OS_PATH : str
        Current absolute path
    framework_instance : str
        Allows controlling which source folder to use for different data; this can eventually become configurable
    framework_instance_data_path : str
        Defines the path to the datasets; controls whether it is live or not
    framework_instance_generated_path : str
        Controls which folder to use to fetch experiments results to be used for comparison
    max_thread_count : int
        Defines the max number of threads to be used
    validate_uuid : bool
        Controls where or not the user UUID is validated against the options defined in the Config.ValidUUID list
    """
    log_times = True
    """used to control whether or not to log (show messages) regarding the execution times"""

    log_status = True
    """used to control whether or not to log (show messages) regarding the status messages (non-error)"""

    max_alert_target_correlation = 0.35
    """used to control whether or not to log (show messages) regarding the execution times"""

    max_alert_dt_feature_importance = 0.35
    """threshold (max) to control whether or not a alert log message is displayed concerning a feature importance in DT algorithms"""

    use_transparency = True
    """controls if generated images have transparent background"""

    publication_format = use_transparency
    """controls if generated images contain detailed information (numbers instead of %, descriptions, ...)"""

    demonstration_mode = False

    force_configurations_log = True

    classification_conf_matrix_show_sample_count = True

    hash_config_basename = True
    """controls if generated files' names are hashed or not"""

    # OS_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
    OS_PATH = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
    """current absolute path"""

    # High-level paths
    framework_instances_path = os.path.join(OS_PATH, 'instances')
    """this variable defines the framework instances dir"""

    framework_configs_path = os.path.join(OS_PATH, 'propheticus', 'configs')
    framework_demo_path = os.path.join(framework_configs_path, 'demo')
    framework_demo_file_path = os.path.join(framework_demo_path, 'complete_demonstration.xlsx')
    framework_temp_path = os.path.join(OS_PATH, '.temp')
    framework_temp_thread_config_file_path = os.path.join(framework_temp_path, '.thread_conf.txt')

    #
    # framework_instance = 'failure_prediction'
    # # framework_instance = 'vulnerability_prediction'
    # """this variable allows defines the instance that will be used for configuring the framework instantiation; this can eventually become configurable"""

    thread_level = 'best'  # NOTE: can be cv, run, batch, algorithm, best
    """defines the the level where threading is to be made"""

    max_thread_count = multiprocessing.cpu_count() - 2
    # max_thread_count = 1
    if max_thread_count > multiprocessing.cpu_count():
       exit('The number of chosen threads is higher than the number of CPUs. Redefining to MAX_CPU - 1')

    """defines the max number of threads to be used"""

    validate_uuid = False
    """controls where or not the user UUID is validated against the options defined in the Config.ValidUUID list"""

    pool_maxtasksperchild = 5

    MetricsByScenarios = {
        'Business-critical': ['f2-score', 'recall'],
        'Heightened-critical': ['informedness', 'recall'],
        'Best effort': ['f1-score', 'recall'],
        'Minimum effort': ['markedness', 'precision']
    }
    ComparisonScenarios = list(MetricsByScenarios.keys())

    ClassesMapping = {
        'Binary_Base': 0,
        'Binary_Error': 9999,

        'RemoveInvalidDeltaTL': 8,  # NOTE: used to samples which do not have enough lead time between injection and failure
        'RemoveDeltaTL': 9,  # NOTE: used to samples which are already after the lead time separation

        'Baseline': 0,
        'Event': 1
    }

    ClassesDescription = {
        ClassesMapping['Binary_Error']: 'General Error',
        ClassesMapping['RemoveInvalidDeltaTL']: 'Invalid DeltaTL',
        ClassesMapping['RemoveDeltaTL']: 'Remove DeltaTL',
    }

    InitialConfigurations = {
            'config_seed_count': 30,
            'config_cv_fold': 10,
            'config_grid_search': False
    }

    ValidUUID = []

    @staticmethod
    def defineDependentPaths(framework_instance):
        Config.framework_instance = framework_instance
        Config.framework_instance_label = framework_instance.replace('_', ' ').title()
        Config.framework_selected_instance_path = os.path.join(Config.framework_instances_path, framework_instance)
        """this variable defines the framework instance dir"""

        Config.framework_instance_artifacts_path = os.path.join(Config.OS_PATH, 'instances_artifacts', framework_instance)
        """this variable defines the framework instance dir"""

        Config.framework_instance_data_path = os.path.join(Config.framework_instance_artifacts_path, 'data')
        """defines the path to the datasets; controls whether it is live or not"""

        Config.framework_instance_generated_path = os.path.join(Config.framework_instance_artifacts_path, 'generated')
        """controls which folder to use to fetch experiments results to be used for comparison"""

        # Low-level paths
        Config.framework_instance_generated_analysis_path = os.path.join(Config.framework_instance_generated_path, '0-analysis')
        Config.framework_instance_generated_classification_path = os.path.join(Config.framework_instance_generated_path, '1-classification')
        Config.framework_instance_generated_clustering_path = os.path.join(Config.framework_instance_generated_path, '2-clustering')
        Config.framework_instance_generated_logs_path = os.path.join(Config.framework_instance_generated_path, '7-log')
        Config.framework_instance_generated_comparisons_path = os.path.join(Config.framework_instance_generated_path, '8-comparison')
        Config.framework_instance_generated_archive_path = os.path.join(Config.framework_instance_generated_path, '9-archive')

# NOTE: THIS IS EXECUTED UPON IMPORT
if os.path.isfile(Config.framework_temp_thread_config_file_path):
    with open(Config.framework_temp_thread_config_file_path, encoding='utf-8') as f:
        ThreadConfigs = json.load(f)
        Config.defineDependentPaths(ThreadConfigs['framework_instance'])
        if ThreadConfigs['thread_level_'] is not None:
            Config.thread_level_ = ThreadConfigs['thread_level_']

