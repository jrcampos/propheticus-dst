"""
Contains the code required to perform batch execution of the framework
"""
import collections
import datetime
import os

import xlrd
import time
import traceback
import openpyxl

import propheticus
import propheticus.core
import propheticus.shared

import sys
sys.path.append(propheticus.Config.framework_selected_instance_path)
if os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, 'InstanceConfig.py')):
    from InstanceConfig import InstanceConfig as Config
else:
    import propheticus.Config as Config

class BatchExecution(object):
    """
    Contains the code required to perform batch execution of the framework

    ...

    Attributes
    ----------
    overwrite_runs : bool
    """

    def __init__(self, Context):
        self.Configurations = []

        self.Context = Context
        self.overwrite_runs = False
        self.parallelize_runs = True
        self.display_visuals = not self.parallelize_runs
        self.Processes = []


    def validateInstanceConfigurations(self):
        """
        Validates the configurations defined by the InstanceBatchExecution

        Returns
        -------

        """
        if len(self.Processes) == 0:
            propheticus.shared.Utils.printFatalMessage('Processes to be executed must be defined and contain at least one item!')

        if len(self.Configurations) == 0:
            propheticus.shared.Utils.printFatalMessage('At least one configuration must be defined!')
        else:
            BaseConfigs = self.Configurations[0].keys()
            for index, Configuration in enumerate(list(self.Configurations)):
                if Configuration.keys() != BaseConfigs:
                    propheticus.shared.Utils.printFatalMessage('All experiments must have the same configurable fields!')

    def _processExistingExperiment(self, logs_path, Configuration, filename):
        # Workbook = xlrd.open_workbook(os.path.join(logs_path, filename))
        # Worksheet = Workbook.sheet_by_index(0)

        Workbook = openpyxl.load_workbook(os.path.join(logs_path, filename))
        Worksheet = Workbook.worksheets[0]
        experiment_configuration = Worksheet.cell(1, 1).value

        ParsedConfigurations = propheticus.shared.Utils.parseLogStringConfiguration(experiment_configuration)
        ExistingConfigurations = {key: value for key, value in ParsedConfigurations.items() if key in Configuration}
        existing_hash = propheticus.shared.Utils.getConfigurationsIdentifier(ExistingConfigurations)

        return existing_hash

    def run(self):
        """
        Runs the selected configuration by performing batch execution of the framework

        Returns
        -------

        """
        self.validateInstanceConfigurations()

        warning_threshold = 200
        print()

        limit_1 = False
        if limit_1 is True:
            self.Configurations = self.Configurations[:1] # TODO REMOVE!!
            propheticus.shared.Utils.printWarningMessage(f'Limiting batch execution to 1 experiment! Typically used for debugging/testing')

        initial_length = len(self.Configurations)
        if self.overwrite_runs is False:
            if initial_length > warning_threshold:
               propheticus.shared.Utils.printWarningMessage(f'The validation of which configurations were already run may take a while ({initial_length})')

            ExistingLogHashesByDataset = {}
            for index, Configuration in enumerate(list(self.Configurations)):
                logs_path = os.path.join(propheticus.Config.framework_instance_generated_logs_path, propheticus.shared.Utils.getDatasetsIdentifiers(''.join(sorted(Configuration['datasets']))))

                if not os.path.isdir(logs_path):
                    continue

                if logs_path not in ExistingLogHashesByDataset:
                    ExistingLogs = sorted([file for file in os.listdir(logs_path) if '.Log.' in file and '~' not in file])
                    if len(ExistingLogs) == 0:
                        continue

                    propheticus.shared.Utils.printStatusMessage(f'Parsing log directory: {logs_path} ({len(ExistingLogs)})')
                    file_pool_count = min(propheticus.Config.max_thread_count, len(ExistingLogs))
                    ExistingLogHashesByDataset[logs_path] = propheticus.shared.Utils.pool(
                        file_pool_count,
                        self._processExistingExperiment,
                        [(logs_path, Configuration, filename) for filename in ExistingLogs]
                    )

                    propheticus.shared.Utils.printStatusMessage('Parsed log directory: #' + str(len(ExistingLogHashesByDataset[logs_path])))

                current_hash = propheticus.shared.Utils.getConfigurationsIdentifier(Configuration)
                if current_hash in ExistingLogHashesByDataset[logs_path]:
                    self.Configurations.remove(Configuration)
                    if initial_length < warning_threshold:
                        propheticus.shared.Utils.printStatusMessage('Skipped: ' + ','.join(propheticus.shared.Utils.getSafeConfigurations(Configuration)), force=True)

            skipped_runs = initial_length - len(self.Configurations)
            if skipped_runs > 0:
                propheticus.shared.Utils.printStatusMessage('Runs skipped: ' + str(skipped_runs))

            existing_experiments = sum([len(Experiments) for logs_path, Experiments in ExistingLogHashesByDataset.items()])
            if skipped_runs != existing_experiments:
                propheticus.shared.Utils.printWarningMessage('The number of skipped runs does not match the number of existing experiments: ' + str(skipped_runs) + ' != ' + str(existing_experiments), acknowledge=True)

        if len(self.Configurations) == 0:
            propheticus.shared.Utils.printWarningMessage('No configurations remain to run')
        else:
            # self.Configurations = [self.Configurations[1]]  # TODO: REMOVE!!
            propheticus.shared.Utils.printStatusMessage('Running batch configurations: ' + str(len(self.Configurations)), force=True)

            confirm = propheticus.shared.Utils.printConfirmationMessage('This will override previous results for the same configurations. Continue?')
            if confirm == 'y':
                thread_level = propheticus.shared.Utils.getBestParallelizationLocation(self.Configurations, self.Context.DatasetsConfigurations)
                if thread_level == propheticus.shared.Utils.THREAD_LEVEL_BATCH:
                    propheticus.shared.Utils.printStatusMessage('Parallelization at level: ' + thread_level)

                ConfigurationsByAlgorithm = [str(Configuration['proc_classification']) for Configuration in self.Configurations if 'proc_classification' in Configuration]
                propheticus.shared.Utils.printStatusMessage('Configurations by algorithm: ' + ','.join([key + ': ' + str(value) for key, value in collections.Counter(ConfigurationsByAlgorithm).items()]))

                self.pool_count = min(propheticus.Config.max_thread_count, len(self.Configurations))

                if self.parallelize_runs is True and thread_level == propheticus.shared.Utils.THREAD_LEVEL_BATCH and self.pool_count > 1:
                    Config.thread_level_ = thread_level
                    propheticus.shared.Utils.pool(self.pool_count, self._run, [(index, Configs, 'batch') for index, Configs in enumerate(self.Configurations)])
                else:
                    [self._run(index, Configs, 'cli') for index, Configs in enumerate(self.Configurations)]

    def _run(self, index, Configuration, mode):
        """
        Runs the selected configuration by performing batch execution of the framework

        Parameters
        ----------
        index : int
        Configuration : dict
        mode : str

        Returns
        -------

        """
        # NOTE: Limited validation, use with caution, mainly with complex structures

        # try:
        prev_log_times = Config.log_times
        prev_log_status = Config.log_status
        prev_hide_demo_popups = propheticus.shared.Utils.hide_demo_popups

        Config.log_times = Config.log_status = True if len(self.Configurations) == 1 or self.pool_count == 1 or mode == 'cli' else False
        propheticus.shared.Utils.hide_demo_popups = True

        start_time = time.time()
        oPropheticus = propheticus.core.GUI()
        oPropheticus.mode = mode  # NOTE: this will make the run behave as cli or batch, parallelizing at the cross validation fold level;
        oPropheticus.display_visuals = self.display_visuals
        for key, value in Configuration.items():
            oPropheticus.DatasetsConfigurations[key] = value

        validated = oPropheticus.validateCurrentConfigurations()
        if validated is False:
            return

        oPropheticus.defineExcludeFeaturesByStaticValues()

        if 'pre_target' in Configuration:
            oPropheticus.DatasetsConfigurations['pre_target'] = oPropheticus.getFeatureDetailsByName(oPropheticus.DatasetsConfigurations['pre_target'])

        if 'pre_excluded_features' in Configuration:
            _ExcludeFeatures = []
            for feature_name in Configuration['pre_excluded_features']:
                _ExcludeFeatures.append(oPropheticus.getFeatureDetailsByName(feature_name))
            oPropheticus.DatasetsConfigurations['pre_excluded_features'] = _ExcludeFeatures

        if 'pre_filter_feature_values' in Configuration:
            _FilterFeatures = []
            for Filter in oPropheticus.DatasetsConfigurations['pre_filter_feature_values']:
                FeatureDetails = oPropheticus.getFeatureDetailsByName(Filter['label'][:Filter['label'].index(':')])
                FeatureDetails['label'] += ': ' + Filter['values']
                FeatureDetails['values'] = Filter['values']
                _FilterFeatures.append(FeatureDetails)
            oPropheticus.DatasetsConfigurations['pre_filter_feature_values'] = _FilterFeatures

        oPropheticus.defineExcludedFeaturesByLabel()

        run_hash_ = oPropheticus.getConfigurationsIdentifier()
        if self.parallelize_runs is True:
            propheticus.shared.Utils.printStatusMessage('Configuration start # ' + str(index + 1) + f' - {run_hash_} - ' + str(datetime.datetime.now()) + ': \n' + propheticus.shared.Utils.toStringCurrentConfigurations(oPropheticus.DatasetsConfigurations, hide_empty=True), force=True)
        else:
            propheticus.shared.Utils.printStatusMessage('Configuration start # ' + str(index + 1) + f' - {run_hash_} - ' + str(datetime.datetime.now()), force=True)

        for process in self.Processes:
            try:
                if isinstance(process, dict):
                    returned = getattr(oPropheticus, process['method'])(
                        *process['arguments'])  # TODO: change this to named args
                else:
                    returned = getattr(oPropheticus, process)()

                if callable(returned):
                    returned()

            except SystemExit as e:
                propheticus.shared.Utils.printWarningMessage(f'Catching SystemExit exception! {e}')
                traceback.print_exc()

        run_hash = oPropheticus.getConfigurationsIdentifier()
        if run_hash_ != run_hash:
            propheticus.shared.Utils.printFatalMessage('Configuration # ' + str(index + 1) + f' hashes differ: {run_hash} > {run_hash_}')

        log_file = os.path.join(Config.framework_instance_generated_logs_path, propheticus.shared.Utils.getDatasetsIdentifiers(''.join(Configuration['datasets'])), run_hash + '.Log.xlsx')
        if not os.path.isfile(log_file):
            propheticus.shared.Utils.printWarningMessage('Configuration failed to generate log files # ' + str(index) + ' - ' + run_hash + f' -- {log_file}')

        if 'proc_classification' in Configuration:
            executed = Configuration['proc_classification']
        elif 'proc_clustering' in Configuration:
            executed = Configuration['proc_clustering']
        else:
            executed = Configuration['config_load_experiment_models']

        propheticus.shared.Utils.printStatusMessage('Configuration end # ' + str(index + 1) + ' - Algorithm: ' + ', '.join(executed) + ' Duration: ' + str(round((time.time() - start_time) / 60, 1)) + ' min - ' + str(datetime.datetime.now()) + ' ' + run_hash, force=True)
        print('')

        Config.log_times = prev_log_times
        Config.log_status = prev_log_status
        propheticus.shared.Utils.hide_demo_popups = prev_hide_demo_popups

        # except Exception as e:
        #     propheticus.shared.Utils.printWarningMessage(f'A terminal exception occurred and was caugth on the most outter call: {e} ; \n{Configuration}')

