"""
Contains the code required to perform batch execution of the framework
"""
import collections
import datetime
import itertools
import multiprocessing
import os
import sys
import time

import xlrd

import propheticus
import propheticus.core
import propheticus.shared

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

    def run(self):
        """
        Runs the selected configuration by performing batch execution of the framework

        Returns
        -------

        """
        self.validateInstanceConfigurations()

        warning_threshold = 200
        print()

        initial_length = len(self.Configurations)
        if self.overwrite_runs is False:
            if initial_length > warning_threshold:
                propheticus.shared.Utils.printWarningMessage('The validation of which configurations were already run may take a while')

            ExistingLogHashesByDataset = {}
            for index, Configuration in enumerate(list(self.Configurations)):
                logs_path = os.path.join(propheticus.Config.framework_instance_generated_logs_path, propheticus.shared.Utils.getDatasetsIdentifiers(''.join(Configuration['datasets'])))
                if not os.path.isdir(logs_path):
                    continue

                if logs_path not in ExistingLogHashesByDataset:
                    propheticus.shared.Utils.printStatusMessage('Parsing log directory: ' + logs_path)
                    ExistingLogHashesByDataset[logs_path] = []

                    ExistingLogs = sorted([file for file in os.listdir(logs_path) if '.Log.' in file])
                    for filename in ExistingLogs:
                        Workbook = xlrd.open_workbook(os.path.join(logs_path, filename))
                        Worksheet = Workbook.sheet_by_index(0)
                        experiment_configuration = Worksheet.cell(0, 0).value

                        ParsedConfigurations = propheticus.shared.Utils.parseLogStringConfiguration(experiment_configuration)
                        ExistingConfigurations = {key: value for key, value in ParsedConfigurations.items() if key in Configuration}
                        existing_hash = propheticus.shared.Utils.getConfigurationsIdentifier(ExistingConfigurations)
                        ExistingLogHashesByDataset[logs_path].append(existing_hash)

                    propheticus.shared.Utils.printStatusMessage('Parsed log directory: #' + str(len(ExistingLogHashesByDataset[logs_path])))

                hash = propheticus.shared.Utils.getConfigurationsIdentifier(Configuration)
                if hash in ExistingLogHashesByDataset[logs_path]:
                    self.Configurations.remove(Configuration)
                    if initial_length < warning_threshold:
                        propheticus.shared.Utils.printStatusMessage('Skipped: ' + ','.join(propheticus.shared.Utils.getSafeConfigurations(Configuration)), force=True)

            skipped_runs = initial_length - len(self.Configurations)
            if skipped_runs > 0:
                propheticus.shared.Utils.printStatusMessage('Runs skipped: ' + str(skipped_runs))

            existing_experiments = sum([len(Experiments) for logs_path, Experiments in ExistingLogHashesByDataset.items()])
            if skipped_runs != existing_experiments:
                propheticus.shared.Utils.printWarningMessage('The number of skipped runs does not match the number of existing experiments: ' + str(skipped_runs) + ' != ' + str(existing_experiments), acknowledge=True)

        propheticus.shared.Utils.printStatusMessage('Running batch configurations: ' + str(len(self.Configurations)), force=True)

        confirm = propheticus.shared.Utils.printConfirmationMessage('This will override previous results for the same configurations. Continue?')
        if confirm == 'y':
            if len(self.Configurations) > 0:
                thread_level = propheticus.shared.Utils.getBestParallelizationLocation(self.Configurations, self.Context.DatasetsConfigurations)
                propheticus.shared.Utils.printStatusMessage('Parallelization at level: ' + thread_level)

                ConfigurationsByAlgorithm = [Configuration['proc_classification'][0] for Configuration in self.Configurations]
                propheticus.shared.Utils.printStatusMessage('Configurations by algorithm: ' + ','.join([key + ': ' + str(value) for key, value in collections.Counter(ConfigurationsByAlgorithm).items()]))

                self.pool_count = min(propheticus.Config.max_thread_count, len(self.Configurations))

                prev_log_times = propheticus.Config.log_times
                prev_log_status = propheticus.Config.log_status
                prev_hide_demo_popups = propheticus.shared.Utils.hide_demo_popups

                propheticus.Config.log_times = propheticus.Config.log_status = True if len(self.Configurations) == 1 or self.pool_count == 1 else False
                propheticus.shared.Utils.hide_demo_popups = True

                if self.parallelize_runs is True and thread_level == propheticus.shared.Utils.THREAD_LEVEL_BATCH:
                    propheticus.shared.Utils.pool(self.pool_count, self._run, [(index, Configs, 'batch') for index, Configs in enumerate(self.Configurations)])
                else:
                    [self._run(index, Configs, 'cli') for index, Configs in enumerate(self.Configurations)]

                propheticus.Config.log_times = prev_log_times
                propheticus.Config.log_status = prev_log_status
                propheticus.shared.Utils.hide_demo_popups = prev_hide_demo_popups

            else:
                propheticus.shared.Utils.printWarningMessage('No configurations remain to run')

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
        # NOTE: Not yet thoroughly tested, use with caution, mainly complex structures

        start_time = time.time()
        oPropheticus = propheticus.core.GUI()
        oPropheticus.mode = mode  # NOTE: this will make the run behave as cli or batch, parallelizing at the cross validation fold level;
        oPropheticus.display_visuals = self.display_visuals
        for key, value in Configuration.items():
            oPropheticus.DatasetsConfigurations[key] = value

        if self.parallelize_runs is True:
            propheticus.shared.Utils.printStatusMessage('Configuration start # ' + str(index + 1) + ' - ' + str(datetime.datetime.now()) + ': \n' + propheticus.shared.Utils.toStringCurrentConfigurations(oPropheticus.DatasetsConfigurations, hide_empty=True), force=True)
        else:
            propheticus.shared.Utils.printStatusMessage('Configuration start # ' + str(index + 1) + ' - ' + str(datetime.datetime.now()), force=True)

        validated = oPropheticus.validateCurrentConfigurations()
        if validated is False:
            return

        oPropheticus.defineExcludeFeaturesByStaticValues()

        if 'pre_target' in Configuration:
            oPropheticus.DatasetsConfigurations['pre_target'] = oPropheticus.getFeatureDetailsByName(oPropheticus.DatasetsConfigurations['pre_target'])

        if 'pre_filter_feature_values' in Configuration:
            _FilterFeatures = []
            for Filter in oPropheticus.DatasetsConfigurations['pre_filter_feature_values']:
                FeatureDetails = oPropheticus.getFeatureDetailsByName(Filter['label'][:Filter['label'].index(':')])
                FeatureDetails['label'] += ': ' + Filter['values']
                FeatureDetails['values'] = Filter['values']
                _FilterFeatures.append(FeatureDetails)
            oPropheticus.DatasetsConfigurations['pre_filter_feature_values'] = _FilterFeatures

        oPropheticus.defineExcludedFeaturesByLabel()

        for process in self.Processes:
            if isinstance(process, dict):
                getattr(oPropheticus, process['method'])(*process['arguments'])  # TODO: change this to named args
            else:
                getattr(oPropheticus, process)()

        run_hash = oPropheticus.getConfigurationsIdentifier()
        log_file = os.path.join(propheticus.Config.framework_instance_generated_logs_path, propheticus.shared.Utils.getDatasetsIdentifiers(''.join(Configuration['datasets'])), run_hash + '.Log.xlsx')
        if not os.path.isfile(log_file):
            propheticus.shared.Utils.printErrorMessage('Configuration failed to generate log files # ' + str(index) + ' - ' + run_hash)

        propheticus.shared.Utils.printStatusMessage('Configuration end # ' + str(index + 1) + ' - Algorithm: ' + ', '.join(Configuration['proc_classification']) + ' Duration: ' + str(round((time.time() - start_time) / 60, 1)) + ' min - ' + str(datetime.datetime.now()) + ' ' + run_hash, force=True)
        print('')
