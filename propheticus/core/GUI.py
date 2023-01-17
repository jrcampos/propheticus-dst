#!/usr/bin/env python
"""
Contains all the code referring to the workflow of the all platform and its (G)UI
"""

import re
import os
import numpy
import json
import time
import datetime
import shutil
import random
import collections
import matplotlib
import pathlib
import xlrd
import sys
import pydoc
import inspect
import copy

import propheticus
import propheticus.core
import propheticus.shared

if propheticus.shared.Utils.getOS() == 'linux':
    matplotlib.use('Agg')

sys.path.append(propheticus.Config.framework_selected_instance_path)
if os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, 'InstanceConfig.py')):
    from InstanceConfig import InstanceConfig as Config
else:
    import Config as Config

if Config.demonstration_mode is True and not propheticus.shared.Utils.isMultiProcess():
    propheticus.shared.Utils.initializeDemonstrationVariables()

numpy.random.seed(propheticus.shared.Utils.RandomSeeds[0])
random.seed(propheticus.shared.Utils.RandomSeeds[0])


# TODO: should allow the system to work with unsupervised problems
class GUI(object):
    def __init__(self):
        self.display_visuals = True if propheticus.shared.Utils.getOS() == 'windows' else False
        self.action_context = None
        self.mode = 'cli'
        self.hash_config_basename = Config.hash_config_basename

        self._DataCache = None
        self.Help = {}

        self.initializeInternalConfigurations()
        self.initializeExperimentsComparisons()
        self.initializeDatasetsConfigurations()
        self.initializeHelpMessages()

        self.oDataManagement = propheticus.core.DataManagement(self)

    def quit(self, choice):
        if Config.demonstration_mode is True:
            propheticus.shared.Utils.storeDemonstrationVariables()

    def start(self):
        if Config.validate_uuid is True:
            propheticus.shared.Utils.validateCurrentUserUUID()

        self.mainMenu()

    def initializeInternalConfigurations(self):
        self.InternalConfigurations = {
            'framework_instance': None,
            'datasets_cache': None
        }

    def initializeExperimentsComparisons(self):
        self.ExperimentsConfigurations = {
            'comparison_experiments': [],
            'comparison_scenario': [],
            'comparison_statistics': [],
            'remove_experiments': []
        }

    def initializeDatasetsDirectConfigurations(self):
        self.DatasetsConfigurations['datasets'] = []
        self.DatasetsConfigurations['pre_excluded_features'] = []
        self.DatasetsConfigurations['pre_selected_features'] = []
        self.DatasetsConfigurations['pre_excluded_features_label'] = []
        self.DatasetsConfigurations['pre_filter_feature_values'] = []
        self.DatasetsConfigurations['pre_target'] = False

        self.clearDataCache()

    def initializeDatasetsConfigurations(self):
        # TODO: these preset values should go to the Config.InitialConfigurations, to allow its direct use in Utils
        self.DatasetsConfigurations = {
            'config_truncate_configurations': True,
            'config_binary_classification': False,
            'config_save_complete_model': False,
            'config_save_experiment_models': False,
            'config_load_experiment_models': None,
            'config_ensemble_algorithms': None,
            'config_ensemble_algorithms_parameters': {},
            'config_ensemble_selection': None,
            'config_ensemble_selection_parameters': None,
            # 'config_seed_count': 30,  # NOTE: FROM InitialConfigurations
            # 'config_grid_search': False,
            'config_data_split': 'stratified_cross_validation',
            'config_data_split_parameters': {},
            'config_grid_inner_cv_fold': 3,
            'config_sequential_data': False,
            'config_sliding_window': 1,
            'config_undersampling_threshold': None,

            'datasets_base_name': '',

            'datasets_exclude_classes': [],
            'datasets_exclude_run_classes': [],
            'datasets_classes_remap': [],
            'datasets_positive_classes': [],

            'pre_excluded_features_static': [],

            'proc_balance_data': [],
            'proc_balance_data_parameters': {},
            'proc_classification': [],
            'proc_classification_grid_params': {},
            'proc_classification_algorithms_parameters': {},
            'proc_clustering': [],
            'proc_clustering_grid_params': {},
            'proc_clustering_algorithms_parameters': {},
            'proc_normalize_data': True,
            'proc_reduce_dimensionality': ['variance'],
            'proc_reduce_dimensionality_parameters': {}
        }

        if hasattr(Config, 'InitialConfigurations'):
            for config, value in Config.InitialConfigurations.items():
                self.DatasetsConfigurations[config] = value

        self.DatasetsStaticConfigs = {
            'config_truncate_configurations': {
                'label': 'Truncate Configurations',
                'type': bool,
                'default': True
            },
            'config_binary_classification': {
                'label': 'Binary Classification',
                'type': bool,
                'reset_cache': True
            },
            'config_save_complete_model': {
                'label': 'Save Trained Model on Complete Dataset to File',
                'type': bool,
            },
            'config_save_experiment_models': {
                'label': 'Save Trained Run/Seed Models to File',
                'type': bool,
            },
            'config_load_experiment_models': {
                'label': 'Use Saved Trained Run/Seed Models from File',
                'type': 'items',
                'values': propheticus.shared.Utils.getPersistedModelsExperimentsList(),
                'default': None,
                'allow_empty': True,
                'callback': self.parseDefineLoadExperimentModels
            },
            'config_grid_inner_cv_fold': {
                'label': 'Inner cross-validation folds (grid-search)',
                'type': int,
                'min': 2
            },
            'config_grid_search': {
                'label': 'Use grid-search',
                'type': bool
            },
            'config_seed_count': {
                'label': 'Seed Count',
                'type': int,
                'min': 1,
                'max': 30
            },
            'config_sequential_data': {
                'label': 'Sequential data',
                'type': bool
            },
            'config_undersampling_threshold': {
                'label': 'Maximum data records',
                'type': int,
                'min': 1,
                'default': None,
                'reset_cache': True,
                'allow_empty': True
            },
            'config_sliding_window': {
                'label': 'Sliding Window',
                'type': int,
                'min': 1,
                'default': 1,
                'reset_cache': True
            },
            'proc_normalize_data': {
                'label': 'Normalize data',
                'type': bool,
                'reset_cache': True
            },
            'config_data_split': {
                'label': 'Data split method',
                'type': 'item',
                'values': list(Config.ClassificationDataSplit.keys()),
                'allow_empty': False,
                'callback': self.parseDefineDatSplitChoice
            },
            'config_ensemble_algorithms': {
                'label': 'Ensemble Classification Algorithms Results',
                'type': 'item',
                'values': list(Config.ClassificationEnsembles.keys()),
                'default': None,
                'allow_empty': True,
                'callback': self.parseDefineEnsembleConfigurationChoice
            },
            'config_ensemble_selection': {
                'label': 'Ensemble Selection Heuristic',
                'type': 'item',
                'values': list(Config.ClassificationEnsemblesSelection.keys()),
                'default': None,
                'allow_empty': True,
                'callback': self.parseDefineEnsembleSelectionChoice
            }
        }

        if hasattr(Config, 'StaticConfigurations'):
            for config, value in Config.StaticConfigurations.items():
                if config in self.DatasetsStaticConfigs:
                    propheticus.shared.Utils.printFatalMessage('Invalid static configuration, already exists! ' + str(config))

                # TODO: improve validation of client defined configs
                self.DatasetsStaticConfigs[config] = value

        self.initializeDatasetsDirectConfigurations()
        self.DatasetsConfigurations = collections.OrderedDict(sorted(self.DatasetsConfigurations.items()))

    def parseDefineLoadExperimentModels(self, processed_config):
        ParsedConfigs = [config.split('-')[0].strip() for config in processed_config]
        self.DatasetsConfigurations['config_load_experiment_models'] = ParsedConfigs

    # # TODO: this was to be used in the static configs as a wrapper; however, it is not possible due to multiprocessing, that cannot pickle it
    # def parseDefineStaticConfigChoiceParameters(self, Options, key):
    #     def _parseDefineStaticConfigChoiceParameters(self, processed_config):
    #         if processed_config is not None:
    #             CallDetails = Options[processed_config]
    #             self.menuConfigureCallArguments(key, CallDetails['parameters'], 'DatasetsConfigurations')
    #
    #     return _parseDefineStaticConfigChoiceParameters

    def parseDefineDatSplitChoice(self, processed_config):
        if processed_config is not None:
            CallDetails = Config.ClassificationDataSplit[processed_config]
            self.menuConfigureCallArguments('config_data_split_parameters', CallDetails['parameters'], 'DatasetsConfigurations')

    def parseDefineEnsembleSelectionChoice(self, processed_config):
        if processed_config is not None:
            CallDetails = Config.ClassificationEnsemblesSelection[processed_config]
            self.menuConfigureCallArguments('config_ensemble_selection_parameters', CallDetails['parameters'], 'DatasetsConfigurations')

    def parseDefineEnsembleConfigurationChoice(self, processed_config):
        if processed_config is not None:
            CallDetails = Config.ClassificationEnsembles[processed_config]
            self.menuConfigureCallArguments('config_ensemble_algorithms_parameters', CallDetails['parameters'], 'DatasetsConfigurations')

    '''
    Propheticus UI menus
    '''

    '''
    Initial Menu
    '''

    def mainMenu(self, choice=0):
        MenuData = {
            '1': {'name': 'Explore Datasets', 'callback': self.menuExploreDatasets},
            '2': {'name': 'Process Classification Results', 'callback': self.menuProcessResults}
        }

        if os.path.isfile(os.path.join(Config.framework_selected_instance_path, 'InstanceBatchExecution.py')):
            MenuData['3'] = {'name': 'Run Batch Configurations', 'callback': self.executeBatchExecution}

        if hasattr(self, 'menuCustom'):
            MenuData['4'] = {'name': 'Framework Instance UI', 'callback': self.menuCustom}

        MenuData['-'] = ''
        MenuData['0'] = {'name': 'Quit', 'callback': self.quit}
        MenuData['h'] = {'name': 'Help', 'callback': self.help()}

        propheticus.shared.Utils.printMenu('Main Menu', MenuData)

    '''
    Explore Datasets Menu
    '''

    def menuExploreDatasets(self, choice=0):
        MenuData = {
            '1': {'name': 'Select Datasets', 'callback': propheticus.shared.Utils.menuData(self, 'Datasets Menu', propheticus.shared.Utils.getAvailableDatasets(), self.parseDatasetsChoice)},
            '2': {'name': 'Pre-process Data', 'callback': self.menuFilterDatasetsData},
            '3': {'name': 'Analysis', 'callback': self.menuAnalysis},
            '4': {'name': 'Processing', 'callback': self.menuProcessing},
            '5': {'name': 'Configuration', 'callback': self.menuConfiguration},
            '-': '',
            '0': {'name': 'Back'},
            'h': {'name': 'Help', 'callback': self.help()}
        }
        propheticus.shared.Utils.printMenu('Explore Datasets Menu', MenuData, 'DatasetsConfigurations', self)

    '''
    Data Analysis Menu
    '''

    def menuAnalysis(self, choice=0):
        MenuDetails = [
            {'name': 'Class Distribution', 'callback': self.DataAnalysis('barPlot')},
            {'name': 'Feature Box Plot', 'callback': self.DataAnalysis('boxPlots')}
        ]

        if self.DatasetsConfigurations['config_sequential_data'] is True:
            MenuDetails += [
                {'name': 'Feature Time Series Plot By Class', 'callback': self.DataAnalysis('seriesValuesByClass')},
                {'name': 'Feature Time Series Plot By Class with Standard Deviation', 'callback': self.DataAnalysis('seriesValuesByClassStd')},
                {'name': 'Feature Time Series By Run', 'callback': self.DataAnalysis('seriesClassByRun')},
            ]

        MenuDetails += [
            {'name': 'Feature Line Plot', 'callback': self.DataAnalysis('lineGraphs')},
            {'name': 'Feature Line Plot with Standard Deviation', 'callback': self.DataAnalysis('lineGraphsStd')},
            {'name': 'Feature Parallel Plot', 'callback': self.DataAnalysis('parallelCoordinates')},
            {'name': 'Feature Scatter Plot', 'callback': self.DataAnalysis('scatterPlotMatrix')},
            {'name': 'Feature Correlation Plot', 'callback': self.DataAnalysis('correlationMatrixPlot')},
            {'name': 'Descriptive Analysis', 'callback': self.DataAnalysis('descriptiveAnalysis')},
            {'name': 'Parse All', 'callback': self.TODO}
        ]

        MenuData = {str(index + 1): value for index, value in enumerate(MenuDetails)}
        MenuData['-'] = ''
        MenuData['0'] = {'name': 'Back'}
        MenuData['h'] = {'name': 'Help', 'callback': self.help()}

        propheticus.shared.Utils.printMenu('Data Analysis Menu', MenuData, 'DatasetsConfigurations', self)

    '''
    Data Configuration Menu
    '''

    def menuConfiguration(self, choice):
        MenuData = {}
        for index, (config, ConfigDetails) in enumerate(self.DatasetsStaticConfigs.items()):
            MenuData[str(index + 1)] = {'name': ConfigDetails['label'], 'callback': self.generalizableGUIConfiguration(config)}

        MenuData['-'] = ''
        MenuData['0'] = {'name': 'Back'}
        MenuData['h'] = {'name': 'Help', 'callback': self.help()}

        propheticus.shared.Utils.printMenu('Processing Menu', MenuData, 'DatasetsConfigurations', self)

    '''
    Data Processing Menu
    '''

    def menuProcessing(self, choice):
        MenuData = {
            '1': {'name': 'Clustering',
                  'callback': propheticus.shared.Utils.generalMenuData(self, 'Clustering Algorithms', propheticus.shared.Utils.AvailableClusteringAlgorithms, Configurations=self.DatasetsConfigurations, menu_key='proc_clustering')},
            '2': {'name': 'Classification',
                  'callback': propheticus.shared.Utils.generalMenuData(self, 'Classification Algorithms', propheticus.shared.Utils.AvailableClassificationAlgorithms, Configurations=self.DatasetsConfigurations, menu_key='proc_classification')},
            '3': {'name': 'Define Selected Algorithms Configuration', 'callback': self.menuCurrentSelectedAlgorithms},
            '4': {'name': 'Run Selected Algorithms', 'callback': self.parseCurrentConfigurationAlgorithms},
            '-': '',
            '0': {'name': 'Back'},
            'h': {'name': 'Help', 'callback': self.help()}
        }
        propheticus.shared.Utils.printMenu('Processing Menu', MenuData, 'DatasetsConfigurations', self)

    def menuCurrentSelectedSampling(self, choice=0):
        self.menuCurrentSelectedPreprocessing('balance_data', Config.SamplingCallDetails, 'Sampling')

    def menuCurrentSelectedDimRed(self, choice=0):
        self.menuCurrentSelectedPreprocessing('reduce_dimensionality', Config.DimensionalityReductionCallDetails, 'Dim. Red.')

    def menuCurrentSelectedPreprocessing(self, key, List, label):
        Techniques = self.DatasetsConfigurations['proc_' + key]
        if len(Techniques) == 0:
            propheticus.shared.Utils.printErrorMessage('At least one technique must be chosen')
            return

        if len(Techniques) == 1:
            self.parseDefineTechniquesConfigurationChoice(key, List, label)('1')
        else:
            MenuData = {str(index + 1): {'name': algorithm, 'callback': self.parseDefineTechniquesConfigurationChoice(key, List, label)} for index, algorithm in enumerate(Techniques)}
            MenuData['-'] = ''
            MenuData['0'] = {'name': 'Back'}
            MenuData['h'] = {'name': 'Help', 'callback': self.help()}
            propheticus.shared.Utils.printMenu(f'{label} Configuration Menu', MenuData)

    def parseDefineTechniquesConfigurationChoice(self, key, List, label):
        def _parseDefineTechniquesConfigurationChoice(choice):
            technique = self.DatasetsConfigurations['proc_' + key][int(choice) - 1]
            if technique in List:
                CallDetails = List[technique]
            else:
                propheticus.shared.Utils.printFatalMessage('Invalid Preprocessing Technique: ' + technique)

            self.menuConfigureCallArguments(technique, CallDetails['parameters'], 'DatasetsConfigurations', f'proc_{key}_parameters')

        return _parseDefineTechniquesConfigurationChoice

    '''
    Process Results Menu
    '''

    def menuCurrentSelectedAlgorithms(self, choice=0):
        Algorithms = self.DatasetsConfigurations['proc_clustering'] + self.DatasetsConfigurations['proc_classification']
        if len(Algorithms) == 1:
            self.parseDefineAlgorithmsConfigurationChoice('1')
        else:
            MenuData = {str(index + 1): {'name': algorithm, 'callback': self.parseDefineAlgorithmsConfigurationChoice} for index, algorithm in enumerate(Algorithms)}
            MenuData['-'] = ''
            MenuData['0'] = {'name': 'Back'}
            MenuData['h'] = {'name': 'Help', 'callback': self.help()}
            propheticus.shared.Utils.printMenu('Algorithms Configuration Menu', MenuData)

    '''
    Process Results Menu
    '''

    def menuProcessResults(self, choice=0):
        MenuData = {
            '1': {'name': 'Compare Experiments', 'callback': self.menuCompareResults},
            '2': {'name': 'Archive Experiments', 'callback': self.menuReduceResults},
            '-': '',
            '0': {'name': 'Back'},
            'h': {'name': 'Help', 'callback': self.help()}
        }
        propheticus.shared.Utils.printMenu('Process Experiments Menu', MenuData, 'ExperimentsConfigurations', self)

    '''
    Compare Results Menu
    '''

    def menuCompareResults(self, choice=0):
        MenuData = {
            '1': {'name': 'Select Experiments for Comparison', 'callback': self.menuSelectExperimentsResults},
            '2': {'name': 'Comparison Scenario', 'callback': propheticus.shared.Utils.generalMenuData(self, 'Comparison Scenario', Config.ComparisonScenarios, Configurations=self.ExperimentsConfigurations, force_choice=False)},
            '3': {'name': 'Compare Selected Experiments', 'callback': self.compareExperiments},
            '-': '',
            '0': {'name': 'Back'},
            'h': {'name': 'Help', 'callback': self.help()}
        }
        propheticus.shared.Utils.printMenu('Compare Experiments Menu', MenuData, 'ExperimentsConfigurations', self)

    def menuSelectExperimentsResults(self, choice):
        AvailableExperiments = propheticus.shared.Utils.getAvailableExperiments(filter_by='classification')  # TODO: in the future this call should not pass skip_config_parse; currently being used for backwards compatibility

        if not AvailableExperiments:
            propheticus.shared.Utils.printErrorMessage('No classification experiments are available for comparison')
            return

        propheticus.shared.Utils.printStatusMessage('Available keys for search:')
        Keys = [key for key, value in random.choice(list(AvailableExperiments.values()))['configuration'].items()]

        items_per_row = 6
        for i in range(int(len(Keys) / items_per_row + 1)):
            propheticus.shared.Utils.printStatusMessage(', '.join(Keys[i * items_per_row:(i + 1) * items_per_row]))
        propheticus.shared.Utils.printNewLine()

        propheticus.shared.Utils.menuData(self, 'Experiments Menu', propheticus.shared.Utils.getAvailableExperimentsList(skip_config_parse=True), self.parseExperimentsChoice)()

    '''
    Reduce Results Menu
    '''

    def menuReduceResults(self, choice=0):
        MenuData = {
            '1': {'name': 'Define Metrics to Archive Experiments', 'callback': propheticus.shared.Utils.menuData(self, 'Metrics Menu', propheticus.shared.Utils.AvailableClassificationMetrics, callback=self.parseRemoveExperimentsChoice)},
            '2': {'name': 'Archive Experiments', 'callback': self.removeExperiments},
            '-': '',
            '0': {'name': 'Back'},
            'h': {'name': 'Help', 'callback': self.help()}
        }
        propheticus.shared.Utils.printMenu('Reduce Experiments Menu', MenuData, 'ExperimentsConfigurations', self)

    '''
    Datasets Filter Menu
    '''

    def menuFilterDatasetsData(self, choice=0):
        if len(self.DatasetsConfigurations['datasets']) == 0:
            propheticus.shared.Utils.printErrorMessage('At least one dataset must be chosen')
            return

        MenuData = {
            '1': {'name': 'Exclude Features', 'callback': self.excludeDatasetFeatures},
            '2': {'name': 'Select Features', 'callback': self.selectDatasetFeatures},
            '3': {'name': 'Filter By Feature Value', 'callback': self.filterFeaturesValues},
            '4': {'name': 'Define Label Field', 'callback': self.defineLabelFeature},
            '5': {'name': 'Data Sampling',
                  'callback': self.generalUIData('Data Balancing', propheticus.shared.Utils.AvailableDataBalancing, Configurations=self.DatasetsConfigurations, force_choice=False, clear_data_cache=True, menu_key='proc_balance_data')},
            '6': {'name': 'Data Sampling Configs', 'callback': self.menuCurrentSelectedSampling},
            '7': {'name': 'Dimen. Reduction', 'callback': self.generalUIData('Dimensionality Reduction', propheticus.shared.Utils.AvailableDimensionalityReduction, Configurations=self.DatasetsConfigurations, force_choice=False, clear_data_cache=True,
                                                                             menu_key='proc_reduce_dimensionality')},
            '8': {'name': 'Dimen. Reduction Configs', 'callback': self.menuCurrentSelectedDimRed},
            '9': {'name': 'Exclude Samples By Class',
                  'callback': self.generalUIData('Data Classes', list(Config.ClassesDescription.values()), Configurations=self.DatasetsConfigurations, force_choice=False, clear_data_cache=True, menu_key='datasets_exclude_classes')}
        }

        if self.DatasetsConfigurations['config_sequential_data'] is True:
            MenuData['10'] = {'name': 'Exclude Runs By Class',
                              'callback': self.generalUIData('Data Classes', list(Config.ClassesDescription.values()), Configurations=self.DatasetsConfigurations, menu_key='datasets_exclude_run_classes', force_choice=False, clear_data_cache=True)}

        MenuData['-'] = ''
        MenuData['0'] = {'name': 'Back'}
        MenuData['h'] = {'name': 'Help', 'callback': self.help()}

        propheticus.shared.Utils.printMenu('Filter Dataset Menu', MenuData, 'DatasetsConfigurations', self)

    def parseDefineAlgorithmsConfigurationChoice(self, choice):
        algorithm = (self.DatasetsConfigurations['proc_clustering'] + self.DatasetsConfigurations['proc_classification'])[int(choice) - 1]
        if algorithm in Config.ClassificationAlgorithmsCallDetails:
            AlgorithmDetails = Config.ClassificationAlgorithmsCallDetails[algorithm]
            key = 'proc_classification_algorithms_parameters'
        elif algorithm in Config.ClusteringAlgorithmsCallDetails:
            AlgorithmDetails = Config.ClusteringAlgorithmsCallDetails[algorithm]
            key = 'proc_clustering_algorithms_parameters'
        else:
            propheticus.shared.Utils.printFatalMessage('Invalid algorithm: ' + algorithm)

        self.menuConfigureCallArguments(algorithm, AlgorithmDetails['parameters'], 'DatasetsConfigurations', key)

        return -1

    def resetConfigurationParameters(self, technique, ParametersDetails, ConfigurationVariable, key):
        _self = self

        def _resetConfigurationParameters(choice):
            TechniquesConfigurations = getattr(_self, ConfigurationVariable)[key]
            if technique in TechniquesConfigurations:
                del TechniquesConfigurations[technique]

                propheticus.shared.Utils.printStatusMessage(f'Custom Configurations Reset')

        return _resetConfigurationParameters

    def defineConfigurationParameter(self, technique, ParametersDetails, ConfigurationVariable, key=None):
        _self = self

        def _defineConfigurationParameter(choice):
            if key is None:
                key_path = []
                AlgorithmConfigurations = getattr(_self, ConfigurationVariable)
            else:
                key_path = key if isinstance(key, list) else [key]
                AlgorithmConfigurations = propheticus.shared.Utils.getNestedDictionaryValue(getattr(_self, ConfigurationVariable), key_path)

            subkey_path = key_path + [technique]

            if technique not in AlgorithmConfigurations:
                AlgorithmConfigurations[technique] = {}

            def _setConfigurationParameter(choice):
                # TODO: VALIDATE FORMAT, type of data, and values (e.g. range); requires an updated list in Utils
                # NOTE: usage example - field_name:type-value
                ChoiceDetails = choice.split(':')

                field_name = ChoiceDetails[0]
                if field_name.isdigit():
                    field_name = list(ParametersDetails.keys())[int(field_name) - 1]
                else:
                    if field_name not in ParametersDetails:
                        propheticus.shared.Utils.printErrorMessage(f'Invalid input. Parameter "{field_name}" does not exist', acknowledge=False)
                        return

                field_type = ParametersDetails[field_name]['type']
                if field_type == 'configuration':
                    if 'data' not in ParametersDetails[field_name]:
                        propheticus.shared.Utils.printFatalMessage('A configuration parameter was defined but no data was provided!')

                    CallDetails = ParametersDetails[field_name]['data']
                    _self.menuConfigureCallArguments(field_name, CallDetails['parameters'], ConfigurationVariable, subkey_path)
                else:
                    if len(ChoiceDetails) != 2:
                        propheticus.shared.Utils.printErrorMessage('Invalid input. It should have the following structure => field_name:datatype-value', acknowledge=False)
                        return

                    ValueDetails = ChoiceDetails[1].split('-')
                    if len(ValueDetails) != 2:
                        propheticus.shared.Utils.printErrorMessage('Invalid input. It should have the following structure => field_name:datatype-value', acknowledge=False)
                        return

                    ValidDataTypes = ['float', 'int', 'bool', 'str']
                    value_type = ValueDetails[0]
                    if value_type not in ValidDataTypes:
                        propheticus.shared.Utils.printErrorMessage('Invalid data type. Valid types are: ' + ', '.join(ValidDataTypes), acknowledge=False)
                        return

                    _value_type = pydoc.locate(value_type)
                    value = ValueDetails[1]
                    _value = _value_type(value)
                    AlgorithmConfigurations[technique][field_name] = _value

                    if 'configuration' in ParametersDetails[field_name]:
                        configuration_key = ParametersDetails[field_name]['configuration']['key']
                        CallDetails = ParametersDetails[field_name]['configuration']['data'][_value]
                        _self.menuConfigureCallArguments(configuration_key, CallDetails['parameters'], ConfigurationVariable, subkey_path)

            MenuData = {}
            for index, (name, ParameterDetails) in enumerate(ParametersDetails.items()):
                if 'hide' in ParameterDetails and ParameterDetails['hide'] is True:
                    continue

                Details = []
                if 'type' in ParameterDetails and ParameterDetails['type'] != '':
                    parameter_type = ParameterDetails['type'] if isinstance(ParameterDetails['type'], list) else [ParameterDetails['type']]
                    Details.append(f"[type]: " + ', '.join(parameter_type))

                if 'values' in ParameterDetails:
                    if isinstance(ParameterDetails['values'], list):
                        ParameterValues = map(str, ParameterDetails['values'])
                    else:
                        ParameterValues = []
                        for parameter_type, Values in ParameterDetails['values'].items():
                            ParameterValues.append(f'{parameter_type}(' + '|'.join(map(str, Values)) + ')')

                    Details.append(f"[values]: " + ', '.join(ParameterValues))

                if 'default' in ParameterDetails:
                    Details.append(f"[default]: {ParameterDetails['default']}")

                MenuData[str(index + 1)] = {'name': f'{name} >>> ' + '; '.join(Details), 'callback': _setConfigurationParameter}

            MenuData['-'] = ''
            MenuData['0'] = {'name': 'Back'}
            MenuData['h'] = {'name': 'Help', 'callback': self.help()}
            MenuData['_regex'] = {'callback': _setConfigurationParameter}

            propheticus.shared.Utils.printMenu(f'Configuration Parameters ({technique})', MenuData, ConfigurationVariable, _self)

        return _defineConfigurationParameter

    def menuConfigureCallArguments(self, technique, ParametersDetails, ConfigurationVariable, key=None):
        MenuData = {
            '1': {'name': 'Define Parameter', 'callback': self.defineConfigurationParameter(technique, ParametersDetails, ConfigurationVariable, key)},
            '2': {'name': 'Reset All Parameters', 'callback': self.resetConfigurationParameters(technique, ParametersDetails, ConfigurationVariable, key)},
            '-': '',
            '0': {'name': 'Back'},
            'h': {'name': 'Help', 'callback': self.help()}
        }
        propheticus.shared.Utils.printMenu(f'Call Parameters Configuration ({technique})', MenuData, 'DatasetsConfigurations', self)

    '''
    GUI Menu Function (required because of clearing cache
    '''

    def generalUIData(self, base_menu_name, MenuData, Configurations, print_menu_configurations=False, show_all_option=True, force_choice=True, clear_data_cache=False, menu_key=None):
        if menu_key is None:
            menu_key = base_menu_name.lower().replace(' ', '_')

        def _parseGeneralChoice(choice):
            valid = propheticus.shared.Utils.parseChoicesSelection(menu_key, base_menu_name, MenuData, choice, Configurations, force_choice=force_choice)
            if valid == -1 and clear_data_cache is True:
                self.clearDataCache()

            return -1 if valid is True else valid

        return propheticus.shared.Utils.menuData(
            Context=self,
            menu_name=f'{base_menu_name} Menu',
            MenuData=MenuData,
            callback=_parseGeneralChoice,
            Configurations=(Configurations if print_menu_configurations else None),
            show_all_option=show_all_option,
            force_choice=force_choice
        )

    '''
    Experiments Choice Parsing
    '''

    def parseExperimentsChoice(self, choice):
        # NOTE: expected string can be as follows: f1:v11,v12&f2:v21,v22
        if ':' in choice:
            AvailableExperiments = propheticus.shared.Utils.getAvailableExperiments(use_cached=False)
            self.ExperimentsConfigurations['comparison_experiments'] = []
            for experiment_identifier, Experiment in AvailableExperiments.items():
                for searches in choice.split('&'):
                    searches = searches.split(':')
                    if len(searches) != 2:
                        propheticus.shared.Utils.printErrorMessage('Invalid search format passed: ' + choice)
                        self.ExperimentsConfigurations['comparison_experiments'] = []
                        return

                    search_key = searches[0]
                    search_values = searches[1]

                    if search_key != 'identifier' and search_key not in Experiment['configuration']:
                        propheticus.shared.Utils.printErrorMessage(f'Search key {search_key} is not valid!')
                        self.ExperimentsConfigurations['comparison_experiments'] = []
                        return

                    add = False
                    for search_value in search_values.split(','):
                        if search_key == 'identifier':
                            if search_value == experiment_identifier:
                                add = True
                                break
                        elif search_value.lower() in json.dumps(Experiment['configuration'][search_key]).lower():
                            add = True
                            break

                    if add is False:
                        break

                if add is True:
                    self.ExperimentsConfigurations['comparison_experiments'].append(experiment_identifier)

            if len(self.ExperimentsConfigurations['comparison_experiments']) == 0:
                propheticus.shared.Utils.printWarningMessage('No selection made')
            else:
                propheticus.shared.Utils.printStatusMessage('Experiments successfully chosen: ' + ','.join(self.ExperimentsConfigurations['comparison_experiments']))

            return -1
        else:
            AvailableExperiments = propheticus.shared.Utils.getAvailableExperimentsList(use_cached=False, skip_config_parse=True, field='identifier')
            return propheticus.shared.Utils.parseChoicesSelection('comparison_experiments', 'Experiments', AvailableExperiments, choice, Configurations=self.ExperimentsConfigurations)

    '''
    Select Experiments by Min Metric
    '''

    def parseRemoveExperimentsChoice(self, choice):
        loop = True
        while loop:
            if choice != '':
                for metrics_details in choice.split(','):
                    MetricDetails = metrics_details.split(':')
                    if len(MetricDetails) <= 1:
                        propheticus.shared.Utils.printErrorMessage('Invalid selection, it must have the following structure => metric1:value11,metric2:value21 \n', acknowledge=False)
                    else:
                        if MetricDetails[0].isdigit():
                            _metric_id = int(MetricDetails[0]) - 1
                            if _metric_id < 0 or _metric_id >= len(propheticus.shared.Utils.AvailableClassificationMetrics):
                                propheticus.shared.Utils.printErrorMessage('Invalid metric choice passed: ' + str(_metric_id + 1))
                            else:
                                loop = False
                        else:
                            if MetricDetails[0] not in propheticus.shared.Utils.AvailableClassificationMetrics:
                                propheticus.shared.Utils.printErrorMessage('Invalid metric choice passed: ' + MetricDetails[0])
                            else:
                                loop = False

                if loop is True:
                    choice = propheticus.shared.Utils.printInputMessage('Define the metrics values to reduce experiments:')

            else:
                loop = False

        self.ExperimentsConfigurations['remove_experiments'] = []
        if choice != '':
            for metrics_details in choice.split(','):
                MetricDetails = metrics_details.split(':')
                if MetricDetails[0].isdigit():
                    metric_id = int(MetricDetails[0])
                    _metric_id = metric_id - 1
                    metric_name = propheticus.shared.Utils.AvailableClassificationMetrics[_metric_id]
                else:
                    metric_name = MetricDetails[0]
                    _metric_id = propheticus.shared.Utils.AvailableClassificationMetrics.index(metric_name)

                self.ExperimentsConfigurations['remove_experiments'].append({
                    'index': _metric_id,
                    'metric_name': metric_name,
                    'label': metric_name + ': ' + MetricDetails[1],
                    'values': float(MetricDetails[1])
                })

        return -1

    # TODO: MOVE TO CompareExperiments module
    def removeExperiments(self, choice=0):
        remove_experiments = self.ExperimentsConfigurations['remove_experiments']
        if len(remove_experiments) > 0:
            confirm = propheticus.shared.Utils.printConfirmationMessage('This procedure will move the experiments mathching the chosen parameters to a new folder. Continue:')
            if confirm == 'y':
                MetricsFilters = {Details['metric_name']: Details['values'] for Details in remove_experiments}

                Experiments = propheticus.shared.Utils.getAvailableExperiments(use_cached=False, filter_by='classification')
                if not Experiments:
                    propheticus.shared.Utils.printErrorMessage('No classification experiments are available for archiving')

                propheticus.shared.Utils.printStatusMessage('Existing experiments #: ' + str(len(Experiments)))

                RemoveExperiments = []
                ValidExperiments = []

                for experiment, ExperimentDetails in Experiments.items():
                    experiment_file_name = ExperimentDetails['filename']
                    subfolder = ExperimentDetails['subfolder']
                    Workbook = xlrd.open_workbook(os.path.join(Config.framework_instance_generated_logs_path, subfolder, experiment_file_name))
                    ClassificationWorksheet = Workbook.sheet_by_index(1)

                    for row_index in range(3, ClassificationWorksheet.nrows):
                        cell_value = ClassificationWorksheet.cell(row_index, 0).value
                        if cell_value.strip() == '':
                            continue

                        algorithm, version = cell_value.strip().split(' - ')
                        if version == 'Final':
                            remove_experiment = False

                            for col_index in range(2, ClassificationWorksheet.ncols):
                                metric = ClassificationWorksheet.cell(2, col_index).value.lower()
                                if metric in MetricsFilters:
                                    value = ClassificationWorksheet.cell(row_index, col_index).value
                                    if value < MetricsFilters[metric]:
                                        remove_experiment = True

                            if remove_experiment is False:
                                ValidExperiments.append(experiment)
                            else:
                                RemoveExperiments.append(experiment)

                            if experiment in RemoveExperiments and experiment in ValidExperiments:
                                propheticus.shared.Utils.printErrorMessage('Experiment is marked for archive although some algorithms within are acceptable for the chosen metrics! ' + experiment)
                                return

                propheticus.shared.Utils.printStatusMessage('Experiments to be archived #: ' + str(len(RemoveExperiments)))
                timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M')

                FolderPaths = [
                    Config.framework_instance_generated_classification_path,
                    Config.framework_instance_generated_logs_path
                ]

                metrics_details = '_'.join([key + '-' + str(value) for key, value in MetricsFilters.items()])
                for folder_path in FolderPaths:
                    for subfolder in next(os.walk(folder_path))[1]:
                        original_path = os.path.join(folder_path, subfolder)
                        for filename in os.listdir(original_path):
                            experiment = filename.split('.')[0]
                            if experiment in RemoveExperiments:
                                removed_path = os.path.join(Config.framework_instance_generated_archive_path, timestamp + '&' + metrics_details, os.path.basename(folder_path), subfolder)
                                pathlib.Path(removed_path).mkdir(parents=True, exist_ok=True)

                                current_path = os.path.join(original_path, filename)
                                new_path = os.path.join(removed_path, filename)
                                shutil.move(current_path, new_path)
                                if not os.path.isfile(new_path):
                                    propheticus.shared.Utils.printFatalMessage('File was not moved! File: ' + new_path)

                propheticus.shared.Utils.printStatusMessage('Experiments successfully archived!')
                propheticus.shared.Utils.printNewLine()
        else:
            propheticus.shared.Utils.printErrorMessage('At least one metric must be selected to reduce the experiments!')

    '''
    Execute Batch Configurations
    '''

    def executeBatchExecution(self, choice=0):
        from InstanceBatchExecution import InstanceBatchExecution
        oBatchExecution = InstanceBatchExecution(self)
        oBatchExecution.run()

    '''
    Compare Experiments
    '''

    def compareExperiments(self, choice):
        # TODO: get this from config
        StatisticalDetails = {
            'confidence_level': 0.05,
            'paired': True,
            'correction_method': 'bonferroni'
        }

        comparison_scenario = self.ExperimentsConfigurations['comparison_scenario']
        if len(comparison_scenario) > 0:
            FocusMetrics = []
            StatisticalDetails['metric'] = Config.MetricsByScenarios[comparison_scenario[0]][0]
            for scenario in comparison_scenario:
                FocusMetrics += Config.MetricsByScenarios[scenario]
        else:
            StatisticalDetails['metric'] = 'recall'
            FocusMetrics = False

        oExperimentsComparison = propheticus.core.ExperimentsComparison(self.display_visuals)
        oExperimentsComparison.compareExperiments(self.ExperimentsConfigurations['comparison_experiments'], StatisticalDetails, FocusMetrics)

    '''
    Datasets selection function
    '''

    def parseDatasetsChoice(self, choice):
        if choice.strip() == '':
            propheticus.shared.Utils.printErrorMessage('At least one dataset must be selected')
            return
        elif re.search('[a-zA-Z]', choice):
            propheticus.shared.Utils.printErrorMessage('Only numeric values are allowed')
            return

        if len(self.DatasetsConfigurations['pre_selected_features']) > 0 or len(self.DatasetsConfigurations['pre_excluded_features']) > 0 or len(self.DatasetsConfigurations['pre_filter_feature_values']) > 0:
            confirm = propheticus.shared.Utils.printConfirmationMessage('Modifying the datasets will erase all feature filters previously applied.')
        else:
            confirm = 'y'

        if confirm == 'y':
            self.initializeDatasetsDirectConfigurations()
            valid = propheticus.shared.Utils.parseChoicesSelection('datasets', 'Datasets', [dataset.replace('.data.txt', '') for dataset in propheticus.shared.Utils.getAvailableDatasets()], choice, Configurations=self.DatasetsConfigurations,
                                                                   show_selection_message=False)
            if valid is False:
                return
            elif valid == -1:
                validated = self.validateChosenDatasets()
                if validated is False:
                    self.DatasetsConfigurations['datasets'] = []
                    return

                self.defineExcludeFeaturesByStaticValues()

            propheticus.shared.Utils.printStatusMessage('Datasets successfully chosen: ' + ','.join(self.DatasetsConfigurations['datasets']))
        else:
            propheticus.shared.Utils.printStatusMessage('Dataset selection aborted')

        return -1

    def validateChosenDatasets(self):
        return self.oDataManagement.validateChosenDatasets(self.DatasetsConfigurations['datasets'])

    '''
    Exclude Features Datasets Filter
    '''

    def defineLabelFeature(self, choice):
        current_excluded_features = [Values['index'] for Values in self.DatasetsConfigurations['pre_excluded_features']]
        self.printCurrentDatasetsHeaders()
        while True:
            choice = propheticus.shared.Utils.printInputMessage('Define the label feature:')
            if choice.strip() != '' and (re.search('[a-zA-Z]', choice) or int(choice) < 1 or int(choice) > len(self.oDataManagement.datasets_headers)):
                propheticus.shared.Utils.printErrorMessage('Invalid feature choice passed: ' + str(choice))
            elif choice.strip() != '' and (int(choice) - 1) in current_excluded_features:
                propheticus.shared.Utils.printErrorMessage('Chosen label feature is set to be excluded: ' + str(choice))
            else:
                break

        # TODO: control here when removing the target to empty if the last column is not in the exclude list
        # TODO: see if selected features should have a part in here
        if choice.strip() != '':
            _choice = int(choice) - 1
            self.DatasetsConfigurations['pre_target'] = {'index': _choice, 'label': self.oDataManagement.datasets_headers[int(_choice)]}
            self.defineExcludedFeaturesByLabel()
        else:
            self.DatasetsConfigurations['pre_target'] = False

    def defineExcludedFeaturesByLabel(self):
        current_excluded_features = [Values['index'] for Values in self.DatasetsConfigurations['pre_excluded_features']]
        self.clearDataCache()
        Label = self.DatasetsConfigurations['pre_target']
        self.DatasetsConfigurations['pre_excluded_features_label'] = []
        if Label is not False:
            for index, header in enumerate(self.oDataManagement.datasets_headers):
                # TODO: this may be "dangerous" and remove unwanted columns/features; somehow fix or give an alert?
                if index != Label['index'] and 'label' in header and index not in current_excluded_features:
                    self.DatasetsConfigurations['pre_excluded_features_label'].append({'index': index, 'label': self.oDataManagement.datasets_headers[int(index)]})

    '''
    Exclude Features Datasets Filter
    '''

    def excludeDatasetFeatures(self, choice):
        self.printCurrentDatasetsHeaders()

        if len(self.DatasetsConfigurations['pre_selected_features']) > 0:
            propheticus.shared.Utils.printErrorMessage('Either Selected Features or Excluded Features can be used, but not both')
            return

        FilterFeaturesIndexes = {Details['index']: Details['values'] for Details in self.DatasetsConfigurations['pre_filter_feature_values']}
        while True:
            print('')
            valid = True
            choice = propheticus.shared.Utils.printInputMessage('Define the features to be excluded:')
            if choice != '':
                if re.search('[a-zA-Z]', choice):
                    propheticus.shared.Utils.printErrorMessage('Only numbers can be entered')
                    valid = False

                for value in choice.split(','):
                    Values = value.split('-')

                    if len(Values) > 1:
                        if self.DatasetsConfigurations['pre_target'] is not False and (int(Values[0]) - 1) <= self.DatasetsConfigurations['pre_target']['index'] <= (int(Values[1]) - 1):
                            propheticus.shared.Utils.printErrorMessage('The feature defined as label can not be removed by the range: ' + value)
                            valid = False

                    for value2 in Values:
                        _value = int(value2) - 1
                        if _value < 0 or _value >= len(self.oDataManagement.datasets_headers):
                            propheticus.shared.Utils.printErrorMessage('Invalid feature choice passed: ' + str(value))
                            valid = False

                        if _value in FilterFeaturesIndexes.items():  # TODO: validate exactly what is intended here; why use 'items()' if just looking for key; if the choice made is a range, only the min and max are being validated
                            propheticus.shared.Utils.printWarningMessage('Feature already used for filtering data. It will first filter, then remove: ' + str(value))
                            # valid = False

                        if int(value2) == 1:
                            propheticus.shared.Utils.printErrorMessage('The run title can not be excluded')
                            valid = False

                        if self.DatasetsConfigurations['pre_target'] is False and int(value2) == len(self.oDataManagement.datasets_headers):
                            propheticus.shared.Utils.printErrorMessage('The last column (label) can only be excluded after a target is defined')
                            valid = False

                if valid:
                    break
            else:
                break

        self.clearDataCache()

        self.DatasetsConfigurations['pre_excluded_features'] = []
        if choice != '':
            for value in choice.split(','):
                Values = value.split('-')
                if len(Values) > 1:
                    for value2 in range(int(Values[0]) - 1, int(Values[1])):
                        self.DatasetsConfigurations['pre_excluded_features'].append({'index': value2, 'label': self.oDataManagement.datasets_headers[int(value2)]})
                else:
                    _value = int(Values[0]) - 1
                    self.DatasetsConfigurations['pre_excluded_features'].append({'index': _value, 'label': self.oDataManagement.datasets_headers[int(_value)]})

        self.defineExcludedFeaturesByLabel()

    '''
    Select Features Datasets Filter
    '''

    def selectDatasetFeatures(self, choice):
        self.printCurrentDatasetsHeaders()

        # TODO: implement validation with feature filter; can not be filtered if not in selected and vice versa

        if len(self.DatasetsConfigurations['pre_excluded_features']) > 0:
            propheticus.shared.Utils.printErrorMessage('Either Selected Features or Excluded Features can be used, but not both')
            return

        while True:
            print('')
            valid = True
            choice = propheticus.shared.Utils.printInputMessage('Define the features to be selected:')
            if choice != '':
                if re.search('[a-zA-Z]', choice):
                    propheticus.shared.Utils.printErrorMessage('Only numbers can be entered')
                    valid = False

                for value in choice.split(','):
                    Values = value.split('-')

                    if len(Values) > 1:
                        if self.DatasetsConfigurations['pre_target'] is not False and (int(Values[0]) - 1) <= self.DatasetsConfigurations['pre_target']['index'] <= (int(Values[1]) - 1):
                            propheticus.shared.Utils.printErrorMessage('The feature defined as label can not selected by the range: ' + value)
                            valid = False

                    for value2 in Values:
                        _value = int(value2) - 1
                        if _value < 0 or _value >= len(self.oDataManagement.datasets_headers):
                            propheticus.shared.Utils.printErrorMessage('Invalid feature choice passed: ' + str(value))
                            valid = False

                        if int(value2) == 1:
                            propheticus.shared.Utils.printErrorMessage('The run title can not be selected')
                            valid = False
                if valid:
                    break
            else:
                break

        self.clearDataCache()

        self.DatasetsConfigurations['pre_selected_features'] = []
        if choice != '':
            for value in choice.split(','):
                Values = value.split('-')
                if len(Values) > 1:
                    for value2 in range(int(Values[0]) - 1, int(Values[1])):
                        self.DatasetsConfigurations['pre_selected_features'].append({'index': value2, 'label': self.oDataManagement.datasets_headers[int(value2)]})
                else:
                    _value = int(Values[0]) - 1
                    self.DatasetsConfigurations['pre_selected_features'].append({'index': _value, 'label': self.oDataManagement.datasets_headers[int(_value)]})

    '''
    Select By Features Value Filter
    '''

    def filterFeaturesValues(self, choice):
        self.printCurrentDatasetsHeaders()

        # TODO: implement validation with feature filter; can not be filtered if not in selected and vice versa

        # if self.DatasetsConfigurations['config_sliding_window'] > 1:
        #     propheticus.shared.Utils.printErrorMessage('It is not possible to define a sliding window and feature filtering')
        #     return

        loop = True
        ExcludedFeaturesIndexes = [Details['index'] for Details in self.DatasetsConfigurations['pre_excluded_features']]
        while loop:
            choice = propheticus.shared.Utils.printInputMessage('Define the features to be filtered:')
            if choice != '':
                for features_details in choice.split('|'):
                    FeatureDetails = features_details.split(':')
                    if len(FeatureDetails) <= 1:
                        propheticus.shared.Utils.printErrorMessage('Invalid selection, it must have the following structure => feature_id1:value11,value12|feature_id2:value21,value22 \n', acknowledge=False)
                    else:
                        _feature_id = int(FeatureDetails[0]) - 1
                        if _feature_id < 0 or _feature_id >= len(self.oDataManagement.datasets_headers):
                            propheticus.shared.Utils.printErrorMessage('Invalid feature choice passed: ' + str(_feature_id + 1))
                        else:
                            if _feature_id in ExcludedFeaturesIndexes:
                                propheticus.shared.Utils.printWarningMessage('Feature already used for excluding data. It will first filter, then exclude: ' + str(_feature_id + 1))

                            loop = False

                            categorical = self.oDataManagement.datasets_headers_details[_feature_id]['categorical']
                            if categorical is not False:
                                AllowedValues = categorical['values']
                                Values = FeatureDetails[1]
                                for value in Values.split(','):
                                    if value not in AllowedValues:
                                        propheticus.shared.Utils.printErrorMessage('Value passed for feature is invalid: ' + str(_feature_id + 1) + ' => ' + value + '. Allowed values are: ' + ",".join(AllowedValues))
                                        loop = True
            else:
                loop = False

        self.clearDataCache()

        self.DatasetsConfigurations['pre_filter_feature_values'] = []
        if choice != '':
            for features_details in choice.split('|'):
                FeatureDetails = features_details.split(':')
                feature_id = FeatureDetails[0]
                _feature_id = int(feature_id) - 1
                feature_name = self.oDataManagement.datasets_headers[int(_feature_id)]
                self.DatasetsConfigurations['pre_filter_feature_values'].append({
                    'index': _feature_id,
                    'feature_name': feature_name,
                    'label': feature_name + ': ' + FeatureDetails[1],
                    'values': FeatureDetails[1]
                })

    # TODO: fix naming convention; why start capital?
    def DataAnalysis(self, method):
        _self = self
        _method = method if method == 'descriptiveAnalysis' else 'DataAnalysis'

        def _DataAnalysis(choice=None):
            _self.action_context = _method
            error = _self.exportDatasets()
            if not error:
                oDataAnalysis = propheticus.core.DataAnalysis(_self.display_visuals, _self.getDatasetsIdentifiers(), _self.getConfigurationsIdentifier(), propheticus.shared.Utils.toStringCurrentConfigurations(_self.DatasetsConfigurations))
                getattr(oDataAnalysis, method)(_self._DataCache)

            _self.action_context = None

        return _DataAnalysis

    def generalizableGUIConfiguration(self, config):
        _self = self

        def _generalizableGUIConfiguration(choice):
            ConfigDetails = _self.DatasetsStaticConfigs[config]
            if 'reset_cache' in ConfigDetails and ConfigDetails['reset_cache'] is True:
                _self.clearDataCache()

            if ConfigDetails['type'] not in [int, bool, 'items', 'item']:
                propheticus.shared.Utils.printFatalMessage('Unexpected configuration type: ' + str(ConfigDetails['type']))

            while True:
                valid = True
                values = ''
                if 'values' in ConfigDetails:
                    if len(ConfigDetails['values']) == 0:
                        propheticus.shared.Utils.printErrorMessage(f'No values were provided for config {config}')
                        return

                    values = '\n' + '\n'.join([f'[{index + 1}] {value}' for index, value in enumerate(ConfigDetails['values'])])
                elif ConfigDetails['type'] == 'item':
                    propheticus.shared.Utils.printFatalMessage('Static configuration of type "item" is only valid with predefined values!')

                _choice = choice = propheticus.shared.Utils.printInputMessage(f'Define the intended value for {config}: {values}')
                if choice.strip() != '':
                    if ConfigDetails['type'] in [int, bool]:
                        if not choice.isdigit():
                            propheticus.shared.Utils.printErrorMessage('Invalid value! Must be numeric')
                            valid = False

                    if ConfigDetails['type'] in [int]:
                        _choice = int(choice)
                        if 'min' in ConfigDetails and _choice < ConfigDetails['min']:
                            propheticus.shared.Utils.printErrorMessage('Invalid value! Must be >= ' + str(ConfigDetails['min']))
                            valid = False
                        elif 'max' in ConfigDetails and _choice > ConfigDetails['max']:
                            propheticus.shared.Utils.printErrorMessage('Invalid value! Must be <= ' + str(ConfigDetails['max']))
                            valid = False

                    elif ConfigDetails['type'] in [bool]:
                        _choice = int(choice)
                        if _choice not in [0, 1]:
                            propheticus.shared.Utils.printErrorMessage('Invalid value! Must be must be either 0 or 1')
                            valid = False

                    elif ConfigDetails['type'] in ['item']:
                        if choice.isdigit():
                            _choice = int(choice)
                            if _choice < 1 or _choice > len(ConfigDetails['values']):
                                propheticus.shared.Utils.printErrorMessage(f"Invalid value! Must be must be between 1 and {len(ConfigDetails['values'])}")
                                valid = False
                            else:
                                _choice = ConfigDetails['values'][_choice - 1]
                        else:
                            if choice not in ConfigDetails['values']:
                                propheticus.shared.Utils.printErrorMessage('Invalid value! Must be one of the predefined values!')
                                valid = False
                            else:
                                _choice = choice

                    elif ConfigDetails['type'] in ['items']:
                        _choice = propheticus.shared.Utils._parseChoicesSelection(ConfigDetails['values'], choice)

                    else:
                        propheticus.shared.Utils.printFatalMessage(f"Invalid datatype provided for static configuration! {ConfigDetails['type']}")

                    if valid is True:
                        break

                elif 'allow_empty' in ConfigDetails and ConfigDetails['allow_empty'] is True:
                    break

                else:
                    propheticus.shared.Utils.printErrorMessage('Invalid value! Must be must be ' + str(ConfigDetails['type']))

            if choice.strip() != '':
                processed_config = ConfigDetails['type'](_choice) if ConfigDetails['type'] in [int, bool] else _choice
            else:
                processed_config = ConfigDetails['default']

            _self.DatasetsConfigurations[config] = processed_config

            if 'callback' in ConfigDetails:
                ConfigDetails['callback'](processed_config)

            propheticus.shared.Utils.printStatusMessage('Configuration successfully defined for ' + config + ': ' + str(_self.DatasetsConfigurations[config]))

        return _generalizableGUIConfiguration

    def menuUndersamplingThreshold(self, choice):
        self.clearDataCache()

        while True:
            choice = propheticus.shared.Utils.printInputMessage('Define the maximum data records value:')
            if choice.strip() != '' and not choice.isdigit() and choice <= 0:
                propheticus.shared.Utils.printErrorMessage('Invalid value, value must be numeric and greater than 0, please try again')
            else:
                break

        self.DatasetsConfigurations['config_undersampling_threshold'] = int(choice) if choice.strip() != '' else None
        propheticus.shared.Utils.printStatusMessage('Maximum data records successfully defined: ' + choice)

    def menuBinaryClassification(self, choice):
        self.clearDataCache()

        while True:
            choice = propheticus.shared.Utils.printInputMessage('Define whether use binary classification or multi-class:')
            if not choice.isdigit() and choice not in [0, 1]:
                propheticus.shared.Utils.printErrorMessage('Invalid value, value must be either 0 or 1, please try again')
            else:
                break

        self.DatasetsConfigurations['config_binary_classification'] = bool(int(choice))
        propheticus.shared.Utils.printStatusMessage('Binary classification successfully defined: ' + choice)

    def menuSlidingWindow(self, choice):
        self.clearDataCache()

        if len(self.DatasetsConfigurations['pre_filter_feature_values']) > 0:
            propheticus.shared.Utils.printErrorMessage('It is not possible to define a sliding window and feature filtering')
            return

        while True:
            choice = propheticus.shared.Utils.printInputMessage('Define the sliding window value:')
            if choice.strip() != '' and not choice.isdigit() and choice <= 0:
                propheticus.shared.Utils.printErrorMessage('Invalid value, sliding window must be numeric and greater than 1, please try again')
            else:
                break

        self.DatasetsConfigurations['config_sliding_window'] = int(choice) if choice.strip() != '' else 1
        propheticus.shared.Utils.printStatusMessage('Sliding window successfully defined: ' + choice)

    def validateCurrentConfigurationAlgorithms(self):
        # TODO: most of this function should be moved to each class (Classification/Clustering) for a validation method
        # TODO: this logic most likely requires updating!

        valid = True

        classification_algorithms = self.DatasetsConfigurations['proc_classification']
        clustering_algorithms = self.DatasetsConfigurations['proc_clustering']
        load_experiment_models = self.DatasetsConfigurations['config_load_experiment_models']
        ensemble_algorithms = self.DatasetsConfigurations['config_ensemble_algorithms']
        ensemble_selection = self.DatasetsConfigurations['config_ensemble_selection']
        datasets = self.DatasetsConfigurations['datasets']

        algorithms_defined = len(clustering_algorithms) > 0 or len(classification_algorithms) > 0
        if load_experiment_models is not None:
            if algorithms_defined is True:
                propheticus.shared.Utils.printErrorMessage('Either algorithms or config_load_experiment_models can be defined, but not both')
                valid = False

            Datasets = None
            for load_experiment in load_experiment_models:
                experiment_id = load_experiment.split('.')[0]
                ExperimentDetails = propheticus.shared.Utils.getAvailableExperiments()[experiment_id]
                ExperimentDatasets = ExperimentDetails['configuration']['datasets']
                if Datasets is None:
                    Datasets = ExperimentDatasets

                if ExperimentDatasets != Datasets:
                    propheticus.shared.Utils.printErrorMessage('The same datasets must be used for all the loaded models')
                    valid = False
                    break

            if len(datasets) > 0 and Datasets != datasets:
                propheticus.shared.Utils.printErrorMessage('The selected datasets must be the same as those used for the loaded models')
                valid = False

            if len(load_experiment_models) > 1 and ensemble_algorithms is None:
                propheticus.shared.Utils.printErrorMessage('More than one experiment was given to load but no ensemble algorithm was provided')
                valid = False

        elif algorithms_defined is False:
            propheticus.shared.Utils.printErrorMessage('At least one algorithm must be chosen')
            valid = False

        if len(datasets) == 0:
            propheticus.shared.Utils.printErrorMessage('At least one dataset must be chosen')
            valid = False

        if ensemble_algorithms is not None:
            if clustering_algorithms:
                propheticus.shared.Utils.printErrorMessage('Ensemble option cannot be used with clustering algorithms!')
                valid = False

            if self.DatasetsConfigurations['config_save_experiment_models'] is True:
                propheticus.shared.Utils.printErrorMessage('When using ensemble models it is not possible to save intermediate models!')
                valid = False

            # TODO: this logic should be associated with a required minimum number of experiments in the ensembles configs: ie stacking can work wiht 1, hard voting should have 3
            # classification_algorithms = classification_algorithms
            # if (not isinstance(classification_algorithms, list) or len(classification_algorithms) < 2) and (not isinstance(load_experiment_models, list) or len(load_experiment_models) < 2):
            #     propheticus.shared.Utils.printErrorMessage('In order to use custom ensembling approaches at least 2 algorithms must be trained/loaded!')
            #     valid = False

        elif ensemble_selection is not None:
            propheticus.shared.Utils.printFatalMessage('Ensemble selection heuristic can only be defined when using ensemble of algorithms')
            valid = False

        # TODO: this validation is specifically internal to the Classification module
        # elif isinstance(classification_algorithms, list):
        #     propheticus.shared.Utils.printErrorMessage('Only a single algorithm can be provided!')
        #     valid = False

        return valid

    def parseCurrentConfigurationAlgorithms(self, choice=None, skip_validation=False):
        validated = self.validateCurrentConfigurationAlgorithms()
        if validated is True:
            if self.DatasetsConfigurations['config_grid_search']:
                if self.DatasetsConfigurations['proc_classification'] and not self.DatasetsConfigurations['proc_classification_grid_params']:
                    propheticus.shared.Utils.printStatusMessage('Classification grid-search is configured but no parameters were passed. Base grid-search parameters will be used!')

                if self.DatasetsConfigurations['proc_clustering'] and not self.DatasetsConfigurations['proc_clustering_grid_params']:
                    propheticus.shared.Utils.printStatusMessage('Clustering grid-search is configured but no parameters were passed. Base grid-search parameters will be used!')

            confirm = 'y' if skip_validation else propheticus.shared.Utils.printConfirmationMessage('This will override previous results for the same configurations. Continue?')
            if confirm == 'y':
                self.action_context = 'algorithms'
                error = self.exportDatasets()
                if not error:
                    start_time = time.time()

                    # TODO: validate that this is the best place to have this
                    if self.DatasetsConfigurations['config_binary_classification'] is True:
                        PositiveClasses = [propheticus.shared.Utils.getClassDescriptionById(Config.ClassesMapping['Binary_Error'])]
                    else:
                        PositiveClasses = self.DatasetsConfigurations['datasets_positive_classes']

                    if self.DatasetsConfigurations['config_grid_search']:
                        if self.DatasetsConfigurations['proc_classification'] and not self.DatasetsConfigurations['proc_classification_grid_params']:
                            GridParameters = {}
                            for algorithm in sorted(self.DatasetsConfigurations['proc_classification']):
                                if 'grid' in Config.ClassificationAlgorithmsCallDetails[algorithm]:
                                    GridParameters[algorithm] = Config.ClassificationAlgorithmsCallDetails[algorithm]['grid']

                            self.DatasetsConfigurations['proc_classification_grid_params'] = GridParameters

                        if self.DatasetsConfigurations['proc_clustering'] and not self.DatasetsConfigurations['proc_clustering_grid_params']:
                            GridParameters = {}
                            for algorithm in sorted(self.DatasetsConfigurations['proc_clustering']):
                                if 'grid' in Config.ClusteringAlgorithmsCallDetails[algorithm]:
                                    GridParameters[algorithm] = Config.ClusteringAlgorithmsCallDetails[algorithm]['grid']

                            self.DatasetsConfigurations['proc_clustering_grid_params'] = GridParameters

                    propheticus.shared.Utils.printCurrentConfigurations(self.DatasetsConfigurations, hide_empty=True, truncate=self.DatasetsConfigurations['config_truncate_configurations'])

                    '''
                    Run Classification Algorithms
                    '''
                    oClassification = propheticus.core.Classification(
                        Context=self,
                        dataset_name=self.getDatasetsIdentifiers(),
                        configurations_id=self.getConfigurationsIdentifier(),
                        description=propheticus.shared.Utils.toStringCurrentConfigurations(self.DatasetsConfigurations),
                        display_visuals=self.display_visuals,
                        balance_data=self.DatasetsConfigurations['proc_balance_data'],
                        balance_data_params=self.DatasetsConfigurations['proc_balance_data_parameters'],
                        reduce_dimensionality=self.DatasetsConfigurations['proc_reduce_dimensionality'],
                        reduce_dimensionality_params=self.DatasetsConfigurations['proc_reduce_dimensionality_parameters'],
                        normalize=self.DatasetsConfigurations['proc_normalize_data'],
                        data_split=self.DatasetsConfigurations['config_data_split'],
                        data_split_parameters=self.DatasetsConfigurations['config_data_split_parameters'],
                        seed_count=self.DatasetsConfigurations['config_seed_count'],
                        cv_fold=self.DatasetsConfigurations['config_cv_fold'] if 'config_cv_fold' in self.DatasetsConfigurations else None,
                        ensemble_algorithms=self.DatasetsConfigurations['config_ensemble_algorithms'],
                        ensemble_algorithms_parameters=self.DatasetsConfigurations['config_ensemble_algorithms_parameters'],
                        ensemble_selection=self.DatasetsConfigurations['config_ensemble_selection'],
                        ensemble_selection_parameters=self.DatasetsConfigurations['config_ensemble_selection_parameters'],
                        grid_search=self.DatasetsConfigurations['config_grid_search'],
                        grid_inner_cv_fold=self.DatasetsConfigurations['config_grid_inner_cv_fold'],
                        mode=self.mode,
                        positive_classes=PositiveClasses,
                        save_complete_model=self.DatasetsConfigurations['config_save_complete_model'],
                        save_experiment_models=self.DatasetsConfigurations['config_save_experiment_models'],
                        load_experiment_models=self.DatasetsConfigurations['config_load_experiment_models'],
                    )

                    ClassificationAlgorithms = self.DatasetsConfigurations['proc_classification']

                    if self.DatasetsConfigurations['config_load_experiment_models'] is not None:
                        if self.mode == 'cli':
                            Config.thread_level_ = propheticus.shared.Utils.getBestParallelizationLocation(self.DatasetsConfigurations)
                            propheticus.shared.Utils.printStatusMessage('Parallelizing at level: ' + Config.thread_level_)

                        task_basename = propheticus.shared.Utils.hash(str(self.DatasetsConfigurations['config_load_experiment_models']))
                        self.ClassificationAlgorithmsResults = {task_basename: []}
                        oClassification.runModel(Dataset=self._DataCache, task_basename=task_basename)

                    elif self.DatasetsConfigurations['config_ensemble_algorithms'] is not None:
                        if self.mode == 'cli':
                            Config.thread_level_ = propheticus.shared.Utils.getBestParallelizationLocation(self.DatasetsConfigurations)
                            propheticus.shared.Utils.printStatusMessage('Parallelizing at level: ' + Config.thread_level_)

                        task_basename = str(ClassificationAlgorithms)
                        self.ClassificationAlgorithmsResults = {task_basename: []}

                        if len(ClassificationAlgorithms) > 0:
                            oClassification.runModel(
                                algorithm=ClassificationAlgorithms,
                                task_basename=str(task_basename),
                                Dataset=self._DataCache,
                                Parameters=self.DatasetsConfigurations['proc_classification_algorithms_parameters'],
                                GridSearchParameters=self.DatasetsConfigurations['proc_classification_grid_params']
                            )

                    else:
                        self.ClassificationAlgorithmsResults = {}
                        for algorithm in sorted(ClassificationAlgorithms):
                            if self.mode == 'cli':
                                AlgConfigs = copy.deepcopy(self.DatasetsConfigurations)
                                AlgConfigs['proc_classification'] = [algorithm]
                                Config.thread_level_ = thread_level = propheticus.shared.Utils.getBestParallelizationLocation(AlgConfigs)
                                propheticus.shared.Utils.printStatusMessage(f'Parallelizing {algorithm} at level: ' + Config.thread_level_)

                            task_basename = algorithm
                            self.ClassificationAlgorithmsResults[task_basename] = []

                            if algorithm in self.DatasetsConfigurations['proc_classification_grid_params']:
                                GridSearchParameters = self.DatasetsConfigurations['proc_classification_grid_params'][algorithm]
                            else:
                                GridSearchParameters = False

                            if algorithm in self.DatasetsConfigurations['proc_classification_algorithms_parameters']:
                                Parameters = self.DatasetsConfigurations['proc_classification_algorithms_parameters'][algorithm]
                            else:
                                Parameters = False

                            oClassification.runModel(
                                algorithm=algorithm,
                                task_basename=task_basename,
                                Dataset=self._DataCache,
                                Parameters=Parameters,
                                GridSearchParameters=GridSearchParameters
                            )

                    '''
                    Run Clustering Algorithms
                    '''
                    oClustering = propheticus.core.Clustering(
                        Context=self,
                        dataset_name=self.getDatasetsIdentifiers(),
                        configurations_id=self.getConfigurationsIdentifier(),
                        description=propheticus.shared.Utils.toStringCurrentConfigurations(self.DatasetsConfigurations),
                        display_visuals=self.display_visuals,
                        balance_data=self.DatasetsConfigurations['proc_balance_data'],
                        balance_data_params=self.DatasetsConfigurations['proc_balance_data_parameters'],
                        reduce_dimensionality=self.DatasetsConfigurations['proc_reduce_dimensionality'],
                        reduce_dimensionality_params=self.DatasetsConfigurations['proc_reduce_dimensionality_parameters'],
                        normalize=self.DatasetsConfigurations['proc_normalize_data'],
                        seed_count=self.DatasetsConfigurations['config_seed_count'],
                        grid_search=self.DatasetsConfigurations['config_grid_search'],
                        mode=self.mode,
                        positive_classes=PositiveClasses
                    )

                    self.ClusteringAlgorithmsResults = {algorithm: [] for algorithm in self.DatasetsConfigurations['proc_clustering']}

                    ClusteringAlgorithms = self.DatasetsConfigurations['proc_clustering']
                    for algorithm in sorted(ClusteringAlgorithms):
                        if self.mode == 'cli':
                            AlgConfigs = copy.deepcopy(self.DatasetsConfigurations)
                            AlgConfigs['proc_clustering'] = [algorithm]
                            Config.thread_level_ = propheticus.shared.Utils.getBestParallelizationLocation(AlgConfigs)
                            propheticus.shared.Utils.printStatusMessage(f'Parallelizing {algorithm} at level: ' + Config.thread_level_)

                        if algorithm in self.DatasetsConfigurations['proc_clustering_grid_params']:
                            GridSearchParameters = self.DatasetsConfigurations['proc_clustering_grid_params'][algorithm]
                        else:
                            GridSearchParameters = False

                        if algorithm in self.DatasetsConfigurations['proc_clustering_algorithms_parameters']:
                            Parameters = self.DatasetsConfigurations['proc_clustering_algorithms_parameters'][algorithm]
                        else:
                            Parameters = False

                        oClustering.runModel(
                            algorithm=algorithm,
                            Dataset=self._DataCache,
                            Parameters=Parameters,
                            GridSearchParameters=GridSearchParameters
                        )

                    self.processResults()

                    propheticus.shared.Utils.printTimeLogMessage('Parsing all the algorithms', start_time)

                self.action_context = None

    '''
    Utility Functions
    '''

    def help(self):
        funcion = inspect.stack()[1][3]
        _self = self

        def _help(choice):
            propheticus.shared.Utils.printBreadCrumb('Propheticus Help')
            # propheticus.shared.Utils.printNewLine()

            if funcion not in _self.Help:
                _self.Help[funcion] = 'May the odds be ever in your favor - Suzanne Collins'

            propheticus.shared.Utils.printMessage(_self.Help[funcion])
            propheticus.shared.Utils.printNewLine()
            propheticus.shared.Utils.printAcknowledgeMessage()
            propheticus.shared.Utils.consoleClear()

        return _help

    def TODO(self, choice):
        print("You'd like that wouldn't you? Still to be done...\n")
        propheticus.shared.Utils.printAcknowledgeMessage()

    def exit(self):
        sys.exit()

    def defineExcludeFeaturesByStaticValues(self):
        Headers = self.oDataManagement.datasets_headers
        if self.DatasetsConfigurations['pre_excluded_features_static'] is not None:
            FeaturesDetails = []
            for feature in self.DatasetsConfigurations['pre_excluded_features_static']:
                if feature not in Headers:
                    propheticus.shared.Utils.printWarningMessage('Static feature does not exist in the chosen datasets: ' + feature)
                    continue
                elif Headers.index(feature) == 0:
                    propheticus.shared.Utils.printErrorMessage(f'Static feature {feature} is the run title and cannot be removed')
                    continue

                FeaturesDetails.append({'index': Headers.index(feature), 'label': feature})

            self.DatasetsConfigurations['pre_excluded_features'] = FeaturesDetails

        return True

    def exportDatasets(self, apply_dimensionality_reduction=False):
        numpy.random.seed(propheticus.shared.Utils.RandomSeeds[0])
        random.seed(propheticus.shared.Utils.RandomSeeds[0])

        if self.InternalConfigurations['datasets_cache'] is not None:
            data_action_context = self.InternalConfigurations['datasets_cache']['action_context']
            if self.action_context != data_action_context:
                self.clearDataCache()
            else:
                propheticus.shared.Utils.printStatusMessage('Dataset cache loaded')
                return False

        if len(self.DatasetsConfigurations['datasets']) == 0:
            propheticus.shared.Utils.printErrorMessage('At least one dataset must be chosen')
            return True

        start_time = time.time()

        ExportedData = self.oDataManagement.exportData(
            self.mode,
            self.DatasetsConfigurations['datasets'],
            self.DatasetsConfigurations['config_sliding_window'],
            self.DatasetsConfigurations['config_sequential_data'],
            self.DatasetsConfigurations['config_binary_classification'],
            self.DatasetsConfigurations['config_undersampling_threshold'],
            self.DatasetsConfigurations['pre_excluded_features_label'],
            self.DatasetsConfigurations['pre_selected_features'],
            self.DatasetsConfigurations['pre_excluded_features'],
            self.DatasetsConfigurations['pre_target'],
            self.DatasetsConfigurations['pre_filter_feature_values'],
            self.DatasetsConfigurations['datasets_exclude_classes'],
            self.DatasetsConfigurations['datasets_exclude_run_classes'],
            self.DatasetsConfigurations['datasets_classes_remap'],
        )

        if ExportedData is False:
            propheticus.shared.Utils.printErrorMessage('Datasets did not pass validation!')
            return True
        else:
            Headers, Classes, DatasetData, DatasetDescriptions, DatasetTargets = ExportedData

        propheticus.shared.Utils.printTimeLogMessage('Processing the dataset files', start_time)

        if len(DatasetData) == 0:
            propheticus.shared.Utils.printErrorMessage('Current features selection does not return any rows. Redefine filters.')
            return True

        Dataset = {
            'headers': Headers,
            'classes': sorted(Classes),
            'data': DatasetData,
            'descriptions': DatasetDescriptions,
            'targets': DatasetTargets
        }

        # NOTE: clean up variables, to allow garbage collection when they are copied and no longer useful
        del Headers
        del DatasetData
        del DatasetDescriptions
        del DatasetTargets

        # NOTE: This validates if any other fields have the term 'label' which may be false predictors
        label_occurrences = sum(header.count('label') for header in Dataset['headers'])
        if label_occurrences > 1:
            propheticus.shared.Utils.printWarningMessage('Multiple features have the term label. Be alert for false predictors! ')

        # Config.thread_level_ = propheticus.shared.Utils.THREAD_LEVEL_ALGORITHM
        if self.action_context != 'algorithms':
            if not hasattr(Config, 'thread_level_'):
                Config.thread_level_ = propheticus.shared.Utils.THREAD_LEVEL_ALGORITHM

            if self.DatasetsConfigurations['proc_reduce_dimensionality'] is not False and 'variance' in self.DatasetsConfigurations['proc_reduce_dimensionality']:
                CallArguments = copy.deepcopy(self.DatasetsConfigurations['proc_reduce_dimensionality_parameters']['variance']) if 'variance' in self.DatasetsConfigurations['proc_reduce_dimensionality_parameters'] else {}
                CallDetails = Config.DimensionalityReductionCallDetails['variance']
                Estimator = propheticus.shared.Utils.dynamicAPICall(CallDetails, CallArguments)
                Dataset['data'] = Estimator.fit_transform(Dataset['data'])
                Dataset['headers'] = propheticus.core.DatasetReduction.removeFeaturesFromHeaders(Estimator.indexes_, Dataset['headers'])

            if self.action_context != 'descriptiveAnalysis' and self.DatasetsConfigurations['proc_normalize_data'] is not False:
                propheticus.shared.Utils.printStatusMessage('Normalizing dataset')
                NormalizeCallDetails = Config.PreprocessingCallDetails['normalize']
                Estimator = propheticus.shared.Utils.dynamicAPICall(NormalizeCallDetails)
                Dataset['data'] = Estimator.fit_transform(Dataset['data'])

            if self.DatasetsConfigurations['proc_reduce_dimensionality'] and (len(self.DatasetsConfigurations['proc_reduce_dimensionality']) > 1 or self.DatasetsConfigurations['proc_reduce_dimensionality'][0] != 'variance'):
                Dataset['data'], RemovedFeatures, Dataset['headers'], _ = propheticus.core.DatasetReduction.dimensionalityReduction(self.getDatasetsIdentifiers(), self.getConfigurationsIdentifier(),
                                                                                                                                    propheticus.shared.Utils.toStringCurrentConfigurations(self.DatasetsConfigurations),
                                                                                                                                    self.DatasetsConfigurations['proc_reduce_dimensionality'], Dataset['data'], Dataset['targets'], Dataset['headers'],
                                                                                                                                    self.DatasetsConfigurations['proc_reduce_dimensionality_parameters'], seed=1234)

        self.loadDataCache(self.action_context, Dataset)

        propheticus.shared.Utils.printStatusMessage('Chosen dataset has ' + str(len(Dataset['data'])) + ' records, with ' + str(len(Dataset['data'][0])) + ' features.')

        propheticus.shared.Utils.printTimeLogMessage('Exporting the data', start_time)

        return False

    def clearDataCache(self):
        self.InternalConfigurations['datasets_cache'] = None
        self._DataCache = None

    def loadDataCache(self, action_context, Dataset):
        self.InternalConfigurations['datasets_cache'] = {'action_context': action_context}
        self._DataCache = Dataset

    def printCurrentDatasetsHeaders(self):
        propheticus.shared.Utils.printStatusMessage('Current datasets headers:')
        Headers = ['[' + str(index + 1) + '] ' + header for index, header in enumerate(self.oDataManagement.datasets_headers)]

        items_per_row = 10
        for i in range(int(len(Headers) / items_per_row + 1)):
            propheticus.shared.Utils.printStatusMessage(', '.join(Headers[i * items_per_row:(i + 1) * items_per_row]))

    def getConfigurationsIdentifier(self):
        if self.hash_config_basename is True:
            basename = propheticus.shared.Utils.getConfigurationsIdentifier(self.DatasetsConfigurations)
        else:
            propheticus.shared.Utils.printFatalMessage('This functionality is not currently maintained. Please contact the author if strictly necessary')
            DescriptiveFields = [
                'config_binary_classification',
                'config_grid_search',
                'config_ensemble_algorithms',
                'config_ensemble_algorithms_parameters',
                'config_ensemble_selection',
                'config_ensemble_selection_parameters',
                'config_sliding_window',
                # 'datasets',
                'pre_target',
                'proc_balance_data',
                'proc_classification',
                'proc_classification_algorithms_parameters',
                'proc_classification_grid_params',
                'proc_clustering',
                'proc_preprocessing_parameters',
                'proc_clustering_algorithms_parameters',
                'proc_reduce_dimensionality'
            ]

            basename = "-".join(sorted([value.replace('"', '') for key, value in propheticus.shared.Utils.getSafeConfigurationsDict(self.DatasetsConfigurations).items() if key in DescriptiveFields]))

        return basename

    def getDatasetsIdentifiers(self):
        identifier = self.DatasetsConfigurations['datasets_base_name'] if self.DatasetsConfigurations['datasets_base_name'] != '' else ''.join(sorted(self.DatasetsConfigurations['datasets']))
        return propheticus.shared.Utils.getDatasetsIdentifiers(identifier)

    # TODO: create method to validate all configurations consistency

    def getFeatureDetailsByName(self, feature):
        return {'index': self.oDataManagement.datasets_headers.index(feature), 'label': feature}

    def processResults(self):
        propheticus.shared.Utils.printStatusMessage('Generating Reports \n')
        RunDetailsData = [[propheticus.shared.Utils.toStringCurrentConfigurations(self.DatasetsConfigurations, truncate=False)]]

        propheticus.shared.Utils.printStatusMessage('- Classification: ')

        if self.DatasetsConfigurations['proc_reduce_dimensionality'] is not False:
            RunDetailsData.append(['Dimensionality Reduced'])

        RunDetailsData.append([])
        RunDetailsData.append(['Custom Details'])
        RunDetailsData.append(['Data Classes', str(self._DataCache["classes"])])

        ClassificationProcessData = []

        ClassificationProcessData.append(['Classification Algorithms'])
        ClassificationProcessData.append(['Datasets:', '-'.join(self.DatasetsConfigurations['datasets'])])
        ClassificationProcessData.append(['Algorithm'] + list(Config.ClassificationReportHeaders.values()))

        for algorithm, Results in self.ClassificationAlgorithmsResults.items():
            propheticus.shared.Utils.printStatusMessage('-- Algorithm: ' + algorithm)
            if isinstance(Results, list):
                for index, Result in enumerate(Results):
                    label = algorithm + ' - ' + (f'seed {index + 1}' if index + 1 != len(Results) else 'Final ')
                    ParsedResults = [label] + self._processResults(Result, Config.ClassificationReportHeaders)
                    ClassificationProcessData.append(ParsedResults)

            else:
                exit('Should not be used?!')

            ClassificationProcessData.append([])

        ClusteringProcessData = []
        ClusteringProcessData.append(['Clustering Algorithms'])
        propheticus.shared.Utils.printStatusMessage('- Clustering: ')
        ClusteringProcessData.append(['Algorithm'] + list(Config.ClusteringReportHeaders.values()))
        for algorithm, Results in self.ClusteringAlgorithmsResults.items():
            propheticus.shared.Utils.printStatusMessage('-- Algorithm: ' + algorithm)
            for Result in Results:
                ParsedResults = [algorithm] + self._processResults(Result, Config.ClusteringReportHeaders)
                ClusteringProcessData.append(ParsedResults)

                ClusteringProcessData.append([])

        propheticus.shared.Utils.saveExcel(os.path.join(Config.framework_instance_generated_logs_path, self.getDatasetsIdentifiers()), self.getConfigurationsIdentifier() + '.Log.xlsx', RunDetailsData, ClassificationProcessData, ClusteringProcessData)

    def _processResults(self, OriginalResult, ReportHeaders):
        ReportHeadersLabels = list(ReportHeaders.values())
        ReportHeadersIndexes = list(ReportHeaders.keys())
        metrics_count = len(ReportHeadersLabels)
        Result = copy.deepcopy(OriginalResult)

        # if len(OriginalResult) != metrics_count:
        #     MissingKeys = list(set(Config.ClassificationReportHeaders.keys()) - set(OriginalResult.keys()))
        #     propheticus.shared.Utils.printWarningMessage(f'Not all results have the same structure! Missing keys: ' + ', '.join(MissingKeys))

        ParsedResults = [propheticus.shared.Utils.escapeExcelValues(Result.pop(key)) if key in Result else '' for key in ReportHeadersIndexes]
        if len(Result) > 0:
            propheticus.shared.Utils.printFatalMessage(f'Not all results have the same structure! Remaining keys: ' + ', '.join(Result.keys()))

        return ParsedResults

    def validateCurrentConfigurations(self):
        """
        Validates the current instance configurations

        Parameters
        ----------

        Returns
        -------

        """
        validated = self.validateChosenDatasets()
        if validated is False:
            return False

        if not isinstance(self.DatasetsConfigurations['config_binary_classification'], bool):
            propheticus.shared.Utils.printErrorMessage('Configuration binary classification must be a boolean value! Current value: ' + str(self.DatasetsConfigurations['config_binary_classification']))

        if self.DatasetsConfigurations['config_seed_count'] < 1:
            propheticus.shared.Utils.printErrorMessage('Configuration seed count must be >= 1! Current value: ' + str(self.DatasetsConfigurations['config_seed_count']))

        if self.DatasetsConfigurations['config_undersampling_threshold'] is not None and self.DatasetsConfigurations['config_undersampling_threshold'] < 0:
            propheticus.shared.Utils.printErrorMessage('Configuration undersampling threshold must be either None or >= 0! Current value: ' + str(self.DatasetsConfigurations['config_undersampling_threshold']))

        ValidateStaticKeys = {
            "proc_reduce_dimensionality": propheticus.shared.Utils.AvailableDimensionalityReduction,
            "proc_classification": propheticus.shared.Utils.AvailableClassificationAlgorithms,
            "proc_clustering": propheticus.shared.Utils.AvailableClusteringAlgorithms,
            "proc_balance_data": propheticus.shared.Utils.AvailableDataBalancing
        }

        for key, Data in ValidateStaticKeys.items():
            if not set(self.DatasetsConfigurations[key]).issubset(Data):
                propheticus.shared.Utils.printErrorMessage(f'Invalid configuration values for key {key} ({self.DatasetsConfigurations[key]})')
                return False

        # TODO: improve these validations
        if 'pre_target' in self.DatasetsConfigurations and not isinstance(self.DatasetsConfigurations['pre_target'], bool):
            if self.DatasetsConfigurations['pre_target'] in self.DatasetsConfigurations['pre_excluded_features']:
                propheticus.shared.Utils.printErrorMessage('Chosen label feature is set to be excluded')
                return False

            if self.DatasetsConfigurations['pre_target'] not in self.oDataManagement.datasets_headers:
                propheticus.shared.Utils.printErrorMessage(f'Chosen label {self.DatasetsConfigurations["pre_target"]} feature does not exist in the datasets chosen')
                return False

        return True

    def initializeHelpMessages(self):
        self.Help['mainMenu'] = '''The most recent documentation for Propheticus can be found at https://jrcampos.github.io/propheticus/
        \n\nMay the odds be ever in your favor - Suzanne Collins
        '''
