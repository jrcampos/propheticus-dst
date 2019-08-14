"""
Contains the code concerned with reducing the data, such as feature selection/extraction, as well as under-/over-sampling
"""
import time
import operator
import copy
import numpy
import math
import os

import propheticus
import propheticus.shared

import sys
sys.path.append(propheticus.Config.framework_selected_instance_path)
if os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, 'InstanceConfig.py')):
    from InstanceConfig import InstanceConfig as Config
else:
    import Config as Config

class DatasetReduction(object):
    """
    Contains the code concerned with reducing the data, such as feature selection/extraction, as well as under-/over-sampling
    """
    def __init__(self):
        a = 0

    @staticmethod
    def removeFeaturesFromHeaders(Features, Headers):
        """
        Removes features from the headers passed in arguments list. The reduction
        of the headers and data is made by copy. Be aware that changes to this method that alter this behavior
        should be carefully considered.

        Parameters
        ----------
        Features : list of int
        Headers : list of str

        Returns
        -------
        Headers : list of str
        """
        return [header for index, header in enumerate(Headers) if index not in Features]

    @staticmethod
    def removeFeaturesFromData(Features, Headers, Items):
        """
        Removes features from the data and headers passed in arguments list. The reduction
        of the headers and data is made by copy. Be aware that changes to this method that alter this behavior
        should be carefully considered.

        Parameters
        ----------
        Features : list of int
        Headers : list of str
        Items : object

        Returns
        -------
        Headers : list of str
        Items : object
        """
        Headers = [header for index, header in enumerate(Headers) if index not in Features]
        Items = numpy.delete(Items, Features, 1)

        return Headers, Items

    @staticmethod
    def dimensionalityReduction(dataset_name, configurations_id, description, method, Data, Target, Headers, Parameters, seed):
        """
        Reduces the dataset dimensionality. This method does not change the parameters values. The reduction
        of the headers and data is made by copy. Be aware that changes to this method that alter this behavior
        should be carefully considered. Available techniques are:

        * ``eigenvalues``
        * ``correlation``
        * ``mic``
        * ``f-score``
        * ``rfe``
        * ``pca``

        Parameters
        ----------
        dataset_name : str
        configurations_id : str
        description : object
        method : list of str
        Data : list of list of float
        Target : list of str
        Headers : list of str
        seed : int

        Returns
        -------
        list of list of float
        list of int
        list of str
        """
        # propheticus.shared.Utils.printStatusMessage('Starting dimensionality reduction')

        # TODO: change order from returned values; use estimator to apply directly to test data
        # TODO: create simple correlation removal with target; only works with supervised problems
        # TODO: shouldn't the dim red log also register features removed by null variance?

        start_time = time.time()

        Reductor = None

        FeatureSelection = []
        FeatureSelection.append([description])

        RemovedFeatures = []

        if 'pca' in method:
            method.append(method.pop(method.index('pca')))  # NOTE: PCA must always be the last method because it changes features/headers

        Estimators = DatasetReduction.buildDimensionalityReductionTransformers(method, Parameters, seed)

        for dim_red_method, Estimator in Estimators.items():
            Data = Estimator.fit_transform(Data, Target)

            if dim_red_method in ['correlation', 'eigenvalues', 'mic', 'rfe']:
                RemovedFeatures += Estimator.indexes_

                FeatureSelection.append([])
                FeatureSelection.append([dim_red_method + ': Removed Features'])
                FeatureSelection.append([header for index, header in enumerate(Headers) if index in Estimator.indexes_])
                Headers = DatasetReduction.removeFeaturesFromHeaders(Estimator.indexes_, Headers)

            elif dim_red_method == 'pca':
                Headers = ['PCA ' + str(i) for i in range(len(Data[0]))]

                FeatureSelection.append([])
                FeatureSelection.append(['PCA: Generated Components'])
                FeatureSelection.append([len(Data[0])])

        FeatureSelection.append([])
        FeatureSelection.append(['Remaining Features'])
        FeatureSelection.append(Headers)

        if dataset_name is not False:
            propheticus.shared.Utils.saveExcel(
                os.path.join(Config.framework_instance_generated_logs_path, dataset_name),
                configurations_id + ".dimensionality_reduction.data.xlsx",
                FeatureSelection,
                show_demo=False
            )

        return Data, list(set(RemovedFeatures)), Headers, Estimators

    @staticmethod
    def buildDimensionalityReductionTransformers(method, Parameters, seed):
        Transformers = {}

        OrderedTechniques = {}
        for _method in method:
            if _method in Parameters and 'propheticus_order' in Parameters[_method]:
                OrderedTechniques[_method] = Parameters[_method]['propheticus_order']

        MethodsOrder = sorted(OrderedTechniques, key=OrderedTechniques.get)
        MethodsOrder += [_method for _method in Config.DimensionalityReductionCallDetails.keys() if _method in method and _method not in MethodsOrder]

        for preprocessing_method in MethodsOrder:
            if preprocessing_method not in Config.DimensionalityReductionCallDetails:
                propheticus.shared.Utils.printFatalMessage('Preprocessing method not available! ' + preprocessing_method)

            CallDetails = copy.deepcopy(Config.DimensionalityReductionCallDetails[preprocessing_method])
            CallArguments = copy.deepcopy(Parameters[preprocessing_method]) if preprocessing_method in Parameters else {}

            if Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_ALGORITHM and 'n_jobs' in CallDetails['parameters']:
                CallArguments['n_jobs'] = -1

            if 'propheticus_order' in CallDetails['parameters']:
                del CallDetails['parameters']['propheticus_order']
            if 'propheticus_order' in CallArguments:
                del CallArguments['propheticus_order']

            Transformers[preprocessing_method] = propheticus.shared.Utils.dynamicAPICall(CallDetails, CallArguments, seed)

        return Transformers

    @staticmethod
    def balanceDataset(Data, Target, Parameters, seed, method):
        """
        Balances the dataset through under-/over-sampling techniques. Accepts multiple methods (max 1 for oversampling
        and another for undersampling) at once. This method does not change the parameters values. The reduction
        of the headers and data is made by copy. Be aware that changes to this method that alter this behavior
        should be carefully considered. Available methods are:

        * ``SMOTE`` - oversamples minority classes to a ration (default 200%) using SMOTE algorithm
        * ``RandomOverSampler`` - blindly oversamples minority classes until the # of the majority class
        * ``RandomUnderSampler`` - blindly undersamples majority classes until the # of the minority class

        Parameters
        ----------
        Data : list of list of float
        Target : list of str
        seed : int
        method : list of str

        Returns
        -------
        Data : list of list of float
        Target : list of str
        """

        Estimators = DatasetReduction.buildDataBalancingTransformers(method, Parameters, Target, seed)
        for data_sampling_method, Estimator in Estimators.items():
            Data, Target = Estimator.fit_sample(Data, Target)

        return Data, Target

    @staticmethod
    def buildDataBalancingTransformers(method, Parameters, Target, seed, grid_search_cv_count=None):
        Transformers = {}

        DistributionByClass = propheticus.shared.Utils.getClassDistribution(Target)
        SamplingTypes = set([Config.SamplingCallDetails[sampling_method]['type'] for sampling_method in method])
        math_fn = math.ceil if 'oversampling' in SamplingTypes else math.floor

        if grid_search_cv_count is not None:
            DistributionByClass = {key: math_fn(value * (grid_search_cv_count - 1)/grid_search_cv_count) for key, value in DistributionByClass.items()}

        OrderedTechniques = {}
        for _method in method:
            if _method in Parameters and 'propheticus_order' in Parameters[_method]:
                OrderedTechniques[_method] = Parameters[_method]['propheticus_order']

        MethodsOrder = sorted(OrderedTechniques, key=OrderedTechniques.get)
        MethodsOrder += [_method for _method in Config.SamplingCallDetails.keys() if _method in method and _method not in MethodsOrder]

        for sampling_method in MethodsOrder:
            if sampling_method not in Config.SamplingCallDetails:
                propheticus.shared.Utils.printFatalMessage('Sampling method not available! ' + sampling_method)

            CallDetails = copy.deepcopy(Config.SamplingCallDetails[sampling_method])
            CallArguments = copy.deepcopy(Parameters[sampling_method]) if sampling_method in Parameters else {}

            propheticus_ratio = CallArguments['propheticus_ratio'] if 'propheticus_ratio' in CallArguments else CallDetails['parameters']['propheticus_ratio']['default']
            if Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_ALGORITHM and 'n_jobs' in CallDetails['parameters']:
                CallArguments['n_jobs'] = -1

            if CallDetails['type'] == 'oversampling':
                high_majority_target = max(DistributionByClass.items(), key=operator.itemgetter(1))[0]
                DistributionByClass = OversamplingRatio = {target: (int(propheticus_ratio * count) if target != high_majority_target else count) for target, count in DistributionByClass.items()}
                CallArguments['sampling_strategy'] = OversamplingRatio

            elif CallDetails['type'] == 'undersampling':
                high_majority_target = max(DistributionByClass.items(), key=operator.itemgetter(1))[0]
                second_high_majority_count = sorted(DistributionByClass.items(), key=operator.itemgetter(1), reverse=True)[1][1]
                UndersamplingRatio = {target: (int(propheticus_ratio * second_high_majority_count) if target == high_majority_target else count) for target, count in DistributionByClass.items()}
                for target, count in UndersamplingRatio.items():
                    if count > DistributionByClass[target]:
                        propheticus.shared.Utils.printFatalMessage(f'There are not enough samples to apply the chosen ratio ({propheticus_ratio}); original: {DistributionByClass[target]}, requested: {count}')

                DistributionByClass = UndersamplingRatio
                CallArguments['sampling_strategy'] = UndersamplingRatio

            else:
                propheticus.shared.Utils.printFatalMessage('Invalid data balancing type: ' + CallDetails['type'])

            if 'propheticus_ratio' in CallDetails['parameters']:
                del CallDetails['parameters']['propheticus_ratio']
            if 'propheticus_ratio' in CallArguments:
                del CallArguments['propheticus_ratio']

            if 'propheticus_order' in CallDetails['parameters']:
                del CallDetails['parameters']['propheticus_order']
            if 'propheticus_order' in CallArguments:
                del CallArguments['propheticus_order']

            Transformers[sampling_method] = propheticus.shared.Utils.dynamicAPICall(CallDetails, CallArguments, seed)

        return Transformers
