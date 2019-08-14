"""
Contains the code for loading and preparing the data
"""
import collections
import numpy
import os
import multiprocessing
import time
import random
import sys
import gc
import json
import sklearn.preprocessing
import operator

import propheticus
import propheticus.shared

# NOTE: imports Client/Data configurations from the data source defined in the propheticus.Config file

import sys
sys.path.append(propheticus.Config.framework_selected_instance_path)
if os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, 'InstanceConfig.py')):
    from InstanceConfig import InstanceConfig as Config
else:
    import propheticus.Config as Config

class DataManagement(object):
    """
    Contains the code for loading and preparing the data

    ...

    Attributes
    ----------

    Parameters
    ----------
    """
    def __init__(self, mode):
        self.mode = mode

        self.Datasets = None
        self.datasets_headers = None
        self.datasets_headers_details = None


    def validateChosenDatasets(self, Datasets):
        self.Datasets = Datasets
        Headers = []
        DataTypes = []
        for dataset in self.Datasets:
            file_path = os.path.join(Config.framework_instance_data_path, dataset + '.headers.txt')
            with open(file_path, encoding='utf-8') as f:
                content = json.load(f)

                CurrentDatasetHeaders = []
                CurrentDatasetHeadersDetails = []
                for HeaderItemDetails in content:
                    header = HeaderItemDetails['name']
                    data_type = HeaderItemDetails['type']
                    categorical = HeaderItemDetails['categorical'] if 'categorical' in HeaderItemDetails else False
                    if categorical is not False:
                        for value in categorical:
                            if value.count(' ') > 0:
                                propheticus.shared.Utils.printErrorMessage('Categorical feature values cannot contain spaces!')
                                return False

                        if data_type == 'string':
                            Binarizer = sklearn.preprocessing.LabelBinarizer()
                            Binarizer.fit(categorical)
                        elif 'int' in data_type:
                            Binarizer = sklearn.preprocessing.OneHotEncoder()
                            Binarizer.fit(categorical)
                        else:
                            propheticus.shared.Utils.printErrorMessage('Dataset contains a categorical feature with invalid datatype! ' + dataset + ' =>  ' + header + ' : ' + data_type)
                            return False

                        categorical = {'encoder': Binarizer, 'values': categorical}

                    CurrentDatasetHeaders.append(header)
                    CurrentDatasetHeadersDetails.append({
                        'type': data_type,
                        'categorical': categorical,
                        'label': header
                    })

                if len(set(CurrentDatasetHeaders)) != len(CurrentDatasetHeaders):
                    propheticus.shared.Utils.printErrorMessage('Dataset has multiple features with the same name')
                    return False

                if len(Headers) == 0:
                    Headers = CurrentDatasetHeaders
                    DataTypes = CurrentDatasetHeadersDetails
                else:
                    if Headers != CurrentDatasetHeaders:
                        propheticus.shared.Utils.printErrorMessage('Datasets chosen do not share the same headers structure')
                        return False

                    if DataTypes != CurrentDatasetHeadersDetails:
                        propheticus.shared.Utils.printErrorMessage('Datasets chosen do not share the same headers data types')
                        return False

        self.datasets_headers = Headers
        self.datasets_headers_details = DataTypes

        return True
    
    def exportData(
            self,
            Datasets,
            config_sliding_window,
            config_sequential_data,
            config_binary_classification,
            config_undersampling_threshold,
            pre_excluded_features_label,
            pre_selected_features,
            pre_excluded_features,
            pre_target,
            pre_filter_feature_values,
            datasets_exclude_classes,
            datasets_exclude_run_classes
        ):

        if Datasets != self.Datasets:
            validated = self.validateChosenDatasets(Datasets)
            if not validated:
                return validated

        # TODO: improve parameters logic; not all are required as class attributes?

        self.Datasets = Datasets
        self.config_sliding_window = config_sliding_window
        self.config_sequential_data = config_sequential_data
        self.config_undersampling_threshold = config_undersampling_threshold
        self.config_binary_classification = config_binary_classification
        self.pre_excluded_features_label = pre_excluded_features_label
        self.pre_selected_features = pre_selected_features
        self.pre_excluded_features = pre_excluded_features
        self.pre_target = pre_target
        self.pre_filter_feature_values = pre_filter_feature_values
        self.datasets_exclude_classes = datasets_exclude_classes
        self.datasets_exclude_run_classes = datasets_exclude_run_classes


        '''
        Count the total number of records from the chosen datasets
        '''
        datasets_records = 0
        for dataset in self.Datasets:
            file_path = os.path.join(Config.framework_instance_data_path, dataset + '.info.txt')
            with open(file_path, encoding='utf-8') as f:
                content = f.readlines()
            datasets_records += int(content[0])

        ''' NOTE: 
        This is done outside the undersampling condition to perform other preprocessing actions, 
        such as removing unwanted runs
        '''
        Classes = {}
        IndexesByClass = {}

        sampling_start_time = time.time()
        propheticus.shared.Utils.printStatusMessage('Preprocessing data files', inline=True)
        pool_count = min(Config.max_thread_count, len(self.Datasets))
        if pool_count > 1 and (self.mode == 'cli' or self.mode == 'batch' and Config.thread_level_ != propheticus.shared.Utils.THREAD_LEVEL_BATCH):
            DatasetsUndersamplingData = propheticus.shared.Utils.pool(pool_count, self._undersampleData, self.Datasets)
        else:
            DatasetsUndersamplingData = [self._undersampleData(dataset) for dataset in self.Datasets]

        propheticus.shared.Utils.printNewLine()

        RemovedIndexes = {dataset: [] for dataset in self.Datasets}

        DatasetsFailureRuns = {}
        DatasetsCleanRuns = {}

        for index, dataset in enumerate(self.Datasets):
            for key, value in DatasetsUndersamplingData[index][0].items():
                if key not in Classes:
                    Classes[key] = 0

                Classes[key] += value

            if self.config_sequential_data is True:
                for clean_run, Details in DatasetsUndersamplingData[index][3].items():
                    DatasetsCleanRuns[dataset + '|' + clean_run] = Details

                for failure_run in DatasetsUndersamplingData[index][4]:
                    DatasetsFailureRuns[dataset + '|' + failure_run] = failure_run
            else:
                for key, Indexes in DatasetsUndersamplingData[index][3].items():
                    if key not in IndexesByClass:
                        IndexesByClass[key] = []

                    IndexesByClass[key] += [dataset + '|' + str(item_index) for item_index in Indexes]

            RemovedIndexes[dataset] += DatasetsUndersamplingData[index][1]

        GRuns = [key for key in DatasetsCleanRuns.keys() if 'G_' in key]

        '''
        Undersample the target data to match the maximum records threshold
        '''
        # TODO: validate that this sampling makes sense; only useful for highly imbalanced datasets?
        if self.config_undersampling_threshold is not None and datasets_records >= self.config_undersampling_threshold:

            propheticus.shared.Utils.printStatusMessage('Undersampling is required. Total # of records: ' + str(datasets_records), inline=True)

            propheticus.shared.Utils.printNewLine()
            Classes = {key: value for key, value in Classes.items() if value > 0}
            distributions = '\n'.join(sorted(['- ' + propheticus.shared.Utils.getClassDescriptionById(key) + ': ' + str(value) for key, value in Classes.items()]))
            propheticus.shared.Utils.printStatusMessage(f'Undersampling Classes distribution in the chosen datasets: \n{distributions}')

            high_majority_target = max(Classes.items(), key=operator.itemgetter(1))[0]
            target_sampling_value = self.config_undersampling_threshold - sum([value for key, value in Classes.items() if key != high_majority_target])
            if target_sampling_value <= 0:
                exit('The undersampling threshold is too low for the amount of non-target samples!')

            count = 0
            min_gold_runs = 0
            current_target_samples = Classes[high_majority_target]
            while current_target_samples > target_sampling_value:
                count += 1
                if count > 100000:
                    propheticus.shared.Utils.printFatalMessage('Possible infinite loop?')

                if self.config_sequential_data is True:
                    remove = random.choice(list(DatasetsCleanRuns.keys()))
                    # TODO: improve, too hardcoded
                    if 'G_' in remove:
                        GRuns.remove(remove)
                        if len(GRuns) == min_gold_runs:
                            for g_run in GRuns:
                                del DatasetsCleanRuns[g_run]

                    current_target_samples -= len(DatasetsCleanRuns[remove])
                    RemovedIndexes[remove.split('|')[0]] += DatasetsCleanRuns[remove]
                    del DatasetsCleanRuns[remove]
                else:
                    SelectedIndexes = random.sample(IndexesByClass[high_majority_target], current_target_samples - target_sampling_value)
                    for item_index in SelectedIndexes:
                        RemovedIndexes[item_index.split('|')[0]].append(int(item_index.split('|')[1]))

                    current_target_samples -= len(SelectedIndexes)

            propheticus.shared.Utils.printStatusMessage(f'Majority samples ({high_majority_target}) reduced to: {current_target_samples}')

            if self.config_sequential_data is True:
                if len(DatasetsCleanRuns) == 0 and current_target_samples > target_sampling_value:
                    propheticus.shared.Utils.printWarningMessage('It was not possible to further reduce the number of the majority samples: ' + str(current_target_samples))

                if len(DatasetsFailureRuns.keys()) > len(GRuns):
                    propheticus.shared.Utils.printWarningMessage('There are more failure runs than gold runs in the data selected: ' + str(len(DatasetsCleanRuns.keys())) + ' ' + str(len(GRuns)) + ' ' + str(len(DatasetsFailureRuns.keys())))

            propheticus.shared.Utils.printTimeLogMessage('Sampling the data', sampling_start_time)

        if self.config_sequential_data is True:
            propheticus.shared.Utils.printStatusMessage('Remaining safe runs: ' + str(len(DatasetsCleanRuns.keys())) + ' ' + ', '.join(DatasetsCleanRuns.keys()))
            propheticus.shared.Utils.printStatusMessage('Remaining golden runs: ' + str(len(GRuns)) + ' ' + ', '.join(GRuns))
            propheticus.shared.Utils.printStatusMessage('Remaining failure runs: ' + str(len(DatasetsFailureRuns.keys())) + ' ' + ', '.join(DatasetsFailureRuns.keys()))

        propheticus.shared.Utils.printStatusMessage('Exporting data', inline=True)

        LabelExcludeFeatures = self.pre_excluded_features_label
        SelectedFeatures = self.pre_selected_features
        SelectedFeaturesHeadersIndexes = list(set([Details['index'] for Details in SelectedFeatures]))
        ExcludeFeatures = self.pre_excluded_features
        ExcludedFeaturesHeadersIndexes = list(set([Details['index'] for Details in (ExcludeFeatures + LabelExcludeFeatures)]))

        # NOTE: this is the label_index without considering categorical features shift;
        label_index = self.pre_target['index'] if self.pre_target is not False else len(self.datasets_headers) - 1
        if len(SelectedFeatures) > 0:
            _Headers = numpy.array([value for _index, value in enumerate(self.datasets_headers) if _index not in [0, label_index] and _index in SelectedFeaturesHeadersIndexes])
        else:
            _Headers = numpy.array([value for _index, value in enumerate(self.datasets_headers) if _index not in [0, label_index] + ExcludedFeaturesHeadersIndexes])

        _EncodedHeaders = []
        for index, header in enumerate(_Headers):
            feature_index = self.datasets_headers.index(header)
            categorical = self.datasets_headers_details[feature_index]['categorical']
            if categorical is not False:
                encoder_classes_len = len(categorical['encoder'].classes_)
                _EncodedHeaders += [header + ' OHE ' + str(i) for i in range(encoder_classes_len)]
            else:
                _EncodedHeaders.append(header)

        _Headers = _EncodedHeaders

        Headers = [None] * len(_Headers) * self.config_sliding_window
        for window in range(self.config_sliding_window):
            Headers[window::self.config_sliding_window] = [header + ((' -' + str((self.config_sliding_window - 1) + window)) if window != (self.config_sliding_window - 1) else '') for header in _Headers]

        '''
        Parse the datasets files into the structure required by the platform
        '''
        pool_count = min(Config.max_thread_count, len(self.Datasets))
        if pool_count > 1 and (self.mode == 'cli' or self.mode == 'batch' and Config.thread_level_ != propheticus.shared.Utils.THREAD_LEVEL_BATCH):
            DatasetsData = propheticus.shared.Utils.pool(pool_count, self._exportDatasetFile, [(dataset, RemovedIndexes[dataset]) for dataset in self.Datasets])
        else:
            DatasetsData = [self._exportDatasetFile(dataset, RemovedIndexes[dataset]) for dataset in self.Datasets]

        DatasetsData = numpy.array(DatasetsData)

        DatasetData = numpy.concatenate(DatasetsData[:, 0])
        DatasetDescriptions = numpy.concatenate(DatasetsData[:, 1])
        DatasetTargets = numpy.concatenate(DatasetsData[:, 2])

        del DatasetsData

        SupportByClass = collections.Counter(DatasetTargets)

        propheticus.shared.Utils.printNewLine()
        propheticus.shared.Utils.printStatusMessage('Classes distribution: \n' + '\n'.join(sorted(['- ' + key + ': ' + str(value) for key, value in SupportByClass.items()])))

        return Headers, set(DatasetTargets), DatasetData, DatasetDescriptions, DatasetTargets

    def _exportDatasetFile(self, dataset, RemovedIndexes):
        """
        NOTE: Due to the fact that indexes start in 0, in order to calculate the position of the target index on the merged
        array for sliding window, accounting that it will be appended to the end of the merges, the easiest way it is by adding 1 to
        the index, multiply by the sliding window, and subtract 1:
        Example:
            target index = 266 (e.g. we want the 267th item)
            self.config_sliding_window = 2

            new_index = ((target_index + 1) * self.config_sliding_window ) - 1
            new_index == 533

            self.config_sliding_window = 3
            0 => ((0 + 1) * 3) - 1 = 2
            1 => ((1 + 1) * 3) - 1 = 5
            2 => ((2 + 1) * 3) - 1 = 8
            ...


        If the intended index is appended as the first, simply multiplying by the sliding window will work.


        Parameters
        ----------
        dataset
        RemovedIndexes

        Returns
        -------

        """
        propheticus.shared.Utils.printInlineStatusMessage('.')
        a = Config.framework_instance_data_path
        file_path = os.path.join(Config.framework_instance_data_path, dataset + '.data.txt')
        with open(file_path, encoding='utf-8') as f:
            content = f.readlines()

        extra_cols_encoder = 0
        CategoricalFeatures = {}
        for index, HeaderDetails in enumerate(self.datasets_headers_details):
            if HeaderDetails['categorical'] is not False:
                CategoricalFeatures[index] = HeaderDetails['categorical']['encoder']
                extra_cols_encoder += len(CategoricalFeatures[index].classes_) - 1

        len_item = len(content[0].split()) + extra_cols_encoder

        if len(CategoricalFeatures) > 0:
            propheticus.shared.Utils.printWarningMessage('Categorical features at runtime is time consuming. Consider doing this when generating the dataset')

        ItemIndexes = sorted(set(range(len(content))) - set(RemovedIndexes))

        FeaturesIndexes = {}
        ContentItem = content[ItemIndexes[0]].split()
        _ContentItem = []
        for feature_index, value in enumerate(ContentItem):
            feature_label = self.datasets_headers[feature_index]
            FeaturesIndexes[feature_label] = []

            if feature_index in CategoricalFeatures:
                Encoder = CategoricalFeatures[feature_index]
                EncodedFeatureValue = Encoder.transform([value])[0]

                for i in range(len(EncodedFeatureValue)):
                    _ContentItem.append(EncodedFeatureValue[i])
                    FeaturesIndexes[feature_label].append(len(_ContentItem) - 1)

            else:
                _ContentItem.append(value)
                FeaturesIndexes[feature_label].append(len(_ContentItem) - 1)

        selected_label = self.pre_target['label'] if self.pre_target is not False else self.datasets_headers[-1]
        if len(FeaturesIndexes[selected_label]) > 1:
            propheticus.shared.Utils.printFatalMessage('Selected label is categorical and as such cannot be chosen as label (allow later)')

        label_index = FeaturesIndexes[selected_label][0]
        slide_label_index = ((label_index + 1) * self.config_sliding_window) - 1  # NOTE: it is necessary to use the target of the intended instance, as the previous in the sliding window may not have the same target

        binary_classification = self.config_binary_classification

        SelectedFeaturesLabels = [Details['label'] for Details in self.pre_selected_features]

        LabelExcludeFeatures = self.pre_excluded_features_label
        ExcludeFeatures = self.pre_excluded_features
        ExcludedFeaturesLabels = list(set([Details['label'] for Details in (ExcludeFeatures + LabelExcludeFeatures)]))

        FilterFeaturesIndexes = {}
        FilterFeatures = self.pre_filter_feature_values
        for Details in FilterFeatures:
            filter_index = Details['index']
            filter_label = Details['feature_name']
            if len(FeaturesIndexes[filter_label]) > 1:
                Encoder = CategoricalFeatures[filter_index]

                FilterIndexes = FeaturesIndexes[filter_label]
                Values = Details['values']
                for value in Values.split(','):
                    EncodedFeatureValue = Encoder.transform([value])[0]
                    for filter_index, encoded_value in zip(FilterIndexes, EncodedFeatureValue):
                        if filter_index not in FilterFeaturesIndexes:
                            FilterFeaturesIndexes[filter_index] = str(encoded_value)
                        else:
                            FilterFeaturesIndexes[filter_index] += ',' + str(encoded_value)
            else:
                FilterFeaturesIndexes[FeaturesIndexes[filter_label][0]] = Details['values']

        if len(SelectedFeaturesLabels) > 0:
            for feature in self.datasets_headers[1:]:
                if feature == selected_label or feature in SelectedFeaturesLabels or feature in ExcludedFeaturesLabels:
                    continue

                ExcludedFeaturesLabels.append(feature)

        len_filter_features = len(FilterFeaturesIndexes)
        len_exclude_features = len(ExcludedFeaturesLabels)

        DatasetData = []
        DatasetDescriptions = []
        DatasetTargets = []

        use_description_field = True if self.datasets_headers[0] == 'Description' else False

        '''Config
        TODO
        - Improve validation of the values (types) for the features according to the configurations
        '''
        for index in ItemIndexes:
            # TODO: review the following if; .splits() can be optimized outside and reused ahead?
            if self.config_sliding_window > 1 and (index < (self.config_sliding_window - 1) or content[index - (self.config_sliding_window - 1)].split()[0] != content[index].split()[0]):
                removing_label = int(content[index].split()[label_index])
                if removing_label != 0 and propheticus.shared.Utils.getClassDescriptionById(removing_label) not in self.datasets_exclude_classes:
                    label_ = propheticus.shared.Utils.getClassDescriptionById(removing_label)
                    propheticus.shared.Utils.printFatalMessage(f'A sample with a relevant target was about to be removed! {label_}')

                continue

            if self.config_sliding_window > 1:
                Item = [None] * len_item * self.config_sliding_window
                for window in range(self.config_sliding_window):
                    ContentItem = content[index - (self.config_sliding_window - 1) + window].split()
                    if len(CategoricalFeatures) > 0:
                        _ContentItem = []
                        for feature_index, value in enumerate(ContentItem):
                            if feature_index in CategoricalFeatures:
                                Encoder = CategoricalFeatures[feature_index]
                                EncodedFeatureValue = Encoder.transform([value])[0]
                                _ContentItem += [EncodedFeatureValue[i] for i in range(len(EncodedFeatureValue))]
                            else:
                                _ContentItem.append(value)

                        ContentItem = _ContentItem

                    Item[window::self.config_sliding_window] = ContentItem  # NOTE: This adds from the oldest to the latest instance
            else:
                ContentItem = content[index].split()
                if len(CategoricalFeatures) > 0:
                    _ContentItem = []
                    for feature_index, value in enumerate(ContentItem):
                        if feature_index in CategoricalFeatures:
                            Encoder = CategoricalFeatures[feature_index]
                            EncodedFeatureValue = Encoder.transform([value])[0]
                            _ContentItem += [EncodedFeatureValue[i] for i in range(len(EncodedFeatureValue))]
                        else:
                            _ContentItem.append(value)

                    ContentItem = _ContentItem

                Item = ContentItem

            # TODO: this needs to change to implement deltas as features
            label = int(Item[slide_label_index])

            if self.config_sliding_window > 1:
                direct_label = int(content[index].split()[label_index])
                if label != direct_label:
                    direct_label_ = propheticus.shared.Utils.getClassDescriptionById(direct_label)
                    label_ = propheticus.shared.Utils.getClassDescriptionById(label)
                    propheticus.shared.Utils.printFatalMessage(f'Slided label is different than the original! Correct: {direct_label_} Slided: {label_}')

            if propheticus.shared.Utils.getClassDescriptionById(label) in self.datasets_exclude_classes:
                continue

            # NOTE: the following line was required to create rec.array, it only accepts a list of tuples
            # DatasetData.append(tuple(Item[self.config_sliding_window:]))

            start_index = self.config_sliding_window if use_description_field else 0
            DatasetData.append(Item[start_index:])
            DatasetDescriptions.append(Item[0] if use_description_field else '-')

            if binary_classification and label != Config.ClassesMapping['Binary_Base']:
                label = Config.ClassesMapping['Binary_Error']

            target = propheticus.shared.Utils.getClassDescriptionById(label)
            DatasetTargets.append(target)

        del content

        '''
        NOTE:
        Do not remove the following code! Initial approach to handle different data types and create recarray from it.
        However, as the data is afterwards normalized, and categorical values must be converted to OneHotEncoding, 
        this is no longer strictly necessary/useful. Nonetheless, leave the code for future consideration.
        '''
        # if len(self.datasets_headers_details) != 0:
        #     HeadersDataTypes = [None] * len(self.datasets_headers_details[1:]) * self.config_sliding_window
        #     for window in range(self.config_sliding_window):
        #         HeadersDataTypes[window::self.config_sliding_window] = [(self.datasets_headers[h_index + 1], DBI.PYTHON_NUMPY_DATA_TYPES_MAP[HeaderDetails['type']]) for h_index, HeaderDetails in enumerate(self.datasets_headers_details[1:])]
        #
        #     dtype = HeadersDataTypes
        # else:
        #     dtype = numpy.float64
        #
        # # NOTE: convert to structured numpy array to keep data types
        # DatasetData = numpy.rec.array(DatasetData, dtype=dtype)

        DatasetData = numpy.array(DatasetData, dtype=numpy.float64)
        if len(DatasetData) > 0:
            if len_filter_features > 0:
                # NOTE: the following line subtracts the self.config_sliding_window value, as this is going to be used in the DatasetData object, which has the first self.config_sliding_window * label removed

                for feature_id, Values in FilterFeaturesIndexes.items():
                    results = numpy.in1d(DatasetData[:, (((feature_id + 1) * self.config_sliding_window) - 1) - self.config_sliding_window], numpy.array(Values.split(','), dtype=numpy.float32))
                    temp = numpy.where(results)

                Filters = numpy.logical_and.reduce([numpy.in1d(DatasetData[:, (((feature_id + 1) * self.config_sliding_window) - 1) - self.config_sliding_window], numpy.array(Values.split(','), dtype=numpy.float32)) for feature_id, Values in FilterFeaturesIndexes.items()])
                _Indexes = numpy.where(Filters)
                DatasetData = DatasetData[_Indexes]
                DatasetTargets = numpy.array(DatasetTargets)[_Indexes].tolist()
                DatasetDescriptions = numpy.array(DatasetDescriptions)[_Indexes].tolist()

            DeleteIndexes = [slide_label_index]
            if len_exclude_features > 0:
                ExcludedFeaturesIndexes = list(set(numpy.hstack([FeaturesIndexes[label] for label in ExcludedFeaturesLabels])))
                DeleteIndexes += list(((numpy.array(ExcludedFeaturesIndexes) + 1) * self.config_sliding_window) - 1)  # NOTE: this calculates the position of the more recent value of the feature

            if self.config_sliding_window > 1:
                DeleteIndexes = numpy.array([numpy.array(DeleteIndexes) - window for window in range(0, self.config_sliding_window)]).ravel()

            DeleteIndexes = list(numpy.array(DeleteIndexes) - self.config_sliding_window)  # NOTE: this line is required as we are going to remove from DatasetData, which already does not have the initial labels
            DatasetData = numpy.delete(DatasetData, DeleteIndexes, 1)
        else:
            propheticus.shared.Utils.printWarningMessage('No samples were chosen from the dataset: ' + dataset)

        return DatasetData, DatasetDescriptions, DatasetTargets, dataset

    def _undersampleData(self, dataset):
        # NOTE: this function does not consider categorical features because it only uses the label index
        propheticus.shared.Utils.printInlineStatusMessage('.')
        return self._undersampleSequentialData(dataset) if self.config_sequential_data is True else self._undersampleIITData(dataset)

    def _undersampleSequentialData(self, dataset):
        label_index = self.pre_target['index'] if self.pre_target is not False else len(self.datasets_headers) - 1

        Classes = {}

        file_path = os.path.join(Config.framework_instance_data_path, dataset + '.data.txt')
        with open(file_path, encoding='utf-8') as f:
            content = f.readlines()

        RemoveIndexes = []

        FailureRuns = {}
        CleanRuns = {}
        ClassesByRun = {}
        IndexesByRun = {}
        RemoveUnwantedRuns = {}

        # TODO: this logic needs to be more abstract
        for index, Item in enumerate(content):
            Item = Item.split()
            if len(Item) <= 1:
                exit('Datasets files cannot contain empty lines: ' + dataset)

            target = int(Item[label_index])
            run = description = Item[0]

            # NOTE: undersampling for sequential data does so by removing complete runs to avoid broken sequences
            if run not in IndexesByRun:
                IndexesByRun[run] = []
            IndexesByRun[run].append(index)

            if target == Config.ClassesMapping['Baseline'] and run not in FailureRuns:
                if run not in CleanRuns:
                    CleanRuns[run] = []
                CleanRuns[run].append(index)
            elif target != Config.ClassesMapping['RemoveDeltaTL'] and run in CleanRuns:
                del CleanRuns[run]
                FailureRuns[run] = True

            if run not in ClassesByRun:
                ClassesByRun[run] = {}

            if propheticus.shared.Utils.getClassDescriptionById(target) in self.datasets_exclude_run_classes and run not in RemoveUnwantedRuns:
                RemoveUnwantedRuns[run] = target
            elif propheticus.shared.Utils.getClassDescriptionById(target) in self.datasets_exclude_classes:  # NOTE: if it doesn't remove the whole run may remove just the sample
                RemoveIndexes.append(index)
            elif run in RemoveUnwantedRuns and target != Config.ClassesMapping['Baseline'] and propheticus.shared.Utils.getClassDescriptionById(target) not in self.datasets_exclude_run_classes:
                unwanted_description = propheticus.shared.Utils.getClassDescriptionById(RemoveUnwantedRuns[run])
                wanted_description = propheticus.shared.Utils.getClassDescriptionById(target)
                propheticus.shared.Utils.printFatalMessage(f'Run contained failures to be removed and kept. All/none failure types must be removed. {run} -> {wanted_description} : {unwanted_description}')
            else:
                if target not in ClassesByRun[run]:
                    ClassesByRun[run][target] = 0

                ClassesByRun[run][target] += 1

        for run, RunClasses in ClassesByRun.items():
            if run not in RemoveUnwantedRuns:
                for target, count in RunClasses.items():
                    if target not in Classes:
                        Classes[target] = 0

                    Classes[target] += count

        for run in RemoveUnwantedRuns:
            if run in FailureRuns:
                del FailureRuns[run]

            RemoveIndexes += IndexesByRun[run]

        return Classes, list(set(RemoveIndexes)), dataset, CleanRuns, sorted(FailureRuns.keys())

    def _undersampleIITData(self, dataset):
        label_index = self.pre_target['index'] if self.pre_target is not False else len(self.datasets_headers) - 1

        file_path = os.path.join(Config.framework_instance_data_path, dataset + '.data.txt')
        with open(file_path, encoding='utf-8') as f:
            content = f.readlines()

        RemoveIndexes = []
        Classes = {}
        IndexesByClass = {}

        for index, Item in enumerate(content):
            Item = Item.split()
            if len(Item) <= 1:
                exit('Datasets files cannot contain empty lines: ' + dataset)

            target = int(Item[label_index])
            if propheticus.shared.Utils.getClassDescriptionById(target) in self.datasets_exclude_classes:
                RemoveIndexes.append(index)
            else:
                if target not in Classes:
                    Classes[target] = 0
                    IndexesByClass[target] = []

                IndexesByClass[target].append(index)
                Classes[target] += 1

        return Classes, list(set(RemoveIndexes)), dataset, IndexesByClass, None
