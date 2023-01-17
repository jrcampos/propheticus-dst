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

import pandas
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
    def __init__(self, Context=None):
        self.Context = Context
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
            mode,
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
            datasets_exclude_run_classes,
            datasets_classes_remap
        ):
        self.mode = mode

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

            if len(content) == 0:
                propheticus.shared.Utils.printFatalMessage(f'{dataset} has no details in info file!?')
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

        # TODO: this logic/naming is not clear; why undersampling if not required? clean up
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

        propheticus.shared.Utils.printStatusMessage(f'Dataset contains {len(DatasetsFailureRuns.keys())} Failure Runs and {len(DatasetsCleanRuns.keys())} Clean Runs ({len(GRuns)} Gold Runs)')

        '''
        Undersample the target data to match the maximum records threshold
        '''
        # TODO: validate that this sampling makes sense; only useful for highly imbalanced datasets?
        if self.config_undersampling_threshold is not None and datasets_records >= self.config_undersampling_threshold:

            propheticus.shared.Utils.printStatusMessage('Undersampling is required. Total # of records: ' + str(datasets_records), inline=True)

            propheticus.shared.Utils.printNewLine()
            Classes = {key: value for key, value in Classes.items() if value > 0}
            distributions = '\n'.join(sorted(['- ' + str(propheticus.shared.Utils.getClassDescriptionById(key)) + ': ' + str(value) for key, value in Classes.items()]))
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
                    if len(DatasetsCleanRuns) == 0:
                        propheticus.shared.Utils.printFatalMessage('No clean runs remain!')

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
                    propheticus.shared.Utils.printWarningMessage(f'There are more failure runs than gold runs in the data selected: {len(DatasetsCleanRuns.keys())} {len(GRuns)} {len(DatasetsFailureRuns.keys())}')

                if len(DatasetsFailureRuns.keys()) > len(DatasetsCleanRuns):
                    propheticus.shared.Utils.printWarningMessage(f'There are more failure runs than clean runs in the data selected: {len(DatasetsCleanRuns.keys())} {len(GRuns)} {len(DatasetsFailureRuns.keys())}')

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
            DatasetsData = propheticus.shared.Utils.pool(pool_count, self._exportDatasetFile, [(dataset, RemovedIndexes[dataset], datasets_classes_remap) for dataset in self.Datasets])
        else:
            DatasetsData = [self._exportDatasetFile(dataset, RemovedIndexes[dataset], datasets_classes_remap) for dataset in self.Datasets]

        DatasetData = []
        DatasetDescriptions = []
        DatasetTargets = []

        for _Data in DatasetsData:
            if _Data is None:
                continue

            DatasetData.append(_Data[0])
            DatasetDescriptions.append(_Data[1])
            DatasetTargets.append(_Data[2])

        del DatasetsData

        DatasetData = numpy.concatenate(DatasetData)
        DatasetDescriptions = numpy.concatenate(DatasetDescriptions)
        DatasetTargets = numpy.concatenate(DatasetTargets)

        SupportByClass = collections.Counter(DatasetTargets)

        propheticus.shared.Utils.printNewLine()
        propheticus.shared.Utils.printStatusMessage('Classes distribution: \n' + '\n'.join(sorted(['- ' + key + ': ' + str(value) for key, value in SupportByClass.items()])))

        UniqueTargets = sorted(set(DatasetTargets))
        ExperimentsByTarget = [f'{unique_target}: ' + ', '.join(sorted(set(DatasetDescriptions[DatasetTargets == unique_target]))) for unique_target in UniqueTargets]
        propheticus.shared.Utils.printStatusMessage('Experiments by target:\n' + '\n'.join(ExperimentsByTarget))

        if len(Headers) == 0:
            propheticus.shared.Utils.printFatalMessage('At least one feature must remain in the dataset!')  # TODO: this validation should also be made earlier in the execution/flow

        return Headers, set(DatasetTargets), DatasetData, DatasetDescriptions, DatasetTargets

    @propheticus.shared.Decorators.custom_hook()
    def _exportDatasetFile(self, dataset, RemovedIndexes, datasets_classes_remap):
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

        start = time.time()
        propheticus.shared.Utils.printInlineStatusMessage('.')
        a = Config.framework_instance_data_path
        file_path = os.path.join(Config.framework_instance_data_path, dataset + '.data.txt')
        with open(file_path, encoding='utf-8') as f:
            # NOTE: consider using f.read().splitlines() ; this automatically handles differences between new lines across OS ;
            # NOTE: currently, if on Windows requires that the .split() afterwards does not receive ' ' as argument, otherwise \n will be included
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
        if len(ItemIndexes) == 0:
            return None

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
            if len(content[index].split()) != len_item:
                propheticus.shared.Utils.printFatalMessage(f'Not all samples in the dataset have the same length! {dataset}: {index} ({len(content[index].split())})({len_item})')

            if self.config_sliding_window > 1 and (index < (self.config_sliding_window - 1) or content[index - (self.config_sliding_window - 1)].split()[0] != content[index].split()[0]):
                # NOTE: tirei isto pq pode acontecer que as features a tirar ja tenham classe; isto diz respeito a 1º condicao, < sliding_window
                # removing_label = int(content[index].split()[label_index])
                # if removing_label != 0 and propheticus.shared.Utils.getClassDescriptionById(removing_label) not in self.datasets_exclude_classes:
                #     label_ = propheticus.shared.Utils.getClassDescriptionById(removing_label)
                #     propheticus.shared.Utils.printFatalMessage(f'A sample with a relevant target was about to be removed! {label_}')

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

            label_data_type = self.datasets_headers_details[label_index]['type']
            label = self.castDataByHeaderDataType(Item[slide_label_index], label_data_type)

            if self.config_sliding_window > 1:
                direct_label = int(content[index].split()[label_index])
                if label != direct_label:
                    direct_label_ = propheticus.shared.Utils.getClassDescriptionById(direct_label)
                    label_ = propheticus.shared.Utils.getClassDescriptionById(label)
                    propheticus.shared.Utils.printFatalMessage(f'Slided label is different than the original! Correct: {direct_label_} Slided: {label_}')

            target_label = propheticus.shared.Utils.getClassDescriptionById(label)
            if target_label in self.datasets_exclude_classes:
                continue

            start_index = self.config_sliding_window if use_description_field else 0

            if len_filter_features > 0 and 0 in FilterFeaturesIndexes:
                if use_description_field is False:
                    propheticus.shared.Utils.printFatalMessage(f'To filter by description "use_description_field" must be True')

                if not propheticus.shared.Utils.inString(Item[0], FilterFeaturesIndexes[0].split(',')):
                    continue

            DatasetData.append(tuple(Item[start_index:]))

            description = self.getDescriptionByItem(Item, dataset, self.datasets_headers_details) if use_description_field else '-'

            DatasetDescriptions.append(description)

            if binary_classification and label != Config.ClassesMapping['Binary_Base']:
                label = Config.ClassesMapping['Binary_Error']

            if label in datasets_classes_remap:
                target = datasets_classes_remap[label]
            else:
                target = target_label

            DatasetTargets.append(target)

        del content

        if len_filter_features > 0 and 0 in FilterFeaturesIndexes:
            del FilterFeaturesIndexes[0]
            len_filter_features = len(FilterFeaturesIndexes)

        # NOTE: create named numpy array to handle mixed data types
        HeadersDataTypes = [None] * len(self.datasets_headers_details[1:]) * self.config_sliding_window
        for window in range(self.config_sliding_window):
            HeadersDataTypes[window::self.config_sliding_window] = [(f'{self.datasets_headers[h_index + 1]}-{window}', self.getDataCastDataType(HeaderDetails['type'])) for h_index, HeaderDetails in enumerate(self.datasets_headers_details[1:])]

        # NOTE: numpy structured arrays require the rows to be immutable, eg using tuples; https://numpy.org/doc/stable/user/basics.rec.html
        DatasetData = numpy.array(DatasetData, dtype=HeadersDataTypes)

        FeatureNames = numpy.array(HeadersDataTypes)[:, 0]

        if len(DatasetData) > 0:
            if len_filter_features > 0:
                # NOTE: the following line subtracts the self.config_sliding_window value, as this is going to be used in the DatasetData object, which has the first self.config_sliding_window * label removed

                # for feature_id, Values in FilterFeaturesIndexes.items():
                #     results = numpy.in1d(DatasetData[:, (((feature_id + 1) * self.config_sliding_window) - 1) - self.config_sliding_window], numpy.array(Values.split(','), dtype=numpy.float32))
                #     temp = numpy.where(results)

                FilteredData = []
                for feature_id, Values in FilterFeaturesIndexes.items():
                    feature_name, feature_data_type = HeadersDataTypes[(((feature_id + 1) * self.config_sliding_window) - 1) - self.config_sliding_window]
                    FilteredData.append(numpy.in1d(DatasetData[feature_name], numpy.array(Values.split(','), dtype=feature_data_type)))

                Filters = numpy.logical_and.reduce(FilteredData)
                _Indexes = numpy.where(Filters)

                DatasetData = DatasetData[_Indexes]
                DatasetTargets = numpy.array(DatasetTargets)[_Indexes].tolist()
                DatasetDescriptions = numpy.array(DatasetDescriptions)[_Indexes].tolist()

            # NOTE: filtering cannot be done before deleting rows because then the indexes will be mismatched
            DeleteIndexes = [slide_label_index]
            if len_exclude_features > 0:
                ExcludedFeaturesIndexes = list(set(numpy.hstack([FeaturesIndexes[label] for label in ExcludedFeaturesLabels])))
                DeleteIndexes += list(((numpy.array(ExcludedFeaturesIndexes) + 1) * self.config_sliding_window) - 1)  # NOTE: this calculates the position of the more recent value of the feature

            if self.config_sliding_window > 1:
                DeleteIndexes = numpy.array([numpy.array(DeleteIndexes) - window for window in range(0, self.config_sliding_window)]).ravel()

            DeleteIndexes = list(numpy.array(DeleteIndexes) - self.config_sliding_window)  # NOTE: this line is required as we are going to remove from DatasetData, which already does not have the initial labels
            DeleteFeatureNames = FeatureNames[DeleteIndexes]
            DatasetData = propheticus.shared.Utils.deleteFromStructuredArray(DatasetData, DeleteFeatureNames)

            ReducedDType = [row for index, row in enumerate(HeadersDataTypes) if index not in DeleteIndexes]  # NOTE: numpy.delete does not work with structured arrays
            UniqueDTypes = list(set(numpy.array(ReducedDType)[:, 1]))

            if len(UniqueDTypes) > 1:
                propheticus.shared.Utils.printFatalMessage(f'Multiple types of data have been detected in the dataset. Currently Propheticus relies on some techniques that cannot handle such data by defaults')
            else:
                # NOTE: this will only cast the whole array to the same data type, instead of being a structured array
                # (although all features have the same type); scikit packages currently cannot handle structured arrays
                DatasetData = numpy.array(DatasetData.tolist(), dtype=UniqueDTypes[0])
        else:
            propheticus.shared.Utils.printWarningMessage('No samples were chosen from the dataset: ' + dataset)

        return DatasetData, DatasetDescriptions, DatasetTargets, dataset

    @propheticus.shared.Decorators.custom_hook()
    def getDescriptionByItem(self, Item, dataset, DatasetHeadersDetails):
        # NOTE: this function is merely for overloading
        return Item[0]

    def getDataCastDataType(self, data_type):
        return getattr(numpy, 'object' if data_type == 'string' else data_type)

    def castDataByHeaderDataType(self, value, data_type):
        cast_value = getattr(numpy, 'str' if data_type == 'string' else data_type)(value)
        return cast_value

    def _undersampleData(self, dataset):
        # NOTE: this function does not consider categorical features because it only uses the label index
        propheticus.shared.Utils.printInlineStatusMessage('.')
        return self._undersampleSequentialData(dataset) if self.config_sequential_data is True else self._undersampleIITData(dataset)

    def _undersampleSequentialData(self, dataset):
        label_index = self.pre_target['index'] if self.pre_target is not False else len(self.datasets_headers) - 1
        label_data_type = self.datasets_headers_details[label_index]['type']

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

            target = self.castDataByHeaderDataType(Item[label_index], label_data_type)
            run = description = Item[0]

            # NOTE: undersampling for sequential data does so by removing complete runs to avoid broken sequences
            if run not in IndexesByRun:
                IndexesByRun[run] = []
            IndexesByRun[run].append(index)

            if target == Config.ClassesMapping['Baseline'] and run not in FailureRuns:
                if run not in CleanRuns:
                    CleanRuns[run] = []
                CleanRuns[run].append(index)
            elif target != Config.ClassesMapping['RemoveDeltaTL']:
                if run in CleanRuns:
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
