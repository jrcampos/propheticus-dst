"""
Contains all the workflow logic to handle the classification tasks
"""
from builtins import setattr

import os
import numpy
import sklearn.pipeline
import sklearn.cluster
import sklearn.metrics
import sklearn.model_selection
import matplotlib.pyplot as plt
import itertools
import scipy
import pathlib
import operator
import random
import time
import collections
import copy
import imblearn.pipeline
import tempfile
import shutil
import warnings
import sklearn.exceptions
import sklearn.tree
import gc
import importlib
import graphviz
import pydotplus
import zipfile
import inspect

import propheticus
import propheticus.core
import propheticus.classification.metrics
import propheticus.shared

import sys
sys.path.append(propheticus.Config.framework_selected_instance_path)
if os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, 'InstanceConfig.py')):
    from InstanceConfig import InstanceConfig as Config
else:
    import propheticus.Config as Config

class Classification(object):
    """
    Contains all the workflow logic to handle the classification tasks

    Parameters
    ----------
    Context : object
    dataset_name : str
    configurations_id : str
    description : str
    display_visuals : bool
    balance_data : list of str
    reduce_dimensionality : list of str
    normalize : bool
    seed_count : int
    cv_fold : int
    mode : str
    positive_classes : list of str
    display_logs : bool, optional
        Default is True
    AlgorithmsParameters : dict, optional
        Default is None
    GridSearch : GridSearch, optional
        Default is None
    """
    def __init__(self, **kwargs):
        self.Context = kwargs['Context']
        self.dataset_name = kwargs['dataset_name']
        self.description = kwargs['description']
        self.configurations_id = kwargs['configurations_id']
        self.generated_files_base_name = kwargs['configurations_id'] + '.'
        self.display_visuals = kwargs['display_visuals']
        self.balance_data = kwargs['balance_data']
        self.sampling_parameters = kwargs['balance_data_params']
        self.reduce_dimensionality = kwargs['reduce_dimensionality']
        self.dim_red_parameters = kwargs['reduce_dimensionality_params']
        self.normalize = kwargs['normalize']
        self.data_split = kwargs['data_split']
        self.data_split_parameters = kwargs['data_split_parameters']
        self.seed_count = kwargs['seed_count']
        self.mode = kwargs['mode']
        self.positive_classes = kwargs['positive_classes']

        self.cv_fold = kwargs['cv_fold']
        self.ensemble_algorithms = kwargs['ensemble_algorithms']
        self.ensemble_algorithms_parameters = kwargs['ensemble_algorithms_parameters']
        self.ensemble_selection = kwargs['ensemble_selection']
        self.ensemble_selection_parameters = kwargs['ensemble_selection_parameters']

        self.grid_search = kwargs['grid_search']
        self.grid_inner_cv_fold = kwargs['grid_inner_cv_fold']

        self.save_complete_model = kwargs['save_complete_model']
        self.save_experiment_models = kwargs['save_experiment_models']
        self.load_experiment_models = kwargs['load_experiment_models']

        OptionalArguments = {'display_logs': True, 'binarize': True}
        for key, value in OptionalArguments.items():
            setattr(self, key, kwargs[key] if key in kwargs else value)

        self.save_items_path = os.path.join(Config.framework_instance_generated_classification_path, self.dataset_name)
        self.save_log_items_path = os.path.join(Config.framework_instance_generated_logs_path, self.dataset_name)
        self.bypass_validation = False

    '''
    Graphical Tools
    '''
    def plotComparisonDirectedGraph(self, Edges):
        filename = self.generated_files_base_name + self.algorithm_key + "_comparison_graph"
        Graph = graphviz.Digraph(format='png', filename=filename, directory=self.save_items_path)

        Graph.attr('graph', dpi='350')

        for edges in Edges.keys():
            for edge in edges:
                Graph.node(edge, fontsize='6.0')

        for nodes, edge_label in Edges.items():
            Graph.edge(nodes[0], nodes[1], label=edge_label, fontsize='6.0', labelfontsize='6.0')

        Graph.render()
        os.remove(os.path.join(self.save_items_path, filename))  # NOTE: an extra file (Graph source?) is generated but not needed

    def plotConfusionMatrix(self, cm, classes, cmap=plt.cm.Blues):
        """
        Plots the corresponding confusion matrix (CM)

        Parameters
        ----------
        cm : list of list of int
        classes : list of str
        cmap : object, optional
            Color map (the default is matplotlib.pyplot.cm.Blues)

        Returns
        -------

        """
        # numpy.set_printoptions(precision=2)

        normalize = True
        plt.figure()
        plt.rcParams.update({'figure.autolayout': True})
        if normalize:
            oldcm = cm
            with numpy.errstate(invalid='ignore'):
                cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

        if numpy.isnan(numpy.sum(cm)):
            # NOTE: Confusion matrix had NaN values. Replacing with 0
            cm = numpy.nan_to_num(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        # plt.suptitle(("Confusion Matrix"), fontsize=14, fontweight='bold')
        plt.title(self.algorithm_key, fontsize=14)
        plt.colorbar()
        tick_marks = numpy.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=40)  # NOTE: 40
        plt.yticks(tick_marks, classes, rotation=40)  # NOTE: 40

        plt.gcf().subplots_adjust(bottom=0.15)

        thresh = 0.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            float_ = cm[i, j]
            int_ = int(round(cm[i, j], 2) * 100)

            if Config.publication_format is False:
                plt.text(j, i, "{0:.2f}".format(round(cm[i, j], 2)) + '(' + str(oldcm[i, j]) + ')', horizontalalignment="center", verticalalignment='center', color="white" if cm[i, j] > thresh else "black")
            else:
                # plt.text(j, i, str(int(round(cm[i, j], 2) * 100)) + '%', horizontalalignment="center", verticalalignment='center', color="white" if cm[i, j] > thresh else "black")
                precision = 1
                value = str(round(cm[i, j] * 100, precision))
                if int(value[-1 * precision]) == 0:
                    value = value[0:-1 * (precision + 1)]

                value += '%'

                if Config.classification_conf_matrix_show_sample_count is True:
                    value += '\n(' + str(oldcm[i, j]) + ')'

                plt.text(j, i, value, horizontalalignment="center", verticalalignment='center', color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if Config.publication_format is False or Config.force_configurations_log is True:
            if Config.force_configurations_log is True:
                plt.annotate(self.description, (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')
            else:
                plt.figtext(.02, .02, self.description, size='xx-small')
        else:
            plt.tight_layout()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # NOTE: can be used like this, but may suppress relevant warnings

            propheticus.shared.Utils.saveImage(self.save_items_path, self.generated_files_base_name + self.algorithm_key + "_cm.png", bbox_inches="tight", dpi=150)

            if self.display_visuals is True:
                propheticus.shared.Utils.showImage()

        plt.close()

    def plotROC(self, Dataset, TargetPredictions, TargetTest, MapClasses):
        """
        Plots the corresponding Receiving Operator Characteristic (ROC) Curve

        Parameters
        ----------
        Dataset : dict
        TargetPredictions : list of list of float
        TargetTest : list of list of float
        MapClasses : list of str
        """
        if numpy.array([list(set(row)) for row in numpy.transpose(TargetTest)]).shape[1] == 1:
            propheticus.shared.Utils.printWarningMessage(f'Only one class in test, ROC cannot be plotted:')
            return

        n_classes = len(Dataset['classes'])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(TargetTest[:, i], TargetPredictions[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(TargetTest.ravel(), TargetPredictions.ravel())
        roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

        all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points and weight them
        weighted_tpr = numpy.zeros_like(all_fpr)
        for i in range(n_classes):
            weighted_tpr += scipy.interp(all_fpr, fpr[i], tpr[i]) * (TargetTest[:, i].tolist().count(1) / len(TargetTest))

        fpr["weighted"] = all_fpr
        tpr["weighted"] = weighted_tpr
        roc_auc["weighted"] = sklearn.metrics.auc(fpr["weighted"], tpr["weighted"])

        # Then interpolate all ROC curves at this points
        mean_tpr = numpy.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

        lw = 2
        # Plot all ROC curves
        plt.figure()
        colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'lightseagreen','limegreen', 'chocolate', 'greenyellow', 'lightgreen'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='Class {0} (area = {1:0.2f})' ''.format(MapClasses[i], roc_auc[i]))

        plt.plot(fpr["micro"], tpr["micro"], label='micro-average (area = {0:0.2f})' ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"], label='macro-average (area = {0:0.2f})' ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
        plt.plot(fpr["weighted"], tpr["weighted"], label='weighted-average (area = {0:0.2f})' ''.format(roc_auc["weighted"]), color='red', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.suptitle(("Receiver Operating Characteristic"), fontsize=14, fontweight='bold')
        plt.title(self.algorithm_key, fontsize=14)
        plt.legend(loc="lower right")

        if Config.publication_format is False or Config.force_configurations_log is True:
            if Config.force_configurations_log is True:
                plt.annotate(self.description, (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')
            else:
                plt.figtext(.02, .02, self.description, size='xx-small')
        else:
            plt.tight_layout()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # NOTE: can be used like this, but may suppress relevant warnings

            propheticus.shared.Utils.saveImage(self.save_items_path, self.generated_files_base_name + self.algorithm_key + "_roc.png", bbox_inches="tight", dpi=150)

            if self.display_visuals is True:
                propheticus.shared.Utils.showImage()

        plt.close()

    def plotRecallPrecisionCurve(self, Dataset, TargetPredictions, TargetTest, MapClasses):
        """
        Plots the corresponding Precision-Recall Curve

        Parameters
        ----------
        Dataset : dict
        TargetPredictions : list of list of float
        TargetTest : list of list of float
        MapClasses : list of str
        """
        if numpy.array([list(set(row)) for row in numpy.transpose(TargetTest)]).shape[1] == 1:
            propheticus.shared.Utils.printWarningMessage(f'Only one class in test, Precision/Recall curve cannot be plotted:')
            return

        n_classes = len(Dataset['classes'])
        precision = dict()
        recall = dict()
        precision_recall = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(TargetTest[:, i], TargetPredictions[:, i])
            precision_recall[i] = sklearn.metrics.average_precision_score(TargetTest[:, i], TargetPredictions[:, i])

        # Compute micro-average
        precision["micro"], recall["micro"], _ = sklearn.metrics.precision_recall_curve(TargetTest.ravel(), TargetPredictions.ravel())
        precision_recall["micro"] = sklearn.metrics.average_precision_score(TargetTest, TargetPredictions, average='micro')  # TODO: confirm why not necessary to use .ravel

        all_precision = numpy.unique(numpy.concatenate([precision[i] for i in range(n_classes)]))

        # Then interpolate all curves at these points and weight them
        weighted_recall = numpy.zeros_like(all_precision)
        for i in range(n_classes):
            temp = TargetTest[:, i].tolist().count(1)
            weighted_recall += scipy.interp(all_precision, precision[i], recall[i]) * (TargetTest[:, i].tolist().count(1) / len(TargetTest))

        precision["weighted"] = all_precision
        recall["weighted"] = weighted_recall
        precision_recall["weighted"] = sklearn.metrics.average_precision_score(TargetTest, TargetPredictions, average='weighted')

        # Then interpolate all curves at these points
        mean_recall = numpy.zeros_like(all_precision)
        for i in range(n_classes):
            mean_recall += scipy.interp(all_precision, precision[i], recall[i])

        # Finally average it and compute score
        mean_recall /= n_classes

        precision["macro"] = all_precision
        recall["macro"] = mean_recall
        precision_recall["macro"] = sklearn.metrics.average_precision_score(TargetTest, TargetPredictions, average='macro')

        lw = 2
        # Plot all ROC curves
        plt.figure()
        colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'lightseagreen','limegreen', 'chocolate', 'greenyellow', 'lightgreen'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=lw, label='Class {0} (area = {1:0.2f})' ''.format(MapClasses[i], precision_recall[i]))

        plt.plot(recall["micro"], precision["micro"], label='micro-average (area = {0:0.2f})' ''.format(precision_recall["micro"]), color='deeppink', linestyle=':', linewidth=4)
        plt.plot(recall["macro"], precision["macro"], label='macro-average (area = {0:0.2f})' ''.format(precision_recall["macro"]), color='navy', linestyle=':', linewidth=4)
        plt.plot(recall["weighted"], precision["weighted"], label='weighted-average (area = {0:0.2f})' ''.format(precision_recall["weighted"]), color='red', linestyle=':', linewidth=4)

        plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.suptitle(("Precision Recall"), fontsize=14, fontweight='bold')
        plt.title(self.algorithm_key, fontsize=14)
        plt.legend(loc="lower right")

        if Config.publication_format is False or Config.force_configurations_log is True:
            if Config.force_configurations_log is True:
                plt.annotate(self.description, (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')
            else:
                plt.figtext(.02, .02, self.description, size='xx-small')
        else:
            plt.tight_layout()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # NOTE: can be used like this, but may suppress relevant warnings

            propheticus.shared.Utils.saveImage(self.save_items_path, self.generated_files_base_name + self.algorithm_key + "_precision_recall.png", bbox_inches="tight", dpi=150)

            if self.display_visuals is True:
                propheticus.shared.Utils.showImage()

        plt.close()

    '''
    Others
    '''
    def logResults(self, BiTargetTest, BiTargetPredictions, TargetProbabilities, TargetTest, TargetPredictions, PositiveClasses=None, Messages=None):
        """
        Stores the results in a dictionary

        Parameters
        ----------
        BiTargetTest : list of list of int
        BiTargetPredictions : list of list of int
        TargetProbabilities : list of list of float
        TargetTest : list of str
        TargetPredictions : list of str
        PositiveClasses : list of str, optional
            Positive classes, i.e. classes that are considered as a failure (the default is None)

        Returns
        -------
        dict
        """

        if len(set(TargetTest)) == 1:
            message = (f'All the samples in test belong to the same class ({set(TargetTest)}). Some performance metrics will not compute properly')
            if Messages is not None:
                Messages['warning'].append(message)
            else:
                propheticus.shared.Utils.printWarningMessage(message)

            single_target_class = True
        else:
            single_target_class = False

        Results = {}
        ClassificationMetrics = Config.ClassificationPerformanceMetrics
        with numpy.errstate(all='raise'):
            for metric, MetricCallDetails in ClassificationMetrics.items():
                CallArguments = {}
                CallArguments['y_true'] = BiTargetTest if 'use_binarized_results' in MetricCallDetails and MetricCallDetails['use_binarized_results'] else TargetTest

                if 'use_prediction_probabilities' in MetricCallDetails and MetricCallDetails['use_prediction_probabilities']:
                    CallArguments['y_score'] = TargetProbabilities
                else:
                    CallArguments['y_pred'] = BiTargetPredictions if 'use_binarized_results' in MetricCallDetails and MetricCallDetails['use_binarized_results'] else TargetPredictions

                CallParameters = MetricCallDetails['parameters']
                if 'labels' in CallParameters and 'default' in CallParameters['labels'] and CallParameters['labels']['default'] == 'propheticus_positive_labels':
                    CallArguments['labels'] = PositiveClasses

                try:
                    Results[metric] = propheticus.shared.Utils.dynamicAPICall(MetricCallDetails, CallArguments)
                except Exception as e:
                    if single_target_class is False or not propheticus.shared.Utils.inString(str(e), ['one class', 'invalid']):
                        message = f'An exception was thrown when calculating {MetricCallDetails}; {e}'
                        if Messages is not None:
                            Messages['warning'].append(message)
                        else:
                            propheticus.shared.Utils.printWarningMessage(message)

        return Results

    @propheticus.shared.Decorators.custom_hook()
    def _initializeResultsObject(self):
        """
        Initializes the dictionary that will contain the classification results

        Returns
        -------

        """
        Results = {
            'confusion_matrix': 0,
            'TargetPredictionsByAlgorithm': [],
            'TargetPredictions': [],
            'TargetProbabilities': [],
            'TargetTest': [],
            'TargetDescriptions': [],
            'GridSearchResults': [],
            'DurationDimensionalityReduction': [],
            'DurationDataBalancing': [],
            'DurationFold': [],
            'DurationTrain': [],
            'DurationTest': []
        }

        return Results

    def _loadExperimentPredictions(self, experiment_basename):
        propheticus.shared.Utils.printInlineStatusMessage('.')

        zip_ref = zipfile.ZipFile(os.path.join(self.save_log_items_path, f'{experiment_basename}_predictions.txt.zip'), 'r')
        zip_ref.extractall(self.save_log_items_path)
        zip_ref.close()

        file_path = os.path.join(self.save_log_items_path, f'{experiment_basename}_predictions.txt')
        with open(file_path, encoding='utf-8') as f:
            content = f.readlines()
        os.remove(file_path)

        Predictions = []
        Target = []
        for row in content[1:]:
            _row = row.split(' ')
            predicted_class = propheticus.shared.Utils.getClassDescriptionById(int(_row[0]))
            target_class = propheticus.shared.Utils.getClassDescriptionById(int(_row[1]))

            Predictions.append(predicted_class)
            Target.append(target_class)

        return experiment_basename, Predictions, Target

    @propheticus.shared.Decorators.custom_hook()
    def runModel(self, Dataset, algorithm=None, _model=False, Parameters=False, GridSearchParameters=False, task_basename=None):
        """
        Runs defined models for passed configurations. Wrapper method.

        Parameters
        ----------
        algorithm
        Dataset
        _model

        Returns
        -------
        dict
        """
        # NOTE: be aware that in order to use multiprocessing for some reason it is required to .close and .join the pool after

        self.algorithm = algorithm
        self.algorithm_key = str(algorithm) if task_basename is None else task_basename
        self.algorithms_parameters = Parameters
        self.grid_search_parameters = GridSearchParameters

        # TODO: this logic should be associated with a required minimum number of experiments in the ensembles configs: ie stacking can work wiht 1, hard voting should have 3
        # if self.ensemble_algorithms is not None:
            # if (not isinstance(self.algorithm, list) or len(self.algorithm) < 2) and (not isinstance(self.load_experiment_models, list) or len(self.load_experiment_models) < 2):
            #     propheticus.shared.Utils.printFatalMessage('In order to use custom ensembling approaches at least 2 algorithms must be trained/loaded!')

        if self.cv_fold is not None:
            propheticus.shared.Utils.printWarningMessage(f'CV fold attribute is deprecated! For defining CV fold data_split use the according attribute! data_split/data_split_parameters')

        elif isinstance(self.algorithm, list):
            propheticus.shared.Utils.printFatalMessage('Only a single algorithm can be provided!')

        file_path = os.path.join(Config.OS_PATH, self.save_items_path)
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

        Target = Dataset['targets']
        if len(set(Target)) < 2:
            propheticus.shared.Utils.printErrorMessage('Provided dataset only contains samples from a single class. This is currently not supported.')
            return False

        Binarizer = propheticus.shared.ExtendedLabelBinarizer()
        Binarizer.fit(Target)

        PositiveClasses = [_class for _class in self.positive_classes if _class in Target] if self.positive_classes else None
        if self.grid_search is True:
            propheticus.shared.Utils.printStatusMessage('Grid search will conducted for the specified algorithms')

        # Seeds = propheticus.shared.Utils.RandomSeeds[:1] if algorithm == 'genetic_programming' else propheticus.shared.Utils.RandomSeeds[:self.seed_count]
        if self.seed_count > len(propheticus.shared.Utils.RandomSeeds):
            propheticus.shared.Utils.printFatalMessage(f'Trying to request more seeds than available! {self.seed_count} vs {len(propheticus.shared.Utils.RandomSeeds)}')

        Seeds = propheticus.shared.Utils.RandomSeeds[:self.seed_count]
        propheticus.shared.Utils.printStatusMessage('Running ' + self.algorithm_key + ' algorithm')
        if not self.bypass_validation:
            if self.ensemble_algorithms is not None and self.load_experiment_models is not None:
                PredictionsByAlgorithms = {}
                PrevEnsembleModelTarget = None

                loading_start_time = time.time()
                propheticus.shared.Utils.printStatusMessage('Loading experiments', inline=True)

                pool_count = min(Config.max_thread_count, len(self.load_experiment_models))
                if pool_count > 1 and (self.mode == 'cli' or self.mode == 'batch' and Config.thread_level_ != propheticus.shared.Utils.THREAD_LEVEL_BATCH):
                    PredictionsData = propheticus.shared.Utils.pool(pool_count, self._loadExperimentPredictions, self.load_experiment_models)
                else:
                    PredictionsData = [self._loadExperimentPredictions(exp) for exp in self.load_experiment_models]

                for experiment_basename, Predictions, PredictionsTarget in PredictionsData:
                    PredictionsByAlgorithms[experiment_basename] = Predictions
                    EnsembleModelTarget = PredictionsTarget

                    if PrevEnsembleModelTarget is None:
                        PrevEnsembleModelTarget = EnsembleModelTarget
                    else:
                        if PrevEnsembleModelTarget != EnsembleModelTarget:
                            propheticus.shared.Utils.printFatalMessage(f'Not all loaded models share the same target values! {experiment_basename}')

                propheticus.shared.Utils.printNewLine()
                propheticus.shared.Utils.printTimeLogMessage('Loading the experiments', loading_start_time)

                if self.ensemble_selection is not None:
                    selection_start_time = time.time()
                    propheticus.shared.Utils.printStatusMessage('Selecting ensemble models')

                    EnsembleSelectionCallDetails = Config.ClassificationEnsemblesSelection[self.ensemble_selection]
                    EnsembleSelectionCallArguments = copy.deepcopy(self.ensemble_selection_parameters) if self.ensemble_selection_parameters is not None else {}

                    EnsembleSelectionCallArguments['Target'] = PrevEnsembleModelTarget
                    for key, value in PredictionsByAlgorithms.items():
                        EnsembleSelectionCallArguments[key] = value
                    selected_models = propheticus.shared.Utils.dynamicAPICall(EnsembleSelectionCallDetails, EnsembleSelectionCallArguments, propheticus.shared.Utils.RandomSeeds[0])

                    # TODO: this should log somewhere which models were in fact used
                    self.load_experiment_models = selected_models
                    propheticus.shared.Utils.printTimeLogMessage('Selecting ensemble models', selection_start_time)

                ComparePredictionsByAlgorithm = {key: value for key, value in PredictionsByAlgorithms.items() if key in self.load_experiment_models}
                self.compareAlgorithmsPredictions(PrevEnsembleModelTarget, **ComparePredictionsByAlgorithm)

            CrossValidatedResults = self._initializeResultsObject()
            start_time = time.time()

            pool_count = min(Config.max_thread_count, self.seed_count)
            if Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_CV:
                propheticus.shared.Utils.printStatusMessage('Parallelizing threads at CV level')
            elif Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_RUN:
                propheticus.shared.Utils.printStatusMessage(f'Parallelizing threads ({pool_count})')

            if self.mode == 'cli' and pool_count > 1 and Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_RUN:
                RunsCrossValidatedResults = propheticus.shared.Utils.pool(pool_count, self._runModel, [(index, seed, Dataset, Binarizer, PositiveClasses, _model) for index, seed in enumerate(Seeds)])
            else:
                RunsCrossValidatedResults = [self._runModel(index, seed, Dataset, Binarizer, PositiveClasses, _model) for index, seed in enumerate(Seeds)]

            for RunCrossValidatedResults in RunsCrossValidatedResults:
                self.Context.ClassificationAlgorithmsResults[self.algorithm_key].append(RunCrossValidatedResults[1])
                for key, value in RunCrossValidatedResults[0].items():
                    try:
                        CrossValidatedResults[key] += value
                    except Exception as e:
                        propheticus.shared.Utils.printFatalMessage(f'Execption occurred {e}:\n{key}:\n{value};\n{CrossValidatedResults[key]}')

            propheticus.shared.Utils.printTimeLogMessage("Processing the algorithm", start_time)

            PredictionsData = [['Predicted', 'Target', 'Description', 'Probabilities']]
            for index in range(len(CrossValidatedResults['TargetTest'])):
                predicted_class = CrossValidatedResults['TargetPredictions'][index]
                predicted_class = propheticus.shared.Utils.getClassIdByDescription(predicted_class)

                predicted_probabilities = CrossValidatedResults['TargetProbabilities'][index] if Config.classification_save_predictions_probabilities else []

                target_class = CrossValidatedResults['TargetTest'][index]
                target_class = propheticus.shared.Utils.getClassIdByDescription(target_class)

                description = CrossValidatedResults['TargetDescriptions'][index] if Config.classification_save_predictions_labels else ''
                PredictionsData.append([predicted_class, target_class, description] + predicted_probabilities)

            pathlib.Path(self.save_log_items_path).mkdir(parents=True, exist_ok=True)
            predictions_filename = self.generated_files_base_name + self.algorithm_key + '_predictions.txt'
            predictions_file_path = os.path.join(self.save_log_items_path, predictions_filename)
            with open(predictions_file_path, "w", encoding="utf-8") as File:
                File.writelines("\n".join([" ".join(map(str, row)) for row in PredictionsData]) + '\n')

            zipfile.ZipFile(os.path.join(self.save_log_items_path, f'{predictions_filename}.zip'), mode='w', compression=zipfile.ZIP_BZIP2).write(predictions_file_path, os.path.basename(predictions_file_path))
            os.remove(predictions_file_path)

            if self.ensemble_algorithms is not None and self.load_experiment_models is None:
                PredictionsByAlgorithms = {}
                for AlgorithmsPredictions in CrossValidatedResults['TargetPredictionsByAlgorithm']:
                    for _algorithm, Predictions in AlgorithmsPredictions.items():
                        if _algorithm not in PredictionsByAlgorithms:
                            PredictionsByAlgorithms[_algorithm] = []
                        PredictionsByAlgorithms[_algorithm] += Predictions.tolist()

                self.compareAlgorithmsPredictions(CrossValidatedResults['TargetTest'], **PredictionsByAlgorithms)

            self.plotConfusionMatrix(CrossValidatedResults['confusion_matrix'], Dataset['classes'])

            BiCVResultsTargetPred = Binarizer.transform(numpy.array(CrossValidatedResults['TargetPredictions']))
            BiCVResultsTargetTest = Binarizer.transform(numpy.array(CrossValidatedResults['TargetTest']))

            '''
            NOTE:
            This aggregates the predictions from every model and then computes the metrics; this means that it is possible that the
            final metrics are considerably different from the average of the individual runs (e.g., if only 1 TP, every other run will
            have 0 precision, and the final will have 1. This is supported by some authors and articles:
            
            Jiawei Han, ... Jian Pei, in Data Mining (Third Edition), 2012

            8.5.3 Cross-Validation
            In -fold cross-validation, the initial data are randomly partitioned into k mutually exclusive subsets or “folds," , each of approximately equal size. Training and testing is performed k times. In iteration i, partition Di is reserved as the test set, and the remaining partitions are collectively used to train the model. That is, in the first iteration, subsets  collectively serve as the training set to obtain a first model, which is tested on D1; the second iteration is trained on subsets  and tested on D2; and so on. Unlike the holdout and random subsampling methods, here each sample is used the same number of times for training and once for testing. For classification, the accuracy estimate is the overall number of correct classifications from the k iterations, divided by the total number of tuples in the initial data.
            
            Thoroughly analyzes the impact of averaging and concludes it is biased
            http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.186.8880
            '''

            Results = self.logResults(
                numpy.array(BiCVResultsTargetTest),
                numpy.array(BiCVResultsTargetPred),
                numpy.array(CrossValidatedResults['TargetProbabilities']),
                numpy.array(CrossValidatedResults['TargetTest']),
                numpy.array(CrossValidatedResults['TargetPredictions']),
                PositiveClasses
            )

            # TODO: improve this logic...
            if len(CrossValidatedResults['DurationDimensionalityReduction']) > 0:
                Results['duration_dimensionality_reduction'] = '|'.join(map(str, [round(numpy.mean(CrossValidatedResults['DurationDimensionalityReduction']), 3), round(numpy.std(CrossValidatedResults['DurationDimensionalityReduction']), 3)]))
            else:
                Results['duration_dimensionality_reduction'] = ''

            if len(CrossValidatedResults['DurationDataBalancing']) > 0:
                Results['duration_data_balancing'] = '|'.join(map(str, [round(numpy.mean(CrossValidatedResults['DurationDataBalancing']), 3), round(numpy.std(CrossValidatedResults['DurationDataBalancing']), 3)]))
            else:
                Results['duration_data_balancing'] = ''

            Results['duration_run'] = '|'.join(map(str, [round(numpy.mean(CrossValidatedResults['DurationFold']), 3), round(numpy.std(CrossValidatedResults['DurationFold']), 3)]))
            Results['duration_train'] = '|'.join(map(str, [round(numpy.mean(CrossValidatedResults['DurationTrain']), 3), round(numpy.std(CrossValidatedResults['DurationTrain']), 3)]))
            Results['duration_test'] = '|'.join(map(str, [round(numpy.mean(CrossValidatedResults['DurationTest']), 3), round(numpy.std(CrossValidatedResults['DurationTest']), 3)]))
            Results['logs'] = ''

            self.Context.ClassificationAlgorithmsResults[self.algorithm_key].append(Results)
            self.plotROC(Dataset, numpy.array(CrossValidatedResults['TargetProbabilities']), BiCVResultsTargetTest, Binarizer.classes_)
            self.plotRecallPrecisionCurve(Dataset, numpy.array(CrossValidatedResults['TargetProbabilities']), BiCVResultsTargetTest, Binarizer.classes_)

            if CrossValidatedResults['GridSearchResults']:
                GridResultsHeaders = ['params']
                ResultsByParameters = {}
                for FoldGridResults in CrossValidatedResults['GridSearchResults']:
                    for GridResults in FoldGridResults:
                        GridResults = copy.deepcopy(GridResults)
                        params = GridResults.pop('parameters')
                        if params not in ResultsByParameters:
                            ResultsByParameters[params] = {}

                        for metric, value in GridResults.items():
                            if metric not in ResultsByParameters[params]:
                                ResultsByParameters[params][metric] = []
                                if metric not in GridResultsHeaders:
                                    GridResultsHeaders.append(metric)

                            ResultsByParameters[params][metric].append(value)

                order_by_index = GridResultsHeaders.index('mean_test')
                ResultsReport = [['(' + params + ')'] + ["{0:.4f}".format(numpy.mean(Values)) for metric, Values in GridParamsResults.items()] for params, GridParamsResults in ResultsByParameters.items()]
                RankedResults = sorted(ResultsReport, key=operator.itemgetter(order_by_index), reverse=True)
                propheticus.shared.Utils.saveExcel(self.save_log_items_path, self.generated_files_base_name + self.algorithm_key + '.grid_search.xlsx', [GridResultsHeaders] + RankedResults)

        if self.save_complete_model is True:
            if not self.save_experiment_models:
                propheticus.shared.Utils.printFatalMessage(f'Flag "save_complete_model" should be used with "save_experiment_models"')

            Returned = self._runCVFold(
                DataTrain=Dataset['data'],
                TargetTrain=Dataset['targets'],
                Headers=Dataset['headers'],
                PositiveClasses=PositiveClasses,
                seed=propheticus.shared.Utils.RandomSeeds[0],
                save_models_prefix=self.algorithm_key,
                run=-1,
                cv_index=-1,
                Binarizer=Binarizer,
                DataTest=None,
                TargetTest=None,
                _model=False,
                DescriptionsTrain=Dataset['descriptions'],
                DescriptionsTest=False,
            )

        return CrossValidatedResults, Results

    @propheticus.shared.Decorators.custom_hook()
    def _runModel(self, index, seed, Dataset, Binarizer, PositiveClasses=None, _model=False):
        """
        Runs defined models for passed configurations.

        Parameters
        ----------
        Dataset
        Binarizer
        Seeds
        PositiveClasses
        _model

        Returns
        -------
        dict
        """

        gc.collect()

        numpy.random.seed(seed)
        random.seed(seed)

        Headers = Dataset['headers']
        Data = Dataset['data']
        Target = Dataset['targets']
        Descriptions = Dataset['descriptions']

        propheticus.shared.Utils.temporaryLogs(
            seed,
            self.algorithm_key,
            self.balance_data,
            self.reduce_dimensionality,
            self.algorithms_parameters,
            self.grid_search_parameters
        )

        RunCrossValidatedResults = self._initializeResultsObject()

        if self.cv_fold is not None:
            cv = sklearn.model_selection.StratifiedKFold(n_splits=self.cv_fold, shuffle=True, random_state=seed)
            splits = cv.split(Data, Target)
        else:
            DataSplitCallDetails = Config.ClassificationDataSplit[self.data_split]
            Splitter = propheticus.shared.Utils.dynamicAPICall(DataSplitCallDetails, self.data_split_parameters, seed=seed)

            SplitFuncDetails = inspect.getfullargspec(Splitter.split)
            if 'Description' in SplitFuncDetails[0]:
                splits = Splitter.split(Data, Target, Descriptions)
            else:
                splits = Splitter.split(Data, Target)

        splits = list(splits)

        pool_count = min(Config.max_thread_count, len(splits))
        propheticus.shared.Utils.printStatusMessage(' - Seeding #' + str(index + 1) + ' ' + str(seed), inline=(False if Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_RUN else True))
        CVFolds = []
        FoldsData = []
        for cv_index, Details in enumerate(splits):
            train = Details[0]
            test = Details[1]

            FoldDetails = (
                index,
                cv_index,
                _model,
                Binarizer,
                Descriptions[train],
                Data[train],
                Target[train],
                Descriptions[test],
                Data[test],
                Target[test],
                Headers,
                seed,
                PositiveClasses,
                self.algorithm_key
            )

            if len(Details) > 2:
                validation = Details[2]
                FoldDetails += (
                    None,
                    Descriptions[validation],
                    Data[validation],
                    Target[validation]
                )

            if self.mode == 'cli' and pool_count > 1 and Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_CV:
                # NOTE: this requires having multiple copies of the data running simultaneously, heavy on memory
                CVFolds.append(FoldDetails)
            else:
                FoldsData.append(self._runCVFold(*FoldDetails))

        if self.mode == 'cli' and pool_count > 1 and Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_CV:
            FoldsData = propheticus.shared.Utils.pool(pool_count, self._runCVFold, CVFolds)

        for FoldData in FoldsData:
            for key, Messages in FoldData[3].items():
                if len(Messages) > 0:
                    if key == 'warning':
                        Messages.insert(0, 'CV Fold Warnings \n')
                        propheticus.shared.Utils.printWarningMessage(Messages)

            for key, value in FoldData[1].items():
                if key == 'confusion_matrix':
                    confusion_matrix = numpy.array(copy.deepcopy(value))
                    CurrentClasses = list(sorted(set(FoldData[1]['TargetTest'] + FoldData[1]['TargetPredictions'])))
                    for target_class in set(Dataset['classes']):
                        if target_class not in CurrentClasses:
                            CurrentClasses.append(target_class)
                            class_index = sorted(CurrentClasses).index(target_class)
                            if class_index > len(confusion_matrix):
                                confusion_matrix = numpy.append(confusion_matrix, 0, axis=0)
                                confusion_matrix = numpy.append(confusion_matrix, 0, axis=1)
                            else:
                                confusion_matrix = numpy.insert(confusion_matrix, class_index, 0, axis=0)
                                confusion_matrix = numpy.insert(confusion_matrix, class_index, 0, axis=1)

                    value = confusion_matrix
                RunCrossValidatedResults[key] += value

        del FoldsData

        if Config.thread_level_ != propheticus.shared.Utils.THREAD_LEVEL_RUN:
            propheticus.shared.Utils.printNewLine()

        '''
        Log Run Performance
        '''
        BiRunCVResultsTargetPred = Binarizer.transform(numpy.array(RunCrossValidatedResults['TargetPredictions']))
        BiRunCVResultsTargetTest = Binarizer.transform(numpy.array(RunCrossValidatedResults['TargetTest']))

        Results = self.logResults(
            numpy.array(BiRunCVResultsTargetTest),
            numpy.array(BiRunCVResultsTargetPred),
            numpy.array(RunCrossValidatedResults['TargetProbabilities']),
            numpy.array(RunCrossValidatedResults['TargetTest']),
            numpy.array(RunCrossValidatedResults['TargetPredictions']),
            PositiveClasses
        )

        Results['duration_dimensionality_reduction'] = '\n'.join(map(str, RunCrossValidatedResults['DurationDimensionalityReduction']))
        Results['duration_data_balancing'] = '\n'.join(map(str, RunCrossValidatedResults['DurationDataBalancing']))
        Results['duration_run'] = '\n'.join(map(str, RunCrossValidatedResults['DurationFold']))
        Results['duration_train'] = '\n'.join(map(str, RunCrossValidatedResults['DurationTrain']))
        Results['duration_test'] = '\n'.join(map(str, RunCrossValidatedResults['DurationTest']))

        # TODO: this should be improved, currently considers only the importances of the last model! how to handle, multiple folds, average? limited..
        if hasattr(FoldData[0], 'feature_importances_'):
            if self.algorithm not in ['decision_tree', 'random_forests', 'xgboost']:
                propheticus.shared.Utils.printWarningMessage(f'Algorithm {self.algorithm} also has property feature_importances but was not defined in the target algorithms')

            FeaturesImportances = {feature_index: importance for feature_index, importance in enumerate(FoldData[0].feature_importances_)}
            Results['logs'] = "\n".join([FoldData[2][feature_index] + ': ' + str(round(importance, 2)) for feature_index, importance in sorted(FeaturesImportances.items(), key=operator.itemgetter(1), reverse=True)])
        else:
            Results['logs'] = ''

        # self.Context.ClassificationAlgorithmsResults[algorithm].append(Results)

        return RunCrossValidatedResults, Results

    def _preprocessData(self, run, cv_index, DataTrain, TargetTrain, DataTest, DataValidation, Headers, seed, FoldResults, Messages, LoadedModels, save_models_prefix):
        dimensionality_reduction_duration = 0
        if self.reduce_dimensionality and 'variance' in self.reduce_dimensionality or 'variance' in LoadedModels:
            dim_red_start_time = time.time()

            if 'variance' in LoadedModels:
                Estimator = LoadedModels['variance']
                DataTrain = Estimator.transform(DataTrain)

            else:
                DimRedCallArguments = copy.deepcopy(self.dim_red_parameters['variance']) if 'variance' in self.dim_red_parameters else {}

                CallDetails = Config.DimensionalityReductionCallDetails['variance']
                Estimator = propheticus.shared.Utils.dynamicAPICall(CallDetails, DimRedCallArguments, seed=seed)
                DataTrain = Estimator.fit_transform(DataTrain)

            if self.save_experiment_models is True:
                propheticus.shared.Utils.saveModelToDisk(
                    Config.framework_instance_generated_persistent_path,
                    f'{self.generated_files_base_name}{save_models_prefix}-{run}.{cv_index}.variance.joblib',
                    Estimator
                )

            if DataTest is not None:
                if len(DataTest) > 1:
                    oVT = sklearn.feature_selection.VarianceThreshold()
                    oVT.fit_transform(DataTest)
                    for index in Estimator.indexes_:
                        if oVT.variances_[index] != 0:
                            # NOTE: when using sliding windows Headers from sliding window will have -X do indicate the second
                            Messages['warning'].append('Feature removed by 0 variance has variance in the test dataset: ' + Headers[index])

                DataTest = Estimator.transform(DataTest)

            if DataValidation is not None:
                oVT = sklearn.feature_selection.VarianceThreshold()
                oVT.fit_transform(DataValidation)
                for index in Estimator.indexes_:
                    if oVT.variances_[index] != 0:
                        # NOTE: when using sliding windows Headers from sliding window will have -X do indicate the second
                        Messages['warning'].append('Feature removed by 0 variance has variance in the validation dataset: ' + Headers[index])

                DataValidation = Estimator.transform(DataValidation)

            Headers = propheticus.core.DatasetReduction.removeFeaturesFromHeaders(Estimator.indexes_, Headers)

            dimensionality_reduction_duration = propheticus.shared.Utils.getTimeDifference(dim_red_start_time)

        if self.normalize is True or 'normalize' in LoadedModels:
            if 'normalize' in LoadedModels:
                Estimator = LoadedModels['normalize']
                DataTrain = Estimator.transform(DataTrain)

            else:
                NormalizeCallDetails = Config.PreprocessingCallDetails['normalize']
                Estimator = propheticus.shared.Utils.dynamicAPICall(NormalizeCallDetails, seed=seed)
                DataTrain = Estimator.fit_transform(DataTrain)

            if self.save_experiment_models is True:
                propheticus.shared.Utils.saveModelToDisk(
                    Config.framework_instance_generated_persistent_path,
                    f'{self.generated_files_base_name}{save_models_prefix}-{run}.{cv_index}.normalize.joblib',
                    Estimator
                )

            if DataTest is not None:
                DataTest = Estimator.transform(DataTest)

        if self.reduce_dimensionality and (len(self.reduce_dimensionality) > 1 or self.reduce_dimensionality[0] != 'variance') or 'reduce_dimensionality' in LoadedModels:
            # TODO: isto aqui volta a executar feature selection por variance; embora n deva fzer diferenca, devia se controlar
            dim_red_start_time = time.time()
            if 'reduce_dimensionality' in LoadedModels:
                DimRedEstimators = LoadedModels['reduce_dimensionality']['estimators']
                Headers = LoadedModels['reduce_dimensionality']['headers']

                for dim_red_method, DimRedEstimator in DimRedEstimators.items():
                    DataTrain = DimRedEstimator.transform(DataTrain)

            else:
                DataTrain, RemoveFeatures, _, DimRedEstimators = propheticus.core.DatasetReduction.dimensionalityReduction(self.dataset_name, self.configurations_id, self.description, self.reduce_dimensionality, DataTrain, TargetTrain, Headers,
                                                                                                                           self.dim_red_parameters, seed)
                Headers = propheticus.core.DatasetReduction.removeFeaturesFromHeaders(RemoveFeatures, Headers)

            if self.save_experiment_models is True:
                propheticus.shared.Utils.saveModelToDisk(
                    Config.framework_instance_generated_persistent_path,
                    f'{self.generated_files_base_name}{save_models_prefix}-{run}.{cv_index}.reduce_dimensionality.joblib',
                    {'estimators': DimRedEstimators, 'headers': Headers}
                )

            if DataTest is not None:
                for dim_red_method, DimRedEstimator in DimRedEstimators.items():
                    DataTest = DimRedEstimator.transform(DataTest)

            if DataValidation is not None:
                for dim_red_method, DimRedEstimator in DimRedEstimators.items():
                    DataValidation = DimRedEstimator.transform(DataValidation)

            dimensionality_reduction_duration += propheticus.shared.Utils.getTimeDifference(dim_red_start_time)
            FoldResults['DurationDimensionalityReduction'].append(dimensionality_reduction_duration)

        # NOTE: this has to be the last method to be called, because of UnsampledDataTrain
        # TODO this undersamplign logic for not sampling is only needed for stacking ensembles; but this should be defined somehow in the configs....
        UnsampledTargetTrain = UnsampledDataTrain = None
        if self.balance_data or 'balance_data' in LoadedModels:
            if self.ensemble_algorithms is not None:
                UnsampledDataTrain = copy.deepcopy(DataTrain)
                UnsampledTargetTrain = copy.deepcopy(TargetTrain)

            balance_data_start_time = time.time()
            DataTrain, TargetTrain, BalanceDataEstimators = self._preprocessBalanceData(DataTrain, TargetTrain, seed, LoadedModels)
            balance_data_duration = propheticus.shared.Utils.getTimeDifference(balance_data_start_time)

            if self.save_experiment_models is True:
                propheticus.shared.Utils.saveModelToDisk(
                    Config.framework_instance_generated_persistent_path,
                    f'{self.generated_files_base_name}{save_models_prefix}-{run}.{cv_index}.balance_data.joblib',
                    BalanceDataEstimators
                )

            FoldResults['DurationDataBalancing'].append(balance_data_duration)

        return DataTrain, UnsampledDataTrain, TargetTrain, UnsampledTargetTrain, DataTest, DataValidation, Headers, FoldResults, Messages

    def _preprocessBalanceData(self, DataTrain, TargetTrain, seed, LoadedModels):
        if 'balance_data' in LoadedModels:
            BalanceDataEstimators = LoadedModels['balance_data']
            for balance_data_method, BalancedDataEstimator in BalanceDataEstimators.items():
                DataTrain, TargetTrain = BalancedDataEstimator.fit_sample(DataTrain, TargetTrain)

        else:
            DataTrain, TargetTrain, BalanceDataEstimators = propheticus.core.DatasetReduction.balanceDataset(DataTrain, TargetTrain, self.sampling_parameters, seed=seed, method=self.balance_data)

        return DataTrain, TargetTrain, BalanceDataEstimators


    def _gridSearchAlgorithmsParameters(self, algorithm, algorithms_parameters, DataTrain, TargetTrain, PositiveClasses, seed, FoldResults):
        if self.load_experiment_models is not None:
            propheticus.shared.Utils.printFatalMessage('It is not possible to use gridsearch with loaded models!')

        if not self.grid_search_parameters and 'grid' in Config.ClassificationAlgorithmsCallDetails[self.algorithm]:
            self.grid_search_parameters = Config.ClassificationAlgorithmsCallDetails[self.algorithm]['grid']

        '''
        If parameters are sent through attribute self.grid_search_parameters those will be used for grid-search;
        otherwise the grid-search parameters from algorithms configuration will be used.
        '''
        # if self.algorithms_parameters:
        #     propheticus.shared.Utils.printWarningMessage('Values in algorithms_parameters may be overwritten by grid-search hyperparameters')

        if not self.grid_search_parameters:
            propheticus.shared.Utils.printFatalMessage('No grid search parameters were passed or are defined in the configs for algorithm ' + self.algorithm + '. Skipping grid search')
        else:
            # TODO: somehow fetch this from arguments / config
            ftwo_scorer = sklearn.metrics.make_scorer(sklearn.metrics.fbeta_score, beta=2, average='weighted', labels=PositiveClasses)
            balanced_acc_informedness = sklearn.metrics.make_scorer(propheticus.classification.metrics.informedness, average='weighted')

            # TODO: this needs to be handled to allow the fine tuning of multiple different parameters?

            GridSearchConfigurations = {}

            ExperimentSteps = []

            if self.reduce_dimensionality and 'variance' in self.reduce_dimensionality:
                DimRedCallArguments = copy.deepcopy(self.dim_red_parameters['variance']) if 'variance' in self.dim_red_parameters else {}
                ExperimentSteps.append(('preproc_variance', propheticus.shared.Utils.dynamicAPICall(Config.DimensionalityReductionCallDetails['variance'], DimRedCallArguments, seed=seed)))

            if self.normalize is True:
                ExperimentSteps.append(('preproc_normalize', propheticus.shared.Utils.dynamicAPICall(Config.PreprocessingCallDetails['normalize'], seed=seed)))

            if self.reduce_dimensionality and (len(self.reduce_dimensionality) > 1 or self.reduce_dimensionality[0] != 'variance'):
                Estimators = propheticus.core.DatasetReduction.buildDimensionalityReductionTransformers(self.reduce_dimensionality, self.dim_red_parameters, seed)
                ExperimentSteps += [('preproc_dr_' + dim_red_method, Estimator) for dim_red_method, Estimator in Estimators.items()]

            if self.balance_data:
                Estimators = propheticus.core.DatasetReduction.buildDataBalancingTransformers(self.balance_data, self.sampling_parameters, TargetTrain, seed, self.grid_inner_cv_fold)
                ExperimentSteps += [('preproc_db_' + data_balancing_method, Estimator) for data_balancing_method, Estimator in Estimators.items()]

            algorithm_pipeline_prefix = 'algorithm'

            for current_algorithm in algorithm:
                GridSearchConfigurations[current_algorithm] = {}

                AlgorithmCallDetails = Config.ClassificationAlgorithmsCallDetails[current_algorithm]
                AlgorithmCallArguments = copy.deepcopy(algorithms_parameters[current_algorithm]) if current_algorithm in algorithms_parameters else {}

                _ExperimentSteps = copy.deepcopy(ExperimentSteps)
                _ExperimentSteps.append((algorithm_pipeline_prefix, propheticus.shared.Utils.dynamicAPICall(AlgorithmCallDetails, AlgorithmCallArguments, seed)))

                pipeline_cache_dir = tempfile.mkdtemp()
                Pipeliner = sklearn.pipeline.Pipeline if not self.balance_data else imblearn.pipeline.Pipeline
                ExperimentPipeline = Pipeliner(_ExperimentSteps)  # , memory=joblib.Memory(cachedir=pipeline_cache_dir, verbose=0)  # TODO: see if it improves performance or not

                GridParameters = [{algorithm_pipeline_prefix + '__' + key: value for key, value in Parameters.items()} for Parameters in self.grid_search_parameters]
                oGridSearch = sklearn.model_selection.GridSearchCV(
                    estimator=ExperimentPipeline,
                    param_grid=GridParameters,
                    scoring=balanced_acc_informedness,
                    cv=sklearn.model_selection.StratifiedKFold(self.grid_inner_cv_fold, shuffle=True, random_state=seed),
                    refit=False,
                    return_train_score=True,  # NOTE: this may increase the computation time,
                    iid=False,
                    n_jobs=-1 if Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_ALGORITHM else None
                )

                oGridSearch.fit(DataTrain, TargetTrain)

                shutil.rmtree(pipeline_cache_dir)

                for hyperparameter, value in oGridSearch.best_params_.items():
                    GridSearchConfigurations[current_algorithm][hyperparameter.replace(algorithm_pipeline_prefix + '__', '')] = value

                mean_train = oGridSearch.cv_results_['mean_train_score']
                std_train = oGridSearch.cv_results_['std_train_score']
                mean_test = oGridSearch.cv_results_['mean_test_score']
                std_test = oGridSearch.cv_results_['std_test_score']

                ResultsLog = []
                for mean_train, std_train, mean_test, std_test, params in zip(mean_train, std_train, mean_test, std_test, oGridSearch.cv_results_['params']):
                    ResultsLog.append({
                        'parameters': f'{current_algorithm}:' + ', '.join([param.replace(algorithm_pipeline_prefix + '__', '') + ': ' + str(value) for param, value in params.items()]),
                        'mean_train': mean_train,
                        'std_train': std_train,
                        'mean_test': mean_test,
                        'std_test': std_test
                    })

                FoldResults['GridSearchResults'].append(ResultsLog)

            return GridSearchConfigurations, FoldResults

    # TODO: this method could probably be improved by passing the train/test indexes and the complete data separately;
    # TODO: then instead of storing in memory the textual data (ie labels, descriptions) only the predictions would need to be stored
    # TODO: this would however require changing some code that later uses such in-memory data

    @propheticus.shared.Decorators.custom_hook()
    def _runCVFold(self, run, cv_index, _model, Binarizer, DescriptionsTrain, DataTrain, TargetTrain, DescriptionsTest, DataTest, TargetTest, Headers, seed, PositiveClasses=None, save_models_prefix=None, thread_level=None, DescriptionsValidation=None, DataValidation=None, TargetValidation=None):
        """
        Runs cross-validation (CV) fold and returns required data for logging

        Parameters
        ----------
        Binarizer : object
        DataTrain : list of list of float
        TargetTrain : list of str
        RunDataTest : list of list of float
        RunTargetTest : list of str
        Headers : list of str
        seed : int
        PositiveClasses : list of str, optional
            Positive classes, i.e. classes that are considered as a failure (the default is None)

        Returns
        -------
        model : object
        FoldResults : dict
        Headers : list of str
        Messages : list of str
        """
        OriginalHeaders = copy.deepcopy(Headers)

        if Config.thread_level_ != propheticus.shared.Utils.THREAD_LEVEL_RUN:
            propheticus.shared.Utils.printInlineStatusMessage('.')

        if self.ensemble_algorithms is None:
            algorithm = [self.algorithm]
            algorithms_parameters = {self.algorithm: self.algorithms_parameters} if self.algorithms_parameters is not False else {}
        else:
            algorithm = self.algorithm
            algorithms_parameters = self.algorithms_parameters

            EnsembleCallDetails = Config.ClassificationEnsembles[self.ensemble_algorithms]
            train_ensemble = True if 'train' in EnsembleCallDetails and EnsembleCallDetails['train'] is True else False

        start_time = time.time()
        FoldResults = self._initializeResultsObject()
        Messages = {'warning': [], 'status': []}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # NOTE: can be used like this, but may suppress relevant warnings
            warnings.filterwarnings(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)

            warnings.filterwarnings("ignore", message="F-score is ill-defined and being set to 0.0 due to no predicted samples")
            warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0 due to no predicted samples")
            warnings.filterwarnings("ignore", message="Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples")
            warnings.filterwarnings("ignore", message="The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.")

            if self.grid_search is not False:
                GridSearchConfigurations, FoldResults = self._gridSearchAlgorithmsParameters(algorithm, algorithms_parameters, DataTrain, TargetTrain, PositiveClasses, seed, FoldResults)

            LoadedModels = {}
            if self.load_experiment_models is not None:
                if len(self.load_experiment_models) > 1 and self.ensemble_algorithms is None:
                    propheticus.shared.Utils.printFatalMessage('More than one experiment was given to load but no ensemble algorithm was provided')

                Estimators = ['variance', 'normalize', 'reduce_dimensionality', 'balance_data', 'classifier']
                for experiment_basename in self.load_experiment_models:
                    # TODO: this should read the configuration file and execute the estimators in the intended order
                    propheticus.shared.Utils.printWarningMessage(f'This process does not take into account the order of existing estimators; if multiple feature selection methods are used, they may be replicated in a wrong order!!')
                    LoadedModels[experiment_basename] = {'experiment_basename': experiment_basename}
                    for item in os.listdir(Config.framework_instance_generated_persistent_path):
                        if f'{experiment_basename}-{run}.{cv_index}.' in item and 'joblib' in item:
                            found_estimator = propheticus.shared.Utils.inString(item, Estimators)
                            if found_estimator is False:
                                propheticus.shared.Utils.printFatalMessage(f'A file matching the existing target experiment ({experiment_basename}) was found but it does not match any intended estimator/model: {item}')

                            LoadedModels[experiment_basename][found_estimator] = propheticus.shared.Utils.loadModelFromDist(Config.framework_instance_generated_persistent_path, item)

                    if 'classifier' not in LoadedModels[experiment_basename]:
                        propheticus.shared.Utils.printFatalMessage(f'No classifier could be found for the experiment >> run >> fold: {experiment_basename}-{run}.{cv_index}')

                DataTrainCopy = copy.deepcopy(DataTrain)
                TargetTrainCopy = copy.deepcopy(TargetTrain)
                DataTestCopy = copy.deepcopy(DataTest)
                DataValidationCopy = copy.deepcopy(DataValidation)
            else:
                DataTrain, UnsampledDataTrain, TargetTrain, UnsampledTargetTrain, DataTest, DataValidation, Headers, FoldResults, Messages = self._preprocessData(run, cv_index, DataTrain, TargetTrain, DataTest, DataValidation, Headers, seed, FoldResults, Messages, LoadedModels, save_models_prefix)

            EnsembleDataTrainProb = {}
            EnsembleDataTrain = {}
            EnsembleDataTargetProb = {}
            EnsembleDataTarget = {}
            AlgorithmsTrainDurations = {}
            AlgorithmsTestDurations = {}
            EnsembleTargetTrain = None

            AlgorithmIterator = algorithm if self.load_experiment_models is None else LoadedModels.values()
            _len_predictions = None
            for Details in AlgorithmIterator:
                if self.load_experiment_models is not None:
                    DataTrain, UnsampledDataTrain, TargetTrain, UnsampledTargetTrain, DataTest, DataValidation, Headers, FoldResults, Messages = self._preprocessData(run, cv_index, DataTrainCopy, TargetTrainCopy, DataTestCopy, DataValidationCopy, OriginalHeaders, seed, FoldResults, Messages, Details, save_models_prefix)
                    model = Details['classifier']
                    current_algorithm = Details['experiment_basename']

                else:
                    train_start_time = time.time()
                    current_algorithm = Details

                    AlgorithmCallDetails = Config.ClassificationAlgorithmsCallDetails[current_algorithm]
                    AlgorithmCallArguments = copy.deepcopy(algorithms_parameters[current_algorithm]) if current_algorithm in algorithms_parameters else {}

                    if self.grid_search is not False:
                        AlgorithmGridSearchBest = GridSearchConfigurations[current_algorithm]
                        for key, value in AlgorithmGridSearchBest.items():
                            AlgorithmCallArguments[key] = value

                    if 'dataset_metadata_argument' in AlgorithmCallDetails and AlgorithmCallDetails['dataset_metadata_argument'] is True:
                        AlgorithmCallArguments['dataset_metadata'] = {
                            'experiment_id': f'{self.generated_files_base_name}{self.algorithm_key}-{run}-{cv_index}',
                            'features_count': len(Headers),
                            'classes_count': len(set(TargetTrain)),
                        }

                    if 'n_jobs' in AlgorithmCallDetails['parameters']:
                        if Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_ALGORITHM:
                            AlgorithmCallArguments['n_jobs'] = -1
                        elif Config.force_n_jobs_1:
                            AlgorithmCallArguments['n_jobs'] = 1

                    # AlgorithmCallArguments['n_jobs'] = -1
                    model = propheticus.shared.Utils.dynamicAPICall(AlgorithmCallDetails, AlgorithmCallArguments, seed)
                    if 'eval_set' in AlgorithmCallDetails['parameters']:
                        if DataValidation is not None:
                            model.fit(DataTrain, TargetTrain, eval_set=[(DataValidation, TargetValidation)], eval_metric=['balanced_accuracy'], early_stopping_rounds=500)
                        else:
                            model.fit(DataTrain, TargetTrain)
                    else:
                        if DataValidation is not None:
                            propheticus.shared.Utils.printWarningMessage(f'Validation set was provided but algorithm has no eval_set argument?!')
                        model.fit(DataTrain, TargetTrain)

                    if current_algorithm == 'decision_tree' and cv_index == 0:
                        Replace = [(',', ' '), ('"', ''), ('{', ' '), ('}', ' '), ('family', '\\nfamily'), ('netdata_', '')]
                        SafeHeaders = [propheticus.shared.Utils.multipleReplace(header, Replace) for header in Headers]
                        ClassNames = sorted(set(TargetTrain))
                        # plt.figure(figsize=(24, 24))

                        dot_data = sklearn.tree.export_graphviz(model, out_file=None, feature_names=SafeHeaders, class_names=ClassNames, filled=True)  #
                        graph = pydotplus.graph_from_dot_data(dot_data)
                        # graph.set_size('"15,5!"')
                        graph.write_png(os.path.join(self.save_items_path, self.generated_files_base_name + self.algorithm_key + "_tree.png"))

                        # sklearn.tree.plot_tree(model, feature_names=Headers, fontsize=10, filled=True)
                        # propheticus.shared.Utils.saveImage(self.save_items_path, self.generated_files_base_name + self.algorithm_key + "_tree.png")
                        # plt.show()
                        # plt.close()

                    AlgorithmsTrainDurations[current_algorithm] = propheticus.shared.Utils.getTimeDifference(train_start_time)

                if not numpy.array_equal(model.classes_, Binarizer.classes_):
                    propheticus.shared.Utils.printFatalMessage(f'The classes mapping between the classifier and the binarizer is not the same! {model.classes_} vs {Binarizer.classes_}')

                if self.ensemble_algorithms is not None and train_ensemble is True:
                    if UnsampledDataTrain is not None:
                        TempDataTrain = UnsampledDataTrain
                        TempTargetTrain = UnsampledTargetTrain
                    else:
                        TempDataTrain = DataTrain
                        TempTargetTrain = TargetTrain

                    EnsembleDataTrain[current_algorithm] = []
                    EnsembleDataTrainProb[current_algorithm] = []
                    AlgorithmTargetTrain = []

                    ensemble_cv = sklearn.model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
                    splits = list(ensemble_cv.split(TempDataTrain, TempTargetTrain))
                    ensemble_weak_model = copy.deepcopy(model)
                    for inner_cv_index, (train, test) in enumerate(splits):
                        if UnsampledDataTrain is not None:
                            SampledTempDataTrain, SampledTempTargetTrain, _ = self._preprocessBalanceData(TempDataTrain[train], TempTargetTrain[train], seed, LoadedModels)
                        else:
                            SampledTempDataTrain = TempDataTrain[train]
                            SampledTempTargetTrain = TempTargetTrain[train]

                        TempDataTest = TempDataTrain[test]
                        TempTargetTest = TempTargetTrain[test]

                        inner_cv_classifier_base_name = f'{Details["experiment_basename"]}-{run}.{cv_index}.{inner_cv_index}.stacking_classifier.joblib'
                        inner_cv_classifier_path = os.path.join(Config.framework_temp_persistent_path, inner_cv_classifier_base_name)
                        if self.load_experiment_models is not None and os.path.isfile(inner_cv_classifier_path):
                            ensemble_weak_model = propheticus.shared.Utils.loadModelFromDist(Config.framework_temp_persistent_path, inner_cv_classifier_base_name)
                        else:
                            ensemble_weak_model.fit(SampledTempDataTrain, SampledTempTargetTrain)

                        # TODO: this logic may need to be refactored
                        if self.load_experiment_models is not None and not os.path.isfile(inner_cv_classifier_path):
                            propheticus.shared.Utils.saveModelToDisk(
                                Config.framework_temp_persistent_path,
                                inner_cv_classifier_base_name,
                                ensemble_weak_model
                            )

                        EnsembleDataTrain[current_algorithm] += ensemble_weak_model.predict(TempDataTest).tolist()
                        EnsembleDataTrainProb[current_algorithm] += ensemble_weak_model.predict_proba(TempDataTest).tolist()
                        AlgorithmTargetTrain += TempTargetTest.tolist()

                    if _len_predictions is None:
                        _len_predictions = len(EnsembleDataTrain[current_algorithm])
                    else:
                        if len(EnsembleDataTrain[current_algorithm]) != _len_predictions:
                            propheticus.shared.Utils.printFatalMessage(f'At least an algorithm trained for the ensemble has a different count of predictions! {_len_predictions} != {len(EnsembleDataTrain[current_algorithm])}')

                    if EnsembleTargetTrain is None:
                        EnsembleTargetTrain = AlgorithmTargetTrain
                    else:
                        if EnsembleTargetTrain != AlgorithmTargetTrain:
                            propheticus.shared.Utils.printFatalMessage(f'The resulting split set used to train the different algorithms for ensembling was not the same!')

                if DataTest is not None:
                    test_start_time = time.time()
                    EnsembleDataTarget[current_algorithm] = model.predict(DataTest)
                    AlgorithmsTestDurations[current_algorithm] = propheticus.shared.Utils.getTimeDifference(test_start_time)
                    EnsembleDataTargetProb[current_algorithm] = model.predict_proba(DataTest)

            if self.ensemble_algorithms is None:
                train_duration = AlgorithmsTrainDurations[list(AlgorithmsTrainDurations.keys())[0]] if len(AlgorithmsTrainDurations) > 0 else -1

                if DataTest is not None:
                    algorithm_key = list(EnsembleDataTarget.keys())[0]
                    TargetPredictions = EnsembleDataTarget[algorithm_key]
                    TargetProbabilities = EnsembleDataTargetProb[algorithm_key]
                    test_duration = AlgorithmsTestDurations[algorithm_key]

            else:
                EnsembleCallDetails = Config.ClassificationEnsembles[self.ensemble_algorithms]
                EnsembleCallArguments = copy.deepcopy(self.ensemble_algorithms_parameters)
                ensemble_model = propheticus.shared.Utils.dynamicAPICall(EnsembleCallDetails, EnsembleCallArguments, seed)
                if train_ensemble is True:
                    ensemble_model.fit(EnsembleDataTrain, EnsembleDataTrainProb, EnsembleTargetTrain)

                    if ensemble_model.classes_ != model.classes_.tolist():
                        propheticus.shared.Utils.printFatalMessage(f'Ensemble learned classes do not match individual models! {ensemble_model.classes_} != {model.classes_}')
                else:
                    ensemble_model.classes_ = model.classes_

                train_duration = sum(list(AlgorithmsTrainDurations.values())) if len(AlgorithmsTrainDurations) > 0 else -1

                if DataTest is not None:
                    TargetPredictions = ensemble_model.predict(EnsembleDataTarget, EnsembleDataTargetProb, TargetTest)
                    TargetProbabilities = ensemble_model.predict_proba(EnsembleDataTarget, EnsembleDataTargetProb)

                    test_duration = sum(list(AlgorithmsTestDurations.values()))

                model = ensemble_model

            FoldResults['DurationTrain'].append(train_duration)

            if DataTest is not None:
                if self.ensemble_algorithms is None:
                    FoldResults['TargetPredictionsByAlgorithm'].append(EnsembleDataTarget)

                FoldResults['TargetPredictions'] += TargetPredictions.tolist()
                FoldResults['TargetProbabilities'] += TargetProbabilities.tolist()
                FoldResults['TargetTest'] += TargetTest.tolist()
                FoldResults['TargetDescriptions'] += [f'{DescriptionsTest[index1]}' for index1, Item in enumerate(DataTest.tolist())]  # ''.join(list(map(str, map(int, Item)))) + f' -
                FoldResults['DurationTest'].append(test_duration)

                BinarizedTargetPredictions = Binarizer.transform(TargetPredictions)
                BinarizedRunTargetTest = Binarizer.transform(TargetTest)

                Results = self.logResults(BinarizedRunTargetTest, BinarizedTargetPredictions, TargetProbabilities, TargetTest, TargetPredictions, PositiveClasses, Messages=Messages)
                FoldResults['confusion_matrix'] += Results['confusion_matrix']

                fold_cv_duration = propheticus.shared.Utils.getTimeDifference(start_time)
                FoldResults['DurationFold'].append(fold_cv_duration)

            if self.load_experiment_models is None and 'callable_package' in AlgorithmCallDetails:
                callback_package = AlgorithmCallDetails['callback_package']
                callback_callable = AlgorithmCallDetails['callback_callable']
                importlib.import_module(callback_package)
                getattr(sys.modules[callback_package], callback_callable)(DataTrain, TargetTrain, DataTest, TargetTest)

        if self.save_experiment_models is True:
            propheticus.shared.Utils.saveModelToDisk(
                Config.framework_instance_generated_persistent_path,
                f'{self.generated_files_base_name}{save_models_prefix}-{run}.{cv_index}.classifier.joblib',
                model
            )

            with open(os.path.join(Config.framework_instance_generated_persistent_path, f'{self.generated_files_base_name}{save_models_prefix}-{run}.{cv_index}.headers.txt'), "w", encoding="utf-8") as File:
                File.writelines("\n".join(OriginalHeaders) + '\n')

        if hasattr(model, 'feature_importances_'):
            _Indexes = numpy.argwhere(model.feature_importances_ > Config.max_alert_dt_feature_importance).ravel()
            if len(_Indexes) > 0:
                Messages['warning'].append('Some features have importances too high: ' + '; '.join(sorted([Headers[feature_index] + ': ' + str(round(model.feature_importances_[feature_index], 2)) for feature_index in _Indexes])))

        return model, FoldResults, Headers, Messages, DataTrain, TargetTrain, DataTest, TargetTest, Results

    def compareAlgorithmsPredictions(self, Target, **kwargs):
        comparing_start_time = time.time()
        propheticus.shared.Utils.printStatusMessage('Comparing experiments')

        PredictionsByAlgorithms = kwargs

        Comparisons = {}
        for _algorithm in sorted(list(PredictionsByAlgorithms.keys())):
            AlgorithmPredictions = PredictionsByAlgorithms[_algorithm]
            for _algorithm2 in sorted(list(PredictionsByAlgorithms.keys())):
                AlgorithmPredictions2 = PredictionsByAlgorithms[_algorithm2]

                compare_algorithms = ' > '.join(sorted([_algorithm, _algorithm2]))
                if _algorithm == _algorithm2 or compare_algorithms in Comparisons:
                    continue

                Comparisons[f'{compare_algorithms}'] = {f'{_algorithm}': [], f'{_algorithm2}': []}

                AlgorithmPredictions = numpy.array(AlgorithmPredictions)
                AlgorithmPredictions2 = numpy.array(AlgorithmPredictions2)

                EqualPredictions = (AlgorithmPredictions == AlgorithmPredictions2)
                if EqualPredictions.all():
                    propheticus.shared.Utils.printWarningMessage(f'Algorithms passed to the ensemble have exactly the same results! {_algorithm} - {_algorithm2}')

                DifferentPredictions = numpy.where(EqualPredictions == numpy.bool_(False))[0]
                for index in DifferentPredictions:
                    if AlgorithmPredictions[index] == Target[index]:
                        Comparisons[f'{compare_algorithms}'][f'{_algorithm}'].append(AlgorithmPredictions[index])
                    elif AlgorithmPredictions2[index] == Target[index]:
                        Comparisons[f'{compare_algorithms}'][f'{_algorithm2}'].append(AlgorithmPredictions2[index])

        Edges = {}

        AvailableExperiments = propheticus.shared.Utils.getAvailableExperiments()
        EdgeLabelParameters = ['proc_classification', 'proc_balance_data', 'proc_reduce_dimensionality']
        EnsembleComparisons = [['Algorithms', 'Correct A', 'Correct B']]
        for compare_algorithms, AlgorithmsComparisons in Comparisons.items():
            AlgorithmCompare = [compare_algorithms]
            for index, Indexes in enumerate(AlgorithmsComparisons.values()):
                exp_src_id = list(AlgorithmsComparisons.keys())[index].split('.')[0]
                exp_dst_id = list(AlgorithmsComparisons.keys())[1 - index].split('.')[0]

                ExpSrcDetails = AvailableExperiments[exp_src_id]['configuration']
                ExpDstDetails = AvailableExperiments[exp_dst_id]['configuration']

                exp_src_lbl = '\n'.join([exp_src_id] + [str(ExpSrcDetails[key]) for key in EdgeLabelParameters])
                exp_dst_lbl = '\n'.join([exp_dst_id] + [str(ExpDstDetails[key]) for key in EdgeLabelParameters])

                Edges[(exp_src_lbl, exp_dst_lbl)] = comparison_differences = '\n'.join(map(str, sorted(collections.Counter(Indexes).items()) + [('Total', len(Indexes))]))
                AlgorithmCompare.append(f'{exp_src_id}\n {comparison_differences} : {len(Indexes)}')

            EnsembleComparisons.append(AlgorithmCompare)

        propheticus.shared.Utils.saveExcel(self.save_log_items_path, self.generated_files_base_name + 'ensemble_comparison.xlsx', EnsembleComparisons)

        self.plotComparisonDirectedGraph(Edges)
        propheticus.shared.Utils.printTimeLogMessage('Comparing experiments', comparing_start_time)
