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
import gc
import importlib

# from CustomEnsembleClassifier import CustomEnsembleClassifier
# from GeneticProgrammingClassifier import GeneticProgrammingClassifier
from statsmodels.stats.power import ftest_anova_power

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
    # def __init__(self, Context, dataset_name, configurations_id, description, display_visuals, balance_data, reduce_dimensionality, normalize, seed_count, mode, positive_classes, display_logs=True):
    def __init__(self, **kwargs):
        self.Context = kwargs['Context']
        self.dataset_name = kwargs['dataset_name']
        self.description = kwargs['description']
        self.configurations_id = kwargs['configurations_id']
        self.generated_files_base_name = kwargs['configurations_id'] + '.'
        self.display_visuals = kwargs['display_visuals']
        self.balance_data = kwargs['balance_data']
        self.reduce_dimensionality = kwargs['reduce_dimensionality']
        self.normalize = kwargs['normalize']
        self.seed_count = kwargs['seed_count']
        self.mode = kwargs['mode']
        self.positive_classes = kwargs['positive_classes']

        self.cv_fold = kwargs['cv_fold']
        self.grid_search = kwargs['grid_search']
        self.grid_inner_cv_fold = kwargs['grid_inner_cv_fold']

        OptionalArguments = {'display_logs': True, 'binarize': True}
        for key, value in OptionalArguments.items():
            setattr(self, key, kwargs[key] if key in kwargs else value)

        self.save_items_path = os.path.join(Config.framework_instance_generated_classification_path, self.dataset_name)
        self.save_log_items_path = os.path.join(Config.framework_instance_generated_logs_path, self.dataset_name)

    '''
    Graphical Tools
    '''
    def plotConfusionMatrix(self, algorithm, cm, classes, cmap=plt.cm.Blues):
        """
        Plots the corresponding confusion matrix (CM)

        Parameters
        ----------
        algorithm : str
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
            cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        # plt.suptitle(("Confusion Matrix"), fontsize=14, fontweight='bold')
        plt.title(algorithm, fontsize=14)
        plt.colorbar()
        tick_marks = numpy.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)  # NOTE: 40
        plt.yticks(tick_marks, classes, rotation=90)  # NOTE: 40

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

            propheticus.shared.Utils.saveImage(self.save_items_path, self.generated_files_base_name + algorithm + "_cm.png", bbox_inches="tight", dpi=150)

        if self.display_visuals is True:
            propheticus.shared.Utils.showImage()

        plt.close()

    def plotROC(self, Dataset, algorithm, TargetPredictions, TargetTest, MapClasses):
        """
        Plots the corresponding Receiving Operator Characteristic (ROC) Curve

        Parameters
        ----------
        Dataset : dict
        algorithm : str
        TargetPredictions : list of list of float
        TargetTest : list of list of float
        MapClasses : list of str
        """
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
        plt.title(algorithm, fontsize=14)
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

            propheticus.shared.Utils.saveImage(self.save_items_path, self.generated_files_base_name + algorithm + "_roc.png", bbox_inches="tight", dpi=150)

        if self.display_visuals is True:
            propheticus.shared.Utils.showImage()

        plt.close()

    def plotRecallPrecisionCurve(self, Dataset, algorithm, TargetPredictions, TargetTest, MapClasses):
        """
        Plots the corresponding Precision-Recall Curve

        Parameters
        ----------
        Dataset : dict
        algorithm : str
        TargetPredictions : list of list of float
        TargetTest : list of list of float
        MapClasses : list of str
        """
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
        plt.title(algorithm, fontsize=14)
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

            propheticus.shared.Utils.saveImage(self.save_items_path, self.generated_files_base_name + algorithm + "_precision_recall.png", bbox_inches="tight", dpi=150)

        if self.display_visuals is True:
            propheticus.shared.Utils.showImage()

        plt.close()

    '''
    Others
    '''
    def logResults(self, BiTargetTest, BiTargetPredictions, TargetProbabilities, TargetTest, TargetPredictions, PositiveClasses=None):
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
        # NOTE: The following code generates the metrics based on the non-binarized predictions; usefull for debugging
        # TempResults = {}
        #
        # # Metrics
        # TempResults['acc_score'] = sklearn.metrics.accuracy_score(TargetTest, TargetPredictions)
        # TempResults['precision_score'] = sklearn.metrics.precision_score(TargetTest, TargetPredictions, average='weighted')
        # TempResults['f1_score'] = sklearn.metrics.f1_score(TargetTest, TargetPredictions, average='weighted')
        # TempResults['recall_score'] = sklearn.metrics.recall_score(TargetTest, TargetPredictions, average='weighted')

        Results = {}
        ClassificationMetrics = Config.ClassificationPerformanceMetrics
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

            Results[metric] = propheticus.shared.Utils.dynamicAPICall(MetricCallDetails, CallArguments)

        return Results

    def _initializeResultsObject(self):
        """
        Initializes the dictionary that will contain the classification results

        Returns
        -------

        """
        Results = {
            'confusion_matrix': 0,
            'TargetPredictions': [],
            'TargetProbabilities': [],
            'TargetTest': [],
            'GridSearchResults': [],
            'DurationDimensionalityReduction': [],
            'DurationDataBalancing': [],
            'DurationFold': [],
            'DurationTrain': [],
            'DurationTest': []
        }

        return Results

    def runModel(self, algorithm, Dataset, _model=False, Parameters=False, GridSearchParameters=False, DimRedParameters=False, SamplingParameters=False):
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

        self.algorithms_parameters = Parameters
        self.grid_search_parameters = GridSearchParameters
        self.dim_red_parameters = DimRedParameters
        self.sampling_parameters = SamplingParameters

        file_path = os.path.join(Config.OS_PATH, self.save_items_path)
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

        thread_count = self.seed_count if Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_RUN else self.cv_fold
        pool_count = min(Config.max_thread_count, thread_count)
        if Config.thread_level_ in [propheticus.shared.Utils.THREAD_LEVEL_CV, propheticus.shared.Utils.THREAD_LEVEL_RUN]:
            propheticus.shared.Utils.printStatusMessage('Parallelizing threads #: ' + str(pool_count))

        Target = Dataset['targets']
        if len(set(Target)) < 2:
            propheticus.shared.Utils.printErrorMessage('Provided dataset only contains samples from a single class. This is currently not supported.')
            return False

        Binarizer = propheticus.shared.ExtendedLabelBinarizer()
        Binarizer.fit(Target)

        PositiveClasses = [_class for _class in self.positive_classes if _class in Target] if self.positive_classes else None

        if self.grid_search is True:
            propheticus.shared.Utils.printStatusMessage('Grid search will conducted for the specified algorithms')

        Seeds = propheticus.shared.Utils.RandomSeeds[:1] if algorithm == 'genetic_programming' else propheticus.shared.Utils.RandomSeeds[:self.seed_count]
        propheticus.shared.Utils.printStatusMessage('Running ' + algorithm + ' algorithm')
        # Results = self._runModel(algorithm, Dataset, Binarizer, Seeds, pool_count, PositiveClasses, _model)

        CrossValidatedResults = self._initializeResultsObject()
        start_time = time.time()

        if self.mode == 'cli' and pool_count > 1 and Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_RUN:
            RunsCrossValidatedResults = propheticus.shared.Utils.pool(pool_count, self._runModel, [(index, seed, algorithm, Dataset, Binarizer, pool_count, PositiveClasses, _model) for index, seed in enumerate(Seeds)])
        else:
            RunsCrossValidatedResults = [self._runModel(index, seed, algorithm, Dataset, Binarizer, pool_count, PositiveClasses, _model) for index, seed in enumerate(Seeds)]

        for RunCrossValidatedResults in RunsCrossValidatedResults:
            self.Context.ClassificationAlgorithmsResults[algorithm].append(RunCrossValidatedResults[1])
            for key, value in RunCrossValidatedResults[0].items():
                CrossValidatedResults[key] += value

        # TODO: this needs to be updated to handle new headers after dimensionality reduction; how to handle it? use last?
        # if algorithm == 'decision_tree_classifier' and propheticus.shared.Utils.getOS() == 'windows':
        #     dot_data = sklearn.tree.export_graphviz(model, out_file=None, feature_names=Dataset['headers'])
        #     graph = pydotplus.graph_from_dot_data(dot_data)
        #     graph.write_png(Config.OS_PATH + self.save_items_path + "/" + self.generated_files_base_name + "alg.decision_tree.tree.png")

        propheticus.shared.Utils.printTimeLogMessage("Processing the algorithm", start_time)

        self.plotConfusionMatrix(algorithm, CrossValidatedResults['confusion_matrix'], Dataset['classes'])

        BiCVResultsTargetPred = Binarizer.transform(numpy.array(CrossValidatedResults['TargetPredictions']))
        BiCVResultsTargetTest = Binarizer.transform(numpy.array(CrossValidatedResults['TargetTest']))

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

        self.Context.ClassificationAlgorithmsResults[algorithm].append(Results)
        self.plotROC(Dataset, algorithm, numpy.array(CrossValidatedResults['TargetProbabilities']), BiCVResultsTargetTest, Binarizer.classes_)
        self.plotRecallPrecisionCurve(Dataset, algorithm, numpy.array(CrossValidatedResults['TargetProbabilities']), BiCVResultsTargetTest, Binarizer.classes_)

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
            propheticus.shared.Utils.saveExcel(self.save_log_items_path, self.generated_files_base_name + algorithm + '.grid_search.xlsx', [GridResultsHeaders] + RankedResults)

        return Results

    # NOTE: this function previously received Headers, Data, and Targets, as separate parameters; was it for when performing gridsearch avoid copy?
    def _runModel(self, index, seed, algorithm, Dataset, Binarizer, pool_count, PositiveClasses=None, _model=False):
        """
        Runs defined models for passed configurations.

        Parameters
        ----------
        algorithm
        Dataset
        Binarizer
        Seeds
        pool_count
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

        propheticus.shared.Utils.temporaryLogs(
            seed,
            algorithm,
            self.balance_data,
            self.reduce_dimensionality,
            self.algorithms_parameters,
            self.grid_search_parameters
        )

        RunCrossValidatedResults = self._initializeResultsObject()

        if algorithm == 'genetic_programming':  # TODO: Define this as configurations; Use 70 / 30 split instead of CV
            DataTrain, DataTest, TargetTrain, TargetTest, IndicesTrain, IndicesTest = sklearn.model_selection.train_test_split(Data, Target, range(Data.shape[0]), test_size=0.3, random_state=seed, stratify=Target)
            splits = [(IndicesTrain, IndicesTest)]
        else:
            cv = sklearn.model_selection.StratifiedKFold(n_splits=self.cv_fold, shuffle=True, random_state=seed)
            splits = cv.split(Data, Target)

        splits = list(splits)
        propheticus.shared.Utils.printStatusMessage(' - Seeding #' + str(index + 1) + ' ' + str(seed), inline=(False if Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_RUN else True))
        if self.mode == 'cli' and pool_count > 1 and Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_CV:
            FoldsData = propheticus.shared.Utils.pool(pool_count, self._runCVFold, [(_model, algorithm, Binarizer, Data[train], Target[train], Data[test], Target[test], Headers, seed, PositiveClasses) for (train, test) in splits])
        else:
            FoldsData = [self._runCVFold(_model, algorithm, Binarizer, Data[train], Target[train], Data[test], Target[test], Headers, seed, PositiveClasses) for (train, test) in splits]

        for FoldData in FoldsData:
            for key, Messages in FoldData[3].items():
                if len(Messages) > 0:
                    if key == 'warning':
                        Messages.insert(0, 'CV Fold Warnings \n')
                        propheticus.shared.Utils.printWarningMessage(Messages)

            for key, value in FoldData[1].items():
                RunCrossValidatedResults[key] += value

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

        # TODO: this should be considered in each cv fold?
        if algorithm == 'decision_tree  ' or algorithm == 'random_forests':
            _Indexes = numpy.argwhere(FoldData[0].feature_importances_ > Config.max_alert_dt_feature_importance).ravel()
            FoldHeaders = FoldData[2]
            if len(_Indexes) > 0:
                propheticus.shared.Utils.printWarningMessage('Some features have importances too high: ' + '; '.join(sorted([FoldHeaders[feature_index] + ': ' + str(round(FoldData[0].feature_importances_[feature_index], 2)) for feature_index in _Indexes])))
            Results['logs'] = "\n".join([FoldHeaders[feature_index] + ': ' + str(round(importance, 2)) for feature_index, importance in enumerate(FoldData[0].feature_importances_)])
        else:
            Results['logs'] = ''

        # self.Context.ClassificationAlgorithmsResults[algorithm].append(Results)

        return RunCrossValidatedResults, Results

    def _runCVFold(self, _model, algorithm, Binarizer, DataTrain, TargetTrain, DataTest, TargetTest, Headers, seed, PositiveClasses=None, thread_level=None):
        """
        Runs cross-validation (CV) fold and returns required data for logging

        Parameters
        ----------
        _model : bool
        algorithm : str
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
        if Config.thread_level_ != propheticus.shared.Utils.THREAD_LEVEL_RUN:
            propheticus.shared.Utils.printInlineStatusMessage('.')

        start_time = time.time()
        FoldResults = self._initializeResultsObject()
        Messages = {'warning': [], 'status': []}

        AlgorithmCallDetails = Config.ClassificationAlgorithmsCallDetails[algorithm]
        AlgorithmCallArguments = copy.deepcopy(self.algorithms_parameters) if self.algorithms_parameters else {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # NOTE: can be used like this, but may suppress relevant warnings
            warnings.filterwarnings(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)

            warnings.filterwarnings("ignore", message="F-score is ill-defined and being set to 0.0 due to no predicted samples")
            warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0 due to no predicted samples")
            warnings.filterwarnings("ignore", message="Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples")
            warnings.filterwarnings("ignore", message="The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.")

            if self.grid_search is not False:
                if not self.grid_search_parameters and 'grid' in Config.ClassificationAlgorithmsCallDetails[algorithm]:
                    self.grid_search_parameters = Config.ClassificationAlgorithmsCallDetails[algorithm]['grid']

                '''
                If parameters are sent through attribute self.grid_search_parameters those will be used for grid-search;
                otherwise the grid-search parameters from algorithms configuration will be used.
                '''
                # if self.algorithms_parameters:
                #     propheticus.shared.Utils.printWarningMessage('Values in algorithms_parameters may be overwritten by grid-search hyperparameters')

                if not self.grid_search_parameters:
                    propheticus.shared.Utils.printErrorMessage('No grid search parameters were passed or are defined in the configs for algorithm ' + algorithm + '. Skipping grid search')
                else:
                    # TODO: somehow fetch this from arguments / config
                    ftwo_scorer = sklearn.metrics.make_scorer(sklearn.metrics.fbeta_score, beta=2, average='weighted', labels=PositiveClasses)
                    balanced_acc_informedness = sklearn.metrics.make_scorer(propheticus.classification.metrics.informedness, average='weighted')

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
                    ExperimentSteps.append((algorithm_pipeline_prefix, propheticus.shared.Utils.dynamicAPICall(AlgorithmCallDetails, AlgorithmCallArguments, seed)))

                    pipeline_cache_dir = tempfile.mkdtemp()
                    Pipeliner = sklearn.pipeline.Pipeline if not self.balance_data else imblearn.pipeline.Pipeline
                    ExperimentPipeline = Pipeliner(ExperimentSteps)  # , memory=joblib.Memory(cachedir=pipeline_cache_dir, verbose=0)  # TODO: see if it improves performance or not

                    # with warnings.catch_warnings():
                    #     warnings.simplefilter("ignore")
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
                        AlgorithmCallArguments[hyperparameter.replace(algorithm_pipeline_prefix + '__', '')] = value

                    mean_train = oGridSearch.cv_results_['mean_train_score']
                    std_train = oGridSearch.cv_results_['std_train_score']
                    mean_test = oGridSearch.cv_results_['mean_test_score']
                    std_test = oGridSearch.cv_results_['std_test_score']

                    ResultsLog = []
                    for mean_train, std_train, mean_test, std_test, params in zip(mean_train , std_train , mean_test , std_test , oGridSearch.cv_results_['params']):
                        ResultsLog.append({
                            'parameters': ', '.join([param.replace(algorithm_pipeline_prefix + '__', '') + ': ' + str(value) for param, value in params.items()]),
                            'mean_train': mean_train,
                            'std_train': std_train,
                            'mean_test': mean_test,
                            'std_test': std_test
                        })

                    FoldResults['GridSearchResults'].append(ResultsLog)

            dimensionality_reduction_duration = 0
            if self.reduce_dimensionality and 'variance' in self.reduce_dimensionality:
                dim_red_start_time = time.time()

                DimRedCallArguments = copy.deepcopy(self.dim_red_parameters['variance']) if 'variance' in self.dim_red_parameters else {}

                CallDetails = Config.DimensionalityReductionCallDetails['variance']
                Estimator = propheticus.shared.Utils.dynamicAPICall(CallDetails, DimRedCallArguments, seed=seed)
                DataTrain = Estimator.fit_transform(DataTrain)
                Headers = propheticus.core.DatasetReduction.removeFeaturesFromHeaders(Estimator.indexes_, Headers)

                oVT = sklearn.feature_selection.VarianceThreshold()
                oVT.fit_transform(DataTest)
                for index in Estimator.indexes_:
                    if oVT.variances_[index] != 0:
                        Messages['warning'].append('Feature removed by 0 variance has variance in the test dataset: ' + Headers[index])

                DataTest = Estimator.transform(DataTest)
                dimensionality_reduction_duration = propheticus.shared.Utils.getTimeDifference(dim_red_start_time)

            if self.normalize is True:
                NormalizeCallDetails = Config.PreprocessingCallDetails['normalize']
                Estimator = propheticus.shared.Utils.dynamicAPICall(NormalizeCallDetails, seed=seed)
                DataTrain = Estimator.fit_transform(DataTrain)
                DataTest = Estimator.transform(DataTest)

            if self.reduce_dimensionality and (len(self.reduce_dimensionality) > 1 or self.reduce_dimensionality[0] != 'variance'):
                dim_red_start_time = time.time()
                DataTrain, RemoveFeatures, _, DimRedEstimators = propheticus.core.DatasetReduction.dimensionalityReduction(self.dataset_name, self.configurations_id, self.description, self.reduce_dimensionality, DataTrain, TargetTrain, Headers, self.dim_red_parameters, seed)
                for dim_red_method, DimRedEstimator in DimRedEstimators.items():
                    DataTest = DimRedEstimator.transform(DataTest)

                Headers = propheticus.core.DatasetReduction.removeFeaturesFromHeaders(RemoveFeatures, Headers)

                dimensionality_reduction_duration += propheticus.shared.Utils.getTimeDifference(dim_red_start_time)
                FoldResults['DurationDimensionalityReduction'].append(dimensionality_reduction_duration)

            if self.balance_data:
                balance_data_start_time = time.time()
                DataTrain, TargetTrain = propheticus.core.DatasetReduction.balanceDataset(DataTrain, TargetTrain, self.sampling_parameters, seed=seed, method=self.balance_data)
                balance_data_duration = propheticus.shared.Utils.getTimeDifference(balance_data_start_time)
                FoldResults['DurationDataBalancing'].append(balance_data_duration)

            if 'dataset_metadata_argument' in AlgorithmCallDetails and AlgorithmCallDetails['dataset_metadata_argument'] is True:
                AlgorithmCallArguments['dataset_metadata'] = {'features_count': len(Headers)}

            if Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_ALGORITHM and 'n_jobs' in AlgorithmCallDetails['parameters']:
                AlgorithmCallArguments['n_jobs'] = -1

            model = propheticus.shared.Utils.dynamicAPICall(AlgorithmCallDetails, AlgorithmCallArguments, seed)

            train_start_time = time.time()
            model.fit(DataTrain, TargetTrain)

            if not numpy.array_equal(model.classes_, Binarizer.classes_):
                propheticus.shared.Utils.printFatalMessage('The classes mapping between the classifier and the binarizer is not the same!')

            train_duration = propheticus.shared.Utils.getTimeDifference(train_start_time)

            test_start_time = time.time()
            TargetPredictions = model.predict(DataTest)
            test_duration = propheticus.shared.Utils.getTimeDifference(test_start_time)

            TargetProbabilities = model.predict_proba(DataTest)

            FoldResults['TargetPredictions'] += TargetPredictions.tolist()
            FoldResults['TargetProbabilities'] += TargetProbabilities.tolist()
            FoldResults['TargetTest'] += TargetTest.tolist()
            FoldResults['DurationTrain'].append(train_duration)
            FoldResults['DurationTest'].append(test_duration)

            BinarizedTargetPredictions = Binarizer.transform(TargetPredictions)
            BinarizedRunTargetTest = Binarizer.transform(TargetTest)

            Results = self.logResults(BinarizedRunTargetTest, BinarizedTargetPredictions, TargetProbabilities, TargetTest, TargetPredictions, PositiveClasses)
            FoldResults['confusion_matrix'] += Results['confusion_matrix']

            fold_cv_duration = propheticus.shared.Utils.getTimeDifference(start_time)
            FoldResults['DurationFold'].append(fold_cv_duration)

            if 'callable_package' in AlgorithmCallDetails:
                callback_package = AlgorithmCallDetails['callback_package']
                callback_callable = AlgorithmCallDetails['callback_callable']
                importlib.import_module(callback_package)
                getattr(sys.modules[callback_package], callback_callable)(DataTrain, TargetTrain, DataTest, TargetTest)


        return model, FoldResults, Headers, Messages





    '''
    Algorithms
    '''
    # def genetic_programming(self, dataset_name, Dataset, seed):
    #     return GeneticProgrammingClassifier(dataset_name, numb_gen=80, pop_size=300, validation_k_fold=5, seed=seed,
    #                                               in_max_depth=6, max_len=1000, prob_mut_node=0.1, prob_cross=0.9,
    #                                               tour_size=3, elite_size=0.02, survivors_func='elite', fitness_func='mdlp')

    # def custom_ensemble_classifier(self, dataset_name, Dataset, seed):
    #
    #     # G_B_G_NN_G_RF_MB_DT_MB_NN_MBS_GAU_MR_SVM
    #     # G_NN_G_RF_MB_B_MBS_GAU_MBS_NN_S
    #
    #     Algorithms = {}
    #     Algorithms['g_bagging'] = True
    #     Algorithms['g_neural_network'] = True
    #     Algorithms['g_random_forest'] = True
    #     Algorithms['g_svm'] = 0
    #
    #     Algorithms['m_gaussian_process'] = 0
    #
    #     Algorithms['mb_bagging'] = 0
    #     Algorithms['mb_decision_tree'] = True
    #     Algorithms['mb_gaussian_process'] = True
    #     Algorithms['mb_logistic_regression'] = 0
    #     Algorithms['mb_naive_bayes'] = 0
    #     Algorithms['mb_neural_network'] = 0
    #     Algorithms['mb_random_forest'] = 0
    #     Algorithms['mb_svm'] = 0
    #
    #     Algorithms['mbs_gaussian_process'] = 0
    #     Algorithms['mbs_knn'] = 0
    #     Algorithms['mbs_logistic_regression'] = 0
    #     Algorithms['mbs_neural_network'] = 0
    #     Algorithms['mbs_svm'] = 0
    #
    #     Algorithms['mr_gaussian_process'] = True
    #     Algorithms['mr_logistic_regression'] = 0
    #     Algorithms['mr_neural_network'] = 0
    #     Algorithms['mr_svm'] = 0
    #
    #     Algorithms['mrs_gaussian_process'] = 0
    #     Algorithms['mrs_knn'] = 0
    #     Algorithms['mrs_logistic_regression'] = 0
    #     Algorithms['mrs_svm'] = 0
    #
    #     Algorithms = collections.OrderedDict(sorted(Algorithms.items()))
    #
    #     Alias = {'bagging': 'B', 'neural_network': 'NN', 'random_forest': 'RF', 'svm': 'SVM', 'decision_tree': 'DT', 'gaussian_process': 'GAU', 'logistic_regression': 'LR', 'naive_bayes': 'NB', 'knn': 'KNN'}
    #
    #     use_simple_voting = False
    #
    #     self.generated_files_base_name = ("_".join([algorithm for algorithm, run in Algorithms.items() if run is True]) + '_').upper()
    #     for find, replace in Alias.items():
    #         self.generated_files_base_name = self.generated_files_base_name.replace(find.upper(), replace.upper())
    #
    #     self.generated_files_base_name += '' if use_simple_voting is False else 'S_'
    #
    #     expertise_weight = 1
    #     prediction_prob_weight = 1
    #     support_weight = 1
    #     self.generated_files_base_name += 'EW_'
    #
    #     # expertise_weight = 1.78
    #     # prediction_prob_weight = 2.5
    #     # support_weight = 1.2
    #
    #     print(self.generated_files_base_name[:-1])
    #
    #     folder_path = Config.OS_PATH + Config.framework_instance_generated_path + "Classification/mitoxt_2_custom_custom/" + self.generated_files_base_name + "custom_ensemble_classifier_cm.png"
    #     folder_path2 = Config.OS_PATH + Config.framework_instance_generated_path + "Classification/mitoxt_2_custom_custom/" + self.generated_files_base_name + "alg_custom_ensemble_classifier_confusion_matrix.png"
    #     if os.path.isfile(folder_path) or os.path.isfile(folder_path2):
    #         exit('Combination already exists! ' + self.generated_files_base_name)
    #
    #     return CustomEnsembleClassifier(random_state=seed, algorithms=Algorithms, expertise_weight=expertise_weight, prediction_prob_weight=prediction_prob_weight, support_weight=support_weight, use_simple_voting=use_simple_voting)

