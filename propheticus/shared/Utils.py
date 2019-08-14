"""
Contains utilitarian methods/functionalities/data
"""
import numpy
import openpyxl
import copy
import os
import re
import hashlib
import time
import multiprocessing
import pathlib
import xlrd
import subprocess
import json
import collections
import importlib
import operator
import gc
import codecs
import itertools
import sklearn.metrics
import datetime
import subprocess
import warnings
import sklearn.exceptions

import matplotlib.pyplot as plt

import propheticus

if hasattr(propheticus.Config, 'framework_selected_instance_path'):
    import sys
    sys.path.append(propheticus.Config.framework_selected_instance_path)
    if os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, 'InstanceConfig.py')):
        from InstanceConfig import InstanceConfig as Config
    else:
        import propheticus.Config as Config
else:
    import propheticus.Config as Config

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


class Utils:
    THREAD_LEVEL_BATCH = 'batch'
    THREAD_LEVEL_RUN = 'run'
    THREAD_LEVEL_CV = 'cv'
    THREAD_LEVEL_ALGORITHM = 'algorithm'
    THREAD_LEVEL_BEST = 'best'

    PYTHON_NUMPY_DATA_TYPES_MAP = {
        'int64': numpy.int64,
        'float64': numpy.float64,
        'string': numpy.object,
        'bool': numpy.bool
    }

    ColourValues = [
                     'FF0000', '00FF00', '0000FF', 'FFFF00', '00FFFF', 'FF00FF', '000000',
                     '800000', '008000', '000080', '808000', '800080', '008080', '808080',
                     'C00000', '00C000', '0000C0', 'C0C000', 'C000C0', '00C0C0', 'C0C0C0',
                     '400000', '004000', '000040', '404000', '400040', '004040', '404040',
                     '200000', '002000', '000020', '202000', '200020', '002020', '202020',
                     '600000', '006000', '000060', '606000', '600060', '006060', '606060',
                     'A00000', '00A000', '0000A0', 'A0A000', 'A000A0', '00A0A0', 'A0A0A0',
                     'E00000', '00E000', '0000E0', 'E0E000', 'E000E0', '00E0E0', 'E0E0E0',
                    ]

    AvailableExperiments = {True: None, False: None}

    LoggedMessages = {'error': [], 'status': [], 'warning': []}

    AvailableDataBalancing = sorted([algorithm for algorithm, Details in Config.SamplingCallDetails.items()])
    AvailableDimensionalityReduction = sorted([algorithm for algorithm, Details in Config.DimensionalityReductionCallDetails.items()])
    AvailableClassificationAlgorithms = sorted([algorithm for algorithm, Details in Config.ClassificationAlgorithmsCallDetails.items()])
    AvailableClusteringAlgorithms = sorted([algorithm for algorithm, Details in Config.ClusteringAlgorithmsCallDetails.items()])

    # TODO: this needs to be more flexible
    AvailableClassificationMetrics = sorted(['f05-score', 'f1-score', 'f2-score', 'roc-auc', 'accuracy', 'precision', 'recall', 'specificity', 'informedness', 'markedness'])

    RandomSeeds = [341187527, 644570282, 786112152, 135708657, 597861104, 880607938, 787554810, 311771549, 717616322, 276791663,
                   852934353, 562682195, 881244661, 954295709, 911889983, 69598916, 359584179, 135103005, 39271315, 137908699,
                   142079615, 448358036, 238576428, 295203401, 170362978, 341448958, 701279747, 914483372, 834810093, 115986650]

    hide_demo_popups = False
    cachedStep = None
    chars_per_second = 15

    '''
    Data Choices for the UI Related Functions
    '''
    @staticmethod
    def getAvailableExperimentsList(use_cached=True, skip_config_parse=False, field='label'):
        return [Experiment[field] for experiment_identifier, Experiment in Utils.getAvailableExperiments(use_cached, skip_config_parse).items()]

    @staticmethod
    def getAvailableExperiments(use_cached=True, skip_config_parse=False, filter_by=None):
        if Utils.AvailableExperiments[skip_config_parse] is None or use_cached is False:
            DescriptiveFields = [
                'pre_target',
                'proc_balance_data',
                'proc_classification',
                'proc_classification_algorithms_parameters',
                'proc_reduce_dimensionality'
            ]

            Experiments = {}
            for root, SubFolders, Files in os.walk(os.path.join(Config.framework_instance_generated_logs_path)):
                for subfolder in SubFolders:
                    for filename in os.listdir(os.path.join(root, subfolder)):
                        if '~' not in filename and 'dimensionality_reduction' not in filename and 'grid_search' not in filename:
                            experiment_identifier = filename.split('.')[0]

                            if not skip_config_parse:
                                Workbook = xlrd.open_workbook(os.path.join(root, subfolder, filename))
                                DetailsWorksheet = Workbook.sheet_by_index(0)
                                experiment_configuration = DetailsWorksheet.cell(0, 0).value
                                ParsedConfigurations = Utils.parseLogStringConfiguration(experiment_configuration)
                                BriefDetails = {key: value for key, value in ParsedConfigurations.items() if key in DescriptiveFields}

                                if filter_by is not None:
                                    if 'proc_' + filter_by not in ParsedConfigurations or not ParsedConfigurations['proc_' + filter_by]:
                                        continue
                            else:
                                BriefDetails = {}
                                ParsedConfigurations = {}

                            Experiments[experiment_identifier] = {
                                'identifier': experiment_identifier,
                                'label': ' ' + subfolder + ' >> ' + ' | '.join(map(str, BriefDetails.values())) + ' * ' + filename,
                                'configuration': ParsedConfigurations,
                                'filename': filename,
                                'subfolder': subfolder
                            }

            Utils.AvailableExperiments[skip_config_parse] = collections.OrderedDict(sorted(Experiments.items()))

        return Utils.AvailableExperiments[skip_config_parse]

    @staticmethod
    def getAvailableDatasets():
        return sorted([file for file in os.listdir(Config.framework_instance_data_path) if '.data.txt' in file])

    ''''###############################################################################################'''



    '''
    Utility Related Functions
    '''
    @staticmethod
    def temporaryLogs(*args):
        pathlib.Path(Config.framework_temp_path).mkdir(parents=True, exist_ok=True)
        with codecs.open(os.path.join(Config.framework_temp_path, 'execution_log.txt'), "w", encoding="utf-8") as File:
            File.writelines(json.dumps(Utils._getSafeConfigurationsDict(args), indent=2))

    @staticmethod
    def showImage():
        if Config.demonstration_mode is True:
            def close_event():
                Utils.parseDemoStep(prepend='figure')
                time.sleep(2)
                plt.close()

            fig = plt.gcf()
            timer = fig.canvas.new_timer(interval=1500)  # creating a timer object and setting an interval of 3000 milliseconds
            timer.add_callback(close_event)
            timer.start()

        plt.show()

    @staticmethod
    def saveImage(path, filename, **kwargs):
        pathlib.Path(os.path.join(Config.OS_PATH, path)).mkdir(parents=True, exist_ok=True)
        kwargs['fname'] = fname = os.path.join(Config.OS_PATH, path, filename)
        kwargs['transparent'] = Config.use_transparency
        plt.savefig(**kwargs)

        if not os.path.isfile(fname):
            Utils.printFatalMessage('Image was not created! File: ' + fname)

    @staticmethod
    def saveExcel(path, filename, *args, SheetNames=False, show_demo=True):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        if SheetNames is False:
            SheetNames = ['Sheet ' + str(index) for index in range(len(args))]

        Workbook = openpyxl.Workbook()
        Sheets = [Workbook.create_sheet(SheetNames[i]) for i in range(1, len(args))]
        Workbook.active.title = SheetNames[0]

        Sheets.insert(0, Workbook.active)

        for index, Data in enumerate(args):
            for Item in Data:
                Sheets[index].append(Item)

        file_path = os.path.join(path, filename)
        Workbook.save(file_path)
        if not os.path.isfile(file_path):
            Utils.printFatalMessage('Excel document was not created! File: ' + file_path)

        if Config.demonstration_mode is True and show_demo and Utils.hide_demo_popups is False:
            Utils.openExternalFileDemo(file_path)

    @staticmethod
    def openExternalFileDemo(file_path):
        with subprocess.Popen(["start", "/WAIT", file_path], shell=True) as doc:
            # use 'doc' here just as you would the file itself
            doc.poll()
            Utils.parseDemoStep(prepend='excel')

    @staticmethod
    def getTimeDifference(start, end=None, precision=3):
        if end is None:
            end = time.time()
        return round((end - start), precision)

    @staticmethod
    def getClassDistribution(Y):
        Y = numpy.array(Y)
        Classes = []
        for index, target in enumerate(Y):
            if target not in Classes:
                Classes.append(target)
        DistributionByClass = {target: int(Y.tolist().count(target)) for target in Classes}

        return DistributionByClass

    @staticmethod
    def getOS():
        if os.name == 'nt':
            system = 'windows'
        elif os.name == 'posix':
            system = 'linux'
        else:
            Utils.printFatalMessage('Unexpected operating system. Change to handle: ' + os.name)

        return system

    @staticmethod
    def getDatasetsIdentifiers(identifier):
        Abrv = {'label_': '', '_': '', 'set_': '', 'XEN': 'X', 'Tomcat': 'Tc', 'vSphere': 'V', 'real': 'R', 'Training': 'Tr', 'Test': 'Ts'}
        for search, replace in Abrv.items():
            identifier = identifier.replace(search, replace)

        if len(identifier) >= 100:
            Utils.printWarningMessage('This would be an hash!')

        return identifier

    @staticmethod
    def validateCurrentUserUUID():
        """
        Validates the current user UUID against the authorized ones defined in the list Config.ValidUUID list.
        WARNING: this varies according to the python version. Python 3.6 has method check_output
        TODO: improve logic
        Returns
        -------

        """
        if Utils.getOS() == 'linux':
            message = subprocess.Popen('dmidecode|grep UUID', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
            if 'denied' in str(message[1]).lower():
                Utils.printFatalMessage('This application must be executed with root privileges')

            uuid = message[0].decode("utf-8").split(' ')[1].split('\n')[0]
            if uuid not in Config.ValidUUID:
                Utils.printFatalMessage('You do not have permission to use this application')
            else:
                Utils.printStatusMessage('User successfully authenticated')
        elif Utils.getOS() == 'windows':
            uuid = subprocess.Popen('wmic csproduct get uuid', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode("utf-8").split('\n')[1].split(' ')[0]
            # uuid = subprocess.check_output('wmic csproduct get uuid').decode("utf-8").split('\n')[1].strip()
            if uuid not in Config.ValidUUID:
                Utils.printFatalMessage('You do not have permission to use this application')
            else:
                Utils.printStatusMessage('User successfully authenticated')
        else:
            Utils.printFatalMessage('Authentication required and operating system is not supported')

    @staticmethod
    def parseConfusionMatrix(confusion_matrix):
        TruePositiveByClass = {}
        FalsePositivesByClass = {}
        TrueNegativeByClass = {}
        FalseNegativeByClass = {}
        TruePositive = numpy.diag(confusion_matrix)
        FalsePositive = []
        FalseNegative = []
        TrueNegative = []

        num_classes = len(confusion_matrix)
        for i in range(num_classes):
            # NOTE: True Positives
            TruePositiveByClass[i] = TruePositive[i]

            # NOTE: False Positives
            FalsePositivesByClass[i] = sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
            FalsePositive.append(sum(confusion_matrix[:, i]) - confusion_matrix[i, i])

            # NOTE: False Negatives
            FalseNegativeByClass[i] = sum(confusion_matrix[i, :]) - confusion_matrix[i, i]
            FalseNegative.append(sum(confusion_matrix[i, :]) - confusion_matrix[i, i])

            # NOTE: True Negatives
            temp = numpy.delete(confusion_matrix, i, 0)  # delete ith row
            temp = numpy.delete(temp, i, 1)  # delete ith column
            TrueNegativeByClass[i] = sum(sum(temp))
            TrueNegative.append(sum(sum(temp)))

        return TrueNegativeByClass, FalsePositivesByClass, FalseNegativeByClass, TruePositiveByClass


    @staticmethod
    def computeMetricsByClass(Y_true, Y_pred, target_metric=None, average='binary', labels=None):
        # TODO: cache results?

        AllowedAverages = ['weighted', 'macro']
        if average is not None and average not in AllowedAverages:
            propheticus.shared.Utils.printFatalMessage('Allowed averages: ' + ', '.join(AllowedAverages))

        MetricsByClass = {'precision': {}, 'recall': {}, 'informedness': {}, 'markedness': {}, 'specificity': {}}

        confusion_matrix = sklearn.metrics.confusion_matrix(Y_true, Y_pred)
        TNC, FPC, FNC, TPC = Utils.parseConfusionMatrix(confusion_matrix)

        ClassSupport = collections.Counter(sorted(Y_true))
        for i in range(len(ClassSupport)):
            if TPC[i] > 0:
                class_recall = TPC[i] / (TPC[i] + FNC[i])
                class_precision = TPC[i] / (TPC[i] + FPC[i])
            else:
                class_recall = 0
                class_precision = 0

            if TNC[i] > 0:
                class_inverse_recall = TNC[i] / (FPC[i] + TNC[i])
                class_inverse_precision = TNC[i] / (FNC[i] + TNC[i])
            else:
                class_inverse_recall = 0
                class_inverse_precision = 0

            class_label = list(ClassSupport.keys())[i]
            MetricsByClass['precision'][class_label] = class_precision
            MetricsByClass['recall'][class_label] = class_recall
            MetricsByClass['informedness'][class_label] = class_recall + class_inverse_recall - 1
            MetricsByClass['markedness'][class_label] = class_precision + class_inverse_precision - 1
            MetricsByClass['specificity'][class_label] = class_inverse_recall

        ClassSupport = collections.Counter(sorted(Y_true))
        if labels is None:
            labels = list(ClassSupport.keys())

        NPositive = sum([ClassSupport[label] for label in labels])
        ComputedMetrics = {}
        if average is not None:
            for metric in MetricsByClass.keys():
                if average == 'weighted':
                    ComputedMetrics[metric] = sum([(ClassSupport[label] / NPositive) * MetricsByClass[metric][label] for label in labels])
                elif average == 'macro':
                    ComputedMetrics[metric] = sum([MetricsByClass[metric][label] for label in labels]) / len(labels)

            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)

                # NOTE: for validation purposes
                sklearn_precision = sklearn.metrics.precision_score(Y_true, Y_pred, average=average, labels=labels)
                sklearn_recall = sklearn.metrics.recall_score(Y_true, Y_pred, average=average, labels=labels)

            sklearn_precision = round(sklearn_precision, 4)
            sklearn_recall = round(sklearn_recall, 4)
            precision = round(ComputedMetrics['precision'], 4)
            recall = round(ComputedMetrics['recall'], 4)
            tolerance = 0.01

            if precision + tolerance < sklearn_precision or sklearn_precision < precision - tolerance \
                    or recall + tolerance < sklearn_recall or sklearn_recall < recall - tolerance:
                Utils.printFatalMessage(f'Invalid computations: {sklearn_precision}:{precision}, {sklearn_recall}:{recall}')
        else:
            ComputedMetrics = MetricsByClass

        return ComputedMetrics if target_metric is None else ComputedMetrics[target_metric]

    @staticmethod
    def cartesianProductDictionaryLists(**kwargs):
        keys, values = zip(*kwargs.items())
        Configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

        return Configs

    @staticmethod
    def isInt(value):
        try:
            int(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def escapeExcelValues(value):
        if isinstance(value, (list, numpy.ndarray, dict)):
            value = str(value)

        return value

    @staticmethod
    def toTitle(value):
        Replace = ['_', '-']
        for search in Replace:
            value = value.replace(search, ' ')

        return value.title()

    @staticmethod
    def getClassDescriptionById(class_id):
        return class_id if class_id not in Config.ClassesDescription else Config.ClassesDescription[class_id]

    @staticmethod
    def isMultiProcess():
        return False if multiprocessing.current_process().name == 'MainProcess' else True

    ''''###############################################################################################'''


    '''
    Demonstration Related Functions
    '''
    @staticmethod
    def storeDemonstrationVariables():
        if len(Utils.DemonstrationLogs) > 0:
            demonstration_start_time = datetime.datetime.utcfromtimestamp(Utils.demonstration_start).strftime('%Y%m%d%H%M%S')

            timestamp = propheticus.shared.Utils.getTimeDifference(Utils.demonstration_start, precision=2)
            Utils.DemonstrationLogs[-1][1] = timestamp - Utils.DemonstrationLogs[-1][0]

            propheticus.shared.Utils.saveExcel(
                Config.framework_instance_generated_logs_path,
                demonstration_start_time + '.demonstration_log.xlsx',
                Utils.DemonstrationLogs
            )

            propheticus.shared.Utils.printStatusMessage('Demonstration log successfully saved')

        RemainingSteps = []
        for RemainingStep in Utils.DemonstrationSteps:
            RemainingSteps.append(', '.join(map(str, RemainingStep)))

        if RemainingSteps:
            Utils.printWarningMessage(['There are still steps remaining in the excel file!'] + RemainingSteps)

    @staticmethod
    def cacheNextDemoStep():
        Step = next(Utils.DemonstrationSteps, None)
        if Step is not None:
            Utils.cachedStep = Step

            delay = 0
            Legends = list(map(str, Step[2:]))
            for index, legend in enumerate(Legends):
                if legend is None or legend == 'None':
                    break

                delay += Utils.getSleepDurationByLegend(legend)

            return delay

        else:
            return None

    @staticmethod
    def getSleepDurationByLegend(legend):
        return max(1, len(legend) / Utils.chars_per_second)

    @staticmethod
    def parseDemoStep(prepend=''):
        if Utils.cachedStep is not None:
            Step = Utils.cachedStep
            Utils.cachedStep = None
        else:
            Step = next(Utils.DemonstrationSteps, None)

        if Step is not None:
            action_input = str(Step[0]) if Step[0] is not None else ''
            if prepend.strip() != '' and action_input.strip() != '':
                Utils.printWarningMessage('Legend step with input defined!')

            label = Step[1]
            Legends = list(map(str, Step[2:]))
            for index, legend in enumerate(Legends):
                if legend is None or legend == 'None':
                    if index == 0:
                        Utils.temporaryDemonstrationLogs(f'{label} {prepend}', legend)  # NOTE: REGISTER ONE ENTRY FOR LOG
                    break

                Utils.temporaryDemonstrationLogs(f'{label} {prepend}', legend, sleep_duration=Utils.getSleepDurationByLegend(legend))
        else:
            action_input = label = Legends = None

        return action_input, label, Legends, Step

    @staticmethod
    def initializeDemonstrationVariables():
        Utils.demonstration_start = time.time()
        Utils.DemonstrationLogs = []
        Utils.DemonstrationSteps = []

        Workbook = openpyxl.load_workbook(Config.framework_demo_file_path)
        StepsWorksheet = Workbook.worksheets[0]
        for i in range(2, StepsWorksheet.max_row + 1):
            Utils.DemonstrationSteps.append([StepsWorksheet.cell(i, j).value for j in range(1, StepsWorksheet.max_column + 1)])

        temp = Utils.DemonstrationSteps
        Utils.DemonstrationSteps = iter(Utils.DemonstrationSteps)

        Utils.parseDemoStep()

    @staticmethod
    def temporaryDemonstrationLogs(message, legend='', sleep_duration=0):
        timestamp = propheticus.shared.Utils.getTimeDifference(Utils.demonstration_start, precision=2)
        Utils.DemonstrationLogs.append([timestamp, sleep_duration, legend, message])
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        # print(legend)

    ''''###############################################################################################'''

    '''
    Messages Related Functions
    '''
    @staticmethod
    def resetLoggedMessages():
        Utils.LoggedMessages = {'error': [], 'status': [], 'warning': []}

    @staticmethod
    def printAcknowledgeMessage(new_line=False):
        Utils.printInputMessage('Acknowledge: (Enter)')
        if new_line is True:
            Utils.printNewLine()

    @staticmethod
    def printErrorMessage(message, acknowledge=True):
        Utils.LoggedMessages['error'].append(message)
        Utils.printBoxedMessage('ERROR', message)
        if acknowledge is True:
            Utils.printAcknowledgeMessage(True)

    @staticmethod
    def printFatalMessage(message):
        Utils.LoggedMessages['error'].append(message)
        Utils.printBoxedMessage('FATAL', message)
        Utils.printAcknowledgeMessage()
        exit('FATAL ERROR OCCURRED')

    @staticmethod
    def printWarningMessage(message, acknowledge=False):
        if isinstance(message, list):
            Utils.LoggedMessages['warning'] += message
        else:
            Utils.LoggedMessages['warning'].append(message)

        Utils.printBoxedMessage('WARNING', message)
        if acknowledge is True:
            Utils.printAcknowledgeMessage()

    @staticmethod
    def printNoticeMessage(message):
        Utils.printBoxedMessage('NOTICE', message)

    @staticmethod
    def printBoxedMessage(base, Messages):
        char_split = 140

        if not isinstance(Messages, list):
            Messages = [Messages]

        FinalMessages = []
        for index, message in enumerate(Messages):
            _message = (base + ': ' if index == 0 else '') + message
            if len(_message) < char_split:
                spacing = "".join([' ' * int(((char_split - len(_message)) / 2))])
                _message = (spacing + _message + spacing)[:char_split - 1]

            FinalMessages.append(re.sub("(.{" + str(char_split) + "})", "\\1\n", _message, 0, re.DOTALL))

        spacer = "".join(['!' * char_split])
        print(f'\n{spacer}\n' + "\n".join(FinalMessages) + f'\n{spacer}\n')

    @staticmethod
    def printNoteMessage(message, force=False):
        if Config.log_status or force is True:
            _message = f'NOTE: {message}'
            print(''.join(['*'] * (len(_message) + 1)))
            print(_message)
            print(''.join(['*'] * (len(_message) + 1)) + '\n')

    @staticmethod
    def printStatusMessage(message, inline=False, force=False, label='STATUS'):
        Utils.LoggedMessages['status'].append(message)
        if Config.log_status or force is True:
            print(f'{label}: {message}', end=("" if inline is True else None), flush=True)

    @staticmethod
    def printMessage(message, inline=False, force=False):
        if Config.log_status or force is True:
            print(message, end=("" if inline is True else None), flush=True)

    @staticmethod
    def printNewLine(count=0):
        if Config.log_status is True:
            print('\n' * count)

    @staticmethod
    def printInlineStatusMessage(message):
        if Config.log_status:
            print(message, end="", flush=True)

    @staticmethod
    def printLogMessage(message):
        print('LOG: ' + message)

    @staticmethod
    def printTimeLogMessage(message, start, end=None, precision=0, force=False):
        if Config.log_times or force is True:
            if end is None:
                end = time.time()
            duration = round((end - start), 1)
            if duration > (60 * 3):
                Utils.printLogMessage(message + " took " + str(datetime.timedelta(seconds=duration)))
            else:
                Utils.printLogMessage(message + " took %s seconds" % duration)

    @staticmethod
    def printInputMessage(message):
        if not Utils.isMultiProcess():
            print('INPUT: ' + message)
            input_prepend = " >>  "
            if Config.demonstration_mode is True:
                Utils.printMessage(input_prepend, inline=True)
                action_input, label, Legends, Step = Utils.parseDemoStep()
                if Step is not None:
                    for char in action_input:
                        Utils.printMessage(char, inline=True)
                        time.sleep(0.2)

                    Utils.printNewLine()
                    time.sleep(2)
                    return action_input
                else:
                    return input()
            else:
                return input(input_prepend)
        else:
            print('SILENCED: Threading')

    @staticmethod
    def printConfirmationMessage(message):
        Mapping = {'y': 'y', '1': 'y', 'n': 'n', '0': 'n'}
        while True:
            choice = Utils.printInputMessage('Confirmation: ' + message + '\nContinue? y : n\n')
            if choice not in Mapping:
                Utils.printErrorMessage('Invalid selection, please try again')
            else:
                break

        return Mapping[choice]

    @staticmethod
    def consoleClear():
        print('\n' * 2)
        print('&' * 140)
        print('&' * 140)
        print('&' * 140)
        print('\n' * 2)
        # os.system('cls')



    ''''###############################################################################################'''



    '''
    GUI Menus Functions
    '''
    @staticmethod
    def generalMenuData(Context, base_menu_name, MenuData, Configurations, print_menu_configurations=False, show_all_option=True, force_choice=True, menu_key=None):
        if menu_key is None:
            menu_key = base_menu_name.lower().replace(' ', '_')

        def _parseGeneralChoice(choice):
            return Utils._parseChoicesSelection(menu_key, base_menu_name, MenuData, choice, Configurations, force_choice=force_choice)

        return propheticus.shared.Utils.menuData(
            Context=Context,
            menu_name=f'{base_menu_name} Menu',
            MenuData=MenuData,
            callback=_parseGeneralChoice,
            Configurations=(Configurations if print_menu_configurations else None),
            show_all_option=show_all_option,
            force_choice=force_choice
        )

    @staticmethod
    def menuData(Context, menu_name, MenuData, callback, Configurations=None, show_all_option=True, force_choice=True):
        Data = Utils.generateMenuSelectionData(Context, MenuData, callback, show_all_option=show_all_option, force_choice=force_choice)
        def _menuData(choice=0):
            Utils.printMenu(menu_name, Data, Configurations, Context)

        return _menuData

    @staticmethod
    def generateMenuSelectionData(Context, Data, callback, show_all_option=True, force_choice=True):
        list_max_digit = len(str(len(Data)))
        MenuData = {}
        for index, value in enumerate(Data):
            MenuData[str(index + 1).zfill(list_max_digit)] = {'name': value, 'callback': callback}

        if show_all_option is True:
            MenuData[str(len(Data) + 1).zfill(list_max_digit)] = {'name': 'All', 'callback': callback}
            MenuData['_regex'] = {'callback': callback}

        if force_choice is False:
            MenuData['_regex'] = {'callback': callback}

        MenuData['-'] = ''
        MenuData['0'] = {'name': 'Back'}
        MenuData['h'] = {'name': 'Help', 'callback': Context.help()}

        return MenuData

    @staticmethod
    def _parseChoicesSelection(property_name, property_display_name, ChoicesData, choice, Configurations, show_selection_message=True, force_choice=True):
        Configurations[property_name] = []

        if choice.strip() == str(len(ChoicesData) + 1):
            choice = '1-' + str(len(ChoicesData))

        valid = True
        silence_error = False
        if choice.strip() != '':
            for index in choice.split(','):
                for value2 in index.split('-'):
                    if not value2:
                        if not silence_error:
                            Utils.printErrorMessage('Invalid input format! The valid format must be integers separated by , (individual choices) or - (ranges)')
                            valid = False
                            silence_error = True

                    else:
                        if not value2.isdigit():
                            Utils.printErrorMessage('Invalid choice passed, must be an integer: ' + str(value2))
                            valid = False
                        else:
                            _index = int(value2) - 1
                            if _index < 0 or _index > len(ChoicesData):
                                Utils.printErrorMessage('Invalid choice passed: ' + str(_index))
                                valid = False

            if valid is True:
                for value in choice.split(','):
                    Values = value.split('-')
                    if len(Values) > 1:
                        for value2 in range(int(Values[0]) - 1, int(Values[1])):
                            Configurations[property_name].append(ChoicesData[value2])
                    else:
                        _value = int(Values[0]) - 1
                        Configurations[property_name].append(ChoicesData[_value])

                if show_selection_message is True:
                    Utils.printStatusMessage(property_display_name + ' successfully chosen: ' + ','.join(Configurations[property_name]))

        elif force_choice is True:
            Utils.printWarningMessage('No selection made')
            valid = False

        return -1 if valid is True else valid

    @staticmethod
    def printBreadCrumb(breadcrumb):
        print(''.join(['-'] * (len(breadcrumb) + 1)))
        print(breadcrumb)
        print(''.join(['-'] * (len(breadcrumb) + 1)) + '\n')

    @staticmethod
    @static_vars(Breadcrumb=['Initialize'])
    def printMenu(label, MenuData, Configurations=None, Context=None, custom_message=None):
        if hasattr(propheticus.Config, 'framework_instance_label'):
            Utils.printMenu.Breadcrumb[0] = Config.framework_instance_label
        while True:
            if Configurations is not None:
                Utils.printCurrentConfigurations(getattr(Context, Configurations), hide_empty=True)

            if custom_message is not None:
                Utils.printStatusMessage(custom_message, label='CUSTOM')

            breadcrumb_separator = ' >> '
            breadcrumb = breadcrumb_separator + breadcrumb_separator.join(Utils.printMenu.Breadcrumb)
            Utils.printBreadCrumb(breadcrumb)

            print(' ---> ' + label)

            Items = collections.OrderedDict(sorted(MenuData.items())).items() if Config.validate_uuid is True else MenuData.items()
            for key, MenuDetails in Items:
                print('') if MenuDetails == '' or key == '_regex' else print(' -----> ' + str(key) + ' - ' + MenuDetails['name'])
            print('')

            while True:
                choice = Utils.printInputMessage('Select an option:')
                if choice not in MenuData and '_regex' not in MenuData:
                    Utils.printErrorMessage('Invalid selection, please try again', acknowledge=False)
                else:
                    break

            Utils.consoleClear()

            if choice.isdigit() and int(choice) == 0:
                if MenuData['0']['name'] not in ['Back', 'Quit']:
                    Utils.printFatalMessage('FATAL: menu choice 0 must be always assigned to Back')

                if 'callback' in MenuData['0']:
                    MenuData['0']['callback'](choice)
                break
            else:
                if choice in MenuData:
                    Utils.printMenu.Breadcrumb.append(MenuData[choice]['name'])

                returned = MenuData[choice if choice in MenuData else '_regex']['callback'](choice)
                if choice in MenuData:
                    del Utils.printMenu.Breadcrumb[-1]

                if returned == -1:
                    break

    ''''###############################################################################################'''




    '''
    Framework Configurations Related Functions
    '''
    @staticmethod
    def printCurrentConfigurations(Configurations, hide_empty=False):
        configurations = Utils.toStringCurrentConfigurations(Configurations, hide_empty=hide_empty)
        Utils.printStatusMessage(configurations)

    @staticmethod
    def toStringCurrentConfigurations(Configurations, truncate=True, hide_empty=False):
        max_chars = 100 if truncate is True else 100000000000000000

        Configs = ['Current configurations:']
        for key, value in Utils.getSafeConfigurationsDict(Configurations, hide_empty).items():
            Configs.append('- [' + key + '] >>> ' + (value[:max_chars] + ', ... ' if len(value) > max_chars else value))

        Configs.append('')

        return '\n'.join(Configs)

    @staticmethod
    def getSafeConfigurations(Configurations):
        SafeConfigurations = [key + '=' + value for key, value in Utils.getSafeConfigurationsDict(Configurations).items()]
        return SafeConfigurations

    @staticmethod
    def getSafeConfigurationsDict(Configurations, hide_empty=False):
        JSONConfigurations = {}
        for key, Values in sorted(Configurations.items()):
            if hide_empty and isinstance(Values, (list, numpy.ndarray, dict, str)) and not Values:
                continue

            JSONConfigurations[key] = json.dumps(Utils._getSafeConfigurationsDict(Values))

        return collections.OrderedDict(sorted(JSONConfigurations.items()))

    @staticmethod
    def _getSafeConfigurationsDict(ConfigValues):
        _ConfigValues = copy.deepcopy(ConfigValues)
        if isinstance(_ConfigValues, (list, numpy.ndarray, dict, range, tuple)):
            if isinstance(_ConfigValues, dict) and 'label' in _ConfigValues:
                _ConfigValues = _ConfigValues['label']
            else:
                iterator = _ConfigValues.items() if isinstance(_ConfigValues, dict) else enumerate(_ConfigValues)
                ShownSublist = {_key: Utils._getSafeConfigurationsDict(value) for _key, value in iterator}
                _ConfigValues = sorted(ShownSublist.items() if isinstance(_ConfigValues, dict) else ShownSublist.values(), key=lambda x: str(x))
                if isinstance(ConfigValues, dict):
                    _ConfigValues = collections.OrderedDict(_ConfigValues)
        elif not isinstance(_ConfigValues, (str, float, int, bool, type(None))):
            Utils.printFatalMessage('Unexpected data type: ' + str(type(_ConfigValues)))

        return _ConfigValues

    @staticmethod
    def parseLogStringConfiguration(experiment_configuration, SelectFields=None):
        Configurations = {}
        for config in experiment_configuration.split('\n'):
            if not config[:3] == '- [':
                continue

            key = config[3:config.index(']')]
            if SelectFields is None or key in SelectFields:
                value = config[config.index('>>>') + 4:]
                Configurations[key] = json.loads(value)

        return Configurations

    @staticmethod
    def getConfigurationsIdentifier(Configurations):
        return hashlib.md5(str.encode(",".join(sorted(Utils.getSafeConfigurations(Configurations))))).hexdigest()

    ''''###############################################################################################'''

    '''
    API Dynamic Calls Related Functions
    '''
    @staticmethod
    def dynamicAPICall(CallDetails, OverrideArguments=None, seed=None):
        MethodParameters = CallDetails['parameters']
        method_package = CallDetails['package']
        method_callable = CallDetails['callable']

        Parameters = {parameter: ParameterDetails['default'] for parameter, ParameterDetails in MethodParameters.items() if 'default' in ParameterDetails}
        if seed is not None and 'random_state' in MethodParameters:
            Parameters['random_state'] = seed

        if OverrideArguments:
            for key, value in OverrideArguments.items():
                Parameters[key] = value

        return Utils.dynamicCall(CallDetails, Parameters)

    @staticmethod
    def dynamicCall(CallDetails, Parameters={}):
        package = CallDetails['package']
        callable = CallDetails['callable']
        loaded_module = importlib.import_module(package)
        return getattr(loaded_module, callable)(**Parameters)

    ''''###############################################################################################'''


    '''
    Parallelization Required Functions 
    '''
    @staticmethod
    def getBestParallelizationLocation(Configurations, BatchContext=False):
        if Config.thread_level != 'best':
            return Config.thread_level

        if BatchContext:
            configurations_grid_search = None
            for Configuration in Configurations:
                if 'config_grid_search' in Configuration and Configuration['config_grid_search'] is True:
                    configurations_grid_search = True
                    break

            config_grid_search = configurations_grid_search if configurations_grid_search is not None else Config.InitialConfigurations['config_grid_search']
            if config_grid_search is True:
                return Utils.THREAD_LEVEL_ALGORITHM

            ExecutionCounts = {
                Utils.THREAD_LEVEL_BATCH: len(Configurations),
                Utils.THREAD_LEVEL_RUN: Configurations[0]['config_seed_count'] if 'config_seed_count' in Configurations[0] else Config.InitialConfigurations['config_seed_count'],
                Utils.THREAD_LEVEL_CV: Configurations[0]['config_cv_fold'] if 'config_cv_fold' in Configurations[0] else Config.InitialConfigurations['config_cv_fold']
            }

            all_algorithms_parallelize = True
            for Configuration in Configurations:
                all_algorithms_parallelize = Utils._validateAlgorithmsParallelization(Configuration)
                if not all_algorithms_parallelize:
                    break
        else:
            if Configurations['config_grid_search'] is True:
                return Utils.THREAD_LEVEL_ALGORITHM

            ExecutionCounts = {
                Utils.THREAD_LEVEL_BATCH: 0,
                Utils.THREAD_LEVEL_RUN: Configurations['config_seed_count'],
                Utils.THREAD_LEVEL_CV: Configurations['config_cv_fold']
            }
            all_algorithms_parallelize = Utils._validateAlgorithmsParallelization(Configurations)

        parallelization_level = Utils.THREAD_LEVEL_ALGORITHM if all_algorithms_parallelize is True else max(ExecutionCounts.items(), key=operator.itemgetter(1))[0]
        return parallelization_level

    @staticmethod
    def _validateAlgorithmsParallelization(AlgorithmConfig):
        if 'proc_classification' in AlgorithmConfig:
            for algorithm in AlgorithmConfig['proc_classification']:
                if 'n_jobs' not in Config.ClassificationAlgorithmsCallDetails[algorithm]['parameters']:
                    return False

        if 'proc_clustering' in AlgorithmConfig:
            for algorithm in AlgorithmConfig['proc_clustering']:
                if 'n_jobs' not in Config.ClusteringAlgorithmsCallDetails[algorithm]['parameters']:
                    return False

        return True


    @staticmethod
    def saveThreadingRequiredProperties():
        ThreadConfigs = {
            'framework_instance': Config.framework_instance,
            'thread_level_': Config.thread_level_ if hasattr(Config, 'thread_level_') else None
        }

        pathlib.Path(Config.framework_temp_path).mkdir(parents=True, exist_ok=True)
        with codecs.open(Config.framework_temp_thread_config_file_path, "w", encoding="utf-8") as File:
            File.writelines(json.dumps(ThreadConfigs, indent=2))

    @staticmethod
    def cleanThreadingRequiredProperties():
        os.remove(Config.framework_temp_thread_config_file_path)

    @staticmethod
    def pool(pool_count, method, arguments):
        Utils.saveThreadingRequiredProperties()

        Pool = multiprocessing.Pool(pool_count, maxtasksperchild=Config.pool_maxtasksperchild)
        if isinstance(arguments[-1], tuple):
            PoolData = Pool.starmap(method, arguments)
        else:
            PoolData = Pool.map(method, arguments)

        Pool.close()
        Pool.join()

        Utils.cleanThreadingRequiredProperties()

        gc.collect()

        return PoolData

    ''''###############################################################################################'''


