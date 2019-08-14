"""
Contains the code for the analysis (exploratory and descriptive) of the data
"""
import collections
import numpy
import pandas
import pandas.tools.plotting
import sklearn
import sklearn.datasets
import sklearn.cluster
import sklearn.tree
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.linear_model
import sklearn.tree
import sklearn.model_selection
import matplotlib.pyplot as plt
import itertools
import math
import seaborn
import pathlib
import matplotlib.font_manager
import os

import propheticus
import propheticus.shared

import sys
sys.path.append(propheticus.Config.framework_selected_instance_path)
if os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, 'InstanceConfig.py')):
    from InstanceConfig import InstanceConfig as Config
else:
    import Config as Config

class DataAnalysis(object):
    """
    Contains the code for the analysis (exploratory and descriptive) of the data

    ...

    Attributes
    ----------
    normalizedDatasets : dict
    reducedDatasets : dict

    Parameters
    ----------
    display_visuals : bool
    datasets : list of str
    configurations_id : str
    description : object
    """
    normalizedDatasets = {}
    reducedDatasets = {}

    def __init__(self, display_visuals, datasets, configurations_id, description):
        self.display_visuals = display_visuals
        self.dataset_name = datasets
        self.configurations_id = configurations_id
        self.description = description

        self.save_items_path = os.path.join(Config.framework_instance_generated_analysis_path, self.dataset_name)


    '''
    Data Analysis
    '''
    def descriptiveAnalysis(self, Dataset):
        """
        Performs a descriptive analysis of the dataset

        Parameters
        ----------
        Dataset : dict

        Returns
        -------

        """
        propheticus.shared.Utils.printStatusMessage('Generating Descriptive Analysis Report')
        Columns = Dataset['headers']
        Target = Dataset['targets']
        Data = Dataset['data']

        DataAnalysis = []
        DataAnalysis.append(['Data Analysis'])
        DataAnalysis.append([self.description])
        DataAnalysis.append([])
        DataAnalysis.append(['Features', 'Standard Deviations:', 'Variance:', 'Averages:', 'Ranges:', 'Correlations with target:'])

        Details = []
        Details.append(Columns)
        # DataAnalysis.append(Columns)

        # DataAnalysis.append([])
        # DataAnalysis.append(['Standard Deviations:'])
        # DataAnalysis.append(["{0:.4f}".format(numpy.std(Measures)) for Measures in numpy.transpose(Data)])
        Details.append(["{0:.4f}".format(numpy.std(Measures)) for Measures in numpy.transpose(Data)])

        # DataAnalysis.append([])
        # DataAnalysis.append(['Variance:'])
        # DataAnalysis.append(["{0:.4f}".format(numpy.var(Measures)) for Measures in numpy.transpose(Data)])
        Details.append(["{0:.4f}".format(numpy.var(Measures)) for Measures in numpy.transpose(Data)])

        # DataAnalysis.append([])
        # DataAnalysis.append(['Averages:'])
        # DataAnalysis.append(["{0:.4f}".format(numpy.mean(Measures)) for Measures in numpy.transpose(Data)])
        Details.append(["{0:.4f}".format(numpy.mean(Measures)) for Measures in numpy.transpose(Data)])

        # DataAnalysis.append([])
        # DataAnalysis.append(['Ranges:'])
        # DataAnalysis.append(["[" + "{0:.1f}".format(min(Measures)) + ", " + "{0:.1f}".format(max(Measures)) + "]" for Measures in numpy.transpose(Data)])
        Details.append(["[" + "{0:.1f}".format(min(Measures)) + ", " + "{0:.1f}".format(max(Measures)) + "]" for Measures in numpy.transpose(Data)])

        SerializedTarget = sklearn.preprocessing.LabelEncoder().fit_transform(Target)

        df = pandas.DataFrame(list(zip(*list(zip(*Data)) + [SerializedTarget])), columns=Columns + ['label'])
        corr = df.corr()  # pearson, kendall, spearman
        if corr.isnull().values.any():
            propheticus.shared.Utils.printErrorMessage('Correlation for the data passed contains NaN values, which will break the correlation analysis. This may be due to features with 0 variance, which will for some reason return NaN')
        else:
            _TargetCorrelations = corr.ix[-1][:-1]
            # propheticus.shared.Utils.printStatusMessage('Features correlation with target')
            TargetCorrelations = _TargetCorrelations[abs(_TargetCorrelations).argsort()[::-1]]

            _Indexes = numpy.argwhere(abs(TargetCorrelations) > Config.max_alert_target_correlation)
            _Correlations = {str(index): str(round(val, 2)) for index, val in TargetCorrelations.items()}  # For some reason directly getting the value sometimes returns series with 2 values instead of direct correlation error
            if len(_Indexes) > 0:
                HighCorrelations = []
                for index in _Indexes:
                    HighCorrelations.append(str(round(TargetCorrelations[index], 2).to_dict()))
                propheticus.shared.Utils.printWarningMessage('Some features have correlations with target too high: \n' + ', '.join(sorted(HighCorrelations)))

            # print(TargetCorrelations)

            # DataAnalysis.append([])
            # DataAnalysis.append(['Correlations with target:'])
            # DataAnalysis.append(["\n".join([str(index) + ': ' + str(val) for index, val in TargetCorrelations.iteritems()])])
            # Details.append(["\n".join([str(index) + ': ' + str(val) for index, val in TargetCorrelations.iteritems()])])

            Details.append([_Correlations[col] for col in Columns])

        Details = numpy.transpose(Details)
        DataAnalysis += Details.tolist()

        propheticus.shared.Utils.saveExcel(self.save_items_path, self.configurations_id + ".analysis.xlsx", DataAnalysis)

        # for line in DataAnalysis:
        #     print(line)

    '''
        Box Plots
    '''
    def boxPlots(self, Dataset):
        """
        Plots the box plots corresponding to the dataset

        Parameters
        ----------
        Dataset : dict

        Returns
        -------

        """
        propheticus.shared.Utils.printStatusMessage('Generating Features Box Plot')

        Data = Dataset['data']
        Headers = Dataset['headers']

        df = pandas.DataFrame(Data, columns=Headers)
        ax = df.plot.box(showfliers=True, sym='k.')  # NOTE: sym property is required possibly do to the use of the seaborn, which ignores the showFliers. Analyse

        # plt.suptitle(('Boxplot: ' + dataset_name), fontsize=14, fontweight='bold')
        # plt.suptitle(('Feature Distribution'), fon    tsize=14, fontweight='bold')

        plt.title('Features Distribution', fontsize=14)

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=40)
        plt.subplots_adjust(bottom=0.25)

        if Config.publication_format is False:
            plt.figtext(.02, .02, self.description, size='xx-small')

        propheticus.shared.Utils.saveImage(self.save_items_path, self.configurations_id + ".data_analysis_boxplot.png")

        if self.display_visuals is True:
            propheticus.shared.Utils.showImage()

        plt.close()

    '''
        Bar Plot
    '''
    def barPlot(self, Dataset):
        """
        Plots the bar plot corresponding to the dataset

        Parameters
        ----------
        Dataset : dict

        Returns
        -------

        """
        propheticus.shared.Utils.printStatusMessage('Generating Class Distribution Plot')

        Target = Dataset['targets']
        Classes = Dataset['classes']

        DataCount = collections.Counter(Target)
        Data = [DataCount[target_class] for target_class in Classes]
        ind = numpy.arange(len(Classes))

        ax = plt.subplot(111)
        ax.bar(ind + 0.02, Data, width=0.95, color=["#" + propheticus.shared.Utils.ColourValues[i] for i in range(len(Classes))])
        ax.set_xticks(ind)
        ax.set_xticklabels(Classes)

        # plt.xticks(rotation=40)

        rects = ax.patches
        for rect, label in zip(rects, Data):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

        # plt.gcf().subplots_adjust(bottom=0.18)

        # plt.suptitle(('Class Distribution: ' + dataset_name), fontsize=14, fontweight='bold')
        # plt.suptitle(('Class Distribution'), fontsize=14, fontweight='bold')
        plt.title('Class Distribution', fontsize=14)

        if Config.publication_format is False:
            plt.figtext(.02, .02, self.description, size='xx-small')

        propheticus.shared.Utils.saveImage(self.save_items_path, self.configurations_id + ".data_analysis_class_distribution.png")

        if self.display_visuals is True:
            propheticus.shared.Utils.showImage()

        plt.close()

    '''
        Scatter Plot
    '''
    def scatterPlotMatrix(self, Dataset):
        """
        Plots the scatter plot matrix corresponding to the dataset

        Parameters
        ----------
        Dataset : dict

        Returns
        -------

        """
        propheticus.shared.Utils.printStatusMessage('Generating Features Scatter Plot')

        Headers = Dataset['headers']
        Data = Dataset['data']
        if len(Data[0]) > 20:
            propheticus.shared.Utils.printErrorMessage('Probably too many features... ?')
            return

        df = pandas.DataFrame(Data, columns=Headers)
        axes = pandas.plotting.scatter_matrix(df, alpha=0.5, diagonal='hist')
        corr = df.corr().as_matrix()

        n = len(df.columns)
        for x in range(n):
            for y in range(n):
                ax = axes[x, y]
                ax.xaxis.label.set_rotation(25)
                ax.yaxis.label.set_rotation(25)
                ax.yaxis.labelpad = 20

        # Correlation factor
        for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
            axes[i, j].annotate("%.3f" % corr[i, j], (0.35, 0.8), xycoords='axes fraction', ha='center', va='center', color="black")

        plt.subplots_adjust(left=0.2, bottom=0.20)
        plt.xticks(rotation=40)

        # plt.suptitle(("Scatter Plot and Correlation: " + dataset_name), fontsize=14, fontweight='bold')
        # plt.suptitle(("Scatter Plot and Correlation"), fontsize=14, fontweight='bold')

        if Config.publication_format is False:
            plt.figtext(.02, .02, self.description, size='xx-small')

        propheticus.shared.Utils.saveImage(self.save_items_path, self.configurations_id + ".data_analysis_scatter_plot.png")

        if self.display_visuals is True:
            propheticus.shared.Utils.showImage()
        plt.close()

    '''
        Correlation Matrix Plot
    '''
    def correlationMatrixPlot(self, Dataset):
        """
        Plots the correlation matrix plot corresponding to the dataset
        Parameters
        ----------
        Dataset : dict

        Returns
        -------

        """
        propheticus.shared.Utils.printStatusMessage('Generating Features Correlation Plot')

        seaborn.set(style="white")

        Headers = Dataset['headers']
        Data = Dataset['data']
        # if len(Data[0]) > 50:
        #     propheticus.shared.Utils.printErrorMessage('Probably too many features... ?')
        #     return

        df = pandas.DataFrame(Data, columns=Headers)
        # axes = pandas.plotting.scatter_matrix(df, alpha=0.5, diagonal='hist')
        # corr = df.corr().as_matrix()
        corr = df.corr()

        corr_with_target = corr.ix[-1][:-1]
        # attributes sorted from the most propheticus
        # propheticus.shared.Utils.printStatusMessage('Features correlation with target')

        # print(corr)

        mask = numpy.zeros_like(corr, dtype=numpy.bool)
        mask[numpy.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = seaborn.diverging_palette(220, 10, as_cmap=True)

        plt.xticks(rotation=40)

        # Draw the heatmap with the mask and correct aspect ratio
        axes = seaborn.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            xticklabels=True,
            yticklabels=True
        )


        if Config.publication_format is False:
            plt.figtext(.02, .02, self.description, size='xx-small')

        plt.title('Correlation Heat Map', fontsize=14)

        plt.yticks(rotation=0)
        # print(axes.get_xticklabels())
        # print(axes.get_yticklabels())

        # ax.yaxis.label.set_rotation(25)

        propheticus.shared.Utils.saveImage(self.save_items_path, self.configurations_id + ".data_analysis_correlation_plot.png")

        if self.display_visuals is True:
            propheticus.shared.Utils.showImage()

        plt.close()

    '''
    Time series analysis
    '''
    def timeSeriesStd(self, Dataset):
        """
        Performs the time series analysis

        Parameters
        ----------
        Dataset : dict

        Returns
        -------

        """
        self.timeSeries(Dataset, True)
    '''
    Time series analysis
    '''
    def timeSeries(self, Dataset, std=False):
        """
        Plots the time series analysis corresponding to the dataset

        Parameters
        ----------
        Dataset : dict
        std : bool, optional
            The default is False

        Returns
        -------

        """
        propheticus.shared.Utils.printStatusMessage('Generating Class Timeseries Plot')

        Headers = Dataset['headers']
        if Headers[0] != 'Index':
            propheticus.shared.Utils.printErrorMessage('The index of the run sample is not present. To draw a timeseries this is required')
            return
        elif len(Headers) == 1:
            propheticus.shared.Utils.printErrorMessage('Only the index feature is present. At least one more is required. Other selected features may have been removed when exporting.')
            return

        Descriptions = Dataset['descriptions']
        Data = Dataset['data']
        Target = Dataset['targets']
        Classes = Dataset['classes']

        # NOTE: the averages for each time step may not be from the same number of samples, as some runs may have failure and thus stop;

        Iterator = range(1, len(Headers))

        ncols = 4
        nrows = 3
        if ncols > len(Iterator):
            ncols = len(Iterator)
            nrows = 1
        elif nrows > math.ceil(len(Iterator) / ncols):
            nrows = math.ceil(len(Iterator) / ncols)

        img_row = -1
        figure_index = 1
        # plt.figure(figure_index)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

        if Config.publication_format is False:
            plt.figtext(.02, .02, self.description, size='xx-small')

        # plt.get_current_fig_manager().window.showMaximized()  # NOTE: this forces plot to showup fullscreen

        file_path = os.path.join(Config.OS_PATH, self.save_items_path)
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

        # TODO: analyze possible optimizations
        # TODO: possibly consider the display of the failures in time? too many though?
        # TODO: if sliding window > 1 it makes no sense

        baseline_description = propheticus.shared.Utils.getClassDescriptionById(Config.ClassesMapping['Baseline'])
        ClassByRun = {}
        for index, Item in enumerate(Data):
            run = Descriptions[index]
            target_class = Target[index]

            if run not in ClassByRun or target_class != baseline_description:
                ClassByRun[run] = target_class

        SupportByClass = collections.Counter([target_class for run, target_class in ClassByRun.items()])
        PlotClasses = [target_class + ' (' + str(SupportByClass[target_class]) + ')' for target_class in Classes]
        img_count = 0
        for feature_index in Iterator:
            img_count += 1
            DataByClasses = {target_class: {} for target_class in Classes}
            StdByClasses = {target_class: {} for target_class in Classes}
            for index, Item in enumerate(Data):
                target_class = ClassByRun[Descriptions[index]]

                record_index = Item[0]
                if record_index not in DataByClasses[target_class]:
                    DataByClasses[target_class][record_index] = []

                DataByClasses[target_class][record_index].append(Item[feature_index])

            for target_class, Steps in DataByClasses.items():
                DataByClasses[target_class] = [numpy.mean(Items) for record_index, Items in sorted(Steps.items())]
                StdByClasses[target_class] = [numpy.std(Items) for record_index, Items in sorted(Steps.items())]

            BaselineData = None
            BaselineStdData = None
            DataFrameData = []
            StdFrameData = []
            for target_class, Averages in DataByClasses.items():
                if target_class == baseline_description:
                    BaselineData = Averages
                    BaselineStdData = StdByClasses[target_class]
                    continue

                StdFrameData.append(StdByClasses[target_class])
                DataFrameData.append(Averages)

            max_length = len(max(DataFrameData, key=len)) + 10
            StdFrameData.insert(0, BaselineStdData[:max_length])  # NOTE: this truncates the control data to the largest of the other classes
            StdFrameData = [Items + [0] * (max_length - len(Items)) for Items in StdFrameData]
            DataFrameData.insert(0, BaselineData[:max_length])  # NOTE: this truncates the control data to the largest of the other classes
            DataFrameData = list(itertools.zip_longest(*DataFrameData))  # NOTE: this handles multiple lenght sublists; replaces non-existing numbers with None, which do not appear in the plot
            df = pandas.DataFrame(DataFrameData, index=[i for i in range(len(DataFrameData))], columns=PlotClasses)
            # plt.xticks(numpy.arange(len(df.index.values)), df.index.values)

            color = ["#" + propheticus.shared.Utils.ColourValues[i] for i in range(len(Classes))] if len(Classes) > 1 else "#" + propheticus.shared.Utils.ColourValues[0]

            _feature_index = img_count - 1
            img_row += 1 if not _feature_index % ncols else 0
            if nrows == 1 and ncols == 1:
                _ax = axes
            else:
                _ax = axes[img_row, _feature_index % ncols] if nrows > 1 else axes[_feature_index % ncols]

            if std is True:
                ax = df.plot(ax=_ax, color=color, yerr=StdFrameData,  capsize=1, capthick=1, elinewidth=0.01)
            else:
                ax = df.plot(ax=_ax, color=color)

            #plt.xticks(numpy.arange(len(df.index.values)), df.index.values, rotation=40)

            #ax.legend(bbox_to_anchor=(1.25, 1.02))
            #ax.set_xticklabels(df.index.values)
            # plt.subplots_adjust(right=0.81, bottom=0.2)

            # plt.suptitle(('Features Means By Class: ' + dataset_name), fontsize=14, fontweight='bold')
            ax.set_title((Headers[feature_index]), fontsize=10)

            LengendFont = matplotlib.font_manager.FontProperties()
            LengendFont.set_size('small')
            # lgd = ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1), prop=LengendFont)
            lgd = ax.legend(prop=LengendFont)

            if img_row + 1 == nrows and not img_count % ncols and (img_count + 1) < len(Iterator):
                propheticus.shared.Utils.saveImage(
                    self.save_items_path,
                    self.configurations_id + '.' + str(figure_index) + ".data_analysis_timeseries.png",
                    additional_artists=[lgd]
                )

                img_row = -1
                figure_index += 1
                # plt.figure(figure_index)
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

                if Config.publication_format is False:
                    plt.figtext(.02, .02, self.description, size='xx-small')

        propheticus.shared.Utils.saveImage(
            self.save_items_path,
            self.configurations_id + '.' + str(figure_index) + '.' + str(int(std)) + ".data_analysis_timeseries.png",
            additional_artists=[lgd]
        )

        if self.display_visuals is True:
            propheticus.shared.Utils.showImage()

        plt.close()



    '''
        Line Graphs with Standard Deviation
    '''
    def lineGraphsStd(self, Dataset):
        """
        Plots line graphs with standard deviation

        Parameters
        ----------
        Dataset : dict

        Returns
        -------

        """
        self.lineGraphs(Dataset, True)

    '''
        Line Graphs
    '''
    def lineGraphs(self, Dataset, std=False):
        """
        Plots line graphs with standard deviation corresponding to the dataset

        Parameters
        ----------
        Dataset : dict
        std : bool, optional
            The default is False

        Returns
        -------

        """
        propheticus.shared.Utils.printStatusMessage('Generating Class Features Line Plot')

        Headers = Dataset['headers']
        Data = Dataset['data']
        Target = Dataset['targets']
        Classes = Dataset['classes']

        DataByClasses = {target_class: [] for target_class in Classes}
        StdByClasses = {target_class: [] for target_class in Classes}

        for index, Item in enumerate(Data):
            target_class = Target[index]
            DataByClasses[target_class].append(Item)

        for target_class, Items in DataByClasses.items():
            DataByClasses[target_class] = [numpy.mean(Measures) for Measures in numpy.transpose(Items)]
            StdByClasses[target_class] = [numpy.std(Measures) for Measures in numpy.transpose(Items)]

        DataFrameData = []
        StdDataFrameData = []
        for target_class, Averages in DataByClasses.items():
            DataFrameData.append(Averages)
            StdDataFrameData.append(StdByClasses[target_class])

        temp = numpy.transpose(DataFrameData)
        df = pandas.DataFrame(numpy.transpose(DataFrameData), index=Headers, columns=Classes)
        # plt.xticks(numpy.arange(len(df.index.values)), df.index.values)

        styles = []
        for col in df.columns:
            style = '-'
            styles.append(style)

        color = ["#" + propheticus.shared.Utils.ColourValues[i] for i in range(len(Classes))] if len(Classes) > 1 else "#" + propheticus.shared.Utils.ColourValues[0]
        if std is True:
            # TODO: validate if StdDataFrameData should be transposed?
            ax = df.plot(color=color, yerr=StdDataFrameData, ls='--', marker='o', capsize=8, capthick=2, elinewidth=0.01)
        else:
            ax = df.plot(style=styles, color=color)

        plt.xticks(numpy.arange(len(df.index.values)), df.index.values, rotation=40)

        #ax.legend(bbox_to_anchor=(1.25, 1.02))
        #ax.set_xticklabels(df.index.values)
        plt.subplots_adjust(right=0.81, bottom=0.2)

        # plt.suptitle(('Features Means By Class: ' + dataset_name), fontsize=14, fontweight='bold')
        # plt.suptitle(('Features Means By Class'), fontsize=14, fontweight='bold')
        plt.title('Features Means By Class', fontsize=14)

        LengendFont = matplotlib.font_manager.FontProperties()
        LengendFont.set_size('small')
        # lgd = ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1), prop=LengendFont)
        lgd = ax.legend(prop=LengendFont)

        if Config.publication_format is False:
            plt.figtext(.02, .02, self.description, size='xx-small')

        # plt.savefig(file_path + "/" + self.configurations_id + '.' + str(int(std)) + ".data_analysis_line_graphs.png", additional_artists=[lgd])  # TODO: estava , bbox_inches="tight"

        propheticus.shared.Utils.saveImage(
            self.save_items_path,
            self.configurations_id + '.' + str(int(std)) + ".data_analysis_line_graphs.png",
            additional_artists=[lgd]
        )

        if self.display_visuals is True:
            propheticus.shared.Utils.showImage()

        plt.close()

    '''
        Parallel Coordinates
    '''
    def parallelCoordinates(self, Dataset):
        """
        Plots the parallel coordinates plot corresponding to the dataset

        Parameters
        ----------
        Dataset : dict

        Returns
        -------

        """
        propheticus.shared.Utils.printStatusMessage('Generating Class Features Parallel Plot')

        Data = Dataset['data']
        Headers = Dataset['headers']
        Target = Dataset['targets']
        Classes = Dataset['classes']

        df = pandas.DataFrame(Data, columns=Headers)
        df['classes'] = Target
        ax = pandas.tools.plotting.parallel_coordinates(df, 'classes')
        # plt.suptitle(('Parallel Composition: ' + dataset_name), fontsize=14, fontweight='bold')

        #ax.legend(bbox_to_anchor=(1.25, 1.02))
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=30)
        plt.subplots_adjust(right=0.81, bottom=0.2)

        LengendFont = matplotlib.font_manager.FontProperties()
        LengendFont.set_size('small')
        lgd = ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1), prop=LengendFont)

        if Config.publication_format is False:
            plt.figtext(.02, .02, self.description, size='xx-small')

        propheticus.shared.Utils.saveImage(
            self.save_items_path,
            self.configurations_id + ".data_analysis_parallel_coordinates.png",
            additional_artists=[lgd],
            bbox_inches="tight"
        )

        if self.display_visuals is True:
            propheticus.shared.Utils.showImage()

        plt.close()
