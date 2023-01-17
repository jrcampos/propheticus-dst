"""
Contains all the workflow logic to handle the clustering tasks
"""
import numpy
import pandas
import random
import pandas.plotting
import sklearn.cluster
import sklearn.metrics
import sklearn.neighbors
import operator
import scipy.cluster.hierarchy
import sklearn.mixture
import scipy.spatial.distance
import collections
import matplotlib.cm
import matplotlib.pyplot as plt
import scipy
import time
import os
import matplotlib.font_manager
import pathlib
import copy
import importlib
import warnings

import propheticus
import propheticus.core
import propheticus.shared

import sys
sys.path.append(propheticus.Config.framework_selected_instance_path)
if os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, 'InstanceConfig.py')):
    from InstanceConfig import InstanceConfig as Config
else:
    import propheticus.Config as Config

class Clustering(object):
    """
    Contains all the workflow logic to handle the clustering tasks

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
    """
    # def __init__(self, Context, dataset_name, configurations_id, description, display_visuals, balance_data, reduce_dimensionality, normalize, seed_count):
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
        self.seed_count = kwargs['seed_count']
        self.mode = kwargs['mode']
        self.positive_classes = kwargs['positive_classes']

        self.grid_search = kwargs['grid_search']

        OptionalArguments = {'display_logs': True}
        for key, value in OptionalArguments.items():
            setattr(self, key, kwargs[key] if key in kwargs else value)

        self.save_items_path = os.path.join(Config.framework_instance_generated_clustering_path, self.dataset_name)

        self.silhouette_threshold = 5000

    def runModel(self, algorithm, Dataset, Parameters=False, GridSearchParameters=False):
        """
        Runs defined models for passed configurations

        Parameters
        ----------
        algorithm : str
        Dataset : dict

        Returns
        -------

        """

        self.algorithms_parameters = Parameters
        self.grid_search_parameters = GridSearchParameters

        file_path = os.path.join(Config.OS_PATH, self.save_items_path)
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

        # pool_count = min(Config.max_thread_count, self.cv_fold)
        # if self.mode == 'cli':
        #     propheticus.shared.Utils.printStatusMessage('Parallelizing threads #: ' + str(pool_count))

        for seed in propheticus.shared.Utils.RandomSeeds[:self.seed_count]:
            numpy.random.seed(seed)
            random.seed(seed)

            Headers = Dataset['headers']
            Data = Dataset['data']
            Target = Dataset['targets']
            Classes = Dataset['classes']
            Descriptions = numpy.array(Dataset['descriptions'])

            if self.reduce_dimensionality:
                if 'variance' in self.reduce_dimensionality:
                    CallArguments = copy.deepcopy(self.dim_red_parameters['variance']) if 'variance' in self.dim_red_parameters else {}
                    CallDetails = Config.DimensionalityReductionCallDetails['variance']
                    Estimator = propheticus.shared.Utils.dynamicAPICall(CallDetails, CallArguments, seed=seed)
                    Data = Estimator.fit_transform(Data)
                    Headers = propheticus.core.DatasetReduction.removeFeaturesFromHeaders(Estimator.indexes_, Headers)

                if len(self.reduce_dimensionality) > 1 or self.reduce_dimensionality[0] != 'variance':
                    Data, RemoveFeatures, Headers, _ = propheticus.core.DatasetReduction.dimensionalityReduction(self.dataset_name, self.configurations_id, self.description, self.reduce_dimensionality, Data, Target, Headers, self.dim_red_parameters, seed)

            if self.balance_data and len(self.balance_data) > 0 and 'custom' not in algorithm:
                Data, Target, BalanceDataEstimators = propheticus.core.DatasetReduction.balanceDataset(Data, Target, self.sampling_parameters, seed=seed, method=self.balance_data)
                for Estimator in BalanceDataEstimators.values():
                    Descriptions = Descriptions[Estimator.sample_indices_]

            CallArguments = copy.deepcopy(self.algorithms_parameters) if self.algorithms_parameters else {}
            AlgorithmCallDetails = Config.ClusteringAlgorithmsCallDetails[algorithm]

            if Config.thread_level_ == propheticus.shared.Utils.THREAD_LEVEL_ALGORITHM and 'n_jobs' in AlgorithmCallDetails['parameters']:
                CallArguments['n_jobs'] = -1

            dist = 'cityblock'

            if self.grid_search is not False:
                if not self.grid_search_parameters and 'grid' in Config.ClusteringAlgorithmsCallDetails[algorithm]:
                    self.grid_search_parameters = Config.ClusteringAlgorithmsCallDetails[algorithm]['grid']

                '''
                If parameters are sent through attribute self.grid_search_parameters those will be used for grid-search;
                otherwise the grid-search parameters from algorithms configuration will be used.
                '''
                if self.algorithms_parameters:
                    propheticus.shared.Utils.printWarningMessage('Values in algorithms_parameters may be overwritten by grid-search hyperparameters')

                if not self.grid_search_parameters:
                    propheticus.shared.Utils.printErrorMessage('No grid search parameters were passed or are defined in the configs for algorithm ' + algorithm + '. Skipping grid search')
                else:
                    cluster_arguments = []
                    cluster_model = []
                    silhouettes = []

                    # TODO: save results for the different parameters?
                    for GridSearchParametersGroups in self.grid_search_parameters:
                        GridSearchParameters = propheticus.shared.Utils.cartesianProductDictionaryLists(**GridSearchParametersGroups)

                        for CallArguments in GridSearchParameters:
                            model = propheticus.shared.Utils.dynamicAPICall(AlgorithmCallDetails, CallArguments, seed)
                            model.fit(Data)
                            Labels = model.labels_

                            if len(Labels) > self.silhouette_threshold:
                                propheticus.shared.Utils.printWarningMessage('Silhouette will not be computed due to computation time, other metric will be used for grid-search')
                                metric_avg = sklearn.metrics.silhouette_score(Data, Labels, metric=dist)
                            else:
                                metric_avg = sklearn.metrics.homogeneity_score(Target, Labels)


                            # TODO: gaussian mixture used this
                            # lowest_bic = numpy.infty
                            # Lowest = {'n_components': None, 'covariance_type': None}
                            # bic = []
                            # model.fit(Data)
                            # bic.append(model.bic(Data))
                            # if bic[-1] < lowest_bic:
                            #     lowest_bic = bic[-1]
                            #     best_gmm = model
                            #     Lowest['n_components'] = n_components
                            #     Lowest['covariance_type'] = cv_type



                            silhouettes.append(metric_avg)

                            cluster_arguments.append(CallArguments)
                            cluster_model.append(model)

                    if len(silhouettes) == 0:
                        propheticus.shared.Utils.printFatalMessage('No configurations were run')

                    max_silhouette_cluster = silhouettes.index(max(silhouettes))
                    CallArguments = cluster_arguments[max_silhouette_cluster]
                    model = cluster_model[max_silhouette_cluster]
                    Clusters = list(model.labels_)
            else:
                model = propheticus.shared.Utils.dynamicAPICall(AlgorithmCallDetails, CallArguments, seed)
                model.fit(Data)
                Clusters = list(model.labels_)

            _return = self.plotClusterDistribution(algorithm, "", Clusters, Data, Target, Classes, Descriptions, algorithm, 0)
            if _return == -1:
                return

            if self.display_visuals is True:
                propheticus.shared.Utils.showImage()
            plt.close()

            self.logResults(algorithm, model.labels_, Data, Target, seed)

            if 'callback' in AlgorithmCallDetails:
                DynamicCallArguments = {
                    'Context': self,
                    'Data': Data,
                    'Target': Target,
                    'Parameters': CallArguments
                }
                propheticus.shared.Utils.dynamicCall(AlgorithmCallDetails['callback'], DynamicCallArguments)




    def plotClusterDistribution(self, base_suptitle, base_title, Clusters, Data, Target, Classes, Descriptions, algorithm, base_index=0, base_filename=''):
        """
        Plots the cluster distribution for passed configurations

        Parameters
        ----------
        base_suptitle : str
        base_title : str
        Clusters : list of int
        Data : list of list of float
            Currently this parameter is not used, however it is required
        Target : list of str
        Classes : dict
        algorithm : str
        base_index : int, optional
            The default is 0
        base_filename : str, optional
            The default is empty str

        Returns
        -------

        """
        ClustersByTarget = dict((target, []) for target in Classes)
        ItemsByCluster = [dict((target, 0) for target in Classes) for i in range(max(Clusters) + numpy.abs(base_index-1))]
        DescriptionsByCluster = [[] for i in range(max(Clusters) + numpy.abs(base_index-1))]

        if len(ItemsByCluster) == 0:
            print("!!! Error! No cluster created!!")
            return -1

        noise_count = 0
        for i in range(len(Clusters)):
            _cluster = Clusters[i] - base_index
            if _cluster == -1:
                # print('Index: ' + str(i) + '; Invalid Cluster -1: Noise')
                noise_count += 1
                continue

            target = Target[i]
            ItemsByCluster[_cluster][target] += 1
            DescriptionsByCluster[_cluster].append(target + '=>' + Descriptions[i].split('v1-NGcpu-')[1])  # TODO: REMOVE THIS SPLI!

        for i in range(max(Clusters) + numpy.abs(base_index-1)):
            #print("\n\nCluster " + str(i+1))

            ItemsByCluster[i] = collections.OrderedDict(sorted(ItemsByCluster[i].items()))
            Items = sorted(ItemsByCluster[i].items(), key=operator.itemgetter(1))
            for key, value in Items:
                #print(key + " => " + str(value))
                if value > 0:
                    ClustersByTarget[key].append(str(i+1) + ":" + str(value))

        save_path = os.path.join(self.save_items_path, algorithm)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        predictions_file_path = os.path.join(save_path, self.generated_files_base_name + '_clusters.txt')
        with open(predictions_file_path, "w", encoding="utf-8") as File:
            File.writelines("\n".join([f'Cluster {i} => {sorted(collections.Counter(DescriptionsByCluster[i]).items())}' for i in range(max(Clusters) + numpy.abs(base_index-1))]) + '\n')

        Data = [[value for key, value in Item.items()] for Item in ItemsByCluster]
        # Columns = [target + " " + "|".join(ClustersByTarget[target]) for target, Values in ItemsByCluster[0].items()]
        Columns = [target for target, Values in ItemsByCluster[0].items()]

        df = pandas.DataFrame(Data, index=["Cluster " + str(i+1) + "-" for i in range(max(Clusters)+numpy.abs(base_index-1))], columns=Columns)

        ax = df.plot.bar(fontsize=10, color=["#" + propheticus.shared.Utils.ColourValues[i] for i in range(len(Columns))])
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=80)
        #ax.get_figure().set_size_inches(len(Columns)*3, 12)
        # plt.suptitle((base_suptitle), fontsize=14, fontweight='bold')

        #plt.title(base_suptitle + ' ' + base_title, fontsize=14)
        plt.title(base_suptitle + base_title, fontsize=14)

        # ax.set_title(base_title)

        if noise_count > 0:
            plt.title('Noise count: ' + str(noise_count))

        plt.ylabel('Count')
        # plt.subplots_adjust(bottom=0.15, right=0.65)
        LengendFont = matplotlib.font_manager.FontProperties()
        LengendFont.set_size('small')
        lgd = ax.legend(prop=LengendFont)
        # lgd = ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1), prop=LengendFont)
        # ax.legend(bbox_to_anchor=(1.60, 1.02), prop=LengendFont)

        if Config.publication_format is False or Config.force_configurations_log is True:
            if Config.force_configurations_log is True:
                plt.annotate(self.description, (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')
            else:
                plt.figtext(.02, .02, self.description, size='xx-small')
        else:
            plt.tight_layout()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # NOTE: can be used like this, but may suppress relevant warnings

            ts = time.time()
            propheticus.shared.Utils.saveImage(os.path.join(self.save_items_path, algorithm), self.generated_files_base_name + "_distribution_" + str(base_filename) + ".png", additional_artists=[lgd], bbox_inches="tight", dpi=150)

            if self.display_visuals is True:
                propheticus.shared.Utils.showImage()

        plt.close()

    def logResults(self, algorithm, Labels, Data, Target, seed):
        """
        Logs results into a dictionary

        Parameters
        ----------
        algorithm : str
        model : object
        Data : list of list of float
        Target : list of str
        Dataset : str
            Currently this parameter is not used, however it is required
        metric : str, optional
            The default is euclidean
        Returns
        -------
        dict
        """
        # Data = propheticus.shared.Utils.getValuesFromData(Dataset)
        # Target = propheticus.shared.Utils.getTargetFromData(Dataset)

        Results = {}

        Clusters = collections.Counter(sorted(Labels))
        if None in Labels.tolist() or len(Clusters) <= 1:
            return False

        Results['clusters'] = len(set(Labels)) - (1 if -1 in Labels else 0)

        ClusteringMetrics = Config.ClusteringPerformanceMetrics
        for metric, MetricCallDetails in ClusteringMetrics.items():
            if metric == 'silhouette' and len(Labels) > self.silhouette_threshold:
                propheticus.shared.Utils.printWarningMessage('Silhouette will not be computed due to computation time')
                Results['silhouette'] = -1
                continue

            CallArguments = {}

            if 'use_data_as_truth' in MetricCallDetails and MetricCallDetails['use_data_as_truth']:
                CallArguments['X'] = Data
                CallArguments['labels'] = Labels
            else:
                CallArguments['labels_true'] = Target
                CallArguments['labels_pred'] = Labels

            Results[metric] = propheticus.shared.Utils.dynamicAPICall(MetricCallDetails, CallArguments)

        self.Context.ClusteringAlgorithmsResults[algorithm].append(Results)
        return Results



            # def optimizeSilhouette(self, algorithm, base_title, Data, Target, Classes, create_model, num_clusters, dist='euclidean', log=True):
    #     """
    #
    #     Parameters
    #     ----------
    #     algorithm : str
    #     base_title : str
    #     Data : list of list of float
    #     Target : list of str
    #     Classes : dict
    #     create_model : object
    #     num_clusters : int
    #     dist : str, optional
    #         The default is euclidean
    #     log : bool, optional
    #         The default is True
    #
    #     Returns
    #     -------
    #     list of str
    #     list of float
    #     """
    #     cluster_labels = []
    #     cluster_model = []
    #     silhouettes = []
    #     for n_clusters in num_clusters:
    #         clustering = create_model(n_clusters)
    #         clustering.fit(Data)
    #         labels = clustering.labels_
    #
    #         if log is True:
    #             returned = self.logResults(algorithm, clustering.labels_, Data, Target, metric=dist)
    #
    #             if returned is False:
    #                 print("Algorithm was returning False: " + algorithm)
    #                 continue
    #
    #         silhouette_avg = sklearn.metrics.silhouette_score(Data, labels, metric=dist)
    #         silhouettes.append(silhouette_avg)
    #
    #         sample_silhouette_values = sklearn.metrics.silhouette_samples(Data, labels)
    #         cluster_labels.append(list(labels))
    #         cluster_model.append(clustering)
    #
    #         if log is True:
    #             self.plotClusterDistribution(algorithm, "", list(labels), Data, Target, Classes, algorithm, 0, "cluster_" + str(n_clusters))
    #
    #         if log is True:
    #             min_range = -0.65
    #             if min(sample_silhouette_values) < min_range:
    #                 min_range = min(sample_silhouette_values)
    #                 print('ERROR - Silhouette Min range is truncating!! ' + str(min_range))
    #             max_range = 0.7
    #
    #             fig, ax1 = plt.subplots(1, 1)
    #             #fig.set_size_inches(18, 7)
    #             ax1.set_xlim([min_range, max_range])
    #             ax1.set_ylim([0, len(Data) + (n_clusters + 1) * 10])
    #             y_lower = 10
    #             for i in range(n_clusters):
    #                 ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
    #                 ith_cluster_silhouette_values.sort()
    #
    #                 size_cluster_i = ith_cluster_silhouette_values.shape[0]
    #                 y_upper = y_lower + size_cluster_i
    #
    #                 color = matplotlib.cm.spectral(float(i) / n_clusters)
    #                 ax1.fill_betweenx(numpy.arange(y_lower, y_upper),
    #                                   0, ith_cluster_silhouette_values,
    #                                   facecolor=color, edgecolor=color, alpha=0.7)
    #
    #                 ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    #                 y_lower = y_upper + 10  # 10 for the 0 samples
    #
    #             ax1.set_title(base_title + " " + dist)
    #             ax1.set_xlabel("The silhouette coefficient values")
    #             ax1.set_ylabel("Cluster label")
    #             ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    #             ax1.set_yticks([])
    #             ax1.set_xticks(numpy.arange(min_range, max_range, 0.1))
    #             plt.suptitle(("Silhouette:  " + algorithm + " #" + str(n_clusters)),
    #                          fontsize=14, fontweight='bold')
    #
    #             propheticus.shared.Utils.saveImage(os.path.join(self.save_items_path, algorithm), self.generated_files_base_name + "_" + str(n_clusters) + ".silhouette.png")
    #             if self.display_visuals is True:
    #                 propheticus.shared.Utils.showImage()
    #
    #             plt.close()
    #
    #     if len(silhouettes) == 0:
    #         return False, False
    #
    #     max_silhouette_cluster = silhouettes.index(max(silhouettes))
    #     Clusters = cluster_labels[max_silhouette_cluster]
    #
    #     return Clusters, cluster_model[max_silhouette_cluster]



def DBSCAN(Context, Data, Target, Parameters):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) data clustering algorithm

    Parameters
    ----------
    Data : list of list of float
    Target : list of str
    Dataset : dict
    seed : int
        Currently this parameter is not used, however it is required

    Returns
    -------

    """
    metric = 'euclidean'
    min_samples = 4
    kNN = sklearn.neighbors.NearestNeighbors(n_neighbors=min_samples, metric=metric).fit(Data)
    Distances, Indices = kNN.kneighbors(Data)
    KDistances = sorted(Distances[:, -1])

    plt.plot(KDistances)
    plt.ylabel('Distance')
    plt.xlabel('Points')
    plt.suptitle((Context.dataset_name + " - k-th (4) Distance to Nearest Neighbor"), fontsize=14, fontweight='bold')

    propheticus.shared.Utils.saveImage(os.path.join(Context.save_items_path, 'dbscan'), Context.generated_files_base_name + "_kth_distance." + metric + ".png")

    if Context.display_visuals is True:
        propheticus.shared.Utils.showImage()

    plt.close()

''' TODO: get this working!
def KMeans(Context, Data, Target, Parameters):
    reduced_data = sklearn.decomposition.PCA(n_components=2, random_state=seed).fit_transform(Data)
    Clusters, model = self.optimizeSilhouette(algorithm, "", reduced_data, Target, Dataset, self.createKMeans(seed), range(2, int(len(Dataset['classes'])*2)), 'cityblock', False)

    _return = self.plotClusterDistribution("PCA KMeans", "", Clusters, Data, Target, Dataset, algorithm, 0, "pca")

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))

    temp1 = numpy.arange(x_min, x_max, h)
    temp2 = numpy.arange(y_min, y_max, h)

    # Obtain labels for each point in mesh. Use last trained model.
    Z = model.predict(numpy.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(2)
    plt.clf()
    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
    plt.title('K-means clustering (PCA-reduced data)')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(Config.OS_PATH + self.save_items_path + "/" + algorithm + "/alg_" + algorithm + "_pca_projection.png")
    if self.display_visuals is True:
        propheticus.shared.Utils.showImage()

    plt.close()
'''

def hierarchical_clustering(Context, Data, Target, Parameters):
    """
    Hierarchichal Clustering

    Parameters
    ----------
    Data : list of list of float
    Target : list of str
    Dataset : dict
    seed : int
        Currently this parameter is not used, however it is required

    Returns
    -------

    """

    max_d = 0  # TODO: attempt to define a cut point based on values
    linkage = 'ward'  # TODO: Attempt to receive this as arg?

    Z = scipy.cluster.hierarchy.linkage(Data, linkage)
    annotate_above = 10   # truncate clusters?

    plt.figure()
    ddata = scipy.cluster.hierarchy.dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90., leaf_font_size=12., show_contracted=True, color_threshold=max_d)

    # plt.suptitle(('Hierarchical Clustering Dendrogram ' + linkage + '- ' + dist), fontsize=14, fontweight='bold')
    # plt.title("Linkage: " + linkage + ", Distance Metric: " + dist)
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        if y > annotate_above:
            plt.plot(x, y, 'o', c=c)
            plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords='offset points', va='top', ha='center')
    if max_d:
        plt.axhline(y=max_d, c='k')

    plt.subplots_adjust(bottom=0.15)
    propheticus.shared.Utils.saveImage(os.path.join(Context.save_items_path, 'hierarchical_clustering'), Context.generated_files_base_name + "_dendogram.png")

    if Context.display_visuals is True:
        propheticus.shared.Utils.showImage()

    plt.close()