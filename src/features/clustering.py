""" Contains the functions implemented to cluster, and visualize the custers. """
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from time import time

customer_feats = ['age', 'gender', 'income', 'missing_demographics',
                  'member_epoch_days']
offer_feats = ['difficulty', 'duration', 'offer_type', 'reward_t',
               'channel_web', 'channel_social', 'channel_mobile',
               'channel_email']


def kmeans_error(X, method, cluster_labels):
    """ Index function for K-Means that returns the SSE"""
    return method.inertia_


def validate_clustering(X, clustering_algo, params, index_fun, n_clust_name):
    """
    Get the Silhouette score and one custom index, and plot the results.
    Args:
        X(array-like): The data to cluster.
        clustering_algo(class): The class of the clustering estimator. Must follow
            scikit-learn conventions.
        params(list of dicts): A list of kwargs to pass in the creation of the clustering
            method.
        index_fun(function): A function that calculates a custom index for the clustering.
            The heading should be index_fun(X, method, cluster_labels) and return a number.
        n_clust_name(str): The name of the parameter that represents the number of clusters.
            If None is given, there will be no plots.
    """
    silhouette = list()
    error = list()
    for i, param_set in enumerate(params):
        tic = time()
        method = clustering_algo(**param_set)
        labels = method.fit_predict(X)
        try:
            silhouette.append(silhouette_score(X, labels))
        except ValueError:
            silhouette.append(0)
        error.append(index_fun(X, method, labels))
        toc = time()
        print('Algorithm {} of {} finished in {} seconds.'.format(
            i + 1, len(params), (toc - tic)))

    best_silhouette_params = params[np.argmax(silhouette)]
    print('The best Silhouette score is for {}, and its value is: {}'.format(
        best_silhouette_params, max(silhouette)))
    print('The error for {} is: {}'.format(
        best_silhouette_params, error[np.argmax(silhouette)]))

    if n_clust_name is not None:
        clusters = [p[n_clust_name] for p in params]
        plt.plot(clusters, silhouette)
        plt.title('Silhouette score')
        plt.vlines(best_silhouette_params[n_clust_name], min(silhouette), max(silhouette), 'r')

        plt.figure()
        plt.plot(clusters, error)
        plt.title(index_fun.__name__)
        plt.vlines(best_silhouette_params[n_clust_name], min(error), max(error), 'r')

    return silhouette, error, best_silhouette_params


def gmm_aic(X, method, cluster_labels):
    """
    Index function that returns the Aikake Information Criterion Index for a
    Gaussian Mixture Model.
    """
    return method.aic(X)


def number_of_clusters(X, method, cluster_labels):
    """ Index function that returns the number of clusters for DBSCAN. """
    return len(method.labels_)
