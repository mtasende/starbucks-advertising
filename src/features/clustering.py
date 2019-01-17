""" Contains the functions implemented to cluster, and visualize the custers. """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import ward, fcluster
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from time import time
import os

import src.data.success_dataset as sd
import src.data.preprocessing as pp
from src import DATA_INTERIM, DATA_PROCESSED

DIST_12 = 40.61273277762122
DIST_3D_3 = 99.68755217120427
DIST_3D_9 = 45.436989981397055
DIST_3D_19 = 23.16997229826248


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


def create_cluster_feats_4d(static_dataset_path=os.path.join(DATA_INTERIM, 'static_data.pkl'),
                            output_path=os.path.join(DATA_PROCESSED, 'static_cluster1.pkl'),
                            save=True):
    """
    Adds the features created by clustering for the selected 4D cases (age, income, gender, memeber_since_epoch).
    The features to add are: kmeans_8, ward_12 and dbscan_10.
    Args:
        static_dataset_path(str): The path to the static dataset to be taken as the initial data.
        output_path(str): The path to save the new dataset.
        save(boolean): Whether to save the new static dataset.
    Returns:
        static_cluster1_dataset(dataframe): The same as the static dataset but with the features added into new
            columns.
        X_train_r(dataframe): X_train (as obtained from time-split with the input static data) with the new features.
        X_test_r(dataframe): X_test (as obtained from time-split with the input static data) with the new features.
        y_train(pd.Series): y_train as obtained from time-split with the input static data.
        y_test(pd.Series): y_test as obtained from time-split with the input static data.
    """
    # Get the data
    X_train, X_test, y_train, y_test, encoder = sd.get_success_data(basic_dataset_path=static_dataset_path,
                                                                    drop_time=False,
                                                                    anon=False)

    # Encode and filter relevant features
    customer_feats = ['age', 'gender', 'income', 'missing_demographics',
                      'member_epoch_days']

    X_train_t = encoder.fit_transform(X_train)
    X_train_t = X_train_t[customer_feats]
    X_test_t = encoder.transform(X_test)
    X_test_t = X_test_t[customer_feats]

    # Drop duplicates and missing data
    X_train_t = X_train_t.dropna().drop_duplicates()
    X_test_t = X_test_t.dropna().drop_duplicates()

    # Keep a copy with the original demographics
    X_train_o = pp.gender_decode(X_train_t.copy())
    X_test_o = pp.gender_decode(X_test_t.copy())

    # Drop the irrelevant column
    X_train_t = X_train_t.drop('missing_demographics', axis=1)
    X_test_t = X_test_t.drop('missing_demographics', axis=1)

    # Normalize
    scaler = StandardScaler()
    scaler.fit(X_train_t)

    X_train_t = pd.DataFrame(scaler.transform(X_train_t),
                             index=X_train_t.index,
                             columns=X_train_t.columns)
    X_test_t = pd.DataFrame(scaler.transform(X_test_t),
                            index=X_test_t.index,
                            columns=X_test_t.columns)

    # Add the clustering labels
    # K-Means (k = 8)
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=2018)
    kmeans.fit(X_train_t)
    X_train_o['kmeans_8'] = kmeans.predict(X_train_t)
    X_test_o['kmeans_8'] = kmeans.predict(X_test_t)

    # Ward 12 clusters
    linkage_matrix = ward(X_train_t)
    dist_12 = DIST_12
    X_train_o['ward_12'] = fcluster(linkage_matrix, dist_12, criterion='distance')
    # Use KNN to determine the test clusters
    knn_ward = KNeighborsClassifier(n_neighbors=5)
    knn_ward.fit(X_train_t, X_train_o['ward_12'])
    X_test_o['ward_12'] = knn_ward.predict(X_test_t)

    # DBSCAN eps=0.3, min_samples=20, 10 clusters
    eps = 0.3
    min_samples = 20
    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    dbs.fit(X_train_t)
    X_train_o['dbscan_10'] = dbs.labels_
    # Use KNN to determine the test clusters
    knn_dbscan = KNeighborsClassifier(n_neighbors=5)
    knn_dbscan.fit(X_train_t, X_train_o['dbscan_10'])
    X_test_o['dbscan_10'] = knn_dbscan.predict(X_test_t)

    # Merge with the original datsets
    X_train_r = X_train.merge(X_train_o, on=customer_feats, how='left')
    X_test_r = X_test.merge(X_test_o, on=customer_feats, how='left')

    # Join the new features with the old static dataset
    static_cluster1 = pd.concat([X_train_r.sort_values(by='time'), X_test_r.sort_values(by='time')])
    old_static = pd.read_pickle(static_dataset_path)
    id_feats = ['person', 'time', 'offer_id']
    cluster_feats = ['kmeans_8', 'ward_12', 'dbscan_10']
    cluster_info = static_cluster1[id_feats + cluster_feats]
    static_cluster1_dataset = old_static.merge(cluster_info, on=id_feats)

    # Save the new static dataset
    if save:
        static_cluster1_dataset.to_pickle(output_path)

    return static_cluster1_dataset, X_train_r, X_test_r, y_train, y_test


def create_cluster_feats_3d(static_dataset_path=os.path.join(DATA_PROCESSED, 'static_cluster1.pkl'),
                            output_path=os.path.join(DATA_PROCESSED, 'static_cluster3d.pkl'),
                            save=True):
    """
    Adds the features created by clustering for the selected 3D cases (age, income, memeber_since_epoch).
    The features to add are: 3d_kmeans_3, 3d_ward_3, 3d_ward_19, 3d_gmm_3, 3d_gmm_16, 3d_dbscan_02_20, 3d_dbscan_05_100
    Args:
        static_dataset_path(str): The path to the static dataset to be taken as the initial data.
        output_path(str): The path to save the new dataset.
        save(boolean): Whether to save the new static dataset.
    Returns:
        static_cluster3d_dataset(dataframe): The same as the static dataset but with the features added into new
            columns.
        X_train_r(dataframe): X_train (as obtained from time-split with the input static data) with the new features.
        X_test_r(dataframe): X_test (as obtained from time-split with the input static data) with the new features.
        y_train(pd.Series): y_train as obtained from time-split with the input static data.
        y_test(pd.Series): y_test as obtained from time-split with the input static data.
    """
    # Get the data
    X_train, X_test, y_train, y_test, encoder = sd.get_success_data(
        basic_dataset_path=static_dataset_path,
        drop_time=False,
        anon=False)

    # Encode and filter relevant features
    customer_feats = ['age', 'income', 'missing_demographics',
                      'member_epoch_days']

    X_train_t = encoder.fit_transform(X_train)
    X_train_t = X_train_t[customer_feats]
    X_test_t = encoder.transform(X_test)
    X_test_t = X_test_t[customer_feats]

    # Drop duplicates and missing data
    X_train_t = X_train_t.dropna().drop_duplicates()
    X_test_t = X_test_t.dropna().drop_duplicates()

    # Keep a copy with the original demographics
    X_train_o = X_train_t.copy()
    X_test_o = X_test_t.copy()

    # Drop the irrelevant column
    X_train_t = X_train_t.drop('missing_demographics', axis=1)
    X_test_t = X_test_t.drop('missing_demographics', axis=1)

    # Normalize
    scaler = StandardScaler()
    scaler.fit(X_train_t)

    X_train_t = pd.DataFrame(scaler.transform(X_train_t),
                             index=X_train_t.index,
                             columns=X_train_t.columns)
    X_test_t = pd.DataFrame(scaler.transform(X_test_t),
                            index=X_test_t.index,
                            columns=X_test_t.columns)

    # Add the clustering labels
    # K-Means (k = 3)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=2018)
    kmeans.fit(X_train_t)
    X_train_o['3d_kmeans_3'] = kmeans.predict(X_train_t)
    X_test_o['3d_kmeans_3'] = kmeans.predict(X_test_t)

    # Ward
    linkage_matrix = ward(X_train_t)

    # Ward 3 clusters
    n_clusters = 3
    feat_name = '3d_ward_3'
    dist = DIST_3D_3
    X_train_o[feat_name] = fcluster(linkage_matrix, dist, criterion='distance')
    # Use KNN to determine the test clusters
    knn_ward = KNeighborsClassifier(n_neighbors=5)
    knn_ward.fit(X_train_t, X_train_o[feat_name])
    X_test_o[feat_name] = knn_ward.predict(X_test_t)

    # Ward 9 clusters
    n_clusters = 9
    feat_name = '3d_ward_9'
    dist = DIST_3D_9
    X_train_o[feat_name] = fcluster(linkage_matrix, dist, criterion='distance')
    # Use KNN to determine the test clusters
    knn_ward = KNeighborsClassifier(n_neighbors=5)
    knn_ward.fit(X_train_t, X_train_o[feat_name])
    X_test_o[feat_name] = knn_ward.predict(X_test_t)

    # Ward 19 clusters
    n_clusters = 19
    feat_name = '3d_ward_19'
    dist = DIST_3D_19
    X_train_o[feat_name] = fcluster(linkage_matrix, dist, criterion='distance')
    # Use KNN to determine the test clusters
    knn_ward = KNeighborsClassifier(n_neighbors=5)
    knn_ward.fit(X_train_t, X_train_o[feat_name])
    X_test_o[feat_name] = knn_ward.predict(X_test_t)

    # GMM 3 clusters
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_train_t)
    X_train_o['3d_gmm_3'] = gmm.predict(X_train_t)
    X_test_o['3d_gmm_3'] = gmm.predict(X_test_t)

    # GMM 16 clusters
    gmm = GaussianMixture(n_components=16)
    gmm.fit(X_train_t)
    X_train_o['3d_gmm_16'] = gmm.predict(X_train_t)
    X_test_o['3d_gmm_16'] = gmm.predict(X_test_t)

    # DBSCAN eps=0.2, min_samples=20
    eps = 0.2
    min_samples = 20
    feat_name = '3d_dbscan_02_20'
    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    dbs.fit(X_train_t)
    X_train_o[feat_name] = dbs.labels_
    # Use KNN to determine the test clusters
    knn_dbscan = KNeighborsClassifier(n_neighbors=5)
    knn_dbscan.fit(X_train_t, X_train_o[feat_name])
    X_test_o[feat_name] = knn_dbscan.predict(X_test_t)

    # DBSCAN eps=0.5, min_samples=100
    eps = 0.5
    min_samples = 100
    feat_name = '3d_dbscan_05_100'
    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    dbs.fit(X_train_t)
    X_train_o[feat_name] = dbs.labels_
    # Use KNN to determine the test clusters
    knn_dbscan = KNeighborsClassifier(n_neighbors=5)
    knn_dbscan.fit(X_train_t, X_train_o[feat_name])
    X_test_o[feat_name] = knn_dbscan.predict(X_test_t)

    # Merge with the original datsets
    X_train_r = X_train.merge(X_train_o, on=customer_feats, how='left')
    X_test_r = X_test.merge(X_test_o, on=customer_feats, how='left')

    # Join the new features with the old static dataset
    cluster_feats = ['3d_kmeans_3', '3d_ward_3', '3d_ward_9', '3d_ward_19',
                     '3d_gmm_3', '3d_gmm_16', '3d_dbscan_02_20', '3d_dbscan_05_100']
    static_cluster3d = pd.concat([X_train_r.sort_values(by='time'), X_test_r.sort_values(by='time')])
    old_static = pd.read_pickle(static_dataset_path)
    id_feats = ['person', 'time', 'offer_id']
    cluster_info = static_cluster3d[id_feats + cluster_feats]
    static_cluster3d_dataset = old_static.merge(cluster_info, on=id_feats)

    # Save the new static dataset
    if save:
        static_cluster3d_dataset.to_pickle(output_path)

    return static_cluster3d_dataset, X_train_r, X_test_r, y_train, y_test
