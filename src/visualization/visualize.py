""" Generic visualization functions. """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.decomposition import PCA


def show_feat_importances(model, X_train):
    """
    Show a barplot with the feature importances for this model's estimator.
    The model is assumed to be a pipeline and the estimator name within the
    pipeline is 'estimator'.
    """
    n_feats = 20
    feat_imp = np.vstack([X_train.columns,
                          model.named_steps[
                              'estimator'].feature_importances_]).T
    feat_imp = pd.DataFrame(feat_imp, columns=['feature', 'importance'])
    feat_imp = feat_imp.sort_values(by='importance').set_index('feature')
    feat_imp.iloc[-n_feats:].plot(kind='barh')
    plt.title('Feature Importances')

    # Use built-in importance plot
    plt.figure()
    plot_importance(model.named_steps['estimator'], max_num_features=n_feats)


def add_bar_labels(values):
    for i, v in enumerate(values):
            plt.text(i, v, str(v), ha='center', fontweight='bold')


def show_imputer_results(data, filled,
                         continuous=['age', 'income'],
                         discrete=['gender']):
    """ Shows some differences between a dataset and a filled dataset. """
    for feat in continuous:
        plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        data[feat].hist(bins=30)
        plt.title('{} original'.format(feat))
        plt.subplot(1, 2, 2, sharey=ax1)
        filled[feat].hist(bins=30)
        plt.title('{} filled'.format(feat))

    for feat in discrete:
        counts1 = data.gender.value_counts(dropna=False)
        counts2 = filled.gender.value_counts(dropna=False)
        plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        counts1.plot(kind='bar')
        plt.title('{} original'.format(feat))
        add_bar_labels(counts1)
        plt.subplot(1, 2, 2, sharey=ax1)
        counts2.plot(kind='bar')
        plt.title('{} filled'.format(feat))
        add_bar_labels(counts2)


def pca_visualize(X, **kwargs):
    """ Applies PCA to get 2-D data and make a scatter plot."""
    extractor = PCA(n_components=2)
    X_pca = extractor.fit_transform(X)

    print('Explained variance ratio for the first two components: {}'.format(
        extractor.explained_variance_ratio_.sum()))

    plt.scatter(X_pca[:, 0], X_pca[:, 1], **kwargs)
    plt.title('PCA scatter plot')
    plt.xlabel('PCA 1')
    _ = plt.ylabel('PCA 2')


def pca_visualize_clusters(X, cluster):
    """ Visualize all the clusters using PCA. """
    for c in np.unique(cluster):
        pca_visualize(X[cluster == c], label='cluster {}'.format(c))
    plt.legend()
