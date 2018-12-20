""" Functions to evaluate models. """
from time import time
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import plot_importance


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Shows some training and test results. Returns the trained model and the
    predictions.
    Args:
        model(sklearn.BaseEstimator): The model to fit and make predictions.
        X_train(array-like): training features.
        X_test(array-like): test features.
        y_train(array-like): training target.
        y_test(array-like): test target.

    Returns:
        model(sklearn.BaseEstimator): The trained model.
        y_train_pred(array-like): The predictions for the training set.
        y_test_pred(array-like): The predictions for the test set.
    """
    # Fit the model
    tic = time()
    model.fit(X_train, y_train)
    toc = time()
    print('Training time: {} seconds.'.format(toc - tic))

    # Predict and show results
    y_train_pred = model.predict(X_train)
    print('-' * 44 + 'TRAIN RESULTS' + '-' * 44)
    print('Confusion Matrix:')
    print(confusion_matrix(y_train, y_train_pred))
    print('Classification Report:')
    print(classification_report(y_train, y_train_pred))
    print('-' * 100)

    y_test_pred = model.predict(X_test)
    print('-' * 44 + 'TEST RESULTS' + '-' * 44)
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_test_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_test_pred))
    print('-' * 100)

    print('\n' + '_' * 51)
    print('| MAIN METRIC (test f1-score): {} |'.format(
        f1_score(y_test, y_test_pred)))
    print('-' * 51)

    return model, y_train_pred, y_test_pred


def show_feat_importances(model, X_train):
    """
    Show a barplot with the feature importances for this model's estimator.
    The model is assumed to be a pipeline and the estimator name within the
    pipeline is 'estimator'.
    """
    feat_imp = np.vstack([X_train.columns,
                          model.named_steps[
                              'estimator'].feature_importances_]).T
    feat_imp = pd.DataFrame(feat_imp, columns=['feature', 'importance'])
    feat_imp = feat_imp.sort_values(by='importance').set_index('feature')
    feat_imp.plot(kind='barh')
    plt.title('Feature Importances')

    # Use built-in importance plot
    plt.figure()
    plot_importance(model.named_steps['estimator'])
