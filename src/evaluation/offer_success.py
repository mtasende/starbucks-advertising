"""
Contains functions and classes that generate validations sets, and validate
models for the 'offer success' problem.
"""
import src.data.success_dataset as sd
import src.utils as utils
import src.data.preprocessing as pp
from time import time
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_time_split_val(val_time=370, **kwargs):
    """
    Returns all the datasets necessary to perform a time-split validation.
    Args:
        val_time(int): The time to make the validation split.
        kwargs(dict): Arguments to be passed to inner functions.

    Returns:
        X_train(pd.DataFrame): Training features.
        X_val(pd.DataFrame): Validation features.
        X_test(pd.DataFrame): Test features.
        X_train_val(pd.DataFrame): Training + Validation features, to use when
            testing.
        y_train(pd.Series): Training target values.
        y_val(pd.Series): Validation target values.
        y_test(pd.Series): Test target values.
        y_train_val(pd.Series): Training + Validation target values, to use
        when testing.
    """

    fun_kwargs = utils.filter_args(sd.get_success_data, kwargs)
    X_train_val, \
    X_test, \
    y_train_val, \
    y_test, \
    encoder = sd.get_success_data(drop_time=False, **fun_kwargs)
    X_test = sd.drop_time_dependent(X_test)
    X_train, X_val, y_train, y_val = sd.time_split(X_train_val, y_train_val,
                                                   val_time)
    return X_train, X_val, X_test, X_train_val, y_train, y_val, y_test, \
           y_train_val


def time_split_validation(model, val_time=370, **kwargs):
    """
    Shows some training and test results, for a time-split validation scheme.
    Returns the trained model and the predictions.
    Args:
        model(sklearn.BaseEstimator): The model to fit and make predictions.
        val_time(int): The time to make the validation split.
        kwargs(dict): Arguments to be passed to inner functions.

    Returns:
        model(sklearn.BaseEstimator): The trained model.
        y_train_pred(array-like): The predictions for the training set.
        y_val_pred(array-like): The predictions for the validation set.
    """
    X_train, X_val, X_test, X_train_val, y_train, y_val, y_test, \
           y_train_val = get_time_split_val(val_time, **kwargs)
    return evaluate_model(model, X_train, X_val, y_train, y_val)


def random_kfold_validation(model, n_splits=3):
    """
    Shows some training and validation results, for a random kfold validation
    scheme.
    Args:
        model(sklearn.BaseEstimator): The model to fit and make predictions.
        n_splits(int): The number of folds for the cross-validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2018)
    X_train_val, X_test, y_train_val, y_test, \
    encoder = sd.get_success_data(drop_time=True)

    # Train and validate for each fold
    f1_train = list()
    f1_val = list()
    i = 0
    for train_index, test_index in skf.split(X_train_val, y_train_val):
        i += 1
        print('Fold - {}'.format(i))
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[
            test_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[
            test_index]
        model, y_train_pred, y_val_pred = evaluate_model(
            model, X_train, X_val, y_train, y_val)
        f1_train.append(f1_score(y_train, y_train_pred))
        f1_val.append(f1_score(y_val, y_val_pred))

    # Show results
    print('Training F1-score: {} +- {}'.format(np.mean(f1_train),
                                               np.std(f1_train)))
    print()
    print('Validation F1-score: {} +- {}'.format(np.mean(f1_val),
                                                 2 * np.std(f1_val)))

def random_1fold_validation(model, **kwargs):
    """
    Shows some training and validation results, for a random train-val-test
    validation scheme.
    Args:
        model(sklearn.BaseEstimator): The model to fit and make predictions.
    """
    X_train_val, X_test, y_train_val, y_test, encoder = sd.get_success_data(
        drop_time=True, **kwargs)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                      y_train_val,
                                                      test_size=0.3,
                                                      random_state=2018)
    model, y_train_pred, y_val_pred = evaluate_model(model, X_train, X_val,
                                                        y_train, y_val)
    print('Training F1-score: {}'.format(f1_score(y_train, y_train_pred)))
    print()
    print('Validation F1-score: {}'.format(f1_score(y_val, y_val_pred)))


def random_1fold_cust_validation(model, **kwargs):
    """
    Shows some training and validation results, for a random train-val-test
    validation scheme. The dataset is divided by customers.
    Args:
        model(sklearn.BaseEstimator): The model to fit and make predictions.
    """
    X_train_val, X_test, y_train_val, y_test, encoder = sd.get_success_data(
        drop_time=True, anon=False, **kwargs)

    # Get random customer splits
    val_size = 0.3
    customers = X_train_val.person.unique()
    n_train = int(np.floor(customers.shape[0] * (1.0 - val_size)))
    np.random.shuffle(customers)
    X_train = X_train_val[X_train_val.person.isin(customers[:n_train])]
    X_val = X_train_val[X_train_val.person.isin(customers[n_train:])]
    y_train = y_train_val[X_train_val.person.isin(customers[:n_train])]
    y_val = y_train_val[X_train_val.person.isin(customers[n_train:])]

    # Anonimize
    X_train = pp.anonimize_data(X_train)
    X_val = pp.anonimize_data(X_val)

    # Evaluate and show results
    model, y_train_pred, y_val_pred = evaluate_model(
        model, X_train, X_val, y_train, y_val)
    print('Training F1-score: {}'.format(f1_score(y_train, y_train_pred)))
    print()
    print('Validation F1-score: {}'.format(f1_score(y_val, y_val_pred)))


def offer_success_test(model, **kwargs):
    """
    Shows some training and test results, for a time-split validation scheme.
    Args:
        model(sklearn.BaseEstimator): The model to fit and make predictions.
    """
    X_train, X_test, y_train, y_test, encoder = sd.get_success_data(**kwargs)
    model, y_train_pred, y_test_pred = evaluate_model(
        model, X_train, X_test, y_train, y_test)
    print('Training F1-score: {}'.format(f1_score(y_train, y_train_pred)))
    print()
    print('Test F1-score: {}'.format(f1_score(y_test, y_test_pred)))


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