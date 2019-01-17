"""
Functions to create and use the 'success' dataset: that is, a dataset to determine
the probability that an offer is viewed and later completed, or just viewed in
the case of informational offers.
"""
import pandas as pd
import os
import src.data.preprocessing as pp
from src.data import DATA_INTERIM


def get_success_data(
        basic_dataset_path=os.path.join(DATA_INTERIM, 'static_data.pkl'),
        time_limit=450,
        informational_success=True,
        drop_time=True,
        anon=True
):
    """
    Generates the dataset to predict whether an offer was successful.
    An offer is considered successful if it is viewed and then completed. In
    the case of informational offers a visualization alone may be considered a
    success or not.
    Args:
        basic_dataset_path(str): The path to the pickle containing the basic
            dataset
        time_limit(int): The limit to split the train and test sets.
        informational_success(boolean): Whether a visualization of an
            informational offer should be considered as a success.
        drop_time(boolean): Whether to drop the absolute time dependent
            features.
        anon(boolean): Whether to drop unique identifiers to customers and
            offers.


    Returns:
        X_train(pd.DataFrame): The training dataset.
        X_test(pd.DataFrame): The test dataset.
        y_train(pd.Series): The training target.
        y_test(pd.Series): The test target.
        BasicEncoder: An encoder to use in an ML pipeline.
    """

    data = pd.read_pickle(basic_dataset_path)
    if anon:
        data = pp.anonimize_data(data)
    if informational_success:
        data.loc[data.offer_type == 'informational', 'success'] = data.loc[
            data.offer_type == 'informational', 'viewed']

    X = data.drop(pp.FUTURE_INFO, axis=1)
    y = data['success']
    X_train, X_test, y_train, y_test = time_split(X, y, time_limit,
                                                  drop_time=drop_time)

    encoder = pp.BasicEncoder()

    return X_train, X_test, y_train, y_test, encoder


def time_split(X, y, time_limit, drop_time=True):
    """
    Splits the features and targets in time. Drops the time dependent features
    if 'drop_time' is True.
    """
    X_train = X[X.time < time_limit]
    y_train = y[X.time < time_limit]
    X_test = X[X.time >= time_limit]
    y_test = y[X.time >= time_limit]

    # Drop the columns that depend on absolute time
    if drop_time:
        X_train = pp.drop_time_dependent(X_train)
        X_test = pp.drop_time_dependent(X_test)

    return X_train, X_test, y_train, y_test
