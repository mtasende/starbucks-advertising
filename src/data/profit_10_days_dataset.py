"""
Functions to create and use the 'profit_10_days' dataset: that is,
a dataset with information about the profits that generated a customer in the 10 days
after an offer is presented to them.
"""
import numpy as np
import pandas as pd
import os
import src.data.preprocessing as pp
import src.data.missing_data as md
from src.data import DATA_PROCESSED
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
from tqdm import tqdm

VIEWCOL_LABEL = 'viewcol'
Z_COLS = ['difficulty',
          'duration',
          'reward_t',
          'channel_web',
          'channel_mobile',
          'channel_email',
          'channel_social']
PROFIT_COLS = Z_COLS + ['offer_id', 'offer_type']
Z_VIEW_COLS = ['{}_{}'.format(col, VIEWCOL_LABEL) for col in Z_COLS]
VIEW_COLS = Z_VIEW_COLS + ['offer_id_{}'.format(VIEWCOL_LABEL),
                           'offer_type_{}'.format(VIEWCOL_LABEL)]


def get_offers_ts(user_received, portfolio, data, delta = 24 * 10, viewed=False):
    """
    Given the received sequence for a user, this function returns a time series
    dataframe containing a 1 in a 10 days period after receiving an offer.
    It can filter out the non-viewed offers.
    Args:
        user_received(dataframe): The received offers for one user.
        portfolio(dataframe): The original portfolio of offers (just to get the offer ids).
        data(dataframe): The original transcript (just to get the time values).
        delta(int): The period of relevance for an offer.
        viewed(boolean): Whether to show only the offers that were viewed.
    """
    offers = portfolio.id.values
    offer_ts = pd.DataFrame(np.zeros((data.time.nunique(), len(offers))),
                            index=data.time.unique(), columns=offers)
    for i, row in user_received.iterrows():
        if viewed:
            if row.viewed:
                offer_ts.loc[row.time: row.time + delta, row.offer_id] = 1
        else:
            offer_ts.loc[row.time: row.time + delta, row.offer_id] = 1
    # Fill the "no-offer" column
    offer_ts['no_offer'] = (offer_ts.sum(axis=1) == 0).astype(int)
    return offer_ts


class BasicEncoderProfits(pp.BasicEncoder):
    """
    Transforms the Basic dataset. Adds the possibility of encoding other custom features, like offer_id,
    for example.
    Args:
        custom_features(list): Names of the custom features to label-encode.
    """

    def __init__(self, custom_features=list()):
        super().__init__()
        self.custom_encoders = {feat: LabelEncoder() for feat in custom_features}

    def fit(self, X, y=None):
        """ Get the encodings for the offer choice. """
        super().fit(X, y)
        for feat, encoder in self.custom_encoders.items():
            encoder.fit(X[feat])
        return self

    def transform(self, X):
        """ Encode offer types and gender, and all the custom features. """
        res = super().transform(X)
        for feat, encoder in self.custom_encoders.items():
            res[feat] = encoder.transform(X[feat])
        return res

    def inverse_transform(self, X):
        """ Transform back to the original encoding. """
        res = super().inverse_transform(X)
        for feat, encoder in self.custom_encoders.items():
            res[feat] = encoder.inverse_transform(X[feat])
        return res


def get_spent_days_static(static_data, preprocessed_data, days=10 * 24):
    """
    Get a static data version of the profit N days dataset.
    Calculates the total money spent for each customer, in the 10 days after
    an offer is shown.
    Args:
        static_data(dataframe): The 'offer success' version of the static data.
        preprocessed_data(dataframe): the result of 'basic_preprocessing'.
        days(int): The number of days to calculate the profits.
    """
    received, viewed, completed, transactions = pp.split_transcript(preprocessed_data)

    results = list()
    for idx, row in tqdm(list(static_data.iterrows())):
        record = dict()

        # Id of the record
        record['person'] = row.person
        record['time'] = row.time
        record['offer_id'] = row.offer_id

        record['spent_10_days'] = transactions[(transactions.person == row.person) &
                                               (transactions.time > row.time) &
                                               (transactions.time <= row.time + days)
                                               ].amount.sum()
        results.append(record)

    return static_data.merge(pd.DataFrame(results),
                             on=['person', 'time', 'offer_id'], how='left')


def fill_null_offer(data):
    """
    Fill the 'null' offer data when an offer was not viewed.
    The 'viewcol' features are generated for the views predictor. That model
    doesn't consider the null offer because it predicts the views themselves.
    Args:
        data(dataframe): A dataframe with sent offers (like the result from 'get_spent_days_static').

    Returns:
        data(dataframe): Like the input but with added / modified columns.
        view_cols(list): The names of the columns for the 'views' estimator.
        profit_cols(list): The names of the columns for the 'profits' estimator.
    """
    viewcol_label = 'viewcol'

    z_cols = ['difficulty',
              'duration',
              'reward_t',
              'channel_web',
              'channel_mobile',
              'channel_email',
              'channel_social']
    profit_cols = z_cols + ['offer_id', 'offer_type']
    z_view_cols = ['{}_{}'.format(col, viewcol_label) for col in z_cols]
    view_cols = z_view_cols + ['offer_id_{}'.format(viewcol_label),
                               'offer_type_{}'.format(viewcol_label)]

    # Fill the "non-view" cols
    data[view_cols] = data[profit_cols].copy()
    data.loc[data['viewed'] == 0, 'offer_id'] = 'no_offer'
    data.loc[data['viewed'] == 0, 'offer_type'] = 'no_offer'
    data.loc[data['viewed'] == 0, data.columns.isin(z_cols)] = 0

    return data, view_cols, profit_cols


def split_view_profit(X, view_cols, profit_cols):
    """
    Splits a features dataset in the features that are used by
    the views predictor and the features that are used by the profits
    predictor.
    Args:
        X(dataframe): Input of the estimator. Each sample is a sent offer. Contains 'view' columns and 'profit'
            columns. Like the result from 'fill_null_offer'.
        view_cols(list): The names of the columns for the 'views' estimator.
        profit_cols(list): The names of the columns for the 'profits' estimator.
    """
    X_view = X.drop(profit_cols, axis=1).copy()
    X_view = X_view.rename(columns={v: p for v, p in zip(view_cols, profit_cols)})
    X_profit = X.drop(view_cols, axis=1).copy()

    return X_view, X_profit


def get_profit_10_days_data(basic_dataset_path=os.path.join(DATA_PROCESSED, 'static_spent_10_days.pkl'),
                            train_times=(0, 168),
                            test_times=(408,),
                            drop_time=True,
                            anon_person=True,
                            drop_offer_id=True,
                            fill_null=True,
                            target='profit_10_days'):
    """
    Generates the dataset to predict the profits in 10 days for each offer.
    The profits are calculated as the money spent minus the paid reward (if any).
    Args:
        basic_dataset_path(str): The path to the pickle containing the basic
            dataset.
        train_times(list): A list (or tuple) with the time values for the training set.
        test_times(list): A list (or tuple) with the time values for the test set.
        drop_time(boolean): Whether to drop or not the absolute time dependent features.
        anon_person(boolean): Whether to drop or not unique identifiers to customers.
        drop_offer_id(boolean): Whether to drop or not the 'offer_id' feature.
        target(list or str): The target feature name (typically, 'viewed' or 'profit_10_days', or both).

    Returns:
        X_train(pd.DataFrame): The training dataset.
        X_test(pd.DataFrame): The test dataset.
        y_train(pd.Series): The training target.
        y_test(pd.Series): The test target.
        encoder(BasicEncoderProfits): An encoder to use in an ML pipeline.
        view_cols(list): The names of the columns for the 'views' estimator.
        profit_cols(list): The names of the columns for the 'profits' estimator.
    """
    data = pd.read_pickle(basic_dataset_path)

    custom_features = ['offer_id']
    view_cols, profit_cols = (None, None)
    if fill_null:
        data, view_cols, profit_cols = fill_null_offer(data)
    if anon_person:
        data = data.drop('person', axis=1)
    if drop_offer_id:
        data = data.drop('offer_id', axis=1)
        custom_features.remove('offer_id')
    data['profit_10_days'] = data.spent_10_days - data.actual_reward
    data = data.drop(['became_member_on', 'spent_10_days'], axis=1)

    X = data.drop(pp.FUTURE_INFO + ['profit_10_days'], axis=1)
    y = data[target]

    # Split the train-test data
    X_train = X[X.time.isin(train_times)]
    X_test = X[X.time.isin(test_times)]
    y_train = y[X.time.isin(train_times)]
    y_test = y[X.time.isin(test_times)]
    if drop_time:
        X_train = pp.drop_time_dependent(X_train)
        X_test = pp.drop_time_dependent(X_test)

    encoder = BasicEncoderProfits(custom_features=custom_features)

    return X_train, X_test, y_train, y_test, encoder, view_cols, profit_cols


def predict_profit_with_offer(model, data, offer, drop_offer_id=False):
    """
    Predicts how much will be the profit in 10 days for a given an offer.
    Args:
        model(ProfitsPredictor): The model to estimate the profits in 10 days.
        data(dataframe): A static dataset, like the result of 'get_profit_10_days_data' (X_train, X_test, ...).
        offer(pd.Series): One row of the portfolio dataframe.
        drop_offer_id(boolean): Whether to drop or not the 'offer_id' column.

    Returns:
        predictions(pd.Series): The predicted profits for the offer and for each sample in 'data'.
    """
    samples = data.copy()
    if drop_offer_id:
        std_offer = offer.drop('id').rename(index={'reward': 'reward_t'})
    else:
        std_offer = offer.rename(index={'reward': 'reward_t', 'id': 'offer_id'}).sort_index()
    view_offer = std_offer.rename(index={old: '{}_viewcol'.format(old) for old in std_offer.index})
    samples.loc[:, sorted(VIEW_COLS)] = np.repeat(view_offer.values.reshape(1, -1), samples.shape[0], axis=0)
    samples.loc[:, sorted(PROFIT_COLS)] = np.repeat(std_offer.values.reshape(1, -1), samples.shape[0], axis=0)

    if offer.offer_type == 'no_offer':
        y_pred = model.predict_profit_alone(samples)
    else:
        y_pred = model.predict(samples)

    return pd.Series(y_pred, name=offer.id).T


def choose_offer(model, X, portfolio, add_null_offer=True):
    """
    Given a model and a features dataframe it returns the offers that maximize the model predictions.
    It calls 'predict_profit_with_offer' for each offer in portfolio, and selects the one with the largest
    predicted profit.
    Args:
        model(ProfitsPredictor): The model to estimate the profits in 10 days.
        X(dataframe): A static dataset, like the result of 'get_profit_10_days_data' (X_train, X_test, ...).
        portfolio(dataframe): The processed portfolio dataframe. Like the result from 'pp.basic_preprocessing'.
        add_null_offer(boolean): Whether to add the null offer (no offer at all) to the portfolio.

    Returns:
        pd.Series: A series with the offer_id of the selected (best) offer for each sample of X.
    """
    complete_portfolio = portfolio.copy()

    # Add the null offer
    if add_null_offer:
        null_offer = pd.Series([0, 0, 'no_offer', 'no_offer', 0, 0, 0, 0, 0],
                               index=complete_portfolio.columns,
                               name=complete_portfolio.shape[0])
        complete_portfolio = complete_portfolio.append(null_offer)

    res = complete_portfolio.apply(
        lambda x: predict_profit_with_offer(model, X, x), axis=1).T
    res.columns = complete_portfolio.id

    return res.idxmax(axis=1), res


class ProfitsPredictor(BaseEstimator, RegressorMixin):
    """
    Predicts the profits in 10 days for any given offer to a specific customer. It uses to models (sklearn Pipelines):
    A classifier to predict the probability of an offer being viewed, and a regressor to predict the expected profit
    that a customer will generate in the 10 days following the reception of an offer. Both results are combined to give
    a total expected profit returns in 10 days, after the reception of an offer.
    Args:
        encoder(BasicEncoderProfits): An encoder to use in an ML pipeline.
        view_cols(list): The names of the columns for the 'views' estimator.
        profit_cols(list): The names of the columns for the 'profits' estimator.
    """

    def __init__(self, encoder=None, view_cols=VIEW_COLS, profit_cols=PROFIT_COLS, **kwargs):
        super().__init__(**kwargs)
        self.view_cols = view_cols
        self.profit_cols = profit_cols
        if encoder is None:
            self.encoder = BasicEncoderProfits()
        else:
            self.encoder = encoder

        # Create the models
        self.views_model = Pipeline([
            ('encoder', self.encoder),
            ('imputer', md.BasicImputer()),
            ('estimator', XGBClassifier(max_depth=7, n_estimators=200, n_jobs=-1,
                                        random_state=2018))
        ])
        self.profits_model = Pipeline([
            ('encoder', self.encoder),
            ('imputer', md.BasicImputer()),
            ('estimator', XGBRegressor(max_depth=4, n_estimators=200, n_jobs=-1,
                                       random_state=2018))
        ])

    def fit(self, X, y):
        """ Fits all the models. """
        y_views = y.iloc[:, 0]
        y_profits = y.iloc[:, 1]
        X_views, X_profits = split_view_profit(X,
                                                   self.view_cols,
                                                   self.profit_cols)
        self.views_model.fit(X_views, y_views)
        self.profits_model.fit(X_profits, y_profits)

        return self

    def predict(self, X):
        """ Gets the predictions from all models and calculates the final prediction."""
        X_views, X_profits = split_view_profit(X,
                                               self.view_cols,
                                               self.profit_cols)
        vis_probas = self.views_model.predict_proba(X_views)[:, 1]
        profits_pred = self.profits_model.predict(X_profits)

        return vis_probas * profits_pred

    def predict_profit_alone(self, X):
        """
        Predicts the profits as if the offer was already seen. It is useful for the
        'no offer' case, that is 'always seen'.
        """
        X_views, X_profits = split_view_profit(X,
                                               self.view_cols,
                                               self.profit_cols)

        return self.profits_model.predict(X_profits)
