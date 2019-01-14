"""
Functions to create and use the 'profit_10_days' dataset: that is,
a dataset with information about the profits that generated a customer in the 10 days
after an offer is presented to them.
"""
import numpy as np
import pandas as pd
import os
import src.data.preprocessing as pp
from src import DATA_PROCESSED
from sklearn.preprocessing import LabelEncoder


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
    Transforms the Basic dataset. Adds the encoding for the offer choice or
    other custom features (may be offer_id for example). """

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
    """ Fill the 'null' offer data when an offer was not viewed"""

    data.loc[data['viewed'] == 0, 'offer_id'] = 'no_offer'
    z_cols = ['difficulty',
              'duration',
              'reward_t',
              'channel_web',
              'channel_mobile',
              'channel_email',
              'channel_social',
              'expected_finish']
    data.loc[data.offer_id == 'no_offer', data.columns.isin(z_cols)] = 0
    data.loc[data.offer_id == 'no_offer', 'offer_type'] = 'no_offer'

    return data


def get_profit_10_days_data(basic_dataset_path=os.path.join(DATA_PROCESSED, 'static_spent_10_days.pkl'),
                            train_times=(0, 168),
                            test_times=(408),
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
            dataset
        time_limit(int): The limit to split the train and test sets.
        drop_time(boolean): Whether to drop the absolute time dependent
            features.
        anon_person(boolean): Whether to drop unique identifiers to customers.
        anon_offer(boolean): Whether to drop unique identifiers to offers.

    Returns:
        X_train(pd.DataFrame): The training dataset.
        X_test(pd.DataFrame): The test dataset.
        y_train(pd.Series): The training target.
        y_test(pd.Series): The test target.
        BasicEncoderProfits: An encoder to use in an ML pipeline.
    """
    data = pd.read_pickle(basic_dataset_path)

    custom_features = ['offer_id']
    if fill_null:
        data = fill_null_offer(data)
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
        X_train = X_train.drop('time', axis=1)
        X_test = X_test.drop('time', axis=1)

    encoder = encoder = BasicEncoderProfits(custom_features=custom_features)

    return X_train, X_test, y_train, y_test, encoder


def predict_profit_with_offer(view_model, profit_model, data, offer, drop_offer_id=False):
    """ Predicts how much will be the profit in 10 days for a given an offer. """
    samples = data.copy()
    if drop_offer_id:
        std_offer = offer.drop('id').rename(index={'reward': 'reward_t'})
    else:
        std_offer = offer.rename(index={'reward': 'reward_t', 'id': 'offer_id'})
    samples.loc[:, std_offer.index] = np.repeat(std_offer.values.reshape(1, -1), samples.shape[0], axis=0)

    if offer.offer_type == 'no_offer':
        vis_probas = np.ones(samples.shape[0])
    else:
        vis_probas = view_model.predict_proba(samples)[:, 1]

    profit_pred = profit_model.predict(samples)

    y_pred = vis_probas * profit_pred

    return pd.Series(y_pred, name=offer.id).T


def choose_offer(view_model, profit_model, X, portfolio, add_null_offer=True):
    """
    Given a model and a features dataframe it returns the
    model that maximizes the model predictions.
    """
    complete_portfolio = portfolio.copy()

    # Add the null offer
    if add_null_offer:
        null_offer = pd.Series([0, 0, 'no_offer', 'no_offer', 0, 0, 0, 0, 0],
                               index=complete_portfolio.columns,
                               name=complete_portfolio.shape[0])
        complete_portfolio = complete_portfolio.append(null_offer)

    res = complete_portfolio.apply(
        lambda x: predict_profit_with_offer(view_model, profit_model, X, x), axis=1).T
    res.columns = complete_portfolio.id

    return res.idxmax(axis=1), res


