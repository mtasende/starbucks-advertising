import numpy as np
import pandas as pd
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

FUTURE_INFO = ['finish', 'success', 'view_time', 'viewed', 'actual_reward',
               'profit_in_duration', 'profit_until_complete',
               'spent_in_duration', 'spent_until_complete', 'completed']


def basic_preprocessing(portfolio, profile, transcript):
    """
    Perform a basic preprocessing. It encodes some features, and fills missing
    data.

    Args:
        portfolio(pd.DataFrame): Raw data with the offers information.
        profile(pd.DataFrame): Raw data with the customers information.
        transcript(pd.FataFrame): Raw data with the events information.

    Returns:
        pd.DataFrame: A preprocessed dataset, for use in ML algorithms or
          previous feature generation.
        pd.DataFrame: The portfolio dataframe, preprocessed.
    """
    portfolio = process_portfolio(portfolio)
    profile = process_profile(profile)
    transcript = process_transcript(transcript)

    data = join_data(transcript, profile, portfolio, static=False)

    return data, portfolio


def process_portfolio(portfolio):
    """ All the preprocessing needed for portfolio alone. """

    return channels_ohe(portfolio)


def process_profile(profile):
    """ All the preprocessing needed for profile alone. """
    profile.age = profile.age.replace(118, np.nan)
    profile.became_member_on = pd.to_datetime(profile.became_member_on,
                                              format='%Y%m%d')
    profile['missing_demographics'] = profile.isnull().any(axis=1).astype(int)
    profile = gender_encode(profile)
    profile = member_epoch_days(profile)

    return profile


def process_transcript(transcript):
    """ All the preprocessing needed for transcript alone. """
    return unwrap_transcript(transcript)


def channels_ohe(portfolio):
    """
    Transforms the 'channels' column of the 'portfolio' dataframe into One-Hot
    encoded columns for the possible channels.
    """
    # Get all the possible channels
    possible_channels = set()
    for c_list in portfolio.channels:
        for channel in c_list:
            possible_channels.add(channel)
    possible_channels = list(possible_channels)

    # Create the channels' columns and fill them
    for channel in possible_channels:
        portfolio['channel_' + channel] = portfolio.channels.apply(
            lambda x: int(channel in x))

    # Drop the old "channels" column
    portfolio = portfolio.drop('channels', axis=1)

    return portfolio


def gender_encode(data):
    """ Encode the gender column. F=0, M=1, O=2. """
    gender_dict = {'F': 0, 'M': 1, 'O': 2, None: np.nan}
    data.gender = data.gender.replace(gender_dict)

    return data


def gender_decode(data):
    """ Decode the gender column. F=0, M=1, O=2. """
    gender_dict_inverse = {0: 'F', 1: 'M', 2: 'O', np.nan: None}
    data.gender = data.gender.replace(gender_dict_inverse)

    return data


def member_epoch_days(profile):
    """
    Adds a column with the date transformed to 'number of days since
    1/1/1970.
    """
    profile['member_epoch_days'] = (
            profile.became_member_on - dt.datetime(1970, 1, 1)).dt.days
    return profile


def unwrap_transcript(transcript):
    """ Reads the 'value' dictionaries and adds the values as columns. """
    values_df = pd.DataFrame(transcript.value.tolist())
    values_df.offer_id.update(values_df['offer id'])
    values_df = values_df.drop('offer id', axis=1)

    return transcript.join(values_df).drop('value', axis=1)


def join_data(transcript, profile, portfolio, static=True):
    """
    Joins the three sources of data in one dataframe.
    Args:
        transcript(pandas dataframe): Contains the events (part of the raw
            data)
        profile(pandas dataframe): Contains the customer's profiles (part of
            the raw data)
        portfolio(pandas dataframe): Contains the offers (part of the raw data)
        static(boolean): If True, remove the customer and offer ids. Otherwise
            keep them for a possible time-dependent analysis.
    """
    merged_df = transcript.merge(profile, left_on='person', right_on='id',
                                 how='left').drop('id', axis=1)
    merged_df = merged_df.merge(
        portfolio.rename(columns={'reward': 'reward_t'}),
        left_on='offer_id', right_on='id', how='left').drop('id', axis=1)
    if static:
        merged_df = merged_df.drop(['person', 'offer_id'], axis=1)

    return merged_df


def split_transcript(transcript):
    """
    Separates the different kinds of events in different dataframes.
    Args:
        transcript(pd.DataFrame): Similar to the raw transcript data.

    Returns:
        received(pd.DataFrame): Contains the reception events.
        veiwed(pd.DataFrame): Contains the offer view events.
        completed(pd.DataFrame): Contains the offer completion events.
        transactions(pd.DataFrame): Contains the transactions.
    """
    received = transcript[transcript.event == 'offer received']
    viewed = transcript[transcript.event == 'offer viewed']
    completed = transcript[transcript.event == 'offer completed']
    transactions = transcript[transcript.event == 'transaction']

    return received, viewed, completed, transactions


def fill_completion(received, completed):
    """
    Looks in the records of one person and checks which offers where completed.
    A 'completed' column is set to 1 when the offer was completed. The finish
    time is also added.
    Args:
        received(pd.DataFrame): As returned from split_transcript
        completed(pd.DataFrame): As returned from split_transcript

    Returns:
        pd.DataFrame: The received dataframe with some new columns.
    """
    results = list()
    for idx, row in received.iterrows():
        record = dict()

        # Identify the record
        record['time'] = row.time
        record['offer_id'] = row.offer_id

        record['expected_finish'] = row.time + row.duration * 24
        completion = completed[(completed.offer_id == row.offer_id) &
                               (completed.time >= row.time) &
                               (completed.time <= record['expected_finish'])]
        if completion.shape[0] > 0:
            record['completed'] = 1
            record['finish'] = completion.time.iloc[0]
        else:
            record['completed'] = 0
            record['finish'] = record['expected_finish']

        results.append(record)

    return received.merge(pd.DataFrame(results), on=['time', 'offer_id'],
                          how='left')


def fill_viewed(data, viewed):
    """
    Checks if the offer was viewed in the active period of the offers.
    Also fills a column called 'success' that tracks whether an offer
    completion happened after a view.
    Args:
        data(pd.DataFrame): As returned from fill_completed
        viewed(pd.DataFrame): As returned from split_transcript

    Returns:
        pd.DataFrame: The received dataframe with some new columns.
    """
    results = list()
    for idx, row in data.iterrows():
        record = dict()

        # Identify the record
        record['time'] = row.time
        record['offer_id'] = row.offer_id

        views = viewed[(viewed.offer_id == viewed.offer_id) &
                       (viewed.time >= row.time) &
                       (viewed.time <= row.finish)]
        if views.shape[0] > 0:
            record['viewed'] = 1
            record['view_time'] = views.time.iloc[0]
            if (record['view_time'] <= row.finish) and row.completed:
                record['success'] = 1
            else:
                record['success'] = 0
        else:
            record['viewed'] = 0
            record['view_time'] = np.nan
            record['success'] = 0

        results.append(record)

    return data.merge(pd.DataFrame(results), on=['time', 'offer_id'],
                      how='left')


def fill_profits(data, transactions):
    """
    Checks if the offer was viewed in the active period of the offers.
    Args:
        data(pd.DataFrame): As returned from fill_completed
        viewed(pd.DataFrame): As returned from split_transcript

    Returns:
        pd.DataFrame: The received dataframe with some new columns.
    """
    results = list()
    for idx, row in data.iterrows():
        record = dict()

        # Identify the record
        record['time'] = row.time
        record['offer_id'] = row.offer_id

        until_complete_tr = transactions[(transactions.time >= row.time) &
                                         (transactions.time <= row.finish)]
        duration_tr = transactions[(transactions.time >= row.time) &
                                   (transactions.time <= row.expected_finish)]
        record['spent_until_complete'] = until_complete_tr.amount.sum()
        record['spent_in_duration'] = duration_tr.amount.sum()
        record['actual_reward'] = row.reward_t if row.completed == 1 else 0
        record['profit_until_complete'] = record['spent_until_complete'] - \
                                          record['actual_reward']
        record['profit_in_duration'] = record['spent_in_duration'] - \
                                       record['actual_reward']

        results.append(record)

    return data.merge(pd.DataFrame(results), on=['time', 'offer_id'],
                      how='left')


def generate_static_dataset(person_data):
    """
    Generates a dataset for one person, that contains a row for each sent
    offer, and adds some 'results' columns, like whether the offer was viewed,
    completed, when did the offer finish, how much was spent by the user while
    the offer was active, the total profit in the period, and the reward paid.
    """
    received, \
    viewed, \
    completed, \
    transactions = split_transcript(person_data)
    if received.shape[0] == 0:
        return None
    data = fill_completion(received, completed)
    data = fill_viewed(data, viewed)
    data = fill_profits(data, transactions)

    return data.drop(['event', 'reward', 'amount'], axis=1)


def anonimize_data(data):
    """
    Takes a 'static data' dataframe and converts it into an anonymized dataset.
    """
    return data.drop(['person', 'offer_id', 'became_member_on'], axis=1)


class BasicEncoder(BaseEstimator, TransformerMixin):
    """ Transforms the Basic dataset. """

    def __init__(self):
        super(BaseEstimator, self).__init__()
        self.offer_type_encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.offer_type_encoder.fit(X['offer_type'])
        return self

    def transform(self, X):
        res = X.copy()
        res['offer_type'] = self.offer_type_encoder.transform(X['offer_type'])
        return res
