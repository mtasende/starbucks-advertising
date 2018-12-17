import numpy as np
import pandas as pd
import datetime as dt


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
