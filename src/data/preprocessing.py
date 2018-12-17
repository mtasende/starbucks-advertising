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

    # Join them all
    data = None

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
