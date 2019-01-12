""" Functions to create lagged features. """
import numpy as np
import pandas as pd


def fill_one_lagged_success(user_data, current_time, portfolio):
    """
    For a given time, and a given user, it counts how many times each offer was shown,
    and how many of those were a success (the rate of success could be easily calculated
    afterwards).
    offer_id_n: keeps track of how many times the offer was shown
    offer_id_success: keeps track of how many times the offer was successful
    """
    feat_names = ['offer_type', 'duration', 'difficulty', 'reward']
    ohe_feats = ['channel_web', 'channel_email', 'channel_social', 'channel_mobile']

    # Some type conversion (data may have NaNs and converts to float)
    portfolio_t = portfolio.copy()
    portfolio_t[['difficulty', 'duration', 'reward']] = portfolio_t[
        ['difficulty', 'duration', 'reward']].astype(float)

    # Create the results containers
    feats = portfolio_t.id.tolist()
    for feat_name in feat_names:
        feats += [feat_name + '_{}'.format(d)
                  for d in portfolio_t[feat_name].unique().tolist()]
    label_feats = np.setdiff1d(feat_names, ['reward']).tolist()
    feats += ohe_feats

    shown = {'{}_n'.format(offer): 0 for offer in feats}
    success = {'{}_success'.format(offer): 0 for offer in feats}
    res = {**shown, **success}

    old_offers = user_data[user_data.time < current_time]
    for i, row in old_offers.iterrows():
        res['{}_n'.format(row.offer_id)] += 1
        for feat_name in label_feats:
            res[feat_name + '_{}_n'.format(row[feat_name])] += 1
        res['reward_{}_n'.format(row['reward_t'])] += 1
        for feat_name in ohe_feats:
            if row[feat_name] == 1:
                res['{}_n'.format(feat_name)] += 1

        if row.success == 1:
            res['{}_success'.format(row.offer_id)] += 1
            for feat_name in label_feats:
                res[feat_name + '_{}_success'.format(row[feat_name])] += 1
            res['reward_{}_success'.format(row['reward_t'])] += 1
            for feat_name in ohe_feats:
                if row[feat_name] == 1:
                    res['{}_success'.format(feat_name)] += 1

    return pd.Series(res)


def fill_user_lagged_success(user_data, portfolio):
    """ Fills the lagged success features for all the records in one customer. """
    return user_data.join(user_data.apply(
        lambda x: fill_one_lagged_success(user_data, x.time, portfolio), axis=1))


def fill_lagged_success(data, portfolio):
    """ Fills the lagged success features for the entire dataset. """
    filled =  data.groupby('person').apply(
        lambda x: fill_user_lagged_success(x, portfolio))

    # Fill the ratios of success / shown
    success_cols = filled.columns.str.extract(
        '(.*)_success').dropna().values.flatten().tolist()
    for col in success_cols:
        filled['{}_success_ratio'.format(col)] = filled['{}_success'.format(col)] / (filled['{}_n'.format(col)] + 1e-5)

    return filled




