""" Contains classes and functions to handle the missing data. """
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import numpy as np


MEMBER_DATE_FEATS = ['member_epoch_days', 'member_day', 'member_month',
                     'member_year', 'member_weekday']


class BasicImputer(BaseEstimator, TransformerMixin):
    """
    Fills the demographics missing data with medians and most frequent values.
    Args:
        fill_mode(list(str)): The names of the columns to fill missing data with
        the most frequent value (other than gender value). This is used if new features
        are added to the dataset, that have missing data.
    """

    def __init__(self, fill_mode=None):
        super(BaseEstimator, self).__init__()
        self.age_value = None
        self.income_value = None
        self.gender_value = None
        if fill_mode is None:
            self.fill_mode = list()
        else:
            self.fill_mode = fill_mode
        self.modes = dict()

    def fit(self, X, y=None):
        """ Get some medians. """
        self.age_value = np.round(X.age.median())
        self.income_value = X.income.median()
        self.gender_value = X.gender.mode().values[0]

        self.modes = {col: X[col].mode().values[0] for col in self.fill_mode}
        return self

    def transform(self, X):
        """ Encode offer types and gender """
        basic_filling = {'age': self.age_value,
                         'income': self.income_value,
                         'gender': self.gender_value}
        filling = {**basic_filling, **self.modes}
        return X.fillna(filling)


def add_date_features(data):
    """ Generates some features from the date the customer became member on,"""
    data['member_day'] = data.became_member_on.dt.day
    data['member_weekday'] = data.became_member_on.dt.weekday
    data['member_year'] = data.became_member_on.dt.year
    data['member_month'] = data.became_member_on.dt.month

    return data


class EstimatorImputer(BaseEstimator, TransformerMixin):
    """
    Fills the demographics missing data with predictions from an estimator.
    """

    def __init__(self, features=MEMBER_DATE_FEATS, keep_date_feats=True):
        super(BaseEstimator, self).__init__()
        self.features = features
        self.age_estimator = XGBRegressor(max_depth=7, n_estimators=200,
                                          random_state=2018)
        self.income_estimator = XGBRegressor(max_depth=7, n_estimators=200,
                                             random_state=2018)
        self.gender_estimator = XGBClassifier(max_depth=7, n_estimators=200,
                                              random_state=2018)
        self.keep_date_feats = keep_date_feats

    def fit(self, X, y=None):
        """ Fit the estimators """
        X = add_date_features(X)
        X_age_clean = X[~X.age.isnull()]
        self.age_estimator.fit(X_age_clean[self.features], X_age_clean.age)
        X_income_clean = X[~X.age.isnull()]
        self.income_estimator.fit(X_income_clean[self.features],
                                  X_income_clean.income)
        X_gender_clean = X[~X.age.isnull()]
        self.gender_estimator.fit(X_gender_clean[self.features],
                                  X_gender_clean.gender)
        return self

    def transform(self, X):
        """ Fill the missing data. """
        res = X.copy()
        res = add_date_features(res)

        # Fill the age values
        age_missing = res[res.age.isnull()]
        age_missing_index = res.index[res.age.isnull()]
        age_values = self.age_estimator.predict(age_missing[self.features])
        res.update(pd.DataFrame(age_values, index=age_missing_index,
                                columns=['age']))

        # Fill the income values
        income_missing = res[res.income.isnull()]
        income_missing_index = res.index[res.income.isnull()]
        income_values = self.income_estimator.predict(
            income_missing[self.features])
        res.update(pd.DataFrame(income_values, index=income_missing_index,
                                columns=['income']))

        # Fill the gender values
        gender_missing = res[res.gender.isnull()]
        gender_missing_index = res.index[res.gender.isnull()]
        gender_values = self.gender_estimator.predict(
            gender_missing[self.features])
        res.update(pd.DataFrame(gender_values, index=gender_missing_index,
                                columns=['gender']))

        if not self.keep_date_feats:
            res.drop(MEMBER_DATE_FEATS, axis=1)

        return res
